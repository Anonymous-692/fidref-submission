"""Offline tests of the approved gap-fill grid and cross-project admission."""
import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_hosted_gapfill_20260926 as suite


class GapfillTests(unittest.TestCase):
    def test_grid(self):
        jobs = suite.jobs()
        self.assertEqual(len(jobs), 940)
        self.assertEqual(len({j['path'] for j in jobs}), 940)
        self.assertEqual(len({(j['model'], j['domain'], j['method']) for j in jobs}), 47)
        self.assertEqual(len({(j['model'], j['domain'], j['trial_id']) for j in jobs}), 320)
        self.assertEqual(Counter(j['model'] for j in jobs), {
            'gpt-5.6-terra': 340, 'gpt-5.6-luna': 280,
            'gpt-5.4-mini': 160, 'gpt-5.4-nano': 160})

    def test_native_fixed_pool(self):
        for d in suite.DOMAINS:
            self.assertEqual(suite.native_method(d, 'fixed_pool'),
                             'sampled_cegis_fixed' if d == 'taubench' else 'sampled_cegis')

    def test_payload_and_logical_budget(self):
        for model in suite.CELLS:
            client = suite.client_for(model, None, 'test')
            request = client.build_request([{'role': 'user', 'content': 'JSON'}],
                temperature=.2, top_p=.95, max_tokens=2048, seed=7)
            self.assertNotIn('seed', request)
            self.assertEqual(request['reasoning_effort'], 'none')
            self.assertEqual(request['service_tier'], suite.tier(model))
        self.assertEqual(suite.spec_for('deployment', 'direct', 0).budgets.query_budget, 1)
        self.assertEqual(suite.spec_for('deployment', 'self_refine', 0).budgets.query_budget, 4)

    def test_routing_shared_cost_and_replay(self):
        seen = []
        class Reply:
            status = 200
            headers = {}
            def __init__(self, request):
                self.payload = json.loads(request.data)
                seen.append((self.payload['model'], request.get_header('Authorization')))
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return json.dumps({'service_tier': self.payload['service_tier'],
                    'choices': [{'message': {'content': '{"ok":true}'}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}).encode()
        with tempfile.TemporaryDirectory() as directory:
            c = suite.RoutingCoordinator(Path(directory), 'TEST_MAIN', 'TEST_FREE', prior_reserve=0)
            with patch('urllib.request.urlopen', side_effect=lambda req, **kw: Reply(req)):
                for model in suite.CELLS:
                    suite.client_for(model, c, model).complete([{'role': 'user', 'content': 'JSON'}])
                cost = c.charged
                for model in suite.CELLS:
                    suite.client_for(model, c, model).complete([{'role': 'user', 'content': 'JSON'}])
            self.assertEqual(len(seen), 4)
            for model, auth in seen:
                self.assertEqual(auth, 'Bearer TEST_FREE' if model.startswith('gpt-5.4') else 'Bearer TEST_MAIN')
            self.assertEqual(c.charged, cost)
            self.assertAlmostEqual(cost, sum((10*suite.api.RATES[m][0] + 5*suite.api.RATES[m][1])/1e6 for m in suite.CELLS))
            self.assertIs(c.main.reservations, c.free.reservations)
            self.assertIs(c.main.lock, c.free.lock)
            self.assertEqual(c.reservations, {})
            resumed = suite.RoutingCoordinator(Path(directory), 'TEST_MAIN', 'TEST_FREE', prior_reserve=0)
            self.assertAlmostEqual(resumed.charged, cost)

    def test_shared_admission_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            c = suite.RoutingCoordinator(Path(directory), 'main', 'free', prior_reserve=0, limit=.012)
            model = 'gpt-5.4-mini'
            payload = suite.client_for(model, None, 'x').build_request(
                [{'role': 'user', 'content': 'JSON'}], temperature=.2, top_p=.95, max_tokens=128)
            c.free.reserve('first', model, payload)
            with self.assertRaises(suite.api.SuitePaused):
                c.main.reserve('second', model, payload)


if __name__ == '__main__':
    unittest.main()
