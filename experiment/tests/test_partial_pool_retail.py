import unittest
from unittest.mock import patch
from scripts.run_st2x2 import build_runner, make_session
from experiment.modeling.runner import RunSpec, Budgets, Decoding, ST2X2_UNIFORM_EXHAUST
from experiment.modeling.client import ChatResponse, Usage


class RetailPoolAccountingTests(unittest.TestCase):
    def session(self, enabled, budget=1100):
        runner, _, _ = build_runner('taubench', 'http://127.0.0.1:1/v1', 'cpu', True)
        if enabled:
            runner.g2_runtime = {'version': 'test'}
        return make_session(runner, RunSpec(method=ST2X2_UNIFORM_EXHAUST,
            decoding=Decoding(max_tokens=2048), budgets=Budgets(48, 4, budget)), 'taubench')

    def test_input_reservation_opt_in_only(self):
        for enabled, expected in ((True, 200), (False, 600)):
            session = self.session(enabled)
            session.ledger.charge_call(Usage(400, 100, 500))
            usage = Usage(400, 200, 600)
            response = ChatResponse('{}', usage, .01, 200, {}, {'usage': usage.to_dict()})
            with patch.object(session.runner.client, 'count_chat_tokens', return_value=400), \
                 patch.object(session.runner.client, 'complete', return_value=response) as call:
                session.ask('revision', 'test')
            self.assertEqual(call.call_args.kwargs['max_tokens'], expected)
            self.assertEqual(session.ledger.usage.total_tokens, 1100)

    def test_no_request_when_input_exhausts_budget(self):
        session = self.session(True, 400)
        with patch.object(session.runner.client, 'count_chat_tokens', return_value=400), \
             patch.object(session.runner.client, 'complete') as call:
            self.assertIsNone(session.ask('synthesis', 'test'))
        call.assert_not_called()
        self.assertEqual(session.stopped_because, 'token_budget_exhausted')

    def test_missing_usage_and_overrun_preserve_response(self):
        for raw, budget, stop in (({}, 1100, 'usage_unavailable'),
                                 ({'usage': Usage(400, 300, 700).to_dict()}, 600, 'token_budget_overrun')):
            session = self.session(True, budget)
            response = ChatResponse('{}', Usage(400, 300, 700), .01, 200, {}, raw)
            with patch.object(session.runner.client, 'count_chat_tokens', return_value=400), \
                 patch.object(session.runner.client, 'complete', return_value=response):
                self.assertIsNone(session.ask('synthesis', 'test'))
            self.assertEqual(session.stopped_because, stop)
            self.assertEqual(session.interactions[-1].response_text, '{}')
