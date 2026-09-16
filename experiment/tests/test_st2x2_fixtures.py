"""Fixture propagation and replay provenance without network/model calls."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts import run_st2x2 as driver
from experiment.modeling.st2x2_domains import deployment_adapter
from experiment.tests.reference_contracts import _exact_dsl_spec
from experiment.modeling.client import ChatClient
from experiment.modeling.runner import RunSpec
from experiment.tests.test_calendar_active_cegis import CalendarMockTransport
from experiment.tests.test_model_runner import ok_response

FIXTURES = Path('analysis/g2_deployment_gate_v1/fixtures')


class FixtureDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runners = {}
        for name in ('D2_tight_quota', 'D3_gate_asymmetry'):
            cls.runners[name] = driver.build_runner('deployment', 'http://localhost:8000/v1', 'test', False,
                sandbox_config=str(FIXTURES / (name + '.json')), fixture_id=name, max_depth=24, max_states=15000)[0]

    def test_d2_d3_config_bounds_and_prompt_are_forwarded(self):
        for name, count in (('D2_tight_quota', 5440), ('D3_gate_asymmetry', 11712)):
            runner = self.runners[name]
            self.assertEqual(len(runner.states), count)
            self.assertEqual((runner.max_depth, runner.max_states), (24, 15000))
            self.assertEqual(runner.fixture_context['fixture_id'], name)
            self.assertEqual(Path(runner.config_path), (FIXTURES / (name + '.json')).resolve())
        self.assertNotEqual(deployment_adapter(self.runners['D2_tight_quota']).direct_prompt(),
                            deployment_adapter(self.runners['D3_gate_asymmetry']).direct_prompt())

    def test_invalid_identity_and_bounds_fail_before_client_creation(self):
        with patch.object(driver, 'ChatClient', side_effect=AssertionError('client must not be created')):
            for kw in ({'max_depth': 24}, {'fixture_id': 'x', 'max_states': 0},
                       {'fixture_id': 'wrong', 'sandbox_config': str(FIXTURES / 'D2_tight_quota.json')}):
                with self.assertRaises(ValueError):
                    driver.build_runner('deployment', 'http://localhost:8000/v1', 'test', False, **kw)

    def test_legacy_default_driver_matches_preserved_version(self):
        old = types.ModuleType('pre_fixture_driver')
        old.__file__ = driver.__file__
        source = Path('experiment/tests/fixtures/legacy_st2x2_driver.py').read_text()
        exec(compile(source, old.__file__, 'exec'), old.__dict__)
        for domain in ('deployment', 'taubench'):
            previous, pa, _ = old.build_runner(domain, 'http://localhost:8000/v1', 'test', False)
            current, ca, _ = driver.build_runner(domain, 'http://localhost:8000/v1', 'test', False)
            self.assertFalse(hasattr(current, 'fixture_context'))
            self.assertEqual(previous.sandbox_context(), current.sandbox_context())
            self.assertEqual(pa.direct_prompt(), ca.direct_prompt())
            spec = RunSpec(method=driver.ST2X2_PARTITIONED_STOP, seed=0)
            self.assertEqual(previous._finish(old.make_session(previous, spec, domain)).hashes,
                             current._finish(driver.make_session(current, spec, domain)).hashes)

    def args(self, tmp):
        return SimpleNamespace(domain='deployment', model='test', guided_json=False,
            state_budget=48, query_budget=4, seeds=[0], condition=driver.ST2X2_PARTITIONED_STOP,
            initial=[str(Path(tmp) / 'initial.json')], output=str(Path(tmp) / 'results'))

    def initial(self, runner, args):
        source = json.dumps(_exact_dsl_spec())
        return {'domain': 'deployment', 'model': 'test', 'seed': 0, 'source': source,
                'source_sha256': driver.sha(source), 'usage': {},
                'initial_context': driver._initial_context(runner, args)}

    def test_other_fixture_missing_context_and_corrupt_source_block_replay(self):
        runner = self.runners['D2_tight_quota']
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp)
            entries = [self.initial(self.runners['D3_gate_asymmetry'], args), self.initial(runner, args), self.initial(runner, args)]
            entries[1].pop('initial_context')
            entries[2]['source_sha256'] = 'wrong'
            for entry in entries:
                Path(args.initial[0]).write_text(json.dumps({'generated': [entry]}))
                with patch.object(driver, '_runner_from_args', return_value=(runner, deployment_adapter(runner), 'deployment')):
                    with patch.object(driver, 'make_session', side_effect=AssertionError('no session before validation')):
                        with self.assertRaises(ValueError):
                            driver.cmd_run(args)

    def test_exact_replay_two_arms_share_fixture_and_have_separate_protocol_hashes(self):
        runner = self.runners['D2_tight_quota']
        hashes = []
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = self.args(tmp)
            entry = self.initial(runner, args)
            Path(args.initial[0]).write_text(json.dumps({'generated': [entry]}))
            for method in (driver.ST2X2_PARTITIONED_STOP, driver.ST2X2_UNIFORM_STOP):
                args.condition = method
                with patch.object(driver, '_runner_from_args', return_value=(runner, deployment_adapter(runner), 'deployment')):
                    with patch.object(runner.client, 'complete', side_effect=AssertionError('no model call')):
                        driver.cmd_run(args)
                artifact = json.loads((Path(args.output) / f'{method}__seed0.json').read_text())
                self.assertEqual(artifact['fixture_context'], runner.fixture_context)
                self.assertEqual(artifact['st2x2']['initial_contract_sha256'], entry['source_sha256'])
                self.assertEqual(artifact['st2x2']['loop_model_calls'], 1)
                hashes.append(artifact['hashes']['protocol_sha256'])
            self.assertNotEqual(*hashes)

    def test_mock_initial_generation_records_context_and_prevents_overwrite(self):
        runner = self.runners['D2_tight_quota']
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = self.args(tmp)
            args.output = args.initial[0]
            transport = CalendarMockTransport([ok_response(json.dumps(_exact_dsl_spec()))])
            client = ChatClient(model='test', base_url='http://mock', allow_remote=True, transport=transport)
            with patch.object(runner, 'client', client), patch.object(driver, '_runner_from_args',
                    return_value=(runner, deployment_adapter(runner), 'deployment')):
                driver.cmd_initial(args)
                row = json.loads(Path(args.output).read_text())['generated'][0]
                self.assertEqual(row['initial_context'], driver._initial_context(runner, args))
                self.assertTrue(row['parsed'])
                self.assertEqual(row['source_sha256'], driver.sha(row['source']))
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    driver.cmd_initial(args)

    def test_custom_truncation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'truncated'):
            driver.build_runner('deployment', 'http://localhost:8000/v1', 'test', False,
                sandbox_config=str(FIXTURES / 'D2_tight_quota.json'), fixture_id='D2_tight_quota',
                max_depth=1, max_states=2)

    def test_duplicate_initial_keys_block_before_execution(self):
        runner = self.runners['D2_tight_quota']
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp)
            entry = self.initial(runner, args)
            Path(args.initial[0]).write_text(json.dumps({'generated': [entry, entry]}))
            with patch.object(driver, '_runner_from_args', return_value=(runner, deployment_adapter(runner), 'deployment')):
                with self.assertRaisesRegex(ValueError, 'duplicate initial'):
                    driver.cmd_run(args)

    def test_existing_output_with_wrong_initial_is_not_silently_skipped(self):
        runner = self.runners['D2_tight_quota']
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp)
            entry = self.initial(runner, args)
            Path(args.initial[0]).write_text(json.dumps({'generated': [entry]}))
            Path(args.output).mkdir()
            artifact = {'initial_context': driver._initial_context(runner, args),
                        'model': args.model, 'method': args.condition, 'seed': 0,
                        'st2x2': {'initial_contract_sha256': 'wrong'}}
            (Path(args.output) / f'{args.condition}__seed0.json').write_text(json.dumps(artifact))
            with patch.object(driver, '_runner_from_args', return_value=(runner, deployment_adapter(runner), 'deployment')):
                with self.assertRaisesRegex(ValueError, 'existing artifact'):
                    driver.cmd_run(args)
