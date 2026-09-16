"""G2 opt-in failure, budget and replay boundaries; no HTTP calls."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_st2x2 as driver
from experiment.modeling import g2_runtime as g2
from experiment.modeling.client import ChatResponse, Usage, TransportError
from experiment.modeling.runner import RunSpec, Budgets, Decoding
from experiment.modeling.st2x2_domains import deployment_adapter
from experiment.tests.reference_contracts import _exact_dsl_spec


class G2RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runners = {}
        for name in ('D2_tight_quota', 'D3_gate_asymmetry'):
            cls.runners[name] = driver.build_runner('deployment', 'http://localhost:1/v1', 'cpu-test', True,
                sandbox_config=f'analysis/g2_deployment_gate_v1/fixtures/{name}.json',
                fixture_id=name, max_depth=24, max_states=15000,
                replay_protocol=g2.VERSION, context_window=8192)[0]
        cls.source = json.dumps(_exact_dsl_spec())

    def args(self, tmp):
        return SimpleNamespace(domain='deployment', model='cpu-test', guided_json=True,
            state_budget=48, query_budget=4, seeds=[0], condition=driver.ST2X2_PARTITIONED_STOP,
            initial=[str(Path(tmp)/'initial.json')], output=str(Path(tmp)/'out'))

    def entry(self, runner, args, missing=False):
        row = {'domain':'deployment','model':'cpu-test','seed':0,
            'source':None if missing else self.source,
            'source_sha256':None if missing else driver.sha(self.source),
            'usage':{} if missing else {'prompt_tokens':400,'completion_tokens':100,'total_tokens':500},
            'usage_known':not missing,'initial_stop':'transport_error' if missing else None,
            'initial_error':'test failure' if missing else None,
            'initial_context':driver._initial_context(runner,args)}
        row['record_sha256'] = g2.record_digest(row)
        return row

    def execute(self, runner, args, row):
        Path(args.initial[0]).write_text(json.dumps({'generated':[row]}))
        with patch.object(driver, '_runner_from_args', return_value=(runner,deployment_adapter(runner),'deployment')):
            with patch.object(runner.client,'complete', side_effect=AssertionError('no model request')):
                with contextlib.redirect_stdout(io.StringIO()):
                    driver.cmd_run(args)
        return json.loads((Path(args.output)/f'{args.condition}__seed0.json').read_text())

    def test_both_fixtures_and_arms_charge_initial_and_sync_states(self):
        for runner in self.runners.values():
            with tempfile.TemporaryDirectory() as tmp:
                args = self.args(tmp)
                row = self.entry(runner,args)
                hashes=[]
                for method in (driver.ST2X2_PARTITIONED_STOP,driver.ST2X2_UNIFORM_STOP):
                    args.condition=method
                    out=self.execute(runner,args,row)
                    self.assertEqual(out['usage']['total_tokens'],500)
                    self.assertEqual(out['spend']['remaining_tokens'],15500)
                    self.assertEqual(out['spend']['model_calls'],1)
                    self.assertEqual(out['spend']['states_observed'],16)
                    self.assertEqual(out['spend']['states_observed'],out['st2x2']['states_observed_loop'])
                    self.assertEqual(out['spend']['sampled_states_checked'],16)
                    self.assertEqual(out['spend']['sampled_feedback_queries'],1)
                    self.assertEqual(out['spend']['oracle_feedback_queries'],0)
                    self.assertTrue(out['g2_accounting']['usage_complete'])
                    hashes.append(out['hashes']['protocol_sha256'])
                self.assertNotEqual(*hashes)

    def test_missing_initial_preserved_without_regeneration_for_both_arms(self):
        for runner in self.runners.values():
            with tempfile.TemporaryDirectory() as tmp:
                args=self.args(tmp)
                row=self.entry(runner,args,True)
                for method in (driver.ST2X2_PARTITIONED_STOP,driver.ST2X2_UNIFORM_STOP):
                    args.condition=method
                    with patch.object(driver,'make_ask', side_effect=lambda *a:lambda *b:self.fail('no redraw')):
                        out=self.execute(runner,args,row)
                    self.assertEqual(out['stopped_because'],'transport_error')
                    self.assertEqual(out['spend']['model_calls'],1)
                    self.assertEqual(out['spend']['states_observed'],0)
                    self.assertEqual(out['contract']['status'],'missing')
                    self.assertFalse(out['g2_accounting']['usage_complete'])
                    self.assertIsNone(out['g2_accounting']['initial_tokens_charged'])

    def session(self, budget=16000):
        runner=self.runners['D2_tight_quota']
        return driver.make_session(runner,RunSpec(method=driver.ST2X2_PARTITIONED_STOP,seed=0,
            decoding=Decoding(max_tokens=2048),
            budgets=Budgets(state_budget=48,query_budget=4,token_budget=budget)),'deployment')

    def response(self, prompt=400, completion=100, raw_known=True):
        usage=Usage(prompt,completion,prompt+completion)
        return ChatResponse(self.source,usage,0.01,200,{},
            {'usage':usage.to_dict()} if raw_known else {})

    def test_prompt_reservation_caps_completion_with_initial_tokens_charged(self):
        session=self.session(1100)
        session.ledger.charge_call(Usage(400,100,500))
        with patch.object(session.runner.client,'count_chat_tokens',return_value=400), \
             patch.object(session.runner.client,'complete',return_value=self.response(completion=200)) as call:
            inter=session.ask('revision_0','test')
        self.assertEqual(call.call_args.kwargs['max_tokens'],200)
        self.assertEqual(inter.requested_max_tokens,2048)
        self.assertEqual(inter.effective_max_tokens,200)
        self.assertEqual(inter.clamp_reason,'token_budget')
        self.assertEqual(session.ledger.usage.total_tokens,1100)
        self.assertEqual(session.ledger.remaining_tokens,0)

    def test_no_completion_when_input_uses_remaining_budget(self):
        session=self.session(400)
        with patch.object(session.runner.client,'count_chat_tokens',return_value=400), \
             patch.object(session.runner.client,'complete',side_effect=AssertionError('no completion')):
            self.assertIsNone(session.ask('synthesis','test'))
        self.assertEqual(session.stopped_because,'token_budget_exhausted')
        self.assertEqual(session.ledger.calls,0)
        self.assertEqual(session.interactions[-1].effective_max_tokens,0)

    def test_context_clamp_active_and_zero_room_blocks(self):
        for prompt in (7800,8000):
            session=self.session()
            with patch.object(session.runner.client,'count_chat_tokens',return_value=prompt), \
                 patch.object(session.runner.client,'complete',return_value=self.response(prompt=prompt)) as call:
                inter=session.ask('synthesis','test')
            if prompt==7800:
                self.assertEqual(call.call_args.kwargs['max_tokens'],136)
                self.assertEqual(inter.clamp_reason,'context_window')
            else:
                call.assert_not_called()
                self.assertIsNone(inter)
                self.assertEqual(session.stopped_because,'context_budget_exhausted')

    def test_missing_usage_and_actual_budget_overrun_stop_preserving_response(self):
        for response,budget,reason in [(self.response(raw_known=False),16000,'usage_unavailable'),
                                      (self.response(completion=300),600,'token_budget_overrun')]:
            session=self.session(budget)
            with patch.object(session.runner.client,'count_chat_tokens',return_value=400), \
                 patch.object(session.runner.client,'complete',return_value=response):
                self.assertIsNone(session.ask('synthesis','test'))
            self.assertEqual(session.stopped_because,reason)
            self.assertEqual(session.interactions[-1].response_text,self.source)
            self.assertEqual(session.ledger.usage,response.usage)

    def test_initial_generation_metadata_and_full_record_integrity(self):
        runner=self.runners['D2_tight_quota']
        for fails in (False,True):
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                args=self.args(tmp)
                args.output=args.initial[0]
                with patch.object(driver,'_runner_from_args',return_value=(runner,deployment_adapter(runner),'deployment')), \
                     patch.object(runner.client,'count_chat_tokens',return_value=400), \
                     patch.object(runner.client,'complete',side_effect=TransportError('test') if fails else None,
                                  return_value=self.response()):
                    driver.cmd_initial(args)
                row=json.loads(Path(args.output).read_text())['generated'][0]
                g2.validate_initial(row)
                self.assertEqual(row['usage_known'],not fails)
                if fails:
                    self.assertIsNone(row['source'])
                    self.assertEqual(row['initial_stop'],'transport_error')
                else:
                    self.assertEqual(row['source'],self.source)
                row['usage']={'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}
                with self.assertRaisesRegex(ValueError,'record hash'):
                    g2.validate_initial(row)

    def test_context_runtime_changes_invalidate_initial_context(self):
        runner=self.runners['D2_tight_quota']
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(tmp)
            row=self.entry(runner,args)
            # JSON round-trip avoids mutating the runner's shared dictionary.
            row=json.loads(json.dumps(row))
            row['initial_context']['runtime']['context_window']=16384
            row['record_sha256']=g2.record_digest(row)
            with self.assertRaisesRegex(ValueError,'initial fixture/protocol mismatch'):
                self.execute(runner,args,row)

    def test_repair_uses_actual_session_and_usage_errors_keep_specific_stop(self):
        runner=self.runners['D2_tight_quota']
        for known in (False,True):
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                args=self.args(tmp)
                row=self.entry(runner,args)
                row['source']='invalid json'
                row['source_sha256']=driver.sha(row['source'])
                row['record_sha256']=g2.record_digest(row)
                Path(args.initial[0]).write_text(json.dumps({'generated':[row]}))
                with patch.object(driver,'_runner_from_args',return_value=(runner,deployment_adapter(runner),'deployment')), \
                     patch.object(runner.client,'count_chat_tokens',return_value=400), \
                     patch.object(runner.client,'complete',return_value=self.response(raw_known=known)) as call:
                    driver.cmd_run(args)
                out=json.loads((Path(args.output)/f'{args.condition}__seed0.json').read_text())
                self.assertEqual(call.call_count,1)
                self.assertEqual(out['spend']['model_calls'],2)
                self.assertEqual(out['usage']['total_tokens'],1000)
                if known:
                    self.assertEqual(out['spend']['states_observed'],16)
                    self.assertTrue(out['g2_accounting']['usage_complete'])
                else:
                    self.assertEqual(out['stopped_because'],'usage_unavailable')
                    self.assertFalse(out['g2_accounting']['usage_complete'])

    def test_changed_initial_usage_cannot_reuse_existing_result(self):
        runner=self.runners['D2_tight_quota']
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(tmp)
            row=self.entry(runner,args)
            self.execute(runner,args,row)
            row['usage']={'prompt_tokens':401,'completion_tokens':100,'total_tokens':501}
            row['record_sha256']=g2.record_digest(row)
            with self.assertRaisesRegex(ValueError,'initial record mismatch'):
                self.execute(runner,args,row)

    def test_invalid_optin_rejected_before_client_creation(self):
        with patch.object(driver,'ChatClient',side_effect=AssertionError('no client')):
            for kw in ({'replay_protocol':g2.VERSION,'fixture_id':'x'},
                       {'context_window':8192},
                       {'replay_protocol':g2.VERSION,'fixture_id':'x','context_window':128,'context_margin':256}):
                with self.assertRaises(ValueError):
                    driver.build_runner('deployment','http://localhost:1/v1','test',False,**kw)


if __name__ == '__main__':
    unittest.main()
