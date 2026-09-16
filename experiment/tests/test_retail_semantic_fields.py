"""Opt-in semantic-field protocol: old paths frozen, new paths shared by all arms."""
import json
from dataclasses import replace
from pathlib import Path
import unittest

from analysis.g1_g2_implementation_v1.capture_legacy import snapshot
from experiment.modeling.client import ChatClient
from experiment.modeling.runner import RunSpec, Budgets
from experiment.modeling.st2x2_domains import taubench_adapter
from experiment.taubench_retail.config import RetailConfig
from experiment.taubench_retail.state import RetailState, SEMANTIC_FIELDS_PROTOCOL as V, SEMANTIC_FIELDS_SCOPE
from experiment.taubench_retail.dsl import parse_contract, contract_json_schema, DslError
from experiment.taubench_retail.env import apply_action, RetailAction, ActionKind
from experiment.taubench_retail.enumeration import enumerate_reachable
from experiment.taubench_retail.ground_truth import exchange_items_postcondition
from experiment.taubench_retail.prompts import system_prompt, counterexample_prompt, counterexample_prompt_v3
from experiment.taubench_retail.runner import RetailExperimentRunner
from experiment.tests.test_taubench_retail_domain import _reference_dsl_spec
from experiment.tests.test_calendar_active_cegis import CalendarMockTransport
from experiment.tests.test_model_runner import ok_response

FIELD = 'order_w1_return_payment_method_id'
DIRECT_METHODS = ('direct', 'self_refine', 'random_probe', 'sampled_cegis', 'sampled_cegis_fixed',
                  'active_cegis', 'active_cegis_no_balance', 'active_cegis_no_coverage', 'active_cegis_uniform')


class SemanticFieldsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = RetailConfig.from_json('experiment/configs/taubench_retail_default.json')
        cls.enum = enumerate_reachable(cls.config)

    def runner(self, **kwargs):
        return RetailExperimentRunner(None, config=self.config, enumeration=self.enum,
                                      vocabulary_protocol=V, **kwargs)

    def test_legacy_and_neutral_snapshots_unchanged(self):
        expected = json.loads(Path('analysis/g1_g2_implementation_v1/legacy_snapshot.json').read_text())
        self.assertEqual(snapshot(), expected)

    def test_new_variable_opt_in_for_var_unchanged_and_feedback_projection(self):
        state = RetailState(order_w1_return_payment_method_id='credit_card_0')
        self.assertNotIn(FIELD, state.to_dict())
        self.assertEqual(state.to_dict(vocabulary_protocol=V)[FIELD], 'credit_card_0')
        self.assertNotIn(FIELD, system_prompt())
        self.assertIn(FIELD, system_prompt(V))
        for formula in ({'op': 'unchanged', 'vars': [FIELD]},
                        {'op': 'eq', 'left': {'var': FIELD}, 'right': {'var': FIELD, 'when': 'after'}}):
            spec = {'precondition': {'op': 'const', 'value': True}, 'postcondition': formula}
            for old in ('legacy_w1_v1', 'order_neutral_v1'):
                with self.assertRaises(DslError):
                    parse_contract(spec, vocabulary_protocol=old)
            c = parse_contract(spec, vocabulary_protocol=V)
            self.assertTrue(c.transition_holds(state, state))
            self.assertFalse(c.transition_holds(state, replace(state, order_w1_return_payment_method_id=None)))

    def test_fourteen_normal_and_fortytwo_injected_frame_errors(self):
        new = parse_contract(_reference_dsl_spec(), self.config, vocabulary_protocol=V)
        old = parse_contract(_reference_dsl_spec(), self.config)
        normal = mutants = 0
        for before in self.enum.states:
            outcome = apply_action(before, RetailAction.make(ActionKind.EXCHANGE_ITEMS), self.config)
            if not outcome.ok:
                continue
            normal += 1
            self.assertTrue(new.transition_holds(before, outcome.state))
            self.assertTrue(exchange_items_postcondition(before, outcome.state, self.config, vocabulary_protocol=V))
            for change in ({'order_w1_return_items': ('shoe_black_9',)},
                           {FIELD: 'credit_card_0'}, {'authenticated': False}):
                after = replace(outcome.state, **change)
                mutants += 1
                self.assertTrue(old.transition_holds(before, after))
                self.assertTrue(exchange_items_postcondition(before, after, self.config))
                self.assertFalse(new.transition_holds(before, after))
                self.assertFalse(exchange_items_postcondition(before, after, self.config, vocabulary_protocol=V))
        self.assertEqual((normal, mutants), (14, 42))

    def test_counterexample_serialization_and_both_refinement_formats(self):
        bad = _reference_dsl_spec()
        bad['postcondition'] = {'op': 'const', 'value': False}
        c = parse_contract(bad, self.config, vocabulary_protocol=V)
        runner = self.runner()
        report = runner.score(c)
        self.assertIn(FIELD, report.to_dict()['counterexamples'][0]['before_state'])
        self.assertIn(FIELD, counterexample_prompt(report, self.config))
        self.assertIn(FIELD, counterexample_prompt_v3(json.dumps(bad), report, self.config))
        adapter = taubench_adapter(runner)
        self.assertEqual(adapter.parse(json.dumps(bad))._vocabulary_protocol, V)
        self.assertIn(FIELD, adapter.counterexample_prompt(json.dumps(bad), adapter.score_subset(c, runner.states)))

    def test_all_registered_methods_share_new_request_schema_and_scope(self):
        requests = []
        for method in DIRECT_METHODS:
            runner = self.runner()
            transport = CalendarMockTransport([ok_response(json.dumps(_reference_dsl_spec()))])
            runner.client = ChatClient(model='test', base_url='http://mock', allow_remote=True, transport=transport)
            artifact = runner.run(RunSpec(method=method, seed=0, budgets=Budgets(query_budget=1), use_guided_json=True))
            requests.append(artifact.interactions[0].messages)
            self.assertEqual(artifact.sandbox['observation_scope'], SEMANTIC_FIELDS_SCOPE)
            self.assertEqual(artifact.sandbox['postcondition_protocol'], V)
            self.assertIn(FIELD, artifact.interactions[0].messages[0]['content'])
            self.assertIn('SemanticFields', json.dumps(transport.calls[0]['payload']))
            self.assertNotEqual(artifact.hashes['protocol_sha256'], snapshot()['protocols']['legacy_w1_v1']['hashes'][method])
        self.assertTrue(all(r == requests[0] for r in requests))

    def test_new_scope_rejects_w2_delivery_configuration(self):
        with self.assertRaisesRegex(ValueError, 'does not support'):
            RetailExperimentRunner(None, config=replace(self.config, deliverable_orders=('#W2',)), vocabulary_protocol=V)
