"""B10 gates: opt-in vocabulary, unchanged legacy, and full W2 closure."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from experiment.modeling.artifacts import sha256_text
from experiment.modeling.client import ChatClient
from experiment.modeling.runner import RunSpec
from experiment.taubench_retail import (
    RetailConfig, ActionKind, RetailAction, apply_action,
    enumerate_reachable, evaluate_contract,
)
from experiment.taubench_retail.dsl import parse_contract, DslError, VOCABULARY_ORDER_NEUTRAL
from experiment.taubench_retail.prompts import system_prompt, direct_prompt
from experiment.taubench_retail.runner import RetailExperimentRunner
from experiment.taubench_retail.model_runner import build_parser, execute_from_args, load_experiment_config
from experiment.tests.test_taubench_retail_domain import _reference_dsl_spec
from experiment.tests.test_calendar_active_cegis import CalendarMockTransport
from experiment.tests.test_model_runner import ok_response

CONFIG = "experiment/configs/taubench_retail_w2del.json"
EXPERIMENT = "experiment/configs/taubench_retail_w2neutral_gemma4_s20.json"
METHODS = ("direct", "sampled_cegis_fixed", "active_cegis", "active_cegis_no_coverage")


class OrderNeutralTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = RetailConfig.from_json(CONFIG)
        cls.enumeration = enumerate_reachable(cls.config)

    def runner(self, **kwargs):
        return RetailExperimentRunner(ChatClient(model="test"), config=self.config,
                                      config_path=CONFIG, enumeration=self.enumeration, **kwargs)

    def test_legacy_prompts_and_protocol_hashes_unchanged(self):
        self.assertEqual(sha256_text(system_prompt()), "2d4983b640a7e09c4d686250923f0c5685138e50a40e46354e4f1253e1147fdf")
        self.assertEqual(sha256_text(direct_prompt(self.config)), "0f66781ca643cfad08cb01f1f71fdef75391739396dbf1f3fc096a600484788e")
        expected = (
            "ee8367fcbbe3a3a01626a0a9c9d04ae2e09591a08eb0ec70481650332cf2680e",
            "f4a5dcb93d404374b420f1388f46afe3b105f0be60ffbb13dc75a0ffa0353cb6",
            "44c3a4fd5bc2fcf027aa69e94fe203289af2cbe8bb9a1bc4b727ae80340ffa94",
            "b38d23bb2283273895a1984a7be32886bb34ebe3f09dad093b6c1630178a2747",
        )
        legacy = self.runner()
        neutral = self.runner(vocabulary_protocol=VOCABULARY_ORDER_NEUTRAL)
        for method, digest in zip(METHODS, expected):
            spec = RunSpec(method=method, seed=0)
            self.assertEqual(legacy.compute_protocol_hash(spec), digest)
            self.assertNotEqual(neutral.compute_protocol_hash(spec), digest)

    def test_reference_exact_over_both_full_closures(self):
        for config, count in ((RetailConfig(), 1294), (self.config, 4500)):
            enumeration = enumerate_reachable(config)
            self.assertFalse(enumeration.truncated)
            self.assertEqual(len(enumeration.states), count)
            neutral = parse_contract(_reference_dsl_spec(), config,
                                     vocabulary_protocol=VOCABULARY_ORDER_NEUTRAL).bind()
            report = evaluate_contract(neutral, enumeration.states, config)
            self.assertTrue(report.exact, report)
            self.assertEqual(report.failures, 0)

    def test_w2_snapshot_rejects_visible_mutations(self):
        target = RetailAction.make(ActionKind.EXCHANGE_ITEMS)
        neutral = parse_contract(_reference_dsl_spec(), self.config,
                                 vocabulary_protocol=VOCABULARY_ORDER_NEUTRAL)
        legacy = parse_contract(_reference_dsl_spec(), self.config)
        successes = 0
        for before in self.enumeration.states:
            result = apply_action(before, target, self.config)
            if not result.ok:
                continue
            if before.draft_order_id == "#W1":
                self.assertEqual(neutral.transition_holds(before, result.state),
                                 legacy.transition_holds(before, result.state))
                continue
            successes += 1
            self.assertTrue(neutral.transition_holds(before, result.state))
            self.assertFalse(legacy.transition_holds(before, result.state))
            for changes in ({"order_w2_status": "delivered"},
                            {"order_w1_status": "broken"}, {"draft_order_id": "#W2"},
                            {"user_gift_card_balance": -100}, {"authenticated": not before.authenticated}):
                self.assertFalse(neutral.transition_holds(before, replace(result.state, **changes)))
        self.assertGreater(successes, 0)

    def test_neutral_prompt_no_order_specific_examples(self):
        prompt = direct_prompt(self.config, VOCABULARY_ORDER_NEUTRAL)
        self.assertNotIn("order_w1_status", prompt)
        self.assertNotIn("order_w2_status", prompt)
        self.assertNotIn("exchange requested", prompt)
        self.assertNotIn('{"const": "#W1"}', system_prompt(VOCABULARY_ORDER_NEUTRAL))

    def test_four_methods_share_initial_prompt_and_record_version(self):
        cfg = load_experiment_config(Path(EXPERIMENT))
        cfg["seeds"] = [0]
        with tempfile.TemporaryDirectory() as output:
            args = build_parser().parse_args(["--output", output])
            transport = CalendarMockTransport([ok_response(json.dumps(_reference_dsl_spec())) for _ in METHODS])
            execute_from_args(args, cfg, transport=transport)
            artifacts = [json.loads((Path(output) / f"{m}__seed0.json").read_text()) for m in METHODS]
        messages = artifacts[0]["interactions"][0]["messages"]
        for artifact in artifacts:
            self.assertEqual(artifact["interactions"][0]["messages"], messages)
            self.assertEqual(artifact["sandbox"]["vocabulary_protocol"], VOCABULARY_ORDER_NEUTRAL)
            self.assertEqual(artifact["outcome"], "exact")
            self.assertEqual(artifact["spend"]["oracle_feedback_queries"], 0)
        self.assertEqual(len({a["hashes"]["protocol_sha256"] for a in artifacts}), 4)

    def test_fixed_pool_is_identical_across_revisions(self):
        bad = _reference_dsl_spec()
        bad["precondition"] = {"op": "const", "value": True}
        runner = self.runner(vocabulary_protocol=VOCABULARY_ORDER_NEUTRAL)
        runner.client = ChatClient(model="test", base_url="http://mock", allow_remote=True,
            transport=CalendarMockTransport([ok_response(json.dumps(bad)) for _ in range(4)]))
        with patch.object(runner, "score", wraps=runner.score) as score:
            artifact = runner.run(RunSpec(method="sampled_cegis_fixed", seed=0))
        subsets = [call.args[1] for call in score.call_args_list if len(call.args) > 1]
        self.assertGreater(len(subsets), 1)
        self.assertTrue(all(subset == subsets[0] for subset in subsets))
        self.assertTrue(all(r["evidence_policy"] == "fixed_pool_unpartitioned" for r in artifact.rounds))

    def test_unknown_version_rejected(self):
        with self.assertRaises(ValueError):
            self.runner(vocabulary_protocol="typo")
        with self.assertRaises(DslError):
            parse_contract(_reference_dsl_spec(), vocabulary_protocol="typo")

    def test_fixed_pool_registration_is_retail_only(self):
        from experiment.modeling.runner import METHODS as shopping_methods
        from experiment.taubench_retail.runner import METHODS as retail_methods
        self.assertNotIn("sampled_cegis_fixed", shopping_methods)
        self.assertIn("sampled_cegis_fixed", retail_methods)


if __name__ == "__main__":
    unittest.main()
