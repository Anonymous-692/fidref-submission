#!/usr/bin/env python3
"""Core gates and verification for the tau-bench retail domain."""

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path

from ..taubench_retail import (
    ActionKind,
    DslError,
    RetailAction,
    RetailConfig,
    RetailState,
    action_space,
    apply_action,
    audit_clause_density,
    enumerate_reachable,
    evaluate_contract,
    exchange_items_postcondition,
    exchange_items_precondition,
    initial_state,
    parse_contract,
)


def _reference_dsl_spec() -> dict[str, object]:
    return {
        "skill": "exchange_delivered_order_items",
        "name": "reference_contract",
        "precondition": {
            "op": "and",
            "args": [
                {"op": "in_set", "value": {"var": "order_id"}, "set": "valid_orders"},
                {
                    "op": "or",
                    "args": [
                        {
                            "op": "and",
                            "args": [
                                {"op": "eq", "left": {"var": "order_id"}, "right": {"const": "#W1"}},
                                {"op": "eq", "left": {"var": "order_w1_status"}, "right": {"const": "delivered"}},
                            ],
                        },
                        {
                            "op": "and",
                            "args": [
                                {"op": "eq", "left": {"var": "order_id"}, "right": {"const": "#W2"}},
                                {"op": "eq", "left": {"var": "order_w2_status"}, "right": {"const": "delivered"}},
                            ],
                        },
                    ],
                },
                {"op": "items_in_order", "items": {"var": "item_ids"}, "order": {"var": "order_id"}},
                {
                    "op": "eq",
                    "left": {"op": "count", "arg": {"var": "item_ids"}},
                    "right": {"op": "count", "arg": {"var": "new_item_ids"}},
                },
                {"op": "valid_variants", "items": {"var": "item_ids"}, "new_items": {"var": "new_item_ids"}},
                {"op": "items_available", "items": {"var": "new_item_ids"}},
                {"op": "in_set", "value": {"var": "payment_method_id"}, "set": "valid_payment_methods"},
                {
                    "op": "sufficient_balance",
                    "payment_method": {"var": "payment_method_id"},
                    "balance": {"var": "gift_card_balance"},
                    "items": {"var": "item_ids"},
                    "new_items": {"var": "new_item_ids"},
                },
            ],
        },
        "postcondition": {
            "op": "exchange_snapshot_matches",
        },
    }


class TauBenchRetailDomainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RetailConfig()

    def test_rejection_does_not_mutate_state(self) -> None:
        state = initial_state(self.config)
        result = apply_action(state, RetailAction.make(ActionKind.EXCHANGE_ITEMS), self.config)
        self.assertFalse(result.ok)
        self.assertIs(result.state, state)

    def test_gift_card_balance_interaction_with_cancellation(self) -> None:
        """Verify that cancelling pending order refunds gift card, enabling higher-priced exchange."""
        state = initial_state(self.config)
        # Draft a high-price difference exchange (+30.00 with gift card having initial balance 15.00)
        actions = (
            RetailAction.make(ActionKind.SET_ORDER, "#W1"),
            RetailAction.make(ActionKind.ADD_ITEM, "shoe_black_9"),
            RetailAction.make(ActionKind.ADD_NEW_ITEM, "shoe_red_9"),
            RetailAction.make(ActionKind.SET_PAYMENT_METHOD, "gift_card_0"),
        )
        for act in actions:
            state = apply_action(state, act, self.config).state

        # Initial balance is 15.00 < 30.00: exchange must be rejected
        target = RetailAction.make(ActionKind.EXCHANGE_ITEMS)
        res_before = apply_action(state, target, self.config)
        self.assertFalse(res_before.ok)
        self.assertIn("insufficient gift card balance", res_before.error or "")

        # Cancel pending order #W2: refunds $25.00 to gift card balance -> $40.00
        cancel_res = apply_action(state, RetailAction.make(ActionKind.CANCEL_PENDING_ORDER), self.config)
        self.assertTrue(cancel_res.ok)
        state_after_cancel = cancel_res.state
        self.assertEqual(state_after_cancel.user_gift_card_balance, 40.0)

        # Now balance is 40.00 >= 30.00: same exchange must succeed
        res_after = apply_action(state_after_cancel, target, self.config)
        self.assertTrue(res_after.ok)
        self.assertEqual(res_after.state.order_w1_status, "exchange requested")
        self.assertTrue(exchange_items_postcondition(state_after_cancel, res_after.state, self.config))

    def test_ground_truth_matches_environment_over_full_closure(self) -> None:
        enumeration = enumerate_reachable(self.config)
        self.assertFalse(enumeration.truncated)
        target = RetailAction.make(ActionKind.EXCHANGE_ITEMS)
        for state in enumeration.states:
            self.assertEqual(
                exchange_items_precondition(state, self.config),
                apply_action(state, target, self.config).ok,
            )

    def test_closure_and_clause_density_gates(self) -> None:
        first = enumerate_reachable(self.config)
        second = enumerate_reachable(self.config)
        self.assertEqual(first.states, second.states)
        self.assertFalse(first.truncated)
        self.assertEqual(len(first.states), 1294)
        self.assertEqual(len(action_space(self.config)), 27)

        rows = audit_clause_density(first, self.config)
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertGreater(row.witnesses, 0, f"Clause {row.name} has no distinguishing states")
            self.assertIsNotNone(row.minimum_depth, f"Clause {row.name} has no min_depth")

        audit_map = {row.name: row for row in rows}
        self.assertLessEqual(audit_map["sufficient_balance_if_gift_card"].minimum_depth or 99, 4)
        self.assertLessEqual(audit_map["new_items_valid_variant"].minimum_depth or 99, 4)

    def test_dsl_can_express_the_exact_contract(self) -> None:
        enumeration = enumerate_reachable(self.config)
        parsed = parse_contract(_reference_dsl_spec(), self.config)
        report = evaluate_contract(parsed.bind(self.config), enumeration.states, self.config)
        self.assertTrue(report.exact)
        self.assertEqual(report.failures, 0)
        self.assertEqual(report.false_accepts, 0)
        self.assertEqual(report.false_rejects, 0)
        self.assertEqual(report.postcondition_violations, 0)

    def test_default_json_matches_default_config(self) -> None:
        loaded = RetailConfig.from_json("experiment/configs/taubench_retail_default.json")
        self.assertEqual(loaded, self.config)

    def test_agent_facing_environment_does_not_import_evaluator_ground_truth(self) -> None:
        env_path = Path("experiment/taubench_retail/env.py")
        source = env_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(env_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn("ground_truth", alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                self.assertNotIn("ground_truth", mod)

    def test_vendored_file_hashes_match_upstream(self) -> None:
        expected = {
            "tau_bench/envs/tool.py": "0a158be298092fc64862ae8843f35f57cfba139aa02c2ee8fbe87b1e4a1de0fc",
            "tau_bench/envs/retail/tools/exchange_delivered_order_items.py": "d547d44b28e97513d4a15714d4bd2bbca13f76827bfe06b71e1b51a742f44489",
            "tau_bench/envs/retail/tools/cancel_pending_order.py": "c33151a896b62ac87ed1bcbcbd62dabcd47c8c1eaf47bbb5e2f15f8b6283b19d",
            "tau_bench/envs/retail/tools/return_delivered_order_items.py": "7d92e8962c17e5fa9c8e3240d21e31fc1adf5e68100c92bb0f67e0a67ea5b7bf",
            "tau_bench/envs/retail/tools/modify_pending_order_payment.py": "6c2d4c329fae0d66d50bd48cb6ae9520167137f8b20d3385fa407aa5b03d2fe5",
            "LICENSE": "243d23d45b80122b5ac575586ccef352fdc9c45e6d7d2605449aca8a17478b42",
        }
        vendor_dir = Path("experiment/taubench_retail/vendor")
        for rel_path, expected_hash in expected.items():
            p = vendor_dir / rel_path
            self.assertTrue(p.exists(), f"Missing vendored file: {rel_path}")
            actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(actual_hash, expected_hash, f"Hash mismatch for {rel_path}")

    def test_runner_end_to_end_mock(self) -> None:
        import json
        from experiment.modeling.client import ChatClient
        from experiment.modeling.runner import Budgets, Decoding, RunSpec
        from experiment.taubench_retail.runner import RetailExperimentRunner
        from experiment.tests.test_calendar_active_cegis import CalendarMockTransport
        from experiment.tests.test_model_runner import ok_response

        transport = CalendarMockTransport([ok_response(json.dumps(_reference_dsl_spec()))])
        client = ChatClient(
            model="mock-model",
            base_url="http://mock",
            transport=transport,
            allow_remote=True,
        )
        runner = RetailExperimentRunner(
            client=client,
            config=self.config,
            config_path="experiment/configs/taubench_retail_default.json",
            max_depth=6,
            max_states=2000,
        )
        spec = RunSpec(
            method="direct",
            seed=0,
            decoding=Decoding(temperature=0.0, top_p=1.0, max_tokens=2048),
            budgets=Budgets(state_budget=48, query_budget=4, token_budget=16000),
        )
        artifact = runner.run_method(spec)
        self.assertEqual(artifact.outcome, "exact")
        self.assertEqual(len(artifact.interactions), 1)
        self.assertIsNotNone(artifact.interactions[0].response_text)
        self.assertTrue(artifact.contract.parsed)
        self.assertIsNotNone(artifact.evaluation)
        self.assertTrue(artifact.evaluation.exact)

    def test_model_runner_execute_from_args_mock(self) -> None:
        import json
        import tempfile
        from experiment.taubench_retail.model_runner import build_parser, execute_from_args
        from experiment.tests.test_calendar_active_cegis import CalendarMockTransport
        from experiment.tests.test_model_runner import ok_response

        transport = CalendarMockTransport([ok_response(json.dumps(_reference_dsl_spec()))])
        with tempfile.TemporaryDirectory() as tmp_dir:
            parser = build_parser()
            args = parser.parse_args([
                "--output", tmp_dir,
                "--methods", "direct",
                "--seeds", "0",
                "--max-depth", "6",
                "--max-states", "2000",
            ])
            summary = execute_from_args(args, {}, transport=transport)
            self.assertIn("context", summary)
            self.assertIn("runs", summary)
            self.assertEqual(len(summary["runs"]), 1)
            self.assertTrue(summary["runs"][0]["exact"])
            artifact_file = Path(tmp_dir) / "direct__seed0.json"
            self.assertTrue(artifact_file.exists())
            summary_file = Path(tmp_dir) / "summary.json"
            self.assertTrue(summary_file.exists())

    def test_in_set_items_parsing_and_semantics(self) -> None:
        spec = {
            "skill": "exchange_delivered_order_items",
            "precondition": {
                "op": "in_set",
                "value": {"var": "item_ids", "when": "before"},
                "set": "valid_items",
            },
            "postcondition": {"op": "const", "value": True},
        }
        parsed = parse_contract(spec, self.config)
        contract = parsed.bind(self.config)
        enumeration = enumerate_reachable(self.config)
        true_count = 0
        false_count = 0
        for state in enumeration.states:
            expected = all(x in self.config.valid_items for x in state.draft_item_ids)
            self.assertEqual(contract.holds_in(state), expected)
            if expected:
                true_count += 1
            else:
                false_count += 1
        self.assertGreater(true_count, 0)
        self.assertGreater(false_count, 0)

    def test_in_set_items_vacuously_true_on_empty(self) -> None:
        spec = {
            "skill": "exchange_delivered_order_items",
            "precondition": {
                "op": "in_set",
                "value": {"var": "item_ids", "when": "before"},
                "set": "valid_items",
            },
            "postcondition": {"op": "const", "value": True},
        }
        contract = parse_contract(spec, self.config).bind(self.config)
        state_empty = RetailState()
        self.assertEqual(state_empty.draft_item_ids, ())
        self.assertTrue(contract.holds_in(state_empty))

    def test_in_set_number_kind_raises_dsl_error(self) -> None:
        spec = {
            "skill": "exchange_delivered_order_items",
            "precondition": {
                "op": "in_set",
                "value": {"var": "gift_card_balance", "when": "before"},
                "set": "valid_orders",
            },
            "postcondition": {"op": "const", "value": True},
        }
        with self.assertRaises(DslError) as ctx:
            parse_contract(spec, self.config)
        self.assertIn("requires an option/text or items value", str(ctx.exception))

    def test_model_runner_runner_error_handling(self) -> None:
        import tempfile
        from unittest.mock import patch
        from experiment.taubench_retail.model_runner import build_parser, execute_from_args
        from experiment.tests.test_calendar_active_cegis import CalendarMockTransport
        from experiment.tests.test_model_runner import ok_response

        transport = CalendarMockTransport([ok_response(json.dumps(_reference_dsl_spec()))])
        with tempfile.TemporaryDirectory() as tmp_dir:
            parser = build_parser()
            args = parser.parse_args([
                "--output", tmp_dir,
                "--methods", "direct",
                "--seeds", "0",
                "--max-depth", "6",
                "--max-states", "2000",
            ])
            with patch("experiment.taubench_retail.runner.RetailExperimentRunner.run_method", side_effect=RuntimeError("simulated crash")):
                summary = execute_from_args(args, {}, transport=transport)
            self.assertEqual(summary["runs"][0]["outcome"], "runner_error")
            self.assertFalse(summary["runs"][0]["exact"])
            artifact_file = Path(tmp_dir) / "direct__seed0.json"
            self.assertTrue(artifact_file.exists())
            artifact_data = json.loads(artifact_file.read_text())
            self.assertEqual(artifact_data["outcome"], "runner_error")
            self.assertIsNone(artifact_data["evaluation"])
            self.assertIn("simulated crash", artifact_data["stopped_because"])


if __name__ == "__main__":
    unittest.main()
