#!/usr/bin/env python3
"""Bounded ``add`` compatibility, numeric misuse rejection, and prompt guidance.

Every test here is offline: the DSL parses model *text*, never executes it, and
no model endpoint is contacted.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from ..deployment import dsl, prompts
from ..deployment.config import DeploymentConfig
from ..deployment.contracts import evaluate_contract
from ..deployment.enumeration import enumerate_reachable

CONFIG = DeploymentConfig(
    seed=2026,
    resource_types=("cpu", "ram"),
    max_quota=2,
    max_allocation_limit=2,
    valid_regions=("us-central", "europe-west"),
    rejected_regions=("unsupported-edge",),
    valid_cluster_tiers=("standard", "premium"),
    rejected_cluster_tiers=("deprecated-v0",),
)


def exact_spec(*, quota_op: str) -> dict[str, Any]:
    """The exact contract with the quota effect written as ``before == after (+) allocated``.

    ``quota_op`` is ``union`` or its ``add`` alias, so the two specs differ by
    exactly one token and must behave identically.
    """
    return {
        "skill": "deploy_service",
        "notes": f"quota effect via {quota_op}",
        "precondition": {
            "op": "and",
            "args": [
                {"op": "is_true", "arg": {"var": "authenticated"}},
                {"op": "eq", "left": {"var": "deployment_status"}, "right": {"const": "idle"}},
                {"op": "not", "arg": {"op": "is_empty", "arg": {"var": "allocated_resources"}}},
                {"op": "in_set", "value": {"var": "target_region"}, "set": "valid_regions"},
                {"op": "in_set", "value": {"var": "cluster_tier"}, "set": "valid_cluster_tiers"},
                {
                    "op": "covers",
                    "available": {"var": "available_quota"},
                    "required": {"var": "allocated_resources"},
                },
            ],
        },
        "postcondition": {
            "op": "and",
            "args": [
                {
                    "op": "eq",
                    "left": {"var": "deployment_status", "when": "after"},
                    "right": {"const": "deployed"},
                },
                {
                    "op": "eq",
                    "left": {"var": "active_deployment", "when": "after"},
                    "right": {"var": "allocated_resources"},
                },
                {"op": "is_empty", "arg": {"var": "allocated_resources", "when": "after"}},
                {
                    "op": "eq",
                    "left": {"var": "available_quota"},
                    "right": {
                        "op": quota_op,
                        "left": {"var": "available_quota", "when": "after"},
                        "right": {"var": "allocated_resources"},
                    },
                },
                {"op": "unchanged", "vars": ["authenticated", "target_region", "cluster_tier"]},
            ],
        },
    }


def numeric_add_spec() -> dict[str, Any]:
    """The shape the 32B model actually produced: ``add`` over two numbers."""
    return {
        "skill": "deploy_service",
        "precondition": {"op": "is_true", "arg": {"var": "authenticated"}},
        "postcondition": {
            "op": "eq",
            "left": {"var": "allocation_size", "when": "after"},
            "right": {
                "op": "add",
                "left": {"var": "allocation_size"},
                "right": {"var": "active_size", "when": "after"},
            },
        },
    }


class AddAliasCompatibilityTest(unittest.TestCase):
    """``add`` is accepted only where it means exactly multiset union."""

    def setUp(self) -> None:
        self.states = enumerate_reachable(CONFIG, max_depth=12, max_states=1000).states
        self.assertTrue(self.states)

    def test_add_over_multisets_is_identical_to_union(self) -> None:
        union_parsed = dsl.parse_contract_text(json.dumps(exact_spec(quota_op="union")))
        add_parsed = dsl.parse_contract_text(json.dumps(exact_spec(quota_op="add")))

        union_report = evaluate_contract(union_parsed.bind(CONFIG), self.states, CONFIG)
        add_report = evaluate_contract(add_parsed.bind(CONFIG), self.states, CONFIG)

        self.assertTrue(union_report.is_exact)
        self.assertTrue(add_report.is_exact)
        self.assertEqual(union_report.summary()["exact"], add_report.summary()["exact"])
        self.assertEqual(union_report.false_accepts, add_report.false_accepts)
        self.assertEqual(union_report.false_rejects, add_report.false_rejects)
        self.assertEqual(
            union_report.postcondition_violations, add_report.postcondition_violations
        )

        # Same verdict on every single enumerated state, not just in aggregate.
        union_contract = union_parsed.bind(CONFIG)
        add_contract = add_parsed.bind(CONFIG)
        for state in self.states:
            self.assertEqual(union_contract.holds_in(state), add_contract.holds_in(state))

    def test_add_alias_use_is_recorded_as_a_compatibility_rewrite(self) -> None:
        add_parsed = dsl.parse_contract_text(json.dumps(exact_spec(quota_op="add")))
        union_parsed = dsl.parse_contract_text(json.dumps(exact_spec(quota_op="union")))
        self.assertEqual(add_parsed.compat_rewrites, ("add->union",))
        self.assertEqual(union_parsed.compat_rewrites, ())

    def test_numeric_add_is_rejected(self) -> None:
        with self.assertRaises(dsl.DslError) as caught:
            dsl.parse_contract_text(json.dumps(numeric_add_spec()))
        message = str(caught.exception)
        self.assertIn("no numeric arithmetic", message)
        self.assertIn("alias for 'union'", message)

    def test_add_with_mixed_operand_kinds_is_rejected(self) -> None:
        spec = {
            "skill": "deploy_service",
            "precondition": {"op": "const", "value": True},
            "postcondition": {
                "op": "is_empty",
                "arg": {
                    "op": "add",
                    "left": {"var": "allocated_resources"},
                    "right": {"var": "allocation_size"},
                },
            },
        }
        with self.assertRaises(dsl.DslError) as caught:
            dsl.parse_contract_dict(spec)
        self.assertIn("multiset", str(caught.exception))

    def test_add_over_text_terms_is_rejected(self) -> None:
        spec = {
            "skill": "deploy_service",
            "precondition": {
                "op": "eq",
                "left": {"var": "deployment_status"},
                "right": {
                    "op": "add",
                    "left": {"const": "de"},
                    "right": {"const": "ployed"},
                },
            },
            "postcondition": {"op": "const", "value": True},
        }
        with self.assertRaises(dsl.DslError):
            dsl.parse_contract_dict(spec)


class NumericMultisetMisuseTest(unittest.TestCase):
    """``total``/``get`` stay multiset-only; the size variables are already numbers."""

    def _reject(self, spec: dict[str, Any]) -> str:
        with self.assertRaises(dsl.DslError) as caught:
            dsl.parse_contract_dict(spec)
        return str(caught.exception)

    def test_total_applied_to_a_number_is_rejected(self) -> None:
        for var_name in ("allocation_size", "quota_size", "active_size"):
            with self.subTest(var=var_name):
                message = self._reject(
                    {
                        "skill": "deploy_service",
                        "precondition": {
                            "op": "lt",
                            "left": {"op": "total", "multiset": {"var": var_name}},
                            "right": {"const": 2},
                        },
                        "postcondition": {"op": "const", "value": True},
                    }
                )
                self.assertIn("already numbers", message)
                self.assertIn("total", message)

    def test_get_applied_to_a_number_is_rejected(self) -> None:
        message = self._reject(
            {
                "skill": "deploy_service",
                "precondition": {
                    "op": "ge",
                    "left": {"op": "get", "multiset": {"var": "quota_size"}, "resource": "cpu"},
                    "right": {"const": 0},
                },
                "postcondition": {"op": "const", "value": True},
            }
        )
        self.assertIn("already numbers", message)

    def test_total_over_a_real_multiset_still_parses(self) -> None:
        parsed = dsl.parse_contract_dict(
            {
                "skill": "deploy_service",
                "precondition": {
                    "op": "lt",
                    "left": {"op": "total", "multiset": {"var": "allocated_resources"}},
                    "right": {"const": 99},
                },
                "postcondition": {"op": "const", "value": True},
            }
        )
        self.assertEqual(parsed.compat_rewrites, ())

    def test_truncated_response_reports_the_token_limit(self) -> None:
        truncated = '{"skill": "deploy_service", "precondition": {"op": "and", "args": [{"op":'
        with self.assertRaises(dsl.DslError) as caught:
            dsl.parse_contract_text(truncated)
        self.assertIn("cut off", str(caught.exception))


class PromptGuidanceTest(unittest.TestCase):
    """The system prompt carries a complete example plus operator guardrails."""

    def setUp(self) -> None:
        self.prompt = prompts.system_prompt()

    def test_prompt_contains_a_complete_parseable_contract_example(self) -> None:
        self.assertIn("```json", self.prompt)
        self.assertIn(prompts.example_contract_json(), self.prompt)

        parsed = dsl.parse_contract_dict(prompts.EXAMPLE_CONTRACT)
        self.assertEqual(parsed.skill, dsl.SKILL)
        self.assertGreater(parsed.node_count, 0)
        # A round-trip through the raw text the model actually sees.
        self.assertIsNotNone(dsl.parse_contract_text(prompts.example_contract_json()))

    def test_prompt_example_is_not_the_ground_truth_contract(self) -> None:
        states = enumerate_reachable(CONFIG, max_depth=12, max_states=1000).states
        report = evaluate_contract(
            dsl.parse_contract_dict(prompts.EXAMPLE_CONTRACT).bind(CONFIG), states, CONFIG
        )
        self.assertFalse(
            report.is_exact,
            "the syntax example must stay deliberately wrong so it cannot leak the oracle",
        )

    def test_prompt_example_has_no_semantic_threshold_or_covers_anchor(self) -> None:
        example = prompts.example_contract_json()
        self.assertNotIn('"covers"', example)
        self.assertNotIn('"allocation_size"', example)
        self.assertNotIn('"quota_size"', example)
        direct = prompts.direct_prompt(CONFIG)
        self.assertIn("only bound generated sandbox states", direct)

    def test_prompt_forbids_numeric_arithmetic_and_total_misuse(self) -> None:
        lowered = self.prompt.lower()
        self.assertIn("no numeric arithmetic", lowered)
        self.assertIn("allocation_size", lowered)
        self.assertIn("never wrap them in `total`", lowered)
        self.assertIn("prefer `union`", lowered)

    def test_prompt_lists_every_supported_operator(self) -> None:
        for op in dsl.FORMULA_OPS:
            self.assertIn(f"`{op}`", self.prompt)
        for op in dsl.TERM_OPS:
            self.assertIn(f"`{op}`", self.prompt)

    def test_parse_failure_note_repeats_the_operator_rules(self) -> None:
        note = prompts.parse_failure_note("unknown term op 'add'")
        self.assertIn("unknown term op 'add'", note)
        for rule in prompts.OPERATOR_RULES:
            self.assertIn(rule, note)

    def test_direct_prompt_warns_about_the_common_failures(self) -> None:
        text = prompts.direct_prompt(CONFIG)
        self.assertIn("no numeric arithmetic", text)
        self.assertIn("allocation_size", text)


if __name__ == "__main__":
    unittest.main()
