#!/usr/bin/env python3
"""Regression gates for price-aware Shopping."""

from __future__ import annotations

import unittest
from pathlib import Path

from ..shopping_price import (
    DEFAULT_PRICE_SHOPPING_CONFIG,
    PriceShoppingAction,
    PriceShoppingConfig,
    apply_action,
    audit_clause_density,
    enumerate_reachable,
    evaluate_contract,
    place_order_postcondition,
    place_order_precondition,
)
from ..shopping_price.dsl import parse_contract
from ..shopping_price.prompts import counterexample_prompt_v3


def _exact_spec() -> dict[str, object]:
    def var(name: str, when: str = "before") -> dict[str, str]:
        return {"var": name, "when": when}

    def const(value: object) -> dict[str, object]:
        return {"const": value}

    def eq(left: object, right: object) -> dict[str, object]:
        return {"op": "eq", "left": left, "right": right}

    payment_valid = {"op": "or", "args": [eq(var("payment_method"), const("card")), eq(var("payment_method"), const("wallet"))]}
    address_valid = {"op": "or", "args": [eq(var("shipping_address"), const("home")), eq(var("shipping_address"), const("remote"))]}
    coupon_valid = {
        "op": "or",
        "args": [
            {"op": "is_null", "arg": var("coupon_code")},
            {
                "op": "and",
                "args": [
                    eq(var("coupon_code"), const("save500")),
                    {"op": "ge", "left": var("cart_subtotal"), "right": const(3000)},
                    eq({"op": "count", "of": var("cart"), "item": "gift"}, const(0)),
                ],
            },
        ],
    }
    capacity = {
        "op": "or",
        "args": [
            {"op": "and", "args": [eq(var("payment_method"), const("wallet")), {"op": "ge", "left": var("wallet_balance"), "right": var("checkout_total")}]},
            {"op": "and", "args": [eq(var("payment_method"), const("card")), {"op": "le", "left": var("checkout_total"), "right": var("card_limit")}]},
        ],
    }
    wallet_effect = {
        "op": "or",
        "args": [
            {
                "op": "and",
                "args": [
                    eq(var("payment_method"), const("wallet")),
                    eq(var("wallet_balance", "after"), {"op": "subtract", "left": var("wallet_balance"), "right": var("checkout_total")}),
                ],
            },
            {"op": "and", "args": [eq(var("payment_method"), const("card")), eq(var("wallet_balance", "after"), var("wallet_balance"))]},
        ],
    }
    return {
        "skill": "place_order",
        "precondition": {
            "op": "and",
            "args": [
                {"op": "is_true", "arg": var("logged_in")},
                eq(var("order_status"), const("none")),
                {"op": "not", "arg": {"op": "is_empty", "arg": var("cart")}},
                payment_valid,
                address_valid,
                {"op": "covers", "available": var("stock"), "required": var("cart")},
                coupon_valid,
                capacity,
            ],
        },
        "postcondition": {
            "op": "and",
            "args": [
                eq(var("order_status", "after"), const("placed")),
                eq(var("order_items", "after"), var("cart")),
                {"op": "is_empty", "arg": var("cart", "after")},
                eq(var("stock", "after"), {"op": "difference", "left": var("stock"), "right": var("cart")}),
                eq(var("order_total", "after"), var("checkout_total")),
                eq(var("order_payment_method", "after"), var("payment_method")),
                {"op": "is_null", "arg": var("coupon_code", "after")},
                wallet_effect,
                {"op": "unchanged", "vars": ["logged_in", "payment_method", "shipping_address", "card_limit"]},
            ],
        },
    }


def _easy_v4_exact_spec() -> dict[str, object]:
    def var(name: str, when: str = "before") -> dict[str, str]:
        return {"var": name, "when": when}

    def const(value: object) -> dict[str, object]:
        return {"const": value}

    def eq(left: object, right: object) -> dict[str, object]:
        return {"op": "eq", "left": left, "right": right}

    return {
        "skill": "place_order",
        "precondition": {
            "op": "and",
            "args": [
                {"op": "is_true", "arg": var("logged_in")},
                eq(var("order_status"), const("none")),
                {"op": "not", "arg": {"op": "is_empty", "arg": var("cart")}},
                {"op": "in_set", "value": var("payment_method"), "set": "payment_methods"},
                {"op": "in_set", "value": var("shipping_address"), "set": "addresses"},
                {"op": "covers", "available": var("stock"), "required": var("cart")},
            ],
        },
        "postcondition": {
            "op": "and",
            "args": [
                eq(var("order_status", "after"), const("placed")),
                eq(var("order_items", "after"), var("cart")),
                {"op": "is_empty", "arg": var("cart", "after")},
                eq(var("stock", "after"), {"op": "difference", "left": var("stock"), "right": var("cart")}),
                eq(var("order_total", "after"), var("checkout_total")),
                eq(var("order_payment_method", "after"), var("payment_method")),
                {"op": "is_null", "arg": var("coupon_code", "after")},
                {"op": "is_null", "arg": var("payment_method", "after")},
                {"op": "is_null", "arg": var("shipping_address", "after")},
                {"op": "unchanged", "vars": ["logged_in", "wallet_balance", "card_limit"]},
            ],
        },
    }


class PriceShoppingDomainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DEFAULT_PRICE_SHOPPING_CONFIG
        self.enumeration = enumerate_reachable(self.config, max_depth=20, max_states=15000)

    def test_default_json_matches_default_config(self) -> None:
        self.assertEqual(PriceShoppingConfig.from_json("experiment/configs/shopping_price_default.json"), self.config)

    def test_full_closure_is_complete_and_deterministic(self) -> None:
        second = enumerate_reachable(self.config, max_depth=20, max_states=15000)
        self.assertFalse(self.enumeration.truncated)
        self.assertEqual(len(self.enumeration.states), 8064)
        self.assertEqual(len(self.enumeration.transitions), 95437)
        self.assertEqual(self.enumeration.states, second.states)

    def test_ground_truth_matches_every_reachable_state(self) -> None:
        action = PriceShoppingAction.place_order()
        for state in self.enumeration.states:
            result = apply_action(state, action, self.config)
            self.assertEqual(place_order_precondition(state, self.config), result.ok)
            if result.ok:
                self.assertTrue(place_order_postcondition(state, result.state, self.config))

    def test_every_clause_has_an_independent_witness(self) -> None:
        for row in audit_clause_density(self.enumeration.states, self.config):
            self.assertGreater(row.distinguishing_states, 0, row.clause)

    def test_dsl_can_express_the_exact_contract(self) -> None:
        parsed = parse_contract(_exact_spec())
        report = evaluate_contract(parsed.bind(self.config), self.enumeration.states, self.config)
        self.assertTrue(report.exact)
        self.assertEqual(report.failures, 0)

    def test_moderate_v2_is_complete_denser_and_dsl_exact(self) -> None:
        config = PriceShoppingConfig.from_json(
            "experiment/configs/shopping_price_moderate_v2.json"
        )
        enumeration = enumerate_reachable(config, max_depth=20, max_states=15000)
        repeated = enumerate_reachable(config, max_depth=20, max_states=15000)
        successes = sum(place_order_precondition(state, config) for state in enumeration.states)
        parsed = parse_contract(_exact_spec())
        report = evaluate_contract(parsed.bind(config), enumeration.states, config)

        self.assertFalse(enumeration.truncated)
        self.assertEqual(enumeration.states, repeated.states)
        self.assertEqual(len(enumeration.states), 397)
        self.assertEqual(successes, 13)
        self.assertGreater(successes / len(enumeration.states), 0.03)
        self.assertTrue(report.exact)

    def test_moderate_v3_preserves_every_clause_witness_and_dsl_exact(self) -> None:
        config = PriceShoppingConfig.from_json(
            "experiment/configs/shopping_price_moderate_v3.json"
        )
        enumeration = enumerate_reachable(config, max_depth=20, max_states=15000)
        repeated = enumerate_reachable(config, max_depth=20, max_states=15000)
        audit = audit_clause_density(enumeration.states, config)
        successes = sum(place_order_precondition(state, config) for state in enumeration.states)
        parsed = parse_contract(_exact_spec())
        report = evaluate_contract(parsed.bind(config), enumeration.states, config)

        self.assertFalse(enumeration.truncated)
        self.assertEqual(enumeration.states, repeated.states)
        self.assertEqual(len(enumeration.states), 462)
        self.assertEqual(successes, 13)
        self.assertTrue(all(row.distinguishing_states > 0 for row in audit), audit)
        self.assertTrue(report.exact)

    def test_easy_v4_has_six_live_clauses_and_dsl_exact(self) -> None:
        config = PriceShoppingConfig.from_json(
            "experiment/configs/shopping_price_easy_v4.json"
        )
        enumeration = enumerate_reachable(config, max_depth=20, max_states=15000)
        repeated = enumerate_reachable(config, max_depth=20, max_states=15000)
        audit = {row.clause: row.distinguishing_states for row in audit_clause_density(enumeration.states, config)}
        successes = sum(place_order_precondition(state, config) for state in enumeration.states)
        parsed = parse_contract(_easy_v4_exact_spec())
        report = evaluate_contract(parsed.bind(config), enumeration.states, config)

        self.assertFalse(enumeration.truncated)
        self.assertEqual(enumeration.states, repeated.states)
        self.assertGreater(successes, 0)
        for clause in (
            "logged_in",
            "empty_order_slot",
            "nonempty_cart",
            "accepted_payment",
            "serviceable_address",
            "sufficient_stock",
        ):
            self.assertGreater(audit[clause], 0, clause)
        self.assertEqual(audit["coupon_eligible"], 0)
        self.assertEqual(audit["payment_capacity"], 0)
        self.assertTrue(report.exact)

    def test_postcondition_counterexample_exposes_observed_transition(self) -> None:
        config = PriceShoppingConfig.from_json(
            "experiment/configs/shopping_price_easy_v4.json"
        )
        enumeration = enumerate_reachable(config, max_depth=20, max_states=15000)
        spec = _easy_v4_exact_spec()
        spec["postcondition"] = {"op": "const", "value": False}
        report = evaluate_contract(parse_contract(spec).bind(config), enumeration.states, config)
        counterexample = next(
            entry
            for entry in report.counterexamples
            if entry.symptom.value == "postcondition_violation"
        )
        self.assertIsNotNone(counterexample.after_state)
        prompt = counterexample_prompt_v3("{}", report, config)
        self.assertIn("- Before:", prompt)
        self.assertIn("- After:", prompt)
        self.assertIn('"order_status": "placed"', prompt)

    def test_prompt_does_not_expose_evaluator_only_policy(self) -> None:
        source = Path("experiment/shopping_price/prompts.py").read_text(encoding="utf-8")
        self.assertNotIn("accepted_coupon_codes", source)
        self.assertNotIn("coupon_excluded_items", source)
        self.assertNotIn("coupon_min_subtotal", source)


if __name__ == "__main__":
    unittest.main()


class RepetitionPenaltyPassThroughTest(unittest.TestCase):
    """`Decoding.repetition_penalty` 는 Shopping-price 호출까지 그대로 전달되어야 한다."""

    class _RecordingClient:
        """`complete` 의 키워드 인자만 기록하는 최소 스텁."""

        model = "stub-model"
        endpoint = "http://127.0.0.1:8000/v1/chat/completions"

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def count_chat_tokens(self, messages) -> int:  # type: ignore[no-untyped-def]
            return 16

        def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(dict(kwargs))
            from ..modeling.client import ChatResponse, Usage

            return ChatResponse(
                text='{"skill": "place_order", "precondition": {"op": "const", "value": true},'
                ' "postcondition": {"op": "const", "value": true}}',
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                latency_s=0.0,
                status=200,
                request={},
                raw={},
                finish_reason="stop",
            )

    def _run_direct(self, repetition_penalty):  # type: ignore[no-untyped-def]
        from ..modeling.runner import Budgets, Decoding
        from ..modeling.runner import RunSpec
        from ..shopping_price.runner import PriceShoppingExperimentRunner

        client = self._RecordingClient()
        runner = PriceShoppingExperimentRunner(
            client=client,  # type: ignore[arg-type]
            config_path="experiment/configs/shopping_price_easy_v4.json",
            max_depth=4,
            max_states=64,
            compact_context=False,
        )
        runner.run_method(
            RunSpec(
                method="direct",
                seed=0,
                decoding=Decoding(
                    temperature=0.2,
                    top_p=0.95,
                    max_tokens=64,
                    repetition_penalty=repetition_penalty,
                ),
                budgets=Budgets(state_budget=2, query_budget=1, token_budget=4000),
            )
        )
        self.assertTrue(client.calls, "모델 호출이 일어나지 않았다")
        return client.calls[0]

    def test_an_unset_penalty_is_forwarded_as_none(self) -> None:
        self.assertIsNone(self._run_direct(None)["repetition_penalty"])

    def test_a_set_penalty_reaches_the_client(self) -> None:
        self.assertEqual(self._run_direct(1.1)["repetition_penalty"], 1.1)
