#!/usr/bin/env python3
"""Tests for the hidden ground truth and its separation from the sandbox."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from .. import environment, evaluation
from ..environment import (
    OrderStatus,
    ShoppingAction,
    ShoppingEnv,
    apply_action,
    enumerate_reachable,
)
from ..evaluation import (
    evaluate_contract,
    ground_truth_contract,
    place_order_postcondition,
    place_order_precondition,
)
from .support import TEST_MAX_DEPTH, TEST_MAX_STATES, ready_to_order_env, small_config

FORBIDDEN_TOKENS = ("ground_truth", "ground truth", "precondition", "postcondition", "contract")


class GroundTruthAgreementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = small_config()
        cls.states = enumerate_reachable(
            cls.config, max_depth=TEST_MAX_DEPTH, max_states=TEST_MAX_STATES
        ).states
        cls.report = evaluate_contract(ground_truth_contract(cls.config), cls.states, cls.config)

    def test_the_state_set_is_not_vacuous(self) -> None:
        self.assertGreater(self.report.successes, 0)
        self.assertGreater(self.report.failures, 0)
        self.assertEqual(self.report.states_checked, len(self.states))

    def test_ground_truth_is_exact_on_every_reachable_state(self) -> None:
        self.assertEqual(self.report.false_accepts, 0)
        self.assertEqual(self.report.false_rejects, 0)
        self.assertEqual(self.report.postcondition_violations, 0)
        self.assertEqual(self.report.symptoms, frozenset())
        self.assertTrue(self.report.is_sound)
        self.assertTrue(self.report.is_complete)
        self.assertTrue(self.report.is_exact)
        self.assertEqual(self.report.counterexamples, ())

    def test_precondition_matches_the_sandbox_state_by_state(self) -> None:
        for state in self.states:
            outcome = apply_action(state, ShoppingAction.place_order(), self.config)
            self.assertEqual(
                place_order_precondition(state, self.config),
                outcome.ok,
                msg=state.describe(),
            )

    def test_postcondition_holds_on_every_success(self) -> None:
        checked = 0
        for state in self.states:
            outcome = apply_action(state, ShoppingAction.place_order(), self.config)
            if outcome.ok:
                checked += 1
                self.assertTrue(place_order_postcondition(state, outcome.state))
        self.assertGreater(checked, 0)

    def test_postcondition_rejects_a_tampered_effect(self) -> None:
        env = ready_to_order_env(self.config)
        before = env.state
        after = env.step(ShoppingAction.place_order()).state
        self.assertTrue(place_order_postcondition(before, after))
        self.assertFalse(place_order_postcondition(before, after.evolve(stock=before.stock)))
        self.assertFalse(place_order_postcondition(before, after.evolve(cart=before.cart)))
        self.assertFalse(
            place_order_postcondition(before, after.evolve(order_status=OrderStatus.CONFIRMED))
        )
        self.assertFalse(place_order_postcondition(before, after.evolve(logged_in=False)))

    def test_insufficient_stock_is_covered_by_the_precondition(self) -> None:
        env = ready_to_order_env(self.config)
        item = self.config.items[0]
        # One unit in stock, two in the cart.
        self.assertTrue(env.step(ShoppingAction.add_to_cart(item)).ok)
        state = env.state
        self.assertGreater(state.cart_quantity(item), state.stock_quantity(item))
        self.assertFalse(place_order_precondition(state, self.config))
        self.assertFalse(apply_action(state, ShoppingAction.place_order(), self.config).ok)


class GroundTruthSeparationTest(unittest.TestCase):
    """The agent-facing package must not carry the answer."""

    def setUp(self) -> None:
        self.environment_dir = Path(environment.__file__).resolve().parent
        self.sources = sorted(self.environment_dir.glob("*.py"))

    def test_there_are_sources_to_inspect(self) -> None:
        self.assertGreaterEqual(len(self.sources), 4)

    def test_environment_never_imports_the_evaluator(self) -> None:
        for path in self.sources:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    self.assertNotIn("evaluation", name, msg=f"{path.name} imports {name}")
                    self.assertNotIn("ground_truth", name, msg=f"{path.name} imports {name}")

    def test_environment_sources_are_free_of_specification_vocabulary(self) -> None:
        for path in self.sources:
            text = path.read_text(encoding="utf-8").lower()
            for token in FORBIDDEN_TOKENS:
                self.assertNotIn(token, text, msg=f"{path.name} mentions {token!r}")

    def test_environment_api_exposes_no_specification(self) -> None:
        names = list(environment.__all__) + dir(ShoppingEnv) + dir(environment)
        for name in names:
            lowered = name.lower()
            for token in ("contract", "precondition", "postcondition", "truth"):
                self.assertNotIn(token, lowered, msg=f"{name} leaks the answer")

    def test_an_env_instance_carries_no_hidden_specification(self) -> None:
        env = ShoppingEnv(small_config())
        for name in vars(env):
            self.assertNotIn("contract", name.lower())
            self.assertNotIn("truth", name.lower())

    def test_the_dependency_runs_one_way_only(self) -> None:
        evaluation_dir = Path(evaluation.__file__).resolve().parent
        imports_environment = False
        for path in sorted(evaluation_dir.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and "environment" in (node.module or ""):
                    imports_environment = True
        self.assertTrue(imports_environment, "the evaluator should drive the sandbox")

    def test_ground_truth_is_marked_evaluator_only(self) -> None:
        self.assertEqual(evaluation.VISIBILITY, "evaluator-only")
        self.assertGreaterEqual(len(evaluation.GROUND_TRUTH_CLAUSES), 6)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
