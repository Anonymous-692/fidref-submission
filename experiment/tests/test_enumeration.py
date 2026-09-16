#!/usr/bin/env python3
"""Tests for the bounded breadth-first reachable-state enumerator."""

from __future__ import annotations

import unittest

from ..environment import (
    OrderStatus,
    ShoppingAction,
    apply_action,
    enumerate_reachable,
    initial_state,
    valid_actions,
)
from .support import TEST_MAX_DEPTH, TEST_MAX_STATES, ready_to_order_env, small_config


class EnumerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = small_config()
        cls.result = enumerate_reachable(
            cls.config, max_depth=TEST_MAX_DEPTH, max_states=TEST_MAX_STATES
        )

    def test_is_reproducible(self) -> None:
        repeat = enumerate_reachable(
            self.config, max_depth=TEST_MAX_DEPTH, max_states=TEST_MAX_STATES
        )
        self.assertEqual(repeat.states, self.result.states)
        self.assertEqual(repeat.transitions, self.result.transitions)
        self.assertEqual(repeat.summary(), self.result.summary())

    def test_has_no_duplicates(self) -> None:
        states = self.result.states
        self.assertEqual(len(set(states)), len(states))
        self.assertEqual(len(self.result.state_set), len(states))
        self.assertEqual(len(self.result.sorted_states), len(states))

    def test_starts_from_the_opening_state(self) -> None:
        opening = initial_state(self.config)
        self.assertEqual(self.result.states[0], opening)
        self.assertEqual(self.result.depth_of(opening), 0)

    def test_transitions_stay_inside_the_state_set(self) -> None:
        known = self.result.state_set
        for transition in self.result.transitions:
            self.assertIn(transition.source, known)
            self.assertIn(transition.target, known)
            outcome = apply_action(transition.source, transition.action, self.config)
            self.assertTrue(outcome.ok)
            self.assertEqual(outcome.state, transition.target)

    def test_depth_one_matches_the_valid_actions_of_the_opening_state(self) -> None:
        opening = initial_state(self.config)
        expected = {
            apply_action(opening, action, self.config).state
            for action in valid_actions(opening, self.config)
        }
        reached = {state for state, depth in self.result.depths if depth == 1}
        self.assertEqual(reached, expected)

    def test_reaches_the_whole_order_lifecycle(self) -> None:
        seen = {state.order_status for state in self.result.states}
        self.assertEqual(seen, set(OrderStatus))

    def test_reaches_states_with_and_without_a_valid_order(self) -> None:
        outcomes = {
            apply_action(state, ShoppingAction.place_order(), self.config).ok
            for state in self.result.states
        }
        self.assertEqual(outcomes, {True, False})

    def test_unbounded_enough_search_is_not_truncated(self) -> None:
        self.assertFalse(self.result.truncated)
        self.assertLessEqual(len(self.result), TEST_MAX_STATES)
        self.assertLessEqual(self.result.deepest_level, TEST_MAX_DEPTH)

    def test_unknown_state_lookup_raises(self) -> None:
        stranger = initial_state(self.config).evolve(payment_method="not_a_method")
        with self.assertRaises(KeyError):
            self.result.depth_of(stranger)


class BoundsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_config()

    def test_zero_depth_returns_only_the_start(self) -> None:
        result = enumerate_reachable(self.config, max_depth=0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.transitions, ())
        self.assertTrue(result.truncated)

    def test_depth_bound_is_respected(self) -> None:
        result = enumerate_reachable(self.config, max_depth=2, max_states=TEST_MAX_STATES)
        self.assertTrue(result.truncated)
        self.assertLessEqual(max(depth for _, depth in result.depths), 2)

    def test_state_bound_is_respected(self) -> None:
        result = enumerate_reachable(self.config, max_depth=TEST_MAX_DEPTH, max_states=5)
        self.assertEqual(len(result), 5)
        self.assertTrue(result.truncated)
        self.assertEqual(len(set(result.states)), 5)

    def test_growing_the_depth_bound_only_adds_states(self) -> None:
        shallow = enumerate_reachable(self.config, max_depth=3, max_states=TEST_MAX_STATES)
        deeper = enumerate_reachable(self.config, max_depth=4, max_states=TEST_MAX_STATES)
        self.assertTrue(shallow.state_set.issubset(deeper.state_set))
        self.assertEqual(deeper.states[: len(shallow.states)], shallow.states)

    def test_custom_start_state(self) -> None:
        start = ready_to_order_env(self.config).state
        result = enumerate_reachable(self.config, start=start, max_depth=1)
        self.assertEqual(result.states[0], start)
        self.assertEqual(
            len(result), 1 + len(set(
                apply_action(start, action, self.config).state
                for action in valid_actions(start, self.config)
            ))
        )

    def test_invalid_bounds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            enumerate_reachable(self.config, max_depth=-1)
        with self.assertRaises(ValueError):
            enumerate_reachable(self.config, max_states=0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
