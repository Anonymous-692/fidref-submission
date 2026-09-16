#!/usr/bin/env python3
"""Tests for deterministic transitions, rejection, and snapshot/restore."""

from __future__ import annotations

import unittest

from ..environment import (
    EnvConfig,
    EnvSnapshot,
    OrderStatus,
    ShoppingAction,
    ShoppingEnv,
    action_space,
    apply_action,
    initial_state,
    valid_actions,
)
from .support import ready_to_order_env, ready_to_order_script, small_config


class ResetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_config()

    def test_reset_is_deterministic_across_instances(self) -> None:
        self.assertEqual(ShoppingEnv(self.config).state, ShoppingEnv(self.config).state)

    def test_reset_clears_state_and_counters(self) -> None:
        env = ShoppingEnv(self.config)
        opening = env.state
        env.step(ShoppingAction.login())
        env.step(ShoppingAction.login())  # rejected
        self.assertEqual(env.step_count, 2)
        self.assertEqual(env.rejected_count, 1)
        self.assertEqual(env.reset(), opening)
        self.assertEqual(env.step_count, 0)
        self.assertEqual(env.rejected_count, 0)

    def test_same_seed_same_stock_across_configs(self) -> None:
        left = ShoppingEnv(EnvConfig(seed=99, max_stock=5))
        right = ShoppingEnv(EnvConfig(seed=99, max_stock=5))
        self.assertEqual(left.state.stock, right.state.stock)


class DeterministicTransitionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_config()

    def test_repeated_application_gives_identical_results(self) -> None:
        state = initial_state(self.config)
        for action in action_space(self.config):
            first = apply_action(state, action, self.config)
            second = apply_action(state, action, self.config)
            self.assertEqual(first, second, msg=str(action))

    def test_identical_scripts_produce_identical_traces(self) -> None:
        script = ready_to_order_script(self.config) + [
            ShoppingAction.place_order(),
            ShoppingAction.confirm_order(),
        ]
        left = ShoppingEnv(self.config).run(script)
        right = ShoppingEnv(self.config).run(script)
        self.assertEqual(left, right)
        self.assertTrue(all(result.ok for result in left))

    def test_transition_function_does_not_mutate_its_input(self) -> None:
        state = initial_state(self.config)
        snapshot_of_state = state.to_dict()
        apply_action(state, ShoppingAction.login(), self.config)
        self.assertEqual(state.to_dict(), snapshot_of_state)

    def test_logout_preserves_cart_payment_and_address(self) -> None:
        env = ready_to_order_env(self.config)
        before = env.state
        self.assertTrue(env.step(ShoppingAction.logout()).ok)
        after = env.state
        self.assertFalse(after.logged_in)
        self.assertEqual(after.cart, before.cart)
        self.assertEqual(after.payment_method, before.payment_method)
        self.assertEqual(after.shipping_address, before.shipping_address)

    def test_add_to_cart_ignores_stock_but_respects_capacity(self) -> None:
        env = ShoppingEnv(self.config)
        item = self.config.items[0]
        self.assertEqual(env.state.stock_quantity(item), 1)
        for _ in range(self.config.cart_capacity):
            self.assertTrue(env.step(ShoppingAction.add_to_cart(item)).ok)
        # The cart may exceed stock; only ordering enforces availability.
        self.assertGreater(env.state.cart_quantity(item), env.state.stock_quantity(item))
        full = env.step(ShoppingAction.add_to_cart(item))
        self.assertFalse(full.ok)
        self.assertEqual(full.error, "cart is full")


class RejectedActionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_config()
        self.env = ShoppingEnv(self.config)

    def assert_rejected(self, action: ShoppingAction, fragment: str) -> None:
        before = self.env.state
        result = self.env.step(action)
        self.assertFalse(result.ok, msg=f"{action} unexpectedly succeeded")
        self.assertIn(fragment, result.error or "")
        self.assertEqual(self.env.state, before, msg=f"{action} mutated the state")

    def test_rejections_from_the_opening_state(self) -> None:
        self.assert_rejected(ShoppingAction.logout(), "not logged in")
        self.assert_rejected(ShoppingAction.remove_from_cart("book"), "not in the cart")
        self.assert_rejected(ShoppingAction.clear_cart(), "already empty")
        self.assert_rejected(ShoppingAction.clear_payment(), "no payment method")
        self.assert_rejected(ShoppingAction.clear_address(), "no address")
        self.assert_rejected(ShoppingAction.place_order(), "not logged in")
        self.assert_rejected(ShoppingAction.confirm_order(), "no placed order")
        self.assert_rejected(ShoppingAction.cancel_order(), "no placed order")
        self.assert_rejected(ShoppingAction.clear_order(), "settled order")
        self.assertEqual(self.env.rejected_count, 9)

    def test_unknown_targets_are_rejected(self) -> None:
        self.assert_rejected(ShoppingAction.add_to_cart("unicorn"), "unknown item")
        self.assert_rejected(ShoppingAction.remove_from_cart("unicorn"), "unknown item")
        self.assert_rejected(ShoppingAction.set_payment("bitcoin"), "unknown payment method")
        self.assert_rejected(ShoppingAction.set_address("moon"), "unknown address")

    def test_redundant_settings_are_rejected(self) -> None:
        self.assertTrue(self.env.step(ShoppingAction.set_payment("card")).ok)
        self.assert_rejected(ShoppingAction.set_payment("card"), "already")
        self.assertTrue(self.env.step(ShoppingAction.set_address("home")).ok)
        self.assert_rejected(ShoppingAction.set_address("home"), "already")

    def test_login_twice_is_rejected(self) -> None:
        self.assertTrue(self.env.step(ShoppingAction.login()).ok)
        self.assert_rejected(ShoppingAction.login(), "already logged in")

    def test_step_rejects_non_actions(self) -> None:
        with self.assertRaises(TypeError):
            self.env.step("login")  # type: ignore[arg-type]


class PlaceOrderGuardTest(unittest.TestCase):
    """Each requirement of ``place_order`` must block it on its own."""

    def setUp(self) -> None:
        self.config = small_config()
        self.ready = ready_to_order_env(self.config).state

    def assert_blocked(self, state, fragment: str) -> None:
        result = apply_action(state, ShoppingAction.place_order(), self.config)
        self.assertFalse(result.ok)
        self.assertIn(fragment, result.error or "")
        self.assertEqual(result.state, state)

    def test_ready_state_succeeds(self) -> None:
        result = apply_action(self.ready, ShoppingAction.place_order(), self.config)
        self.assertTrue(result.ok)
        self.assertIsNone(result.error)

    def test_logged_out_blocks(self) -> None:
        self.assert_blocked(self.ready.evolve(logged_in=False), "not logged in")

    def test_empty_cart_blocks(self) -> None:
        self.assert_blocked(self.ready.evolve(cart=()), "cart is empty")

    def test_missing_payment_blocks(self) -> None:
        self.assert_blocked(self.ready.evolve(payment_method=None), "payment method")

    def test_rejected_payment_blocks(self) -> None:
        self.assert_blocked(self.ready.evolve(payment_method="expired_card"), "payment method")

    def test_missing_address_blocks(self) -> None:
        self.assert_blocked(self.ready.evolve(shipping_address=None), "shipping address")

    def test_unserviceable_address_blocks(self) -> None:
        self.assert_blocked(self.ready.evolve(shipping_address="po_box"), "shipping address")

    def test_insufficient_stock_blocks(self) -> None:
        self.assert_blocked(self.ready.evolve(stock=()), "insufficient stock")

    def test_occupied_order_slot_blocks(self) -> None:
        self.assert_blocked(self.ready.evolve(order_status=OrderStatus.PLACED), "already placed")


class OrderLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_config()
        self.env = ready_to_order_env(self.config)
        self.before_order = self.env.state

    def test_place_moves_goods_from_stock_to_the_order(self) -> None:
        self.assertTrue(self.env.step(ShoppingAction.place_order()).ok)
        state = self.env.state
        self.assertIs(state.order_status, OrderStatus.PLACED)
        self.assertEqual(state.order_items, self.before_order.cart)
        self.assertEqual(state.cart, ())
        for item, quantity in self.before_order.cart:
            self.assertEqual(
                state.stock_quantity(item), self.before_order.stock_quantity(item) - quantity
            )

    def test_confirm_then_clear_frees_the_order_slot(self) -> None:
        self.env.step(ShoppingAction.place_order())
        self.assertTrue(self.env.step(ShoppingAction.confirm_order()).ok)
        self.assertIs(self.env.state.order_status, OrderStatus.CONFIRMED)
        self.assertFalse(self.env.step(ShoppingAction.cancel_order()).ok)
        self.assertTrue(self.env.step(ShoppingAction.clear_order()).ok)
        self.assertIs(self.env.state.order_status, OrderStatus.NONE)
        self.assertEqual(self.env.state.order_items, ())

    def test_cancel_returns_the_goods_to_stock(self) -> None:
        self.env.step(ShoppingAction.place_order())
        self.assertTrue(self.env.step(ShoppingAction.cancel_order()).ok)
        state = self.env.state
        self.assertIs(state.order_status, OrderStatus.CANCELLED)
        self.assertEqual(state.stock, self.before_order.stock)
        self.assertEqual(state.order_items, ())
        self.assertTrue(self.env.step(ShoppingAction.clear_order()).ok)
        self.assertIs(self.env.state.order_status, OrderStatus.NONE)


class ValidActionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_config()

    def test_valid_actions_agree_with_step(self) -> None:
        env = ready_to_order_env(self.config)
        allowed = set(env.valid_actions())
        self.assertTrue(allowed)
        for action in env.action_space():
            outcome = apply_action(env.state, action, self.config)
            self.assertEqual(action in allowed, outcome.ok, msg=str(action))

    def test_valid_actions_are_ordered_and_stable(self) -> None:
        state = ready_to_order_env(self.config).state
        first = valid_actions(state, self.config)
        second = valid_actions(state, self.config)
        self.assertEqual(first, second)
        self.assertEqual(list(first), sorted(first, key=lambda action: action.sort_key))

    def test_action_space_is_a_stable_superset(self) -> None:
        space = action_space(self.config)
        self.assertEqual(space, action_space(self.config))
        self.assertEqual(len(set(space)), len(space))
        state = ready_to_order_env(self.config).state
        self.assertTrue(set(valid_actions(state, self.config)).issubset(set(space)))
        self.assertIn(ShoppingAction.place_order(), space)
        self.assertIn(ShoppingAction.set_payment("expired_card"), space)

    def test_place_order_appears_only_when_it_is_allowed(self) -> None:
        env = ShoppingEnv(self.config)
        self.assertNotIn(ShoppingAction.place_order(), env.valid_actions())
        env = ready_to_order_env(self.config)
        self.assertIn(ShoppingAction.place_order(), env.valid_actions())


class SnapshotRestoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_config()

    def test_round_trip_restores_state_and_counters(self) -> None:
        env = ready_to_order_env(self.config)
        snapshot = env.snapshot()
        marked_state = env.state
        marked_steps = env.step_count

        env.step(ShoppingAction.place_order())
        env.step(ShoppingAction.confirm_order())
        self.assertNotEqual(env.state, marked_state)

        self.assertEqual(env.restore(snapshot), marked_state)
        self.assertEqual(env.state, marked_state)
        self.assertEqual(env.step_count, marked_steps)
        self.assertEqual(env.rejected_count, snapshot.rejected_count)

    def test_snapshot_is_unaffected_by_later_steps(self) -> None:
        env = ready_to_order_env(self.config)
        snapshot = env.snapshot()
        captured = snapshot.state
        env.step(ShoppingAction.place_order())
        self.assertEqual(snapshot.state, captured)

    def test_restoring_replays_identically(self) -> None:
        env = ready_to_order_env(self.config)
        snapshot = env.snapshot()
        first = env.step(ShoppingAction.place_order())
        env.restore(snapshot)
        second = env.step(ShoppingAction.place_order())
        self.assertEqual(first, second)

    def test_restore_across_configurations_is_refused(self) -> None:
        snapshot = ShoppingEnv(self.config).snapshot()
        other = ShoppingEnv(EnvConfig(seed=1, items=("book", "pen"), max_stock=1))
        with self.assertRaises(ValueError):
            other.restore(snapshot)

    def test_restore_rejects_foreign_objects(self) -> None:
        env = ShoppingEnv(self.config)
        with self.assertRaises(TypeError):
            env.restore(env.state)  # type: ignore[arg-type]

    def test_snapshot_carries_the_configuration_fingerprint(self) -> None:
        env = ShoppingEnv(self.config)
        snapshot = env.snapshot()
        self.assertIsInstance(snapshot, EnvSnapshot)
        self.assertEqual(snapshot.config_fingerprint, self.config.fingerprint)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
