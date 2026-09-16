#!/usr/bin/env python3
"""Tests for the immutable state and action values."""

from __future__ import annotations

import dataclasses
import unittest

from ..environment import (
    ActionKind,
    EnvConfig,
    OrderStatus,
    ShoppingAction,
    ShoppingState,
    counts_combine,
    counts_covers,
    counts_from_mapping,
    counts_get,
    counts_to_mapping,
    counts_total,
    initial_state,
    sort_states,
)
from ..environment.state import counts_add
from .support import DEFAULT_CONFIG_PATH, SMALL_CONFIG_PATH, small_config


class ItemCountsTest(unittest.TestCase):
    def test_canonical_order_and_zero_dropping(self) -> None:
        counts = counts_from_mapping({"pen": 2, "book": 1, "lamp": 0})
        self.assertEqual(counts, (("book", 1), ("pen", 2)))

    def test_equal_multisets_are_identical_tuples(self) -> None:
        self.assertEqual(
            counts_from_mapping({"a": 1, "b": 2}), counts_from_mapping({"b": 2, "a": 1})
        )

    def test_negative_quantities_rejected(self) -> None:
        with self.assertRaises(ValueError):
            counts_from_mapping({"book": -1})
        with self.assertRaises(ValueError):
            counts_add((("book", 1),), "book", -2)
        with self.assertRaises(ValueError):
            counts_combine((("book", 1),), (("book", 2),), sign=-1)

    def test_helpers(self) -> None:
        counts = counts_from_mapping({"book": 1, "pen": 3})
        self.assertEqual(counts_get(counts, "book"), 1)
        self.assertEqual(counts_get(counts, "missing"), 0)
        self.assertEqual(counts_total(counts), 4)
        self.assertEqual(counts_to_mapping(counts), {"book": 1, "pen": 3})
        self.assertTrue(counts_covers(counts, (("book", 1),)))
        self.assertFalse(counts_covers(counts, (("book", 2),)))
        self.assertEqual(counts_combine(counts, (("book", 1),)), (("book", 2), ("pen", 3)))
        self.assertEqual(counts_add(counts, "pen", -3), (("book", 1),))

    def test_combine_rejects_bad_sign(self) -> None:
        with self.assertRaises(ValueError):
            counts_combine((), (), sign=0)


class ShoppingActionTest(unittest.TestCase):
    def test_target_requirements(self) -> None:
        with self.assertRaises(ValueError):
            ShoppingAction(ActionKind.ADD_TO_CART)
        with self.assertRaises(ValueError):
            ShoppingAction(ActionKind.LOGIN, "book")

    def test_value_equality_and_hashing(self) -> None:
        self.assertEqual(ShoppingAction.add_to_cart("book"), ShoppingAction.add_to_cart("book"))
        self.assertEqual(len({ShoppingAction.login(), ShoppingAction.login()}), 1)

    def test_rendering_and_sort_key(self) -> None:
        self.assertEqual(str(ShoppingAction.login()), "login")
        self.assertEqual(str(ShoppingAction.add_to_cart("book")), "add_to_cart(book)")
        self.assertLess(ShoppingAction.login().sort_key, ShoppingAction.place_order().sort_key)
        self.assertEqual(
            ShoppingAction.set_payment("card").to_dict(),
            {"kind": "SET_PAYMENT", "target": "card"},
        )

    def test_constructors_cover_every_kind(self) -> None:
        built = {
            ShoppingAction.login().kind,
            ShoppingAction.logout().kind,
            ShoppingAction.add_to_cart("book").kind,
            ShoppingAction.remove_from_cart("book").kind,
            ShoppingAction.clear_cart().kind,
            ShoppingAction.set_payment("card").kind,
            ShoppingAction.clear_payment().kind,
            ShoppingAction.set_address("home").kind,
            ShoppingAction.clear_address().kind,
            ShoppingAction.place_order().kind,
            ShoppingAction.confirm_order().kind,
            ShoppingAction.cancel_order().kind,
            ShoppingAction.clear_order().kind,
        }
        self.assertEqual(built, set(ActionKind))


class ShoppingStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_config()
        self.state = initial_state(self.config)

    def test_state_is_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.state.logged_in = True  # type: ignore[misc]

    def test_evolve_leaves_the_original_untouched(self) -> None:
        changed = self.state.evolve(logged_in=True)
        self.assertTrue(changed.logged_in)
        self.assertFalse(self.state.logged_in)
        self.assertNotEqual(changed, self.state)

    def test_states_are_hashable_and_compare_by_value(self) -> None:
        twin = initial_state(self.config)
        self.assertEqual(self.state, twin)
        self.assertEqual(len({self.state, twin}), 1)

    def test_initial_state_covers_every_condition_family(self) -> None:
        self.assertFalse(self.state.logged_in)
        self.assertTrue(self.state.cart_is_empty)
        self.assertEqual(self.state.cart_size, 0)
        self.assertIsNone(self.state.payment_method)
        self.assertIsNone(self.state.shipping_address)
        self.assertIs(self.state.order_status, OrderStatus.NONE)
        self.assertEqual(self.state.order_items, ())
        for item in self.config.items:
            self.assertGreaterEqual(self.state.stock_quantity(item), 1)
            self.assertEqual(self.state.cart_quantity(item), 0)

    def test_order_status_covers_the_lifecycle(self) -> None:
        self.assertEqual(
            {status.value for status in OrderStatus},
            {"none", "placed", "confirmed", "cancelled"},
        )

    def test_key_and_dict_are_serialisable(self) -> None:
        state = self.state.evolve(logged_in=True, payment_method="card")
        self.assertEqual(state.key[0], True)
        self.assertEqual(state.key[5], "none")
        payload = state.to_dict()
        self.assertEqual(payload["payment_method"], "card")
        self.assertEqual(payload["order_status"], "none")
        self.assertIn("login=y", state.describe())

    def test_sort_states_is_a_total_stable_order(self) -> None:
        states = [self.state.evolve(logged_in=True), self.state]
        self.assertEqual(sort_states(states), sort_states(reversed(states)))


class EnvConfigTest(unittest.TestCase):
    def test_json_configs_load(self) -> None:
        for path in (SMALL_CONFIG_PATH, DEFAULT_CONFIG_PATH):
            config = EnvConfig.from_json_file(path)
            self.assertGreaterEqual(len(config.items), 1)

    def test_round_trip(self) -> None:
        config = small_config()
        self.assertEqual(EnvConfig.from_dict(config.to_dict()), config)

    def test_normalisation_and_validation(self) -> None:
        config = EnvConfig(items=("pen", "book", "pen"))
        self.assertEqual(config.items, ("book", "pen"))
        with self.assertRaises(ValueError):
            EnvConfig(max_stock=0)
        with self.assertRaises(ValueError):
            EnvConfig(cart_capacity=0)
        with self.assertRaises(ValueError):
            EnvConfig(items=())
        with self.assertRaises(ValueError):
            EnvConfig(valid_payment_methods=("card",), rejected_payment_methods=("card",))
        with self.assertRaises(ValueError):
            EnvConfig(valid_addresses=("home",), rejected_addresses=("home",))
        with self.assertRaises(ValueError):
            EnvConfig.from_dict({"items": ["book"], "nonsense": 1})

    def test_seed_determines_initial_stock(self) -> None:
        left = EnvConfig(seed=11, max_stock=4)
        right = EnvConfig(seed=11, max_stock=4)
        self.assertEqual(left.initial_stock(), right.initial_stock())
        self.assertEqual(left.initial_stock(), left.initial_stock())
        for _, quantity in left.initial_stock():
            self.assertTrue(1 <= quantity <= 4)

    def test_options_include_rejected_values(self) -> None:
        config = small_config()
        self.assertIn("expired_card", config.payment_options)
        self.assertIn("po_box", config.address_options)
        self.assertFalse(config.is_valid_payment("expired_card"))
        self.assertFalse(config.is_valid_payment(None))
        self.assertTrue(config.is_valid_payment("card"))
        self.assertFalse(config.is_valid_address("po_box"))
        self.assertFalse(config.is_valid_address(None))
        self.assertTrue(config.is_valid_address("home"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
