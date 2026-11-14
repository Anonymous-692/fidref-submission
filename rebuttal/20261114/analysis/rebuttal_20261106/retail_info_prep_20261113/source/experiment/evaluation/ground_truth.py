#!/usr/bin/env python3
"""Evaluator-only ground-truth contract for the ``place_order`` skill.

This module is the hidden reference specification. It is imported by scoring
and reporting code only; ``experiment.environment`` never imports it, so an
agent driving the sandbox cannot read the answer out of the environment API.
"""

from __future__ import annotations

from ..environment import (
    EnvConfig,
    OrderStatus,
    ShoppingAction,
    ShoppingState,
    counts_combine,
    counts_covers,
)
from .contracts import Contract

VISIBILITY = "evaluator-only"

GROUND_TRUTH_CLAUSES = (
    "the shopper is logged in",
    "no order occupies the order slot",
    "the cart is not empty",
    "the selected payment method is accepted",
    "the shipping address is serviceable",
    "stock covers every cart line",
)


def place_order_precondition(state: ShoppingState, config: EnvConfig) -> bool:
    """True exactly where the sandbox will accept ``place_order``."""
    return (
        state.logged_in
        and state.order_status is OrderStatus.NONE
        and not state.cart_is_empty
        and config.is_valid_payment(state.payment_method)
        and config.is_valid_address(state.shipping_address)
        and counts_covers(state.stock, state.cart)
    )


def place_order_postcondition(before: ShoppingState, after: ShoppingState) -> bool:
    """True exactly when the observed effect matches the intended one."""
    return (
        after.order_status is OrderStatus.PLACED
        and after.order_items == before.cart
        and after.cart == ()
        and after.stock == counts_combine(before.stock, before.cart, sign=-1)
        and after.logged_in == before.logged_in
        and after.payment_method == before.payment_method
        and after.shipping_address == before.shipping_address
    )


def ground_truth_contract(config: EnvConfig) -> Contract:
    """Build the reference contract bound to one sandbox configuration."""
    return Contract(
        name="ground_truth.place_order",
        action=ShoppingAction.place_order(),
        precondition=lambda state: place_order_precondition(state, config),
        postcondition=place_order_postcondition,
        description="Reference specification: " + "; ".join(GROUND_TRUTH_CLAUSES),
    )
