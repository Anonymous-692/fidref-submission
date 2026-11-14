#!/usr/bin/env python3
"""Evaluator-only contract for ``place_order`` in price-aware shopping."""

from __future__ import annotations

from .config import PriceShoppingConfig
from .state import OrderStatus, PriceShoppingState, counts_combine, counts_covers

VISIBILITY = "evaluator-only"

CLAUSE_NAMES = (
    "logged_in",
    "empty_order_slot",
    "nonempty_cart",
    "accepted_payment",
    "serviceable_address",
    "sufficient_stock",
    "coupon_eligible",
    "payment_capacity",
)


def clause_values(state: PriceShoppingState, config: PriceShoppingConfig) -> tuple[bool, ...]:
    coupon_ok = state.coupon_code is None or config.coupon_eligible(state.coupon_code, state.cart)
    payable = config.payable_total(state)
    payment_capacity = (
        (state.payment_method == "wallet" and state.wallet_balance >= payable)
        or (state.payment_method != "wallet" and payable <= config.card_limit)
    )
    return (
        state.logged_in,
        state.order_status is OrderStatus.NONE,
        not state.cart_is_empty,
        state.payment_method in config.accepted_payment_methods,
        state.shipping_address in config.serviceable_addresses,
        counts_covers(state.stock, state.cart),
        coupon_ok,
        payment_capacity,
    )


def place_order_precondition(state: PriceShoppingState, config: PriceShoppingConfig) -> bool:
    return all(clause_values(state, config))


def place_order_postcondition(
    before: PriceShoppingState,
    after: PriceShoppingState,
    config: PriceShoppingConfig,
) -> bool:
    payable = config.payable_total(before)
    expected_wallet = before.wallet_balance - payable if before.payment_method == "wallet" else before.wallet_balance
    expected_payment = None if config.clear_checkout_selections_after_order else before.payment_method
    expected_address = None if config.clear_checkout_selections_after_order else before.shipping_address
    return (
        after.order_status is OrderStatus.PLACED
        and after.order_items == before.cart
        and after.cart == ()
        and after.stock == counts_combine(before.stock, before.cart, sign=-1)
        and after.order_total == payable
        and after.order_payment_method == before.payment_method
        and after.wallet_balance == expected_wallet
        and after.card_limit == before.card_limit
        and after.cart_subtotal == 0
        and after.checkout_total == 0
        and after.coupon_code is None
        and after.logged_in == before.logged_in
        and after.payment_method == expected_payment
        and after.shipping_address == expected_address
    )
