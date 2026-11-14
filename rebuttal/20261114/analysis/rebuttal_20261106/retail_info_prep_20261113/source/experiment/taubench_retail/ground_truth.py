#!/usr/bin/env python3
"""Evaluator-only reference contract and ground truth for exchange_delivered_order_items."""

from __future__ import annotations

from .config import RetailConfig
from .state import RetailState

VISIBILITY = "evaluator-only"

CLAUSE_NAMES = (
    "order_exists",
    "order_delivered",
    "items_in_order",
    "item_count_matches",
    "new_items_valid_variant",
    "new_items_available",
    "payment_method_exists",
    "sufficient_balance_if_gift_card",
)

# Product variant catalog mapping item_id -> product_id
ITEM_TO_PRODUCT: dict[str, str] = {
    "shoe_black_9": "prod_shoes",
    "shoe_black_10": "prod_shoes",
    "shoe_red_9": "prod_shoes",
    "shoe_blue_9": "prod_shoes",
    "shirt_blue_m": "prod_shirt",
    "shirt_blue_l": "prod_shirt",
    "socks_white": "prod_socks",
}

# Product variant prices
ITEM_PRICES: dict[str, float] = {
    "shoe_black_9": 100.0,
    "shoe_black_10": 100.0,
    "shoe_red_9": 130.0,
    "shoe_blue_9": 100.0,
    "shirt_blue_m": 40.0,
    "shirt_blue_l": 60.0,
    "socks_white": 25.0,
}

# Product variant availability
ITEM_AVAILABLE: dict[str, bool] = {
    "shoe_black_9": True,
    "shoe_black_10": True,
    "shoe_red_9": True,
    "shoe_blue_9": False,
    "shirt_blue_m": True,
    "shirt_blue_l": True,
    "socks_white": True,
}

# Items in each order
ORDER_ITEMS: dict[str, tuple[str, ...]] = {
    "#W1": ("shoe_black_9", "shirt_blue_m"),
    "#W2": ("socks_white",),
}


def calculate_price_difference(items: tuple[str, ...], new_items: tuple[str, ...]) -> float:
    """Calculate the net price difference for an item exchange."""
    diff = 0.0
    for old_it, new_it in zip(items, new_items):
        old_price = ITEM_PRICES.get(old_it, 0.0)
        new_price = ITEM_PRICES.get(new_it, 0.0)
        diff += new_price - old_price
    return round(diff, 2)


def clause_values(state: RetailState, config: RetailConfig) -> tuple[bool, ...]:
    """Evaluate each reference clause independently on the state."""
    # 1. order_exists: draft order is in known orders
    order_exists = state.draft_order_id in config.valid_orders

    # 2. order_delivered: if order exists, it must be currently delivered
    if state.draft_order_id == "#W1":
        order_delivered = state.order_w1_status == "delivered"
    elif state.draft_order_id == "#W2":
        order_delivered = state.order_w2_status == "delivered"
    else:
        # Nonexistent / null order: do not penalize twice so order_exists is distinguishable
        order_delivered = True

    # 3. items_in_order: all drafted items are contained in the order
    ref_order = state.draft_order_id if order_exists else "#W1"
    order_items = ORDER_ITEMS.get(ref_order or "", ())
    items_in_order = all(
        state.draft_item_ids.count(it) <= order_items.count(it)
        for it in state.draft_item_ids
    )

    # 4. item_count_matches: count of items equals count of new items
    item_count_matches = len(state.draft_item_ids) == len(state.draft_new_item_ids)

    # 5. new_items_valid_variant: each new item is a variant of the same product as the old item
    new_items_valid_variant = True
    for old_it, new_it in zip(state.draft_item_ids, state.draft_new_item_ids):
        old_prod = ITEM_TO_PRODUCT.get(old_it)
        new_prod = ITEM_TO_PRODUCT.get(new_it)
        if old_prod is None or new_prod is None or old_prod != new_prod:
            new_items_valid_variant = False
            break

    # 6. new_items_available: each new item is currently available (in stock)
    new_items_available = all(
        ITEM_AVAILABLE.get(nit, False)
        for nit in state.draft_new_item_ids
    )

    # 7. payment_method_exists: payment method exists for the user
    payment_method_exists = state.draft_payment_method_id in config.valid_payment_methods

    # 8. sufficient_balance_if_gift_card: gift card has sufficient balance for positive price diff
    if state.draft_payment_method_id == "gift_card_0":
        diff = calculate_price_difference(state.draft_item_ids, state.draft_new_item_ids)
        sufficient_balance_if_gift_card = round(state.user_gift_card_balance, 2) >= diff
    else:
        sufficient_balance_if_gift_card = True

    return (
        order_exists,
        order_delivered,
        items_in_order,
        item_count_matches,
        new_items_valid_variant,
        new_items_available,
        payment_method_exists,
        sufficient_balance_if_gift_card,
    )


def exchange_items_precondition(state: RetailState, config: RetailConfig) -> bool:
    return all(clause_values(state, config))


def exchange_items_postcondition(
    before: RetailState,
    after: RetailState,
    config: RetailConfig,
    *,
    vocabulary_protocol: str | None = None,
) -> bool:
    from .state import SEMANTIC_FIELDS_PROTOCOL
    if vocabulary_protocol == SEMANTIC_FIELDS_PROTOCOL:
        if before.draft_order_id != "#W1" or not (
            before.order_w1_return_items == after.order_w1_return_items
            and before.order_w1_return_payment_method_id == after.order_w1_return_payment_method_id
            and before.authenticated == after.authenticated
        ):
            return False
    expected_diff = calculate_price_difference(before.draft_item_ids, before.draft_new_item_ids)
    return (
        after.order_w1_status == "exchange requested"
        and after.order_w1_exchange_items == tuple(sorted(before.draft_item_ids))
        and after.order_w1_exchange_new_items == tuple(sorted(before.draft_new_item_ids))
        and after.order_w1_exchange_payment_method_id == before.draft_payment_method_id
        and after.order_w1_exchange_price_difference == expected_diff
        and after.order_w2_status == before.order_w2_status
        and after.order_w2_cancel_reason == before.order_w2_cancel_reason
        and round(after.user_gift_card_balance, 2) == round(before.user_gift_card_balance, 2)
        and after.draft_order_id is None
        and after.draft_item_ids == ()
        and after.draft_new_item_ids == ()
        and after.draft_payment_method_id is None
    )
