#!/usr/bin/env python3
"""Immutable state and actions for the price-aware shopping domain."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

ItemCounts = tuple[tuple[str, int], ...]


def counts_from_mapping(mapping: Mapping[str, int]) -> ItemCounts:
    for item, quantity in mapping.items():
        if quantity < 0:
            raise ValueError(f"negative quantity for {item!r}: {quantity}")
    return tuple(sorted((str(item), int(qty)) for item, qty in mapping.items() if qty > 0))


def counts_to_mapping(counts: ItemCounts) -> dict[str, int]:
    return dict(counts)


def counts_get(counts: ItemCounts, item: str) -> int:
    return dict(counts).get(item, 0)


def counts_total(counts: ItemCounts) -> int:
    return sum(quantity for _, quantity in counts)


def counts_add(counts: ItemCounts, item: str, delta: int) -> ItemCounts:
    updated = counts_to_mapping(counts)
    quantity = updated.get(item, 0) + delta
    if quantity < 0:
        raise ValueError(f"quantity for {item!r} would become negative")
    updated[item] = quantity
    return counts_from_mapping(updated)


def counts_combine(left: ItemCounts, right: ItemCounts, sign: int = 1) -> ItemCounts:
    if sign not in (1, -1):
        raise ValueError("sign must be 1 or -1")
    result = counts_to_mapping(left)
    for item, quantity in right:
        value = result.get(item, 0) + sign * quantity
        if value < 0:
            raise ValueError(f"quantity for {item!r} would become negative")
        result[item] = value
    return counts_from_mapping(result)


def counts_covers(available: ItemCounts, required: ItemCounts) -> bool:
    return all(counts_get(available, item) >= quantity for item, quantity in required)


class OrderStatus(str, Enum):
    NONE = "none"
    PLACED = "placed"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


class ActionKind(Enum):
    LOGIN = 1
    LOGOUT = 2
    ADD_TO_CART = 3
    REMOVE_FROM_CART = 4
    CLEAR_CART = 5
    SET_PAYMENT = 6
    CLEAR_PAYMENT = 7
    SET_ADDRESS = 8
    CLEAR_ADDRESS = 9
    APPLY_COUPON = 10
    CLEAR_COUPON = 11
    PLACE_ORDER = 12
    CONFIRM_ORDER = 13
    CANCEL_ORDER = 14
    CLEAR_ORDER = 15


PARAMETERISED_KINDS = frozenset(
    {
        ActionKind.ADD_TO_CART,
        ActionKind.REMOVE_FROM_CART,
        ActionKind.SET_PAYMENT,
        ActionKind.SET_ADDRESS,
        ActionKind.APPLY_COUPON,
    }
)


@dataclass(frozen=True)
class PriceShoppingAction:
    kind: ActionKind
    target: str | None = None

    def __post_init__(self) -> None:
        if self.kind in PARAMETERISED_KINDS and self.target is None:
            raise ValueError(f"{self.kind.name} requires a target")
        if self.kind not in PARAMETERISED_KINDS and self.target is not None:
            raise ValueError(f"{self.kind.name} does not accept a target")

    @property
    def sort_key(self) -> tuple[int, str]:
        return self.kind.value, self.target or ""

    @classmethod
    def make(cls, kind: ActionKind, target: str | None = None) -> "PriceShoppingAction":
        return cls(kind, target)

    @classmethod
    def place_order(cls) -> "PriceShoppingAction":
        return cls(ActionKind.PLACE_ORDER)


@dataclass(frozen=True)
class PriceShoppingState:
    logged_in: bool
    cart: ItemCounts
    stock: ItemCounts
    payment_method: str | None
    shipping_address: str | None
    coupon_code: str | None
    wallet_balance: int
    card_limit: int
    cart_subtotal: int
    checkout_total: int
    order_status: OrderStatus
    order_items: ItemCounts
    order_total: int
    order_payment_method: str | None

    def __post_init__(self) -> None:
        if min(self.wallet_balance, self.card_limit, self.cart_subtotal, self.checkout_total, self.order_total) < 0:
            raise ValueError("money values must not be negative")

    @property
    def cart_size(self) -> int:
        return counts_total(self.cart)

    @property
    def cart_is_empty(self) -> bool:
        return not self.cart

    def cart_quantity(self, item: str) -> int:
        return counts_get(self.cart, item)

    def evolve(self, **changes: object) -> "PriceShoppingState":
        return dataclasses.replace(self, **changes)

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.logged_in,
            self.cart,
            self.stock,
            self.payment_method or "",
            self.shipping_address or "",
            self.coupon_code or "",
            self.wallet_balance,
            self.card_limit,
            self.cart_subtotal,
            self.checkout_total,
            self.order_status.value,
            self.order_items,
            self.order_total,
            self.order_payment_method or "",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "logged_in": self.logged_in,
            "cart": counts_to_mapping(self.cart),
            "stock": counts_to_mapping(self.stock),
            "payment_method": self.payment_method,
            "shipping_address": self.shipping_address,
            "coupon_code": self.coupon_code,
            "wallet_balance": self.wallet_balance,
            "card_limit": self.card_limit,
            "cart_subtotal": self.cart_subtotal,
            "checkout_total": self.checkout_total,
            "order_status": self.order_status.value,
            "order_items": counts_to_mapping(self.order_items),
            "order_total": self.order_total,
            "order_payment_method": self.order_payment_method,
        }

    def describe(self) -> str:
        cart = ",".join(f"{item}x{qty}" for item, qty in self.cart) or "-"
        stock = ",".join(f"{item}x{qty}" for item, qty in self.stock) or "-"
        return (
            f"login={'y' if self.logged_in else 'n'} cart={cart} stock={stock} "
            f"pay={self.payment_method or '-'} addr={self.shipping_address or '-'} "
            f"coupon={self.coupon_code or '-'} subtotal={self.cart_subtotal} "
            f"checkout={self.checkout_total} wallet={self.wallet_balance} limit={self.card_limit} "
            f"order={self.order_status.value} total={self.order_total}"
        )


def sort_states(states: Iterable[PriceShoppingState]) -> tuple[PriceShoppingState, ...]:
    return tuple(sorted(states, key=lambda state: state.key))
