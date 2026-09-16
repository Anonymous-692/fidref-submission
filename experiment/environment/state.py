#!/usr/bin/env python3
"""Immutable state and action values for the controlled shopping sandbox."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping


# An item multiset, always stored sorted by item name with strictly positive
# quantities so that two equal multisets are always the same tuple.
ItemCounts = tuple[tuple[str, int], ...]


def counts_from_mapping(mapping: Mapping[str, int]) -> ItemCounts:
    """Build a canonical item multiset, dropping non-positive quantities."""
    for item, quantity in mapping.items():
        if quantity < 0:
            raise ValueError(f"negative quantity for {item!r}: {quantity}")
    return tuple(sorted((item, qty) for item, qty in mapping.items() if qty > 0))


def counts_to_mapping(counts: ItemCounts) -> dict[str, int]:
    return {item: quantity for item, quantity in counts}


def counts_get(counts: ItemCounts, item: str) -> int:
    for name, quantity in counts:
        if name == item:
            return quantity
    return 0


def counts_total(counts: ItemCounts) -> int:
    return sum(quantity for _, quantity in counts)


def counts_add(counts: ItemCounts, item: str, delta: int) -> ItemCounts:
    """Return a copy of ``counts`` with ``delta`` added to ``item``."""
    updated = counts_to_mapping(counts)
    new_quantity = updated.get(item, 0) + delta
    if new_quantity < 0:
        raise ValueError(f"quantity for {item!r} would become negative")
    updated[item] = new_quantity
    return counts_from_mapping(updated)


def counts_combine(left: ItemCounts, right: ItemCounts, sign: int = 1) -> ItemCounts:
    """Return ``left + sign * right`` as a canonical multiset."""
    if sign not in (1, -1):
        raise ValueError("sign must be 1 or -1")
    combined = counts_to_mapping(left)
    for item, quantity in right:
        new_quantity = combined.get(item, 0) + sign * quantity
        if new_quantity < 0:
            raise ValueError(f"quantity for {item!r} would become negative")
        combined[item] = new_quantity
    return counts_from_mapping(combined)


def counts_covers(available: ItemCounts, required: ItemCounts) -> bool:
    """True when ``available`` holds at least every quantity in ``required``."""
    return all(counts_get(available, item) >= qty for item, qty in required)


class OrderStatus(str, Enum):
    """Lifecycle position of the single order slot owned by the shopper."""

    NONE = "none"
    PLACED = "placed"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class ActionKind(Enum):
    """Every operation the agent-facing API exposes.

    The integer values fix a total order so that action enumeration, and
    therefore reachable-state enumeration, is reproducible across runs.
    """

    LOGIN = 1
    LOGOUT = 2
    ADD_TO_CART = 3
    REMOVE_FROM_CART = 4
    CLEAR_CART = 5
    SET_PAYMENT = 6
    CLEAR_PAYMENT = 7
    SET_ADDRESS = 8
    CLEAR_ADDRESS = 9
    PLACE_ORDER = 10
    CONFIRM_ORDER = 11
    CANCEL_ORDER = 12
    CLEAR_ORDER = 13


# Kinds that carry a target argument; all others must leave ``target`` unset.
PARAMETERISED_KINDS = frozenset(
    {ActionKind.ADD_TO_CART, ActionKind.REMOVE_FROM_CART, ActionKind.SET_PAYMENT, ActionKind.SET_ADDRESS}
)


@dataclass(frozen=True)
class ShoppingAction:
    """A single agent operation, optionally parameterised by a target."""

    kind: ActionKind
    target: str | None = None

    def __post_init__(self) -> None:
        if self.kind in PARAMETERISED_KINDS and self.target is None:
            raise ValueError(f"{self.kind.name} requires a target")
        if self.kind not in PARAMETERISED_KINDS and self.target is not None:
            raise ValueError(f"{self.kind.name} does not accept a target")

    @property
    def sort_key(self) -> tuple[int, str]:
        return (self.kind.value, self.target or "")

    @property
    def name(self) -> str:
        return self.kind.name.lower()

    def __str__(self) -> str:
        if self.target is None:
            return self.name
        return f"{self.name}({self.target})"

    def to_dict(self) -> dict[str, str | None]:
        return {"kind": self.kind.name, "target": self.target}

    # Convenience constructors keep call sites readable in tests and demos.
    @classmethod
    def login(cls) -> ShoppingAction:
        return cls(ActionKind.LOGIN)

    @classmethod
    def logout(cls) -> ShoppingAction:
        return cls(ActionKind.LOGOUT)

    @classmethod
    def add_to_cart(cls, item: str) -> ShoppingAction:
        return cls(ActionKind.ADD_TO_CART, item)

    @classmethod
    def remove_from_cart(cls, item: str) -> ShoppingAction:
        return cls(ActionKind.REMOVE_FROM_CART, item)

    @classmethod
    def clear_cart(cls) -> ShoppingAction:
        return cls(ActionKind.CLEAR_CART)

    @classmethod
    def set_payment(cls, method: str) -> ShoppingAction:
        return cls(ActionKind.SET_PAYMENT, method)

    @classmethod
    def clear_payment(cls) -> ShoppingAction:
        return cls(ActionKind.CLEAR_PAYMENT)

    @classmethod
    def set_address(cls, address: str) -> ShoppingAction:
        return cls(ActionKind.SET_ADDRESS, address)

    @classmethod
    def clear_address(cls) -> ShoppingAction:
        return cls(ActionKind.CLEAR_ADDRESS)

    @classmethod
    def place_order(cls) -> ShoppingAction:
        return cls(ActionKind.PLACE_ORDER)

    @classmethod
    def confirm_order(cls) -> ShoppingAction:
        return cls(ActionKind.CONFIRM_ORDER)

    @classmethod
    def cancel_order(cls) -> ShoppingAction:
        return cls(ActionKind.CANCEL_ORDER)

    @classmethod
    def clear_order(cls) -> ShoppingAction:
        return cls(ActionKind.CLEAR_ORDER)


@dataclass(frozen=True)
class ShoppingState:
    """A complete, hashable snapshot of the storefront world.

    Every field is either a scalar or a canonical tuple, so instances are
    immutable, comparable by value, and safe to share between the environment,
    the enumerator, and any downstream analysis without defensive copying.
    """

    logged_in: bool
    cart: ItemCounts
    stock: ItemCounts
    payment_method: str | None
    shipping_address: str | None
    order_status: OrderStatus
    order_items: ItemCounts

    def cart_quantity(self, item: str) -> int:
        return counts_get(self.cart, item)

    def stock_quantity(self, item: str) -> int:
        return counts_get(self.stock, item)

    @property
    def cart_size(self) -> int:
        return counts_total(self.cart)

    @property
    def cart_is_empty(self) -> bool:
        return self.cart_size == 0

    def evolve(self, **changes: object) -> ShoppingState:
        """Return a new state with the given fields replaced."""
        return dataclasses.replace(self, **changes)

    @property
    def key(self) -> tuple[object, ...]:
        """A canonical, sortable identity for the state."""
        return (
            self.logged_in,
            self.cart,
            self.stock,
            self.payment_method or "",
            self.shipping_address or "",
            self.order_status.value,
            self.order_items,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "logged_in": self.logged_in,
            "cart": counts_to_mapping(self.cart),
            "stock": counts_to_mapping(self.stock),
            "payment_method": self.payment_method,
            "shipping_address": self.shipping_address,
            "order_status": self.order_status.value,
            "order_items": counts_to_mapping(self.order_items),
        }

    def describe(self) -> str:
        cart = ",".join(f"{item}x{qty}" for item, qty in self.cart) or "-"
        stock = ",".join(f"{item}x{qty}" for item, qty in self.stock) or "-"
        return (
            f"login={'y' if self.logged_in else 'n'} cart={cart} stock={stock} "
            f"pay={self.payment_method or '-'} addr={self.shipping_address or '-'} "
            f"order={self.order_status.value}"
        )


def sort_states(states: Iterable[ShoppingState]) -> tuple[ShoppingState, ...]:
    """Order states canonically; used to compare enumerations set-wise."""
    return tuple(sorted(states, key=lambda state: state.key))
