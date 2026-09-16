#!/usr/bin/env python3
"""Finite configuration for price-aware shopping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .state import ItemCounts, counts_from_mapping, counts_get


def _unique_strings(values: tuple[str, ...], name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    normalised = tuple(str(value) for value in values)
    if not allow_empty and not normalised:
        raise ValueError(f"{name} must not be empty")
    if any(not value for value in normalised) or len(set(normalised)) != len(normalised):
        raise ValueError(f"{name} must contain unique non-empty strings")
    return normalised


@dataclass(frozen=True)
class PriceShoppingConfig:
    items: tuple[str, ...] = ("book", "gift")
    item_prices: tuple[tuple[str, int], ...] = (("book", 2400), ("gift", 2500))
    opening_stock: tuple[tuple[str, int], ...] = (("book", 2), ("gift", 1))
    cart_capacity: int = 2
    payment_methods: tuple[str, ...] = ("card", "wallet", "expired_card")
    accepted_payment_methods: tuple[str, ...] = ("card", "wallet")
    addresses: tuple[str, ...] = ("home", "remote", "po_box")
    serviceable_addresses: tuple[str, ...] = ("home", "remote")
    coupon_codes: tuple[str, ...] = ("save500", "invalid_coupon")
    accepted_coupon_codes: tuple[str, ...] = ("save500",)
    coupon_min_subtotal: int = 3000
    coupon_discount: int = 500
    coupon_excluded_items: tuple[str, ...] = ("gift",)
    free_shipping_threshold: int = 4000
    shipping_fee: int = 600
    remote_shipping_surcharge: int = 400
    initial_wallet_balance: int = 4300
    card_limit: int = 5000
    terminal_after_order: bool = False
    allow_post_order_cart_add: bool = False
    clear_checkout_selections_after_order: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "item_prices",
            tuple(sorted((str(item), int(value)) for item, value in self.item_prices)),
        )
        object.__setattr__(
            self,
            "opening_stock",
            tuple(sorted((str(item), int(value)) for item, value in self.opening_stock)),
        )
        for name in (
            "items", "payment_methods", "accepted_payment_methods", "addresses",
            "serviceable_addresses", "coupon_codes", "accepted_coupon_codes",
            "coupon_excluded_items",
        ):
            object.__setattr__(self, name, _unique_strings(getattr(self, name), name, allow_empty=name == "coupon_excluded_items"))
        prices = dict(self.item_prices)
        stock = dict(self.opening_stock)
        if set(prices) != set(self.items) or set(stock) != set(self.items):
            raise ValueError("item_prices and opening_stock must cover every item exactly once")
        if any(type(value) is not int or value <= 0 for value in prices.values()):
            raise ValueError("item prices must be positive integer cents")
        if any(type(value) is not int or value < 0 for value in stock.values()):
            raise ValueError("opening stock must be non-negative integers")
        if not set(self.accepted_payment_methods).issubset(self.payment_methods):
            raise ValueError("accepted payment methods must be selectable")
        if not set(self.serviceable_addresses).issubset(self.addresses):
            raise ValueError("serviceable addresses must be selectable")
        if not set(self.accepted_coupon_codes).issubset(self.coupon_codes):
            raise ValueError("accepted coupon codes must be selectable")
        if not set(self.coupon_excluded_items).issubset(self.items):
            raise ValueError("coupon exclusions must name configured items")
        for name in (
            "coupon_min_subtotal", "coupon_discount", "free_shipping_threshold",
            "shipping_fee", "remote_shipping_surcharge", "initial_wallet_balance", "card_limit",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.cart_capacity < 1:
            raise ValueError("cart_capacity must be positive")
        if type(self.terminal_after_order) is not bool:
            raise ValueError("terminal_after_order must be a boolean")
        if type(self.allow_post_order_cart_add) is not bool:
            raise ValueError("allow_post_order_cart_add must be a boolean")
        if type(self.clear_checkout_selections_after_order) is not bool:
            raise ValueError("clear_checkout_selections_after_order must be a boolean")
        if self.allow_post_order_cart_add and not self.terminal_after_order:
            raise ValueError("allow_post_order_cart_add requires terminal_after_order")
        if self.coupon_discount > self.coupon_min_subtotal:
            raise ValueError("coupon_discount must not exceed coupon_min_subtotal")

    @property
    def fingerprint(self) -> tuple[Any, ...]:
        return (
            self.items,
            self.item_prices,
            self.opening_stock,
            self.cart_capacity,
            self.payment_methods,
            self.accepted_payment_methods,
            self.addresses,
            self.serviceable_addresses,
            self.coupon_codes,
            self.accepted_coupon_codes,
            self.coupon_min_subtotal,
            self.coupon_discount,
            self.coupon_excluded_items,
            self.free_shipping_threshold,
            self.shipping_fee,
            self.remote_shipping_surcharge,
            self.initial_wallet_balance,
            self.card_limit,
            self.terminal_after_order,
            self.allow_post_order_cart_add,
            self.clear_checkout_selections_after_order,
        )

    def initial_stock(self) -> ItemCounts:
        return counts_from_mapping(dict(self.opening_stock))

    def item_price(self, item: str) -> int:
        return dict(self.item_prices)[item]

    def subtotal(self, cart: ItemCounts) -> int:
        return sum(self.item_price(item) * quantity for item, quantity in cart)

    def coupon_eligible(self, code: str | None, cart: ItemCounts) -> bool:
        return (
            code in self.accepted_coupon_codes
            and self.subtotal(cart) >= self.coupon_min_subtotal
            and not any(counts_get(cart, item) for item in self.coupon_excluded_items)
        )

    def discount(self, code: str | None, cart: ItemCounts) -> int:
        return self.coupon_discount if code is not None and self.coupon_eligible(code, cart) else 0

    def delivery_charge(self, address: str | None, cart: ItemCounts) -> int:
        if not cart:
            return 0
        base = 0 if self.subtotal(cart) >= self.free_shipping_threshold else self.shipping_fee
        return base + (self.remote_shipping_surcharge if address == "remote" else 0)

    def payable_total(self, state: Any) -> int:
        return self.subtotal(state.cart) - self.discount(state.coupon_code, state.cart) + self.delivery_charge(state.shipping_address, state.cart)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "items": list(self.items),
            "item_prices": dict(self.item_prices),
            "opening_stock": dict(self.opening_stock),
            "cart_capacity": self.cart_capacity,
            "payment_methods": list(self.payment_methods),
            "accepted_payment_methods": list(self.accepted_payment_methods),
            "addresses": list(self.addresses),
            "serviceable_addresses": list(self.serviceable_addresses),
            "coupon_codes": list(self.coupon_codes),
            "accepted_coupon_codes": list(self.accepted_coupon_codes),
            "coupon_min_subtotal": self.coupon_min_subtotal,
            "coupon_discount": self.coupon_discount,
            "coupon_excluded_items": list(self.coupon_excluded_items),
            "free_shipping_threshold": self.free_shipping_threshold,
            "shipping_fee": self.shipping_fee,
            "remote_shipping_surcharge": self.remote_shipping_surcharge,
            "initial_wallet_balance": self.initial_wallet_balance,
            "card_limit": self.card_limit,
        }
        if self.terminal_after_order:
            data["terminal_after_order"] = True
        if self.allow_post_order_cart_add:
            data["allow_post_order_cart_add"] = True
        if self.clear_checkout_selections_after_order:
            data["clear_checkout_selections_after_order"] = True
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PriceShoppingConfig":
        defaults = cls()
        return cls(
            items=tuple(data.get("items", defaults.items)),
            item_prices=tuple(sorted((str(k), int(v)) for k, v in data.get("item_prices", dict(defaults.item_prices)).items())),
            opening_stock=tuple(sorted((str(k), int(v)) for k, v in data.get("opening_stock", dict(defaults.opening_stock)).items())),
            cart_capacity=int(data.get("cart_capacity", defaults.cart_capacity)),
            payment_methods=tuple(data.get("payment_methods", defaults.payment_methods)),
            accepted_payment_methods=tuple(data.get("accepted_payment_methods", defaults.accepted_payment_methods)),
            addresses=tuple(data.get("addresses", defaults.addresses)),
            serviceable_addresses=tuple(data.get("serviceable_addresses", defaults.serviceable_addresses)),
            coupon_codes=tuple(data.get("coupon_codes", defaults.coupon_codes)),
            accepted_coupon_codes=tuple(data.get("accepted_coupon_codes", defaults.accepted_coupon_codes)),
            coupon_min_subtotal=int(data.get("coupon_min_subtotal", defaults.coupon_min_subtotal)),
            coupon_discount=int(data.get("coupon_discount", defaults.coupon_discount)),
            coupon_excluded_items=tuple(data.get("coupon_excluded_items", defaults.coupon_excluded_items)),
            free_shipping_threshold=int(data.get("free_shipping_threshold", defaults.free_shipping_threshold)),
            shipping_fee=int(data.get("shipping_fee", defaults.shipping_fee)),
            remote_shipping_surcharge=int(data.get("remote_shipping_surcharge", defaults.remote_shipping_surcharge)),
            initial_wallet_balance=int(data.get("initial_wallet_balance", defaults.initial_wallet_balance)),
            card_limit=int(data.get("card_limit", defaults.card_limit)),
            terminal_after_order=data.get("terminal_after_order", defaults.terminal_after_order),
            allow_post_order_cart_add=data.get(
                "allow_post_order_cart_add", defaults.allow_post_order_cart_add
            ),
            clear_checkout_selections_after_order=data.get(
                "clear_checkout_selections_after_order",
                defaults.clear_checkout_selections_after_order,
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PriceShoppingConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


DEFAULT_PRICE_SHOPPING_CONFIG = PriceShoppingConfig()
