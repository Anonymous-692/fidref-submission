#!/usr/bin/env python3
"""Finite configuration for the tau-bench retail domain."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _strings(values: tuple[str, ...], name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    normalised = tuple(values)
    if not allow_empty and not normalised:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value for value in normalised):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"{name} must not contain duplicates")
    return normalised


@dataclass(frozen=True)
class RetailConfig:
    valid_orders: tuple[str, ...] = ("#W1", "#W2")
    rejected_orders: tuple[str, ...] = ("#W99",)
    valid_items: tuple[str, ...] = ("shoe_black_9", "shirt_blue_m")
    rejected_items: tuple[str, ...] = ("shoe_red_9",)
    valid_new_items: tuple[str, ...] = ("shoe_black_10", "shoe_red_9", "shirt_blue_l")
    rejected_new_items: tuple[str, ...] = ("shoe_blue_9",)
    valid_payment_methods: tuple[str, ...] = ("gift_card_0", "credit_card_0")
    rejected_payment_methods: tuple[str, ...] = ("invalid_pm",)
    max_items: int = 1
    max_new_items: int = 1
    max_depth: int = 20
    max_states: int = 5000
    deliverable_orders: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "valid_orders", "rejected_orders",
            "valid_items", "rejected_items",
            "valid_new_items", "rejected_new_items",
            "valid_payment_methods", "rejected_payment_methods",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name, allow_empty=name.startswith("rejected")))
        object.__setattr__(self, "deliverable_orders", _strings(getattr(self, "deliverable_orders"), "deliverable_orders", allow_empty=True))
        if set(self.valid_orders) & set(self.rejected_orders):
            raise ValueError("valid and rejected orders must be disjoint")
        if set(self.valid_items) & set(self.rejected_items):
            raise ValueError("valid and rejected items must be disjoint")
        if set(self.valid_new_items) & set(self.rejected_new_items):
            raise ValueError("valid and rejected new items must be disjoint")
        if set(self.valid_payment_methods) & set(self.rejected_payment_methods):
            raise ValueError("valid and rejected payment methods must be disjoint")

    @property
    def order_options(self) -> tuple[str, ...]:
        return self.valid_orders + self.rejected_orders

    @property
    def item_options(self) -> tuple[str, ...]:
        return self.valid_items + self.rejected_items

    @property
    def new_item_options(self) -> tuple[str, ...]:
        return self.valid_new_items + self.rejected_new_items

    @property
    def payment_method_options(self) -> tuple[str, ...]:
        return self.valid_payment_methods + self.rejected_payment_methods

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_orders": list(self.valid_orders),
            "rejected_orders": list(self.rejected_orders),
            "valid_items": list(self.valid_items),
            "rejected_items": list(self.rejected_items),
            "valid_new_items": list(self.valid_new_items),
            "rejected_new_items": list(self.rejected_new_items),
            "valid_payment_methods": list(self.valid_payment_methods),
            "rejected_payment_methods": list(self.rejected_payment_methods),
            "max_items": self.max_items,
            "max_new_items": self.max_new_items,
            "max_depth": self.max_depth,
            "max_states": self.max_states,
            "deliverable_orders": list(self.deliverable_orders),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RetailConfig:
        return cls(
            valid_orders=tuple(data.get("valid_orders", cls.valid_orders)),
            rejected_orders=tuple(data.get("rejected_orders", cls.rejected_orders)),
            valid_items=tuple(data.get("valid_items", cls.valid_items)),
            rejected_items=tuple(data.get("rejected_items", cls.rejected_items)),
            valid_new_items=tuple(data.get("valid_new_items", cls.valid_new_items)),
            rejected_new_items=tuple(data.get("rejected_new_items", cls.rejected_new_items)),
            valid_payment_methods=tuple(data.get("valid_payment_methods", cls.valid_payment_methods)),
            rejected_payment_methods=tuple(data.get("rejected_payment_methods", cls.rejected_payment_methods)),
            max_items=int(data.get("max_items", cls.max_items)),
            max_new_items=int(data.get("max_new_items", cls.max_new_items)),
            max_depth=int(data.get("max_depth", cls.max_depth)),
            max_states=int(data.get("max_states", cls.max_states)),
            deliverable_orders=tuple(data.get("deliverable_orders", cls.deliverable_orders)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> RetailConfig:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


DEFAULT_RETAIL_CONFIG = RetailConfig()
