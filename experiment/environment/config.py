#!/usr/bin/env python3
"""Seeded, immutable configuration for the controlled shopping sandbox."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .state import ItemCounts, counts_from_mapping


@dataclass(frozen=True)
class EnvConfig:
    """Catalog, capacity, and seed settings for one sandbox instance.

    The seed fixes the initial stock levels, which is the only place where the
    sandbox draws random numbers. Everything downstream of ``reset`` is a pure
    function of this configuration.
    """

    seed: int = 20260901
    items: tuple[str, ...] = ("book", "pen")
    max_stock: int = 2
    cart_capacity: int = 2
    valid_payment_methods: tuple[str, ...] = ("card", "voucher")
    rejected_payment_methods: tuple[str, ...] = ("expired_card",)
    valid_addresses: tuple[str, ...] = ("home",)
    rejected_addresses: tuple[str, ...] = ("po_box",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", _normalise(self.items, "items"))
        object.__setattr__(
            self, "valid_payment_methods", _normalise(self.valid_payment_methods, "valid_payment_methods")
        )
        object.__setattr__(
            self, "rejected_payment_methods", _normalise(self.rejected_payment_methods, "rejected_payment_methods", allow_empty=True)
        )
        object.__setattr__(self, "valid_addresses", _normalise(self.valid_addresses, "valid_addresses"))
        object.__setattr__(
            self, "rejected_addresses", _normalise(self.rejected_addresses, "rejected_addresses", allow_empty=True)
        )
        if self.max_stock < 1:
            raise ValueError("max_stock must be at least 1")
        if self.cart_capacity < 1:
            raise ValueError("cart_capacity must be at least 1")
        overlap = set(self.valid_payment_methods) & set(self.rejected_payment_methods)
        if overlap:
            raise ValueError(f"payment methods cannot be valid and rejected: {sorted(overlap)}")
        overlap = set(self.valid_addresses) & set(self.rejected_addresses)
        if overlap:
            raise ValueError(f"addresses cannot be valid and rejected: {sorted(overlap)}")

    @property
    def payment_options(self) -> tuple[str, ...]:
        """Every payment value the agent may set, valid or not."""
        return tuple(sorted(self.valid_payment_methods + self.rejected_payment_methods))

    @property
    def address_options(self) -> tuple[str, ...]:
        """Every address value the agent may set, valid or not."""
        return tuple(sorted(self.valid_addresses + self.rejected_addresses))

    def is_valid_payment(self, method: str | None) -> bool:
        return method is not None and method in self.valid_payment_methods

    def is_valid_address(self, address: str | None) -> bool:
        return address is not None and address in self.valid_addresses

    def initial_stock(self) -> ItemCounts:
        """Draw the opening stock levels from the configured seed."""
        rng = random.Random(self.seed)
        return counts_from_mapping({item: rng.randint(1, self.max_stock) for item in self.items})

    @property
    def fingerprint(self) -> tuple[Any, ...]:
        """Identity used to reject snapshots taken under a different setup."""
        return (
            self.seed,
            self.items,
            self.max_stock,
            self.cart_capacity,
            self.valid_payment_methods,
            self.rejected_payment_methods,
            self.valid_addresses,
            self.rejected_addresses,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "items": list(self.items),
            "max_stock": self.max_stock,
            "cart_capacity": self.cart_capacity,
            "valid_payment_methods": list(self.valid_payment_methods),
            "rejected_payment_methods": list(self.rejected_payment_methods),
            "valid_addresses": list(self.valid_addresses),
            "rejected_addresses": list(self.rejected_addresses),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EnvConfig:
        known = {field: payload[field] for field in _FIELD_NAMES if field in payload}
        unknown = sorted(set(payload) - set(_FIELD_NAMES) - {"description"})
        if unknown:
            raise ValueError(f"unknown configuration keys: {unknown}")
        for field in ("items", "valid_payment_methods", "rejected_payment_methods", "valid_addresses", "rejected_addresses"):
            if field in known:
                known[field] = tuple(known[field])
        return cls(**known)

    @classmethod
    def from_json_file(cls, path: str | Path) -> EnvConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


_FIELD_NAMES = (
    "seed",
    "items",
    "max_stock",
    "cart_capacity",
    "valid_payment_methods",
    "rejected_payment_methods",
    "valid_addresses",
    "rejected_addresses",
)


def _normalise(values: tuple[str, ...], label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    cleaned = tuple(sorted({str(value) for value in values}))
    if "" in cleaned:
        raise ValueError(f"{label} must not contain empty strings")
    if not cleaned and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    return cleaned


DEFAULT_CONFIG = EnvConfig()
