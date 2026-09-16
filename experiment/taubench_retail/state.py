#!/usr/bin/env python3
"""Immutable values for the tau-bench retail domain."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

SEMANTIC_FIELDS_PROTOCOL = "w1_semantic_fields_v1"
SEMANTIC_FIELDS_SCOPE = "sandbox_semantic_fields_v1"


class ActionKind(Enum):
    SET_ORDER = 1
    CLEAR_ORDER = 2
    ADD_ITEM = 3
    REMOVE_ITEM = 4
    CLEAR_ITEMS = 5
    ADD_NEW_ITEM = 6
    REMOVE_NEW_ITEM = 7
    CLEAR_NEW_ITEMS = 8
    SET_PAYMENT_METHOD = 9
    CLEAR_PAYMENT_METHOD = 10
    CANCEL_PENDING_ORDER = 11
    RETURN_DELIVERED_ORDER = 12
    EXCHANGE_ITEMS = 13
    DELIVER_ORDER = 14


PARAMETERISED_KINDS = frozenset(
    {
        ActionKind.SET_ORDER,
        ActionKind.ADD_ITEM,
        ActionKind.REMOVE_ITEM,
        ActionKind.ADD_NEW_ITEM,
        ActionKind.REMOVE_NEW_ITEM,
        ActionKind.SET_PAYMENT_METHOD,
        ActionKind.DELIVER_ORDER,
    }
)


@dataclass(frozen=True)
class RetailAction:
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

    @classmethod
    def make(cls, kind: ActionKind, target: str | None = None) -> RetailAction:
        return cls(kind, target)


@dataclass(frozen=True)
class RetailState:
    # DB state (mutable fields from the minimal slice)
    order_w1_status: str = "delivered"
    order_w1_exchange_items: tuple[str, ...] = ()
    order_w1_exchange_new_items: tuple[str, ...] = ()
    order_w1_exchange_payment_method_id: str | None = None
    order_w1_exchange_price_difference: float | None = None
    order_w1_return_items: tuple[str, ...] = ()
    order_w1_return_payment_method_id: str | None = None
    order_w2_status: str = "pending"
    order_w2_cancel_reason: str | None = None
    user_gift_card_balance: float = 15.0

    # Draft arguments for the target skill (exchange_delivered_order_items)
    draft_order_id: str | None = None
    draft_item_ids: tuple[str, ...] = ()
    draft_new_item_ids: tuple[str, ...] = ()
    draft_payment_method_id: str | None = None
    authenticated: bool = True

    def evolve(self, **changes: object) -> RetailState:
        return dataclasses.replace(self, **changes)

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.order_w1_status,
            self.order_w1_exchange_items,
            self.order_w1_exchange_new_items,
            self.order_w1_exchange_payment_method_id or "",
            self.order_w1_exchange_price_difference if self.order_w1_exchange_price_difference is not None else -999.0,
            self.order_w1_return_items,
            self.order_w1_return_payment_method_id or "",
            self.order_w2_status,
            self.order_w2_cancel_reason or "",
            round(self.user_gift_card_balance, 2),
            self.draft_order_id or "",
            self.draft_item_ids,
            self.draft_new_item_ids,
            self.draft_payment_method_id or "",
            self.authenticated,
        )

    def to_dict(self, *, vocabulary_protocol: str | None = None) -> dict[str, object]:
        payload = {
            "order_id": self.draft_order_id,
            "item_ids": list(self.draft_item_ids),
            "new_item_ids": list(self.draft_new_item_ids),
            "payment_method_id": self.draft_payment_method_id,
            "order_w1_status": self.order_w1_status,
            "order_w1_exchange_items": list(self.order_w1_exchange_items),
            "order_w1_exchange_new_items": list(self.order_w1_exchange_new_items),
            "order_w1_exchange_payment_method_id": self.order_w1_exchange_payment_method_id,
            "order_w1_exchange_price_difference": self.order_w1_exchange_price_difference,
            "order_w1_return_items": list(self.order_w1_return_items),
            "order_w2_status": self.order_w2_status,
            "order_w2_cancel_reason": self.order_w2_cancel_reason,
            "gift_card_balance": round(self.user_gift_card_balance, 2),
            "authenticated": self.authenticated,
        }
        if vocabulary_protocol == SEMANTIC_FIELDS_PROTOCOL:
            payload["order_w1_return_payment_method_id"] = self.order_w1_return_payment_method_id
        return payload

    def describe(self) -> str:
        return (
            f"w1={self.order_w1_status}, w2={self.order_w2_status}, "
            f"gc_bal={self.user_gift_card_balance:.2f}, "
            f"draft_order={self.draft_order_id}, "
            f"draft_items={list(self.draft_item_ids)}, "
            f"draft_new_items={list(self.draft_new_item_ids)}, "
            f"draft_pm={self.draft_payment_method_id}"
        )


def sort_states(states: Iterable[RetailState]) -> tuple[RetailState, ...]:
    return tuple(sorted(states, key=lambda state: state.key))
