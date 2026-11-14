#!/usr/bin/env python3
"""Pure transition rules for the tau-bench retail domain."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

# Ensure vendored tau-bench packages are importable with zero external dependencies
_vendor_path = str((Path(__file__).parent / "vendor").resolve())
if _vendor_path not in sys.path:
    sys.path.insert(0, _vendor_path)

from tau_bench.envs.retail.tools.cancel_pending_order import CancelPendingOrder
from tau_bench.envs.retail.tools.exchange_delivered_order_items import ExchangeDeliveredOrderItems
from tau_bench.envs.retail.tools.return_delivered_order_items import ReturnDeliveredOrderItems

from .config import DEFAULT_RETAIL_CONFIG, RetailConfig
from .fixtures import get_initial_data
from .state import ActionKind, RetailAction, RetailState


@dataclass(frozen=True)
class StepResult:
    state: RetailState
    ok: bool
    action: RetailAction
    error: str | None = None

    def __post_init__(self) -> None:
        if self.ok and self.error is not None:
            raise ValueError("a successful step cannot carry an error")
        if not self.ok and not self.error:
            raise ValueError("a rejected step must explain why")


def initial_state(config: RetailConfig = DEFAULT_RETAIL_CONFIG) -> RetailState:
    return RetailState()


def action_space(config: RetailConfig = DEFAULT_RETAIL_CONFIG) -> tuple[RetailAction, ...]:
    actions: list[RetailAction] = []
    actions += [RetailAction.make(ActionKind.SET_ORDER, value) for value in config.order_options]
    actions.append(RetailAction.make(ActionKind.CLEAR_ORDER))
    actions += [RetailAction.make(ActionKind.ADD_ITEM, value) for value in config.item_options]
    actions += [RetailAction.make(ActionKind.REMOVE_ITEM, value) for value in config.item_options]
    actions.append(RetailAction.make(ActionKind.CLEAR_ITEMS))
    actions += [RetailAction.make(ActionKind.ADD_NEW_ITEM, value) for value in config.new_item_options]
    actions += [RetailAction.make(ActionKind.REMOVE_NEW_ITEM, value) for value in config.new_item_options]
    actions.append(RetailAction.make(ActionKind.CLEAR_NEW_ITEMS))
    actions += [RetailAction.make(ActionKind.SET_PAYMENT_METHOD, value) for value in config.payment_method_options]
    actions.append(RetailAction.make(ActionKind.CLEAR_PAYMENT_METHOD))
    actions.append(RetailAction.make(ActionKind.CANCEL_PENDING_ORDER))
    actions.append(RetailAction.make(ActionKind.RETURN_DELIVERED_ORDER))
    actions.append(RetailAction.make(ActionKind.EXCHANGE_ITEMS))
    if config.deliverable_orders:
        actions += [RetailAction.make(ActionKind.DELIVER_ORDER, value) for value in config.deliverable_orders]
    return tuple(sorted(actions, key=lambda a: a.sort_key))


def _reject(state: RetailState, action: RetailAction, reason: str) -> StepResult:
    return StepResult(state, False, action, reason)


def _accept(state: RetailState, action: RetailAction, **changes: object) -> StepResult:
    return StepResult(state.evolve(**changes), True, action)


def state_to_tau_data(state: RetailState) -> Dict[str, Any]:
    """Map RetailState dynamic fields to a fresh copy of tau-bench database slice."""
    data = get_initial_data()
    w1 = data["orders"]["#W1"]
    w1["status"] = state.order_w1_status
    if state.order_w1_exchange_items:
        w1["exchange_items"] = list(state.order_w1_exchange_items)
    if state.order_w1_exchange_new_items:
        w1["exchange_new_items"] = list(state.order_w1_exchange_new_items)
    if state.order_w1_exchange_payment_method_id:
        w1["exchange_payment_method_id"] = state.order_w1_exchange_payment_method_id
    if state.order_w1_exchange_price_difference is not None:
        w1["exchange_price_difference"] = state.order_w1_exchange_price_difference
    if state.order_w1_return_items:
        w1["return_items"] = list(state.order_w1_return_items)
    if state.order_w1_return_payment_method_id:
        w1["return_payment_method_id"] = state.order_w1_return_payment_method_id

    w2 = data["orders"]["#W2"]
    w2["status"] = state.order_w2_status
    if state.order_w2_cancel_reason:
        w2["cancel_reason"] = state.order_w2_cancel_reason

    data["users"]["user_1"]["payment_methods"]["gift_card_0"]["balance"] = state.user_gift_card_balance
    return data


def _draftable(state: RetailState, action: RetailAction) -> StepResult | None:
    if state.order_w1_status != "delivered" and state.order_w2_status != "delivered":
        return _reject(state, action, "cannot draft for non-delivered order")
    return None


def apply_action(
    state: RetailState,
    action: RetailAction,
    config: RetailConfig = DEFAULT_RETAIL_CONFIG,
) -> StepResult:
    kind = action.kind

    if kind in (
        ActionKind.SET_ORDER, ActionKind.CLEAR_ORDER,
        ActionKind.ADD_ITEM, ActionKind.REMOVE_ITEM, ActionKind.CLEAR_ITEMS,
        ActionKind.ADD_NEW_ITEM, ActionKind.REMOVE_NEW_ITEM, ActionKind.CLEAR_NEW_ITEMS,
        ActionKind.SET_PAYMENT_METHOD, ActionKind.CLEAR_PAYMENT_METHOD,
    ):
        blocked = _draftable(state, action)
        if blocked is not None:
            return blocked

    if kind is ActionKind.SET_ORDER:
        if action.target not in config.order_options:
            return _reject(state, action, "unknown order")
        if state.draft_order_id == action.target:
            return _reject(state, action, "order already selected")
        return _accept(state, action, draft_order_id=action.target)

    if kind is ActionKind.CLEAR_ORDER:
        if state.draft_order_id is None:
            return _reject(state, action, "no order selected")
        return _accept(state, action, draft_order_id=None)

    if kind is ActionKind.ADD_ITEM:
        if action.target not in config.item_options:
            return _reject(state, action, "unknown item")
        if action.target in state.draft_item_ids:
            return _reject(state, action, "item already added")
        if len(state.draft_item_ids) >= config.max_items:
            return _reject(state, action, "item limit reached")
        return _accept(state, action, draft_item_ids=(*state.draft_item_ids, action.target))

    if kind is ActionKind.REMOVE_ITEM:
        if action.target not in state.draft_item_ids:
            return _reject(state, action, "item is not in the draft")
        return _accept(state, action, draft_item_ids=tuple(x for x in state.draft_item_ids if x != action.target))

    if kind is ActionKind.CLEAR_ITEMS:
        if not state.draft_item_ids:
            return _reject(state, action, "draft items are empty")
        return _accept(state, action, draft_item_ids=())

    if kind is ActionKind.ADD_NEW_ITEM:
        if action.target not in config.new_item_options:
            return _reject(state, action, "unknown new item")
        if action.target in state.draft_new_item_ids:
            return _reject(state, action, "new item already added")
        if len(state.draft_new_item_ids) >= config.max_new_items:
            return _reject(state, action, "new item limit reached")
        return _accept(state, action, draft_new_item_ids=(*state.draft_new_item_ids, action.target))

    if kind is ActionKind.REMOVE_NEW_ITEM:
        if action.target not in state.draft_new_item_ids:
            return _reject(state, action, "new item is not in the draft")
        return _accept(state, action, draft_new_item_ids=tuple(x for x in state.draft_new_item_ids if x != action.target))

    if kind is ActionKind.CLEAR_NEW_ITEMS:
        if not state.draft_new_item_ids:
            return _reject(state, action, "draft new items are empty")
        return _accept(state, action, draft_new_item_ids=())

    if kind is ActionKind.SET_PAYMENT_METHOD:
        if action.target not in config.payment_method_options:
            return _reject(state, action, "unknown payment method")
        if state.draft_payment_method_id == action.target:
            return _reject(state, action, "payment method already selected")
        return _accept(state, action, draft_payment_method_id=action.target)

    if kind is ActionKind.CLEAR_PAYMENT_METHOD:
        if state.draft_payment_method_id is None:
            return _reject(state, action, "no payment method selected")
        return _accept(state, action, draft_payment_method_id=None)

    # Auxiliary action 1: cancel_pending_order (vendored tool execution)
    if kind is ActionKind.CANCEL_PENDING_ORDER:
        data = state_to_tau_data(state)
        res = CancelPendingOrder.invoke(data, order_id="#W2", reason="no longer needed")
        if res.startswith("Error:"):
            return _reject(state, action, res)
        order = json.loads(res)
        new_balance = data["users"]["user_1"]["payment_methods"]["gift_card_0"]["balance"]
        return _accept(
            state,
            action,
            order_w2_status=order["status"],
            order_w2_cancel_reason=order.get("cancel_reason"),
            user_gift_card_balance=new_balance,
        )

    # Auxiliary action 2: return_delivered_order_items (vendored tool execution)
    if kind is ActionKind.RETURN_DELIVERED_ORDER:
        data = state_to_tau_data(state)
        res = ReturnDeliveredOrderItems.invoke(
            data,
            order_id="#W1",
            item_ids=["shoe_black_9"],
            payment_method_id="credit_card_0",
        )
        if res.startswith("Error:"):
            return _reject(state, action, res)
        order = json.loads(res)
        return _accept(
            state,
            action,
            order_w1_status=order["status"],
            order_w1_return_items=tuple(order.get("return_items", ())),
            order_w1_return_payment_method_id=order.get("return_payment_method_id"),
        )

    # Target skill: exchange_delivered_order_items (vendored tool execution)
    if kind is ActionKind.EXCHANGE_ITEMS:
        data = state_to_tau_data(state)
        res = ExchangeDeliveredOrderItems.invoke(
            data,
            order_id=state.draft_order_id or "",
            item_ids=list(state.draft_item_ids),
            new_item_ids=list(state.draft_new_item_ids),
            payment_method_id=state.draft_payment_method_id or "",
        )
        if res.startswith("Error:"):
            return _reject(state, action, res)
        order = json.loads(res)
        if order.get("order_id") == "#W2":
            return _accept(
                state,
                action,
                order_w2_status=order["status"],
                draft_order_id=None,
                draft_item_ids=(),
                draft_new_item_ids=(),
                draft_payment_method_id=None,
            )
        return _accept(
            state,
            action,
            order_w1_status=order["status"],
            order_w1_exchange_items=tuple(order.get("exchange_items", ())),
            order_w1_exchange_new_items=tuple(order.get("exchange_new_items", ())),
            order_w1_exchange_payment_method_id=order.get("exchange_payment_method_id"),
            order_w1_exchange_price_difference=order.get("exchange_price_difference"),
            draft_order_id=None,
            draft_item_ids=(),
            draft_new_item_ids=(),
            draft_payment_method_id=None,
        )

    if kind is ActionKind.DELIVER_ORDER:
        if action.target not in config.deliverable_orders:
            return _reject(state, action, f"order {action.target} cannot be delivered")
        if action.target == "#W1":
            if state.order_w1_status != "pending":
                return _reject(state, action, f"order {action.target} is not pending")
            return _accept(state, action, order_w1_status="delivered")
        elif action.target == "#W2":
            if state.order_w2_status != "pending":
                return _reject(state, action, f"order {action.target} is not pending")
            return _accept(state, action, order_w2_status="delivered")
        else:
            return _reject(state, action, f"unknown deliverable order {action.target}")

    raise AssertionError(f"unhandled action kind: {kind}")


def valid_actions(state: RetailState, config: RetailConfig = DEFAULT_RETAIL_CONFIG) -> tuple[RetailAction, ...]:
    return tuple(action for action in action_space(config) if apply_action(state, action, config).ok)


class RetailEnv:
    """Stateful wrapper over pure retail transitions."""

    def __init__(self, config: RetailConfig | None = None) -> None:
        self._config = config or DEFAULT_RETAIL_CONFIG
        self._state = initial_state(self._config)
        self._step_count = 0
        self._rejected_count = 0

    @property
    def state(self) -> RetailState:
        return self._state

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def rejected_count(self) -> int:
        return self._rejected_count

    def reset(self) -> RetailState:
        self._state = initial_state(self._config)
        self._step_count = 0
        self._rejected_count = 0
        return self._state

    def step(self, action: RetailAction) -> StepResult:
        if not isinstance(action, RetailAction):
            raise TypeError(f"expected RetailAction, got {type(action).__name__}")
        result = apply_action(self._state, action, self._config)
        self._state = result.state
        self._step_count += 1
        self._rejected_count += int(not result.ok)
        return result
