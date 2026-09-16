#!/usr/bin/env python3
"""Deterministic transition rules and the agent-facing shopping environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import DEFAULT_CONFIG, EnvConfig
from .state import (
    ActionKind,
    OrderStatus,
    ShoppingAction,
    ShoppingState,
    counts_add,
    counts_combine,
    counts_covers,
)


@dataclass(frozen=True)
class StepResult:
    """Outcome of applying one action.

    ``state`` is the state after the action when ``ok`` is true, and the
    unchanged input state otherwise: rejected actions never mutate the world.
    """

    state: ShoppingState
    ok: bool
    action: ShoppingAction
    error: str | None = None

    def __post_init__(self) -> None:
        if self.ok and self.error is not None:
            raise ValueError("a successful step cannot carry an error")
        if not self.ok and not self.error:
            raise ValueError("a rejected step must explain why")


@dataclass(frozen=True)
class EnvSnapshot:
    """Opaque, immutable capture of everything ``restore`` needs."""

    state: ShoppingState
    step_count: int
    rejected_count: int
    config_fingerprint: tuple[Any, ...]


def initial_state(config: EnvConfig = DEFAULT_CONFIG) -> ShoppingState:
    """The seeded opening state: logged out, empty cart, stocked shelves."""
    return ShoppingState(
        logged_in=False,
        cart=(),
        stock=config.initial_stock(),
        payment_method=None,
        shipping_address=None,
        order_status=OrderStatus.NONE,
        order_items=(),
    )


def action_space(config: EnvConfig = DEFAULT_CONFIG) -> tuple[ShoppingAction, ...]:
    """Every action the agent may attempt, in a fixed reproducible order."""
    actions: list[ShoppingAction] = [ShoppingAction.login(), ShoppingAction.logout()]
    actions.extend(ShoppingAction.add_to_cart(item) for item in config.items)
    actions.extend(ShoppingAction.remove_from_cart(item) for item in config.items)
    actions.append(ShoppingAction.clear_cart())
    actions.extend(ShoppingAction.set_payment(method) for method in config.payment_options)
    actions.append(ShoppingAction.clear_payment())
    actions.extend(ShoppingAction.set_address(address) for address in config.address_options)
    actions.append(ShoppingAction.clear_address())
    actions.extend(
        [
            ShoppingAction.place_order(),
            ShoppingAction.confirm_order(),
            ShoppingAction.cancel_order(),
            ShoppingAction.clear_order(),
        ]
    )
    return tuple(sorted(actions, key=lambda action: action.sort_key))


def apply_action(
    state: ShoppingState, action: ShoppingAction, config: EnvConfig = DEFAULT_CONFIG
) -> StepResult:
    """Pure transition function shared by the environment and the enumerator.

    The rules below are the *implementation* of the storefront. They are
    deliberately written as plain guards so that the sandbox behaviour, not a
    declarative specification, decides what an action does.
    """
    handler = _HANDLERS[action.kind]
    return handler(state, action, config)


def _reject(state: ShoppingState, action: ShoppingAction, reason: str) -> StepResult:
    return StepResult(state=state, ok=False, action=action, error=reason)


def _accept(state: ShoppingState, action: ShoppingAction) -> StepResult:
    return StepResult(state=state, ok=True, action=action, error=None)


def _login(state: ShoppingState, action: ShoppingAction, config: EnvConfig) -> StepResult:
    if state.logged_in:
        return _reject(state, action, "already logged in")
    return _accept(state.evolve(logged_in=True), action)


def _logout(state: ShoppingState, action: ShoppingAction, config: EnvConfig) -> StepResult:
    if not state.logged_in:
        return _reject(state, action, "not logged in")
    # The cart, payment method, and address survive a logout, which is what
    # makes "everything ready but signed out" a reachable situation.
    return _accept(state.evolve(logged_in=False), action)


def _add_to_cart(state: ShoppingState, action: ShoppingAction, config: EnvConfig) -> StepResult:
    item = action.target
    if item not in config.items:
        return _reject(state, action, f"unknown item {item!r}")
    if state.cart_size >= config.cart_capacity:
        return _reject(state, action, "cart is full")
    # Stock is intentionally not checked here; availability is only enforced
    # when the order is placed.
    return _accept(state.evolve(cart=counts_add(state.cart, item, 1)), action)


def _remove_from_cart(state: ShoppingState, action: ShoppingAction, config: EnvConfig) -> StepResult:
    item = action.target
    if item not in config.items:
        return _reject(state, action, f"unknown item {item!r}")
    if state.cart_quantity(item) == 0:
        return _reject(state, action, f"{item!r} is not in the cart")
    return _accept(state.evolve(cart=counts_add(state.cart, item, -1)), action)


def _clear_cart(state: ShoppingState, action: ShoppingAction, config: EnvConfig) -> StepResult:
    if state.cart_is_empty:
        return _reject(state, action, "cart is already empty")
    return _accept(state.evolve(cart=()), action)


def _set_payment(state: ShoppingState, action: ShoppingAction, config: EnvConfig) -> StepResult:
    method = action.target
    if method not in config.payment_options:
        return _reject(state, action, f"unknown payment method {method!r}")
    if state.payment_method == method:
        return _reject(state, action, f"payment method is already {method!r}")
    # Rejected methods can still be selected; they only fail at order time.
    return _accept(state.evolve(payment_method=method), action)


def _clear_payment(state: ShoppingState, action: ShoppingAction, config: EnvConfig) -> StepResult:
    if state.payment_method is None:
        return _reject(state, action, "no payment method is set")
    return _accept(state.evolve(payment_method=None), action)


def _set_address(state: ShoppingState, action: ShoppingAction, config: EnvConfig) -> StepResult:
    address = action.target
    if address not in config.address_options:
        return _reject(state, action, f"unknown address {address!r}")
    if state.shipping_address == address:
        return _reject(state, action, f"address is already {address!r}")
    return _accept(state.evolve(shipping_address=address), action)


def _clear_address(state: ShoppingState, action: ShoppingAction, config: EnvConfig) -> StepResult:
    if state.shipping_address is None:
        return _reject(state, action, "no address is set")
    return _accept(state.evolve(shipping_address=None), action)


def _place_order(state: ShoppingState, action: ShoppingAction, config: EnvConfig) -> StepResult:
    if not state.logged_in:
        return _reject(state, action, "not logged in")
    if state.order_status is not OrderStatus.NONE:
        return _reject(state, action, f"an order is already {state.order_status.value}")
    if state.cart_is_empty:
        return _reject(state, action, "cart is empty")
    if not config.is_valid_payment(state.payment_method):
        return _reject(state, action, "payment method is missing or not accepted")
    if not config.is_valid_address(state.shipping_address):
        return _reject(state, action, "shipping address is missing or not serviceable")
    if not counts_covers(state.stock, state.cart):
        return _reject(state, action, "insufficient stock for the cart")
    return _accept(
        state.evolve(
            cart=(),
            stock=counts_combine(state.stock, state.cart, sign=-1),
            order_status=OrderStatus.PLACED,
            order_items=state.cart,
        ),
        action,
    )


def _confirm_order(state: ShoppingState, action: ShoppingAction, config: EnvConfig) -> StepResult:
    if state.order_status is not OrderStatus.PLACED:
        return _reject(state, action, "no placed order to confirm")
    return _accept(state.evolve(order_status=OrderStatus.CONFIRMED), action)


def _cancel_order(state: ShoppingState, action: ShoppingAction, config: EnvConfig) -> StepResult:
    if state.order_status is not OrderStatus.PLACED:
        return _reject(state, action, "no placed order to cancel")
    return _accept(
        state.evolve(
            stock=counts_combine(state.stock, state.order_items, sign=1),
            order_status=OrderStatus.CANCELLED,
            order_items=(),
        ),
        action,
    )


def _clear_order(state: ShoppingState, action: ShoppingAction, config: EnvConfig) -> StepResult:
    if state.order_status not in (OrderStatus.CONFIRMED, OrderStatus.CANCELLED):
        return _reject(state, action, "only a settled order can be cleared")
    return _accept(state.evolve(order_status=OrderStatus.NONE, order_items=()), action)


_HANDLERS = {
    ActionKind.LOGIN: _login,
    ActionKind.LOGOUT: _logout,
    ActionKind.ADD_TO_CART: _add_to_cart,
    ActionKind.REMOVE_FROM_CART: _remove_from_cart,
    ActionKind.CLEAR_CART: _clear_cart,
    ActionKind.SET_PAYMENT: _set_payment,
    ActionKind.CLEAR_PAYMENT: _clear_payment,
    ActionKind.SET_ADDRESS: _set_address,
    ActionKind.CLEAR_ADDRESS: _clear_address,
    ActionKind.PLACE_ORDER: _place_order,
    ActionKind.CONFIRM_ORDER: _confirm_order,
    ActionKind.CANCEL_ORDER: _cancel_order,
    ActionKind.CLEAR_ORDER: _clear_order,
}


class ShoppingEnv:
    """Agent-facing sandbox: reset, step, inspect, snapshot, restore.

    This class is the entire surface an agent or search procedure sees. It
    exposes no declarative description of what ``place_order`` is supposed to
    do; that lives with the evaluator and is never importable from here.
    """

    def __init__(self, config: EnvConfig | None = None) -> None:
        self._config = config if config is not None else DEFAULT_CONFIG
        self._action_space = action_space(self._config)
        self._state = initial_state(self._config)
        self._step_count = 0
        self._rejected_count = 0

    @property
    def config(self) -> EnvConfig:
        return self._config

    @property
    def state(self) -> ShoppingState:
        """The current state; immutable, so the caller cannot corrupt it."""
        return self._state

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def rejected_count(self) -> int:
        return self._rejected_count

    def reset(self) -> ShoppingState:
        """Return to the seeded opening state and clear all bookkeeping."""
        self._state = initial_state(self._config)
        self._step_count = 0
        self._rejected_count = 0
        return self._state

    def step(self, action: ShoppingAction) -> StepResult:
        """Attempt ``action``; rejected actions leave the state untouched."""
        if not isinstance(action, ShoppingAction):
            raise TypeError(f"expected ShoppingAction, got {type(action).__name__}")
        result = apply_action(self._state, action, self._config)
        self._state = result.state
        self._step_count += 1
        if not result.ok:
            self._rejected_count += 1
        return result

    def action_space(self) -> tuple[ShoppingAction, ...]:
        """Every attemptable action, valid here or not, in a fixed order."""
        return self._action_space

    def valid_actions(self) -> tuple[ShoppingAction, ...]:
        """The subset of the action space that would succeed right now."""
        return valid_actions(self._state, self._config)

    def snapshot(self) -> EnvSnapshot:
        """Capture the current state and counters for later restoration."""
        return EnvSnapshot(
            state=self._state,
            step_count=self._step_count,
            rejected_count=self._rejected_count,
            config_fingerprint=self._config.fingerprint,
        )

    def restore(self, snapshot: EnvSnapshot) -> ShoppingState:
        """Rewind to a snapshot taken from an identically configured env."""
        if not isinstance(snapshot, EnvSnapshot):
            raise TypeError(f"expected EnvSnapshot, got {type(snapshot).__name__}")
        if snapshot.config_fingerprint != self._config.fingerprint:
            raise ValueError("snapshot was taken under a different configuration")
        self._state = snapshot.state
        self._step_count = snapshot.step_count
        self._rejected_count = snapshot.rejected_count
        return self._state

    def run(self, actions: list[ShoppingAction]) -> list[StepResult]:
        """Apply a scripted sequence, returning every individual outcome."""
        return [self.step(action) for action in actions]


def valid_actions(
    state: ShoppingState, config: EnvConfig = DEFAULT_CONFIG
) -> tuple[ShoppingAction, ...]:
    """Actions that succeed in ``state``, derived from the transition rules.

    Deriving this by trial application rather than by a second set of guards
    keeps ``valid_actions`` and ``step`` from ever disagreeing.
    """
    return tuple(
        action for action in action_space(config) if apply_action(state, action, config).ok
    )


__all__ = [
    "EnvSnapshot",
    "ShoppingEnv",
    "StepResult",
    "action_space",
    "apply_action",
    "initial_state",
    "valid_actions",
]
