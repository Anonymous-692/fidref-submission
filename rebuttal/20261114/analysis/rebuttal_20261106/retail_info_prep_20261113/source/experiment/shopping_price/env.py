#!/usr/bin/env python3
"""Deterministic transition rules for price-aware shopping."""

from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_PRICE_SHOPPING_CONFIG, PriceShoppingConfig
from .state import (
    ActionKind,
    OrderStatus,
    PriceShoppingAction,
    PriceShoppingState,
    counts_add,
    counts_combine,
    counts_covers,
)


@dataclass(frozen=True)
class StepResult:
    state: PriceShoppingState
    ok: bool
    action: PriceShoppingAction
    error: str | None = None


def initial_state(config: PriceShoppingConfig = DEFAULT_PRICE_SHOPPING_CONFIG) -> PriceShoppingState:
    return PriceShoppingState(
        logged_in=False,
        cart=(),
        stock=config.initial_stock(),
        payment_method=None,
        shipping_address=None,
        coupon_code=None,
        wallet_balance=config.initial_wallet_balance,
        card_limit=config.card_limit,
        cart_subtotal=0,
        checkout_total=0,
        order_status=OrderStatus.NONE,
        order_items=(),
        order_total=0,
        order_payment_method=None,
    )


def action_space(config: PriceShoppingConfig = DEFAULT_PRICE_SHOPPING_CONFIG) -> tuple[PriceShoppingAction, ...]:
    actions = [
        PriceShoppingAction.make(ActionKind.LOGIN),
        PriceShoppingAction.make(ActionKind.LOGOUT),
    ]
    actions.extend(PriceShoppingAction.make(ActionKind.ADD_TO_CART, item) for item in config.items)
    actions.extend(PriceShoppingAction.make(ActionKind.REMOVE_FROM_CART, item) for item in config.items)
    actions.append(PriceShoppingAction.make(ActionKind.CLEAR_CART))
    actions.extend(PriceShoppingAction.make(ActionKind.SET_PAYMENT, method) for method in config.payment_methods)
    actions.append(PriceShoppingAction.make(ActionKind.CLEAR_PAYMENT))
    actions.extend(PriceShoppingAction.make(ActionKind.SET_ADDRESS, address) for address in config.addresses)
    actions.append(PriceShoppingAction.make(ActionKind.CLEAR_ADDRESS))
    actions.extend(PriceShoppingAction.make(ActionKind.APPLY_COUPON, code) for code in config.coupon_codes)
    actions.extend(
        [
            PriceShoppingAction.make(ActionKind.CLEAR_COUPON),
            PriceShoppingAction.make(ActionKind.PLACE_ORDER),
        ]
    )
    return tuple(sorted(actions, key=lambda action: action.sort_key))


def _accept(state: PriceShoppingState, action: PriceShoppingAction) -> StepResult:
    return StepResult(state, True, action)


def _reject(state: PriceShoppingState, action: PriceShoppingAction, error: str) -> StepResult:
    return StepResult(state, False, action, error)


def _refresh_pricing(state: PriceShoppingState, config: PriceShoppingConfig) -> PriceShoppingState:
    return state.evolve(
        cart_subtotal=config.subtotal(state.cart),
        checkout_total=config.payable_total(state),
    )


def apply_action(
    state: PriceShoppingState,
    action: PriceShoppingAction,
    config: PriceShoppingConfig = DEFAULT_PRICE_SHOPPING_CONFIG,
) -> StepResult:
    kind = action.kind
    post_order_cart_probe = config.allow_post_order_cart_add and (
        kind is ActionKind.ADD_TO_CART
        or (
            config.clear_checkout_selections_after_order
            and kind in (ActionKind.SET_PAYMENT, ActionKind.SET_ADDRESS)
        )
    )
    if (
        config.terminal_after_order
        and state.order_status is not OrderStatus.NONE
        and not post_order_cart_probe
    ):
        return _reject(state, action, "episode ended after order placement")
    if kind is ActionKind.LOGIN:
        return _reject(state, action, "already logged in") if state.logged_in else _accept(state.evolve(logged_in=True), action)
    if kind is ActionKind.LOGOUT:
        return _reject(state, action, "not logged in") if not state.logged_in else _accept(state.evolve(logged_in=False), action)
    if kind is ActionKind.ADD_TO_CART:
        if action.target not in config.items:
            return _reject(state, action, "unknown item")
        if state.cart_size >= config.cart_capacity:
            return _reject(state, action, "cart is full")
        return _accept(_refresh_pricing(state.evolve(cart=counts_add(state.cart, action.target or "", 1)), config), action)
    if kind is ActionKind.REMOVE_FROM_CART:
        if action.target not in config.items or state.cart_quantity(action.target or "") == 0:
            return _reject(state, action, "item is not in the cart")
        return _accept(_refresh_pricing(state.evolve(cart=counts_add(state.cart, action.target or "", -1)), config), action)
    if kind is ActionKind.CLEAR_CART:
        return _reject(state, action, "cart is already empty") if state.cart_is_empty else _accept(_refresh_pricing(state.evolve(cart=()), config), action)
    if kind is ActionKind.SET_PAYMENT:
        if action.target not in config.payment_methods:
            return _reject(state, action, "unknown payment method")
        if state.payment_method == action.target:
            return _reject(state, action, "payment method already selected")
        return _accept(state.evolve(payment_method=action.target), action)
    if kind is ActionKind.CLEAR_PAYMENT:
        return _reject(state, action, "no payment method selected") if state.payment_method is None else _accept(state.evolve(payment_method=None), action)
    if kind is ActionKind.SET_ADDRESS:
        if action.target not in config.addresses:
            return _reject(state, action, "unknown address")
        if state.shipping_address == action.target:
            return _reject(state, action, "address already selected")
        return _accept(_refresh_pricing(state.evolve(shipping_address=action.target), config), action)
    if kind is ActionKind.CLEAR_ADDRESS:
        return _reject(state, action, "no address selected") if state.shipping_address is None else _accept(_refresh_pricing(state.evolve(shipping_address=None), config), action)
    if kind is ActionKind.APPLY_COUPON:
        if action.target not in config.coupon_codes:
            return _reject(state, action, "unknown coupon")
        if state.coupon_code == action.target:
            return _reject(state, action, "coupon already selected")
        return _accept(_refresh_pricing(state.evolve(coupon_code=action.target), config), action)
    if kind is ActionKind.CLEAR_COUPON:
        return _reject(state, action, "no coupon selected") if state.coupon_code is None else _accept(_refresh_pricing(state.evolve(coupon_code=None), config), action)
    if kind is ActionKind.PLACE_ORDER:
        return _place_order(state, action, config)
    if kind is ActionKind.CONFIRM_ORDER:
        if state.order_status is not OrderStatus.PLACED:
            return _reject(state, action, "no placed order")
        return _accept(state.evolve(order_status=OrderStatus.CONFIRMED), action)
    if kind is ActionKind.CANCEL_ORDER:
        if state.order_status is not OrderStatus.PLACED:
            return _reject(state, action, "no placed order")
        refunded = state.order_total if state.order_payment_method == "wallet" else 0
        return _accept(
            state.evolve(
                stock=counts_combine(state.stock, state.order_items),
                wallet_balance=state.wallet_balance + refunded,
                order_status=OrderStatus.CANCELLED,
                order_items=(),
                order_total=0,
                order_payment_method=None,
            ),
            action,
        )
    if kind is ActionKind.CLEAR_ORDER:
        if state.order_status not in (OrderStatus.PLACED, OrderStatus.CONFIRMED, OrderStatus.CANCELLED):
            return _reject(state, action, "no order can be archived")
        return _accept(
            state.evolve(
                order_status=OrderStatus.NONE,
                order_items=(),
                order_total=0,
                order_payment_method=None,
            ),
            action,
        )
    raise ValueError(f"unhandled action kind: {kind}")


def _place_order(state: PriceShoppingState, action: PriceShoppingAction, config: PriceShoppingConfig) -> StepResult:
    if not state.logged_in:
        return _reject(state, action, "not logged in")
    if state.order_status is not OrderStatus.NONE:
        return _reject(state, action, "order slot is occupied")
    if state.cart_is_empty:
        return _reject(state, action, "cart is empty")
    if state.payment_method not in config.accepted_payment_methods:
        return _reject(state, action, "payment method is not accepted")
    if state.shipping_address not in config.serviceable_addresses:
        return _reject(state, action, "address is not serviceable")
    if not counts_covers(state.stock, state.cart):
        return _reject(state, action, "insufficient stock")
    if state.coupon_code is not None and not config.coupon_eligible(state.coupon_code, state.cart):
        return _reject(state, action, "coupon is invalid or ineligible")
    payable = config.payable_total(state)
    if state.payment_method == "wallet" and state.wallet_balance < payable:
        return _reject(state, action, "insufficient wallet balance")
    if state.payment_method == "card" and payable > config.card_limit:
        return _reject(state, action, "card limit exceeded")
    wallet_after = state.wallet_balance - payable if state.payment_method == "wallet" else state.wallet_balance
    payment_after = None if config.clear_checkout_selections_after_order else state.payment_method
    address_after = None if config.clear_checkout_selections_after_order else state.shipping_address
    return _accept(
        _refresh_pricing(state.evolve(
            cart=(),
            stock=counts_combine(state.stock, state.cart, sign=-1),
            coupon_code=None,
            wallet_balance=wallet_after,
            order_status=OrderStatus.PLACED,
            order_items=state.cart,
            order_total=payable,
            order_payment_method=state.payment_method,
            payment_method=payment_after,
            shipping_address=address_after,
        ), config),
        action,
    )


def valid_actions(state: PriceShoppingState, config: PriceShoppingConfig = DEFAULT_PRICE_SHOPPING_CONFIG) -> tuple[PriceShoppingAction, ...]:
    return tuple(action for action in action_space(config) if apply_action(state, action, config).ok)


class PriceShoppingEnv:
    def __init__(self, config: PriceShoppingConfig | None = None) -> None:
        self.config = config or DEFAULT_PRICE_SHOPPING_CONFIG
        self.state = initial_state(self.config)

    def reset(self) -> PriceShoppingState:
        self.state = initial_state(self.config)
        return self.state

    def step(self, action: PriceShoppingAction) -> StepResult:
        result = apply_action(self.state, action, self.config)
        self.state = result.state
        return result
