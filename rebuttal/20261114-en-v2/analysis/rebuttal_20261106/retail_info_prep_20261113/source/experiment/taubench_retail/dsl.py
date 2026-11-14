#!/usr/bin/env python3
"""A safe, declarative JSON DSL for candidate exchange_delivered_order_items contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import DEFAULT_RETAIL_CONFIG, RetailConfig
from .contracts import Contract
from .ground_truth import ITEM_AVAILABLE, ITEM_PRICES, ITEM_TO_PRODUCT, ORDER_ITEMS, calculate_price_difference
from .state import ActionKind, RetailAction, RetailState, SEMANTIC_FIELDS_PROTOCOL, SEMANTIC_FIELDS_SCOPE

SKILL = "exchange_delivered_order_items"

MAX_NODES = 200
MAX_DEPTH = 12
MAX_SOURCE_CHARS = 40000

CONTRACT_KEYS = ("skill", "name", "notes", "precondition", "postcondition")
REQUIRED_CONTRACT_KEYS = ("precondition", "postcondition")

SCOPE_PRE = "precondition"
SCOPE_POST = "postcondition"

VOCABULARY_LEGACY = "legacy_w1_v1"
VOCABULARY_ORDER_NEUTRAL = "order_neutral_v1"
VOCABULARY_SEMANTIC_FIELDS = SEMANTIC_FIELDS_PROTOCOL
VOCABULARY_PROTOCOLS = (VOCABULARY_LEGACY, VOCABULARY_ORDER_NEUTRAL, VOCABULARY_SEMANTIC_FIELDS)


class DslError(ValueError):
    """Model output is not a well-formed, in-grammar contract."""


class _Invalid:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<invalid>"


INVALID = _Invalid()


@dataclass(frozen=True)
class Binding:
    before: RetailState
    after: RetailState
    config: RetailConfig
    vocabulary_protocol: str = VOCABULARY_LEGACY


Term = Callable[[Binding], Any]
Formula = Callable[[Binding], bool]

_VAR_READERS: dict[str, Callable[[RetailState], Any]] = {
    "order_id": lambda state: state.draft_order_id,
    "item_ids": lambda state: list(state.draft_item_ids),
    "item_ids_count": lambda state: len(state.draft_item_ids),
    "new_item_ids": lambda state: list(state.draft_new_item_ids),
    "new_item_ids_count": lambda state: len(state.draft_new_item_ids),
    "payment_method_id": lambda state: state.draft_payment_method_id,
    "order_w1_status": lambda state: state.order_w1_status,
    "order_w2_status": lambda state: state.order_w2_status,
    "gift_card_balance": lambda state: round(state.user_gift_card_balance, 2),
    "order_w1_exchange_items": lambda state: list(state.order_w1_exchange_items),
    "order_w1_exchange_new_items": lambda state: list(state.order_w1_exchange_new_items),
    "order_w1_exchange_payment_method_id": lambda state: state.order_w1_exchange_payment_method_id,
    "order_w1_exchange_price_difference": lambda state: state.order_w1_exchange_price_difference,
    "order_w1_return_items": lambda state: list(state.order_w1_return_items),
    "order_w2_cancel_reason": lambda state: state.order_w2_cancel_reason,
    "authenticated": lambda state: state.authenticated,
}

_VAR_KINDS: dict[str, str] = {
    "order_id": "option",
    "item_ids": "items",
    "item_ids_count": "number",
    "new_item_ids": "items",
    "new_item_ids_count": "number",
    "payment_method_id": "option",
    "order_w1_status": "option",
    "order_w2_status": "option",
    "gift_card_balance": "number",
    "order_w1_exchange_items": "items",
    "order_w1_exchange_new_items": "items",
    "order_w1_exchange_payment_method_id": "option",
    "order_w1_exchange_price_difference": "number",
    "order_w1_return_items": "items",
    "order_w2_cancel_reason": "option",
    "authenticated": "bool",
}

STATE_VARS: tuple[str, ...] = tuple(sorted(_VAR_READERS))
VAR_KINDS: Mapping[str, str] = dict(_VAR_KINDS)

_SEMANTIC_READERS = {**_VAR_READERS,
    "order_w1_return_payment_method_id": lambda state: state.order_w1_return_payment_method_id}
_SEMANTIC_KINDS = {**_VAR_KINDS, "order_w1_return_payment_method_id": "option"}


def state_var_kinds(vocabulary_protocol: str = VOCABULARY_LEGACY) -> Mapping[str, str]:
    return _SEMANTIC_KINDS if vocabulary_protocol == VOCABULARY_SEMANTIC_FIELDS else VAR_KINDS

CONFIG_SETS: tuple[str, ...] = (
    "valid_orders",
    "rejected_orders",
    "order_options",
    "valid_items",
    "rejected_items",
    "item_options",
    "valid_new_items",
    "rejected_new_items",
    "new_item_options",
    "valid_payment_methods",
    "rejected_payment_methods",
    "payment_method_options",
)

_TEXTUAL = frozenset({"text", "option", "null"})

FORMULA_OPS: dict[str, str] = {
    "const": '{"op": "const", "value": true} — a literal truth value.',
    "and": '{"op": "and", "args": [f, ...]} — every listed formula holds.',
    "or": '{"op": "or", "args": [f, ...]} — at least one listed formula holds.',
    "not": '{"op": "not", "arg": f} — the formula does not hold.',
    "is_true": '{"op": "is_true", "arg": t} — a boolean term is true.',
    "is_null": '{"op": "is_null", "arg": t} — an optional term is unset (null).',
    "is_empty": '{"op": "is_empty", "arg": t} — an item list has no members.',
    "eq": '{"op": "eq", "left": t, "right": t} — two terms of the same kind are equal.',
    "ne": '{"op": "ne", "left": t, "right": t} — two terms of the same kind differ.',
    "lt": '{"op": "lt", "left": t, "right": t} — a number is strictly smaller.',
    "le": '{"op": "le", "left": t, "right": t} — a number is smaller or equal.',
    "gt": '{"op": "gt", "left": t, "right": t} — a number is strictly greater.',
    "ge": '{"op": "ge", "left": t, "right": t} — a number is greater or equal.',
    "in_set": '{"op": "in_set", "value": t, "set": "<config set>"} — member of set (for option/text) or all members in set (for items list).',
    "contains": '{"op": "contains", "container": t, "item": t} — list contains item.',
    "items_in_order": '{"op": "items_in_order", "items": t, "order": t} — all items belong to order.',
    "valid_variants": '{"op": "valid_variants", "items": t, "new_items": t} — new items match product variants.',
    "items_available": '{"op": "items_available", "items": t} — all items in list are available.',
    "sufficient_balance": '{"op": "sufficient_balance", "payment_method": t, "balance": t, "items": t, "new_items": t} — balance covers diff.',
    "exchange_snapshot_matches": '{"op": "exchange_snapshot_matches"} — after-state reflects draft exchange execution.',
    "unchanged": '{"op": "unchanged", "vars": ["..."]} — postcondition variables unchanged.',
}

TERM_OPS: dict[str, str] = {
    "var": '{"var": "<name>", "when": "before"|"after"} — read a state variable.',
    "const": '{"const": ...} — literal number, string, boolean, null, or list.',
    "count": '{"op": "count", "arg": t} — count of items in a list.',
    "price_difference": '{"op": "price_difference", "items": t, "new_items": t} — price diff of exchange.',
}


def _config_set(name: str, config: RetailConfig) -> set[str]:
    if hasattr(config, name):
        val = getattr(config, name)
        if isinstance(val, (tuple, list, set)):
            return set(val)
    raise DslError(f"unknown config set: {name!r}; choose from {list(CONFIG_SETS)}")


class _Counter:
    __slots__ = ("nodes", "readers", "kinds")

    def __init__(self, vocabulary_protocol: str = VOCABULARY_LEGACY) -> None:
        self.nodes = 0
        self.readers = _SEMANTIC_READERS if vocabulary_protocol == VOCABULARY_SEMANTIC_FIELDS else _VAR_READERS
        self.kinds = state_var_kinds(vocabulary_protocol)

    def tick(self) -> None:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise DslError(f"contract exceeds {MAX_NODES} AST nodes")


def _read_var(var_name: str, when: str, binding: Binding) -> Any:
    state = binding.before if when == "before" else binding.after
    readers = _SEMANTIC_READERS if binding.vocabulary_protocol == VOCABULARY_SEMANTIC_FIELDS else _VAR_READERS
    reader = readers.get(var_name)
    if reader is None:
        return INVALID
    return reader(state)


def _compile_term(node: Any, scope: str, depth: int, counter: _Counter) -> tuple[Term, str]:
    counter.tick()
    if depth > MAX_DEPTH:
        raise DslError(f"AST exceeds maximum depth of {MAX_DEPTH}")

    if not isinstance(node, dict):
        raise DslError(f"expected object for term, got {type(node).__name__}: {node!r}")

    if "var" in node:
        var_name = node["var"]
        if not isinstance(var_name, str) or var_name not in counter.readers:
            raise DslError(f"unknown state variable {var_name!r}; choose from {STATE_VARS}")
        when = node.get("when", "before")
        if when not in ("before", "after"):
            raise DslError(f"'when' must be 'before' or 'after', got {when!r}")
        if when == "after" and scope != SCOPE_POST:
            raise DslError("when: 'after' is only valid inside postconditions")
        kind = counter.kinds[var_name]

        def var_term(binding: Binding, v=var_name, w=when) -> Any:
            return _read_var(v, w, binding)

        return var_term, kind

    if "const" in node:
        val = node["const"]
        if isinstance(val, bool):
            kind = "bool"
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            val = float(val) if isinstance(val, float) else int(val)
            kind = "number"
        elif isinstance(val, str):
            kind = "option"
        elif val is None:
            kind = "null"
        elif isinstance(val, (list, tuple)) and all(isinstance(x, str) for x in val):
            val = list(val)
            kind = "items"
        else:
            raise DslError(f"unsupported constant value: {val!r}")

        def const_term(binding: Binding, c=val) -> Any:
            return c

        return const_term, kind

    op = node.get("op")
    if not isinstance(op, str):
        raise DslError(f"term object must contain 'var', 'const', or 'op': {node!r}")

    if op == "count":
        arg_node = node.get("arg")
        if arg_node is None:
            raise DslError("'count' requires 'arg'")
        arg_term, arg_kind = _compile_term(arg_node, scope, depth + 1, counter)
        if arg_kind not in ("items", "null"):
            raise DslError(f"'count' requires items list, got {arg_kind}")

        def count_term(binding: Binding) -> int:
            val = arg_term(binding)
            return len(val) if isinstance(val, (list, tuple)) else 0

        return count_term, "number"

    if op == "price_difference":
        items_node = node.get("items")
        new_items_node = node.get("new_items")
        if items_node is None or new_items_node is None:
            raise DslError("'price_difference' requires 'items' and 'new_items'")
        it_term, it_kind = _compile_term(items_node, scope, depth + 1, counter)
        nit_term, nit_kind = _compile_term(new_items_node, scope, depth + 1, counter)

        def diff_term(binding: Binding) -> float:
            its = it_term(binding) or ()
            nits = nit_term(binding) or ()
            return calculate_price_difference(tuple(its), tuple(nits))

        return diff_term, "number"

    raise DslError(f"unknown term operator: {op!r}; choose from {list(TERM_OPS)}")


def _compile_formula(node: Any, scope: str, depth: int, counter: _Counter) -> Formula:
    counter.tick()
    if depth > MAX_DEPTH:
        raise DslError(f"AST exceeds maximum depth of {MAX_DEPTH}")

    if not isinstance(node, dict):
        raise DslError(f"expected object for formula, got {type(node).__name__}: {node!r}")

    op = node.get("op")
    if not isinstance(op, str):
        raise DslError(f"formula object must contain 'op': {node!r}")

    if op == "const":
        val = node.get("value")
        if not isinstance(val, bool):
            raise DslError(f"'const' formula requires boolean 'value', got {val!r}")
        return lambda binding, v=val: v

    if op == "and":
        args = node.get("args")
        if not isinstance(args, list) or not args:
            raise DslError("'and' requires non-empty list 'args'")
        sub_formulas = tuple(_compile_formula(arg, scope, depth + 1, counter) for arg in args)
        return lambda binding, sf=sub_formulas: all(f(binding) for f in sf)

    if op == "or":
        args = node.get("args")
        if not isinstance(args, list) or not args:
            raise DslError("'or' requires non-empty list 'args'")
        sub_formulas = tuple(_compile_formula(arg, scope, depth + 1, counter) for arg in args)
        return lambda binding, sf=sub_formulas: any(f(binding) for f in sf)

    if op == "not":
        arg = node.get("arg")
        if arg is None:
            raise DslError("'not' requires 'arg'")
        sub_formula = _compile_formula(arg, scope, depth + 1, counter)
        return lambda binding, sf=sub_formula: not sf(binding)

    if op == "is_true":
        arg_node = node.get("arg")
        if arg_node is None:
            raise DslError("'is_true' requires 'arg'")
        term, kind = _compile_term(arg_node, scope, depth + 1, counter)
        if kind != "bool":
            raise DslError(f"'is_true' requires bool term, got {kind}")
        return lambda binding, t=term: bool(t(binding))

    if op == "is_null":
        arg_node = node.get("arg")
        if arg_node is None:
            raise DslError("'is_null' requires 'arg'")
        term, kind = _compile_term(arg_node, scope, depth + 1, counter)
        return lambda binding, t=term: t(binding) is None

    if op == "is_empty":
        arg_node = node.get("arg")
        if arg_node is None:
            raise DslError("'is_empty' requires 'arg'")
        term, kind = _compile_term(arg_node, scope, depth + 1, counter)
        return lambda binding, t=term: len(t(binding) or ()) == 0

    if op in ("eq", "ne"):
        left_node = node.get("left")
        right_node = node.get("right")
        if left_node is None or right_node is None:
            raise DslError(f"{op!r} requires 'left' and 'right'")
        lt, lk = _compile_term(left_node, scope, depth + 1, counter)
        rt, rk = _compile_term(right_node, scope, depth + 1, counter)
        if op == "eq":
            return lambda binding, l=lt, r=rt: l(binding) == r(binding)
        return lambda binding, l=lt, r=rt: l(binding) != r(binding)

    if op in ("lt", "le", "gt", "ge"):
        left_node = node.get("left")
        right_node = node.get("right")
        if left_node is None or right_node is None:
            raise DslError(f"{op!r} requires 'left' and 'right'")
        lt, lk = _compile_term(left_node, scope, depth + 1, counter)
        rt, rk = _compile_term(right_node, scope, depth + 1, counter)
        if lk not in ("number", "null") or rk not in ("number", "null"):
            raise DslError(f"{op!r} requires number terms, got {lk} and {rk}")
        if op == "lt":
            return lambda binding, l=lt, r=rt: (l(binding) or 0) < (r(binding) or 0)
        if op == "le":
            return lambda binding, l=lt, r=rt: (l(binding) or 0) <= (r(binding) or 0)
        if op == "gt":
            return lambda binding, l=lt, r=rt: (l(binding) or 0) > (r(binding) or 0)
        return lambda binding, l=lt, r=rt: (l(binding) or 0) >= (r(binding) or 0)

    if op == "in_set":
        val_node = node.get("value")
        set_name = node.get("set")
        if val_node is None or not isinstance(set_name, str):
            raise DslError("'in_set' requires 'value' term and string 'set' name")
        vt, vk = _compile_term(val_node, scope, depth + 1, counter)
        if set_name not in CONFIG_SETS:
            raise DslError(f"unknown config set {set_name!r}; choose from {CONFIG_SETS}")

        if vk in _TEXTUAL:
            def check_in_set_scalar(binding: Binding) -> bool:
                val = vt(binding)
                if val is None:
                    return False
                return val in _config_set(set_name, binding.config)

            return check_in_set_scalar

        if vk == "items":
            def check_in_set_items(binding: Binding) -> bool:
                val = vt(binding)
                if not val:
                    return True
                target = _config_set(set_name, binding.config)
                return all(x in target for x in val)

            return check_in_set_items

        raise DslError(f"'in_set' requires an option/text or items value, got {vk}")

    if op == "contains":
        c_node = node.get("container")
        i_node = node.get("item")
        if c_node is None or i_node is None:
            raise DslError("'contains' requires 'container' and 'item'")
        ct, ck = _compile_term(c_node, scope, depth + 1, counter)
        it, ik = _compile_term(i_node, scope, depth + 1, counter)
        return lambda binding, c=ct, i=it: i(binding) in (c(binding) or ())

    if op == "items_in_order":
        items_node = node.get("items")
        order_node = node.get("order")
        if items_node is None or order_node is None:
            raise DslError("'items_in_order' requires 'items' and 'order'")
        it_term, _ = _compile_term(items_node, scope, depth + 1, counter)
        ord_term, _ = _compile_term(order_node, scope, depth + 1, counter)

        def check_items_in_order(binding: Binding) -> bool:
            its = it_term(binding) or ()
            ord_id = ord_term(binding)
            order_items = ORDER_ITEMS.get(ord_id if ord_id in ORDER_ITEMS else "#W1", ())
            return all(its.count(x) <= order_items.count(x) for x in its)

        return check_items_in_order

    if op == "valid_variants":
        items_node = node.get("items")
        new_items_node = node.get("new_items")
        if items_node is None or new_items_node is None:
            raise DslError("'valid_variants' requires 'items' and 'new_items'")
        it_term, _ = _compile_term(items_node, scope, depth + 1, counter)
        nit_term, _ = _compile_term(new_items_node, scope, depth + 1, counter)

        def check_valid_variants(binding: Binding) -> bool:
            its = it_term(binding) or ()
            nits = nit_term(binding) or ()
            for old_it, new_it in zip(its, nits):
                p1 = ITEM_TO_PRODUCT.get(old_it)
                p2 = ITEM_TO_PRODUCT.get(new_it)
                if p1 is None or p2 is None or p1 != p2:
                    return False
            return True

        return check_valid_variants

    if op == "items_available":
        items_node = node.get("items")
        if items_node is None:
            raise DslError("'items_available' requires 'items'")
        it_term, _ = _compile_term(items_node, scope, depth + 1, counter)

        def check_items_available(binding: Binding) -> bool:
            its = it_term(binding) or ()
            return all(ITEM_AVAILABLE.get(x, False) for x in its)

        return check_items_available

    if op == "sufficient_balance":
        pm_node = node.get("payment_method")
        bal_node = node.get("balance")
        items_node = node.get("items")
        new_items_node = node.get("new_items")
        if pm_node is None or bal_node is None or items_node is None or new_items_node is None:
            raise DslError("'sufficient_balance' requires 'payment_method', 'balance', 'items', 'new_items'")
        pm_term, _ = _compile_term(pm_node, scope, depth + 1, counter)
        bal_term, _ = _compile_term(bal_node, scope, depth + 1, counter)
        it_term, _ = _compile_term(items_node, scope, depth + 1, counter)
        nit_term, _ = _compile_term(new_items_node, scope, depth + 1, counter)

        def check_balance(binding: Binding) -> bool:
            pm = pm_term(binding)
            if pm != "gift_card_0":
                return True
            bal = bal_term(binding) or 0.0
            its = it_term(binding) or ()
            nits = nit_term(binding) or ()
            diff = calculate_price_difference(tuple(its), tuple(nits))
            return round(bal, 2) >= diff

        return check_balance

    if op == "exchange_snapshot_matches":
        if scope != SCOPE_POST:
            raise DslError("'exchange_snapshot_matches' is only valid in postcondition")

        def snapshot_matches(binding: Binding) -> bool:
            before = binding.before
            after = binding.after
            if binding.vocabulary_protocol == VOCABULARY_SEMANTIC_FIELDS:
                if before.draft_order_id != "#W1" or not (
                    before.order_w1_return_items == after.order_w1_return_items
                    and before.order_w1_return_payment_method_id == after.order_w1_return_payment_method_id
                    and before.authenticated == after.authenticated
                ):
                    return False
            if binding.vocabulary_protocol == VOCABULARY_ORDER_NEUTRAL:
                if before.draft_order_id == "#W2":
                    # W2 payload fields are not represented by this sandbox. Check
                    # its observable status, the draft reset, and preserved fields.
                    preserved = (
                        "order_w1_status", "order_w1_exchange_items",
                        "order_w1_exchange_new_items", "order_w1_exchange_payment_method_id",
                        "order_w1_exchange_price_difference", "order_w1_return_items",
                        "order_w2_cancel_reason", "gift_card_balance", "authenticated",
                    )
                    return (
                        after.order_w2_status == "exchange requested"
                        and all(_VAR_READERS[v](before) == _VAR_READERS[v](after) for v in preserved)
                        and after.draft_order_id is None
                        and after.draft_item_ids == ()
                        and after.draft_new_item_ids == ()
                        and after.draft_payment_method_id is None
                    )
                if before.draft_order_id != "#W1":
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

        return snapshot_matches

    if op == "unchanged":
        if scope != SCOPE_POST:
            raise DslError("'unchanged' is only valid in postcondition")
        vars_list = node.get("vars")
        if not isinstance(vars_list, list) or not vars_list:
            raise DslError("'unchanged' requires non-empty list 'vars'")
        for v in vars_list:
            if v not in counter.readers:
                raise DslError(f"unknown variable in unchanged: {v!r}")

        def unchanged_formula(binding: Binding, vl=tuple(vars_list)) -> bool:
            before = binding.before
            after = binding.after
            for var_name in vl:
                reader = counter.readers[var_name]
                if reader(before) != reader(after):
                    return False
            return True

        return unchanged_formula

    raise DslError(f"unknown formula operator: {op!r}; choose from {list(FORMULA_OPS)}")


class ParsedContract:
    def __init__(
        self,
        name: str,
        notes: str,
        pre_fn: Formula,
        post_fn: Formula,
        default_config: RetailConfig = DEFAULT_RETAIL_CONFIG,
        vocabulary_protocol: str = VOCABULARY_LEGACY,
    ) -> None:
        self.name = name
        self.notes = notes
        self._pre_fn = pre_fn
        self._post_fn = post_fn
        self._default_config = default_config
        self._vocabulary_protocol = vocabulary_protocol
        self.action = RetailAction.make(ActionKind.EXCHANGE_ITEMS)

    def bind(self, config: RetailConfig | None = None) -> Contract:
        effective_config = config or self._default_config

        def precondition(state: RetailState) -> bool:
            binding = Binding(before=state, after=state, config=effective_config,
                              vocabulary_protocol=self._vocabulary_protocol)
            return bool(self._pre_fn(binding))

        def postcondition(before: RetailState, after: RetailState) -> bool:
            binding = Binding(before=before, after=after, config=effective_config,
                              vocabulary_protocol=self._vocabulary_protocol)
            return bool(self._post_fn(binding))

        return Contract(
            name=self.name,
            action=self.action,
            precondition=precondition,
            postcondition=postcondition,
            description=self.notes,
        )

    def holds_in(self, state: RetailState) -> bool:
        return self.bind().holds_in(state)

    def transition_holds(self, before: RetailState, after: RetailState) -> bool:
        return self.bind().transition_holds(before, after)


def parse_contract(
    source: str | Mapping[str, Any],
    config: RetailConfig | None = None,
    *,
    vocabulary_protocol: str = VOCABULARY_LEGACY,
) -> ParsedContract:
    """Parse JSON or dict into an executable Retail Contract."""
    if vocabulary_protocol not in VOCABULARY_PROTOCOLS:
        raise DslError(f"unknown vocabulary protocol: {vocabulary_protocol!r}")
    if isinstance(source, str):
        if len(source) > MAX_SOURCE_CHARS:
            raise DslError(f"contract source exceeds {MAX_SOURCE_CHARS} characters")
        text = source.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DslError(f"invalid JSON: {exc}") from exc
    elif isinstance(source, Mapping):
        data = dict(source)
    else:
        raise DslError(f"expected str or Mapping, got {type(source).__name__}")

    if not isinstance(data, dict):
        raise DslError(f"contract root must be a JSON object, got {type(data).__name__}")

    for key in REQUIRED_CONTRACT_KEYS:
        if key not in data:
            raise DslError(f"missing required key: {key!r}")

    skill = data.get("skill", SKILL)
    if skill != SKILL:
        raise DslError(f"expected skill {SKILL!r}, got {skill!r}")

    name = str(data.get("name", "candidate_exchange_delivered_order_items"))
    description = str(data.get("notes", ""))

    counter = _Counter(vocabulary_protocol)
    pre_fn = _compile_formula(data["precondition"], SCOPE_PRE, 1, counter)
    post_fn = _compile_formula(data["postcondition"], SCOPE_POST, 1, counter)

    return ParsedContract(
        name=name,
        notes=description,
        pre_fn=pre_fn,
        post_fn=post_fn,
        default_config=config or DEFAULT_RETAIL_CONFIG,
        vocabulary_protocol=vocabulary_protocol,
    )


def contract_json_schema(vocabulary_protocol: str = VOCABULARY_LEGACY) -> dict[str, Any]:
    """JSON Schema for guided decoding."""
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "RetailContract",
        "type": "object",
        "required": ["skill", "precondition", "postcondition"],
        "properties": {
            "skill": {"type": "string", "const": SKILL},
            "name": {"type": "string"},
            "notes": {"type": "string"},
            "precondition": {"type": "object"},
            "postcondition": {"type": "object"},
        },
        "additionalProperties": False,
    }
    if vocabulary_protocol == VOCABULARY_SEMANTIC_FIELDS:
        schema["title"] = "RetailContractSemanticFieldsV1"
        schema["description"] = f"Vocabulary {vocabulary_protocol}; observation scope {SEMANTIC_FIELDS_SCOPE}."
    return schema
