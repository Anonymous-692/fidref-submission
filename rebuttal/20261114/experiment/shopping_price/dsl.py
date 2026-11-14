#!/usr/bin/env python3
"""A safe, declarative JSON DSL for candidate ``place_order`` contracts.

Model output is *data*, never code. A contract arrives as a JSON object whose
operators are drawn from the closed tables below; parsing walks that object and
builds a tree of small closures. Nothing is ``eval``-ed, ``exec``-ed, imported,
or otherwise executed, so a malformed or hostile response can only ever become a
:class:`DslError` — that is, a recorded parse failure — and never a side effect.

Two further properties matter for the experiment:

* **Bounded.** Node count and nesting depth are capped, so a pathological
  response cannot make parsing or scoring expensive.
* **Total.** Compiled formulas never raise on a well-typed state: operators that
  cannot produce a value (a multiset difference that would go negative, a
  comparison between incomparable values) yield ``False`` rather than an
  exception, so one odd state cannot abort a whole evaluation sweep.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .config import PriceShoppingConfig
from .state import (
    OrderStatus,
    PriceShoppingAction,
    PriceShoppingState,
    counts_covers,
    counts_from_mapping,
    counts_get,
    counts_total,
)
from .contracts import Contract

# Only ``place_order`` has a reference specification, so it is the only skill a
# contract may name.
SKILL = "place_order"

MAX_NODES = 200
MAX_DEPTH = 12
MAX_SOURCE_CHARS = 40000

CONTRACT_KEYS = ("skill", "name", "notes", "precondition", "postcondition")
REQUIRED_CONTRACT_KEYS = ("precondition", "postcondition")

SCOPE_PRE = "precondition"
SCOPE_POST = "postcondition"


class DslError(ValueError):
    """Model output is not a well-formed, in-grammar contract."""


class _Invalid:
    """A term with no value, e.g. a multiset difference that went negative."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<invalid>"


INVALID = _Invalid()


@dataclass(frozen=True)
class Binding:
    """Everything a compiled formula is allowed to look at."""

    before: PriceShoppingState
    after: PriceShoppingState
    config: PriceShoppingConfig


Term = Callable[[Binding], Any]
Formula = Callable[[Binding], bool]


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

_VAR_READERS: dict[str, Callable[[PriceShoppingState], Any]] = {
    "logged_in": lambda state: state.logged_in,
    "cart": lambda state: state.cart,
    "stock": lambda state: state.stock,
    "order_items": lambda state: state.order_items,
    "payment_method": lambda state: state.payment_method,
    "shipping_address": lambda state: state.shipping_address,
    "coupon_code": lambda state: state.coupon_code,
    "wallet_balance": lambda state: state.wallet_balance,
    "card_limit": lambda state: state.card_limit,
    "cart_subtotal": lambda state: state.cart_subtotal,
    "checkout_total": lambda state: state.checkout_total,
    "order_status": lambda state: state.order_status.value,
    "order_total": lambda state: state.order_total,
    "order_payment_method": lambda state: state.order_payment_method,
    "cart_size": lambda state: state.cart_size,
    "order_size": lambda state: counts_total(state.order_items),
}

# Kinds are checked while parsing so that a mistyped contract is reported as a
# parse failure with a useful message rather than silently scoring as False.
_VAR_KINDS: dict[str, str] = {
    "logged_in": "bool",
    "cart": "multiset",
    "stock": "multiset",
    "order_items": "multiset",
    "payment_method": "option",
    "shipping_address": "option",
    "coupon_code": "option",
    "wallet_balance": "number",
    "card_limit": "number",
    "cart_subtotal": "number",
    "checkout_total": "number",
    "order_status": "text",
    "order_total": "number",
    "order_payment_method": "option",
    "cart_size": "number",
    "order_size": "number",
}

STATE_VARS: tuple[str, ...] = tuple(sorted(_VAR_READERS))
VAR_KINDS: Mapping[str, str] = dict(_VAR_KINDS)

ORDER_STATUS_VALUES: tuple[str, ...] = tuple(status.value for status in OrderStatus)

CONFIG_SETS: tuple[str, ...] = (
    "addresses",
    "coupon_codes",
    "items",
    "payment_methods",
)

# Values of ``text``/``option``/``null`` kind are mutually comparable; the other
# kinds only compare with themselves.
_TEXTUAL = frozenset({"text", "option", "null"})

FORMULA_OPS: dict[str, str] = {
    "const": '{"op": "const", "value": true} — a literal truth value.',
    "and": '{"op": "and", "args": [f, ...]} — every listed formula holds.',
    "or": '{"op": "or", "args": [f, ...]} — at least one listed formula holds.',
    "not": '{"op": "not", "arg": f} — the formula does not hold.',
    "is_true": '{"op": "is_true", "arg": t} — a boolean term is true.',
    "is_null": '{"op": "is_null", "arg": t} — an optional term is unset (null).',
    "is_empty": '{"op": "is_empty", "arg": t} — an item multiset has no units.',
    "eq": '{"op": "eq", "left": t, "right": t} — two terms of the same kind are equal.',
    "ne": '{"op": "ne", "left": t, "right": t} — two terms of the same kind differ.',
    "lt": '{"op": "lt", "left": t, "right": t} — a number is strictly smaller.',
    "le": '{"op": "le", "left": t, "right": t} — a number is smaller or equal.',
    "gt": '{"op": "gt", "left": t, "right": t} — a number is strictly greater.',
    "ge": '{"op": "ge", "left": t, "right": t} — a number is greater or equal.',
    "in_set": (
        '{"op": "in_set", "value": t, "set": "<config set>"} — a text/optional '
        "term is a member of a configured set; an unset value is never a member."
    ),
    "covers": (
        '{"op": "covers", "available": t, "required": t} — the first multiset '
        "holds at least every quantity of the second."
    ),
    "unchanged": (
        '{"op": "unchanged", "vars": ["<state var>", ...]} — every listed '
        "variable has the same value before and after (postcondition only)."
    ),
}

TERM_OPS: dict[str, str] = {
    "count": '{"op": "count", "of": t, "item": "<item>"} — units of one item in a multiset.',
    "total": '{"op": "total", "of": t} — total units in a multiset.',
    "difference": (
        '{"op": "difference", "left": t, "right": t} — the first multiset minus '
        "the second; has no value if any line would go negative."
    ),
    "subtract": '{"op": "subtract", "left": t, "right": t} — numeric subtraction.',
}

TERM_LITERALS: dict[str, str] = {
    "var": (
        '{"var": "<state var>", "when": "before"|"after"} — a state variable. '
        '"when" defaults to "before" and may only be "after" in a postcondition.'
    ),
    "const": '{"const": true|1|"text"|null} — a literal boolean, integer, string, or null.',
    "multiset": '{"multiset": {"<item>": <count>, ...}} — a literal item multiset.',
}


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


class _Guard:
    """Enforces the node-count and nesting bounds while parsing."""

    def __init__(self) -> None:
        self.nodes = 0

    def enter(self, depth: int, path: str) -> None:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise DslError(f"contract has more than {MAX_NODES} nodes")
        if depth > MAX_DEPTH:
            raise DslError(f"{path} nests deeper than {MAX_DEPTH} levels")


def _object(node: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        raise DslError(f"{path}: expected a JSON object, got {type(node).__name__}")
    for key in node:
        if not isinstance(key, str):
            raise DslError(f"{path}: object keys must be strings")
    return node


def _reject_unknown_keys(node: Mapping[str, Any], path: str, allowed: Sequence[str]) -> None:
    unknown = sorted(set(node) - set(allowed))
    if unknown:
        raise DslError(f"{path}: unknown keys {unknown}; allowed keys are {sorted(allowed)}")


def _require(node: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in node:
        raise DslError(f"{path}: missing required key {key!r}")
    return node[key]


def parse_formula(node: Any, path: str, scope: str, depth: int, guard: _Guard) -> Formula:
    """Compile one boolean node into a closure over a :class:`Binding`."""
    guard.enter(depth, path)
    node = _object(node, path)
    if "op" not in node:
        raise DslError(
            f"{path}: expected a formula with an 'op' key, one of {sorted(FORMULA_OPS)}"
        )
    op = node["op"]
    if not isinstance(op, str):
        raise DslError(f"{path}: 'op' must be a string")
    parser = _FORMULA_PARSERS.get(op)
    if parser is None:
        if op in TERM_OPS:
            raise DslError(f"{path}: {op!r} produces a value, not a truth value")
        raise DslError(f"{path}: unknown formula operator {op!r}; allowed: {sorted(FORMULA_OPS)}")
    return parser(node, path, scope, depth, guard)


def parse_term(node: Any, path: str, scope: str, depth: int, guard: _Guard) -> tuple[Term, str]:
    """Compile one value node, returning the closure and its kind."""
    guard.enter(depth, path)
    node = _object(node, path)
    literal_keys = sorted(set(node) & set(TERM_LITERALS))
    if literal_keys and "op" in node:
        raise DslError(f"{path}: a term has either 'op' or one of {sorted(TERM_LITERALS)}, not both")
    if len(literal_keys) > 1:
        raise DslError(f"{path}: a term carries exactly one of {sorted(TERM_LITERALS)}")
    if literal_keys:
        return _TERM_LITERAL_PARSERS[literal_keys[0]](node, path, scope)
    if "op" not in node:
        raise DslError(
            f"{path}: expected a term with 'op' or one of {sorted(TERM_LITERALS)}"
        )
    op = node["op"]
    if not isinstance(op, str):
        raise DslError(f"{path}: 'op' must be a string")
    parser = _TERM_PARSERS.get(op)
    if parser is None:
        if op in FORMULA_OPS:
            raise DslError(f"{path}: {op!r} produces a truth value, not a value")
        raise DslError(f"{path}: unknown term operator {op!r}; allowed: {sorted(TERM_OPS)}")
    return parser(node, path, scope, depth, guard)


# -- term literals ---------------------------------------------------------


def _parse_var(node: Mapping[str, Any], path: str, scope: str) -> tuple[Term, str]:
    _reject_unknown_keys(node, path, ("var", "when"))
    name = node["var"]
    if not isinstance(name, str) or name not in _VAR_READERS:
        raise DslError(f"{path}: unknown state variable {name!r}; allowed: {list(STATE_VARS)}")
    when = node.get("when", "before")
    if when not in ("before", "after"):
        raise DslError(f"{path}: 'when' must be \"before\" or \"after\", got {when!r}")
    if when == "after" and scope == SCOPE_PRE:
        raise DslError(
            f"{path}: a precondition speaks about the state before the action, "
            "so it may not read \"after\" variables"
        )
    read = _VAR_READERS[name]
    if when == "before":
        term: Term = lambda binding: read(binding.before)
    else:
        term = lambda binding: read(binding.after)
    return term, _VAR_KINDS[name]


def _parse_const(node: Mapping[str, Any], path: str, scope: str) -> tuple[Term, str]:
    _reject_unknown_keys(node, path, ("const",))
    value = node["const"]
    if isinstance(value, bool):
        kind = "bool"
    elif value is None:
        kind = "null"
    elif isinstance(value, int):
        kind = "number"
    elif isinstance(value, str):
        kind = "text"
    else:
        raise DslError(
            f"{path}: a constant must be a boolean, integer, string, or null, "
            f"got {type(value).__name__}"
        )
    return (lambda binding: value), kind


def _parse_multiset(node: Mapping[str, Any], path: str, scope: str) -> tuple[Term, str]:
    _reject_unknown_keys(node, path, ("multiset",))
    mapping = node["multiset"]
    if not isinstance(mapping, Mapping):
        raise DslError(f"{path}: 'multiset' must be an object of item -> count")
    counts: dict[str, int] = {}
    for item, quantity in mapping.items():
        if not isinstance(item, str):
            raise DslError(f"{path}: multiset item names must be strings")
        if isinstance(quantity, bool) or not isinstance(quantity, int):
            raise DslError(f"{path}: count for {item!r} must be an integer")
        if quantity < 0:
            raise DslError(f"{path}: count for {item!r} must not be negative")
        counts[item] = quantity
    value = counts_from_mapping(counts)
    return (lambda binding: value), "multiset"


_TERM_LITERAL_PARSERS: dict[str, Callable[[Mapping[str, Any], str, str], tuple[Term, str]]] = {
    "var": _parse_var,
    "const": _parse_const,
    "multiset": _parse_multiset,
}


# -- term operators --------------------------------------------------------


def _child_term(
    node: Mapping[str, Any],
    key: str,
    path: str,
    scope: str,
    depth: int,
    guard: _Guard,
    *,
    kinds: Sequence[str] | None = None,
) -> Term:
    child_path = f"{path}.{key}"
    term, kind = parse_term(_require(node, key, path), child_path, scope, depth + 1, guard)
    if kinds is not None and kind not in kinds:
        raise DslError(f"{child_path}: expected a {' or '.join(kinds)} term, got {kind}")
    return term


def _term_count(
    node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
) -> tuple[Term, str]:
    _reject_unknown_keys(node, path, ("op", "of", "item"))
    of = _child_term(node, "of", path, scope, depth, guard, kinds=("multiset",))
    item = _require(node, "item", path)
    if not isinstance(item, str) or not item:
        raise DslError(f"{path}: 'item' must be a non-empty string")

    def term(binding: Binding) -> Any:
        counts = of(binding)
        if counts is INVALID:
            return INVALID
        return counts_get(counts, item)

    return term, "number"


def _term_total(
    node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
) -> tuple[Term, str]:
    _reject_unknown_keys(node, path, ("op", "of"))
    of = _child_term(node, "of", path, scope, depth, guard, kinds=("multiset",))

    def term(binding: Binding) -> Any:
        counts = of(binding)
        if counts is INVALID:
            return INVALID
        return counts_total(counts)

    return term, "number"


def _term_difference(
    node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
) -> tuple[Term, str]:
    _reject_unknown_keys(node, path, ("op", "left", "right"))
    left = _child_term(node, "left", path, scope, depth, guard, kinds=("multiset",))
    right = _child_term(node, "right", path, scope, depth, guard, kinds=("multiset",))

    def term(binding: Binding) -> Any:
        available = left(binding)
        required = right(binding)
        if available is INVALID or required is INVALID:
            return INVALID
        remaining = dict(available)
        for item, quantity in required:
            new_quantity = remaining.get(item, 0) - quantity
            if new_quantity < 0:
                # No value rather than an exception: a contract that subtracts
                # more than is there is simply wrong about this state.
                return INVALID
            remaining[item] = new_quantity
        return counts_from_mapping(remaining)

    return term, "multiset"


def _term_subtract(
    node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
) -> tuple[Term, str]:
    _reject_unknown_keys(node, path, ("op", "left", "right"))
    left = _child_term(node, "left", path, scope, depth, guard, kinds=("number",))
    right = _child_term(node, "right", path, scope, depth, guard, kinds=("number",))

    def term(binding: Binding) -> Any:
        left_value = left(binding)
        right_value = right(binding)
        if left_value is INVALID or right_value is INVALID:
            return INVALID
        return left_value - right_value

    return term, "number"


_TERM_PARSERS: dict[
    str, Callable[[Mapping[str, Any], str, str, int, _Guard], tuple[Term, str]]
] = {
    "count": _term_count,
    "total": _term_total,
    "difference": _term_difference,
    "subtract": _term_subtract,
}


# -- formula operators -----------------------------------------------------


def _formula_const(
    node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
) -> Formula:
    _reject_unknown_keys(node, path, ("op", "value"))
    value = _require(node, "value", path)
    if not isinstance(value, bool):
        raise DslError(f"{path}: 'value' must be true or false")
    return lambda binding: value


def _formula_args(
    node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
) -> tuple[Formula, ...]:
    _reject_unknown_keys(node, path, ("op", "args"))
    args = _require(node, "args", path)
    if not isinstance(args, list) or not args:
        raise DslError(f"{path}: 'args' must be a non-empty list of formulas")
    return tuple(
        parse_formula(arg, f"{path}.args[{index}]", scope, depth + 1, guard)
        for index, arg in enumerate(args)
    )


def _formula_and(
    node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
) -> Formula:
    args = _formula_args(node, path, scope, depth, guard)
    return lambda binding: all(arg(binding) for arg in args)


def _formula_or(
    node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
) -> Formula:
    args = _formula_args(node, path, scope, depth, guard)
    return lambda binding: any(arg(binding) for arg in args)


def _formula_not(
    node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
) -> Formula:
    _reject_unknown_keys(node, path, ("op", "arg"))
    inner = parse_formula(_require(node, "arg", path), f"{path}.arg", scope, depth + 1, guard)
    return lambda binding: not inner(binding)


def _formula_is_true(
    node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
) -> Formula:
    _reject_unknown_keys(node, path, ("op", "arg"))
    term = _child_term(node, "arg", path, scope, depth, guard, kinds=("bool",))
    return lambda binding: term(binding) is True


def _formula_is_null(
    node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
) -> Formula:
    _reject_unknown_keys(node, path, ("op", "arg"))
    term = _child_term(node, "arg", path, scope, depth, guard, kinds=("option", "text", "null"))
    return lambda binding: term(binding) is None


def _formula_is_empty(
    node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
) -> Formula:
    _reject_unknown_keys(node, path, ("op", "arg"))
    term = _child_term(node, "arg", path, scope, depth, guard, kinds=("multiset",))

    def formula(binding: Binding) -> bool:
        counts = term(binding)
        return counts is not INVALID and counts_total(counts) == 0

    return formula


def _comparable(left_kind: str, right_kind: str) -> bool:
    if left_kind in _TEXTUAL and right_kind in _TEXTUAL:
        return True
    return left_kind == right_kind


def _formula_equality(negated: bool) -> Callable[..., Formula]:
    def build(
        node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
    ) -> Formula:
        _reject_unknown_keys(node, path, ("op", "left", "right"))
        left_term, left_kind = parse_term(
            _require(node, "left", path), f"{path}.left", scope, depth + 1, guard
        )
        right_term, right_kind = parse_term(
            _require(node, "right", path), f"{path}.right", scope, depth + 1, guard
        )
        if not _comparable(left_kind, right_kind):
            raise DslError(
                f"{path}: cannot compare a {left_kind} term with a {right_kind} term"
            )

        def formula(binding: Binding) -> bool:
            left = left_term(binding)
            right = right_term(binding)
            if left is INVALID or right is INVALID:
                return False
            try:
                equal = bool(left == right)
            except TypeError:  # pragma: no cover - defensive, kinds already match
                return False
            return not equal if negated else equal

        return formula

    return build


_ORDERINGS: dict[str, Callable[[int, int], bool]] = {
    "lt": lambda left, right: left < right,
    "le": lambda left, right: left <= right,
    "gt": lambda left, right: left > right,
    "ge": lambda left, right: left >= right,
}


def _formula_ordering(op: str) -> Callable[..., Formula]:
    compare = _ORDERINGS[op]

    def build(
        node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
    ) -> Formula:
        _reject_unknown_keys(node, path, ("op", "left", "right"))
        left_term = _child_term(node, "left", path, scope, depth, guard, kinds=("number",))
        right_term = _child_term(node, "right", path, scope, depth, guard, kinds=("number",))

        def formula(binding: Binding) -> bool:
            left = left_term(binding)
            right = right_term(binding)
            if left is INVALID or right is INVALID:
                return False
            return compare(left, right)

        return formula

    return build


def _formula_in_set(
    node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
) -> Formula:
    _reject_unknown_keys(node, path, ("op", "value", "set"))
    term = _child_term(node, "value", path, scope, depth, guard, kinds=("option", "text", "null"))
    name = _require(node, "set", path)
    if not isinstance(name, str) or name not in CONFIG_SETS:
        raise DslError(f"{path}: unknown configured set {name!r}; allowed: {list(CONFIG_SETS)}")

    def formula(binding: Binding) -> bool:
        value = term(binding)
        if value is INVALID or value is None:
            return False
        return value in getattr(binding.config, name)

    return formula


def _formula_covers(
    node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
) -> Formula:
    _reject_unknown_keys(node, path, ("op", "available", "required"))
    available_term = _child_term(node, "available", path, scope, depth, guard, kinds=("multiset",))
    required_term = _child_term(node, "required", path, scope, depth, guard, kinds=("multiset",))

    def formula(binding: Binding) -> bool:
        available = available_term(binding)
        required = required_term(binding)
        if available is INVALID or required is INVALID:
            return False
        return counts_covers(available, required)

    return formula


def _formula_unchanged(
    node: Mapping[str, Any], path: str, scope: str, depth: int, guard: _Guard
) -> Formula:
    _reject_unknown_keys(node, path, ("op", "vars"))
    if scope == SCOPE_PRE:
        raise DslError(f"{path}: 'unchanged' compares before with after, so it needs a postcondition")
    names = _require(node, "vars", path)
    if not isinstance(names, list) or not names:
        raise DslError(f"{path}: 'vars' must be a non-empty list of state variables")
    readers = []
    for index, name in enumerate(names):
        if not isinstance(name, str) or name not in _VAR_READERS:
            raise DslError(
                f"{path}.vars[{index}]: unknown state variable {name!r}; "
                f"allowed: {list(STATE_VARS)}"
            )
        readers.append(_VAR_READERS[name])
    frozen = tuple(readers)
    return lambda binding: all(read(binding.before) == read(binding.after) for read in frozen)


_FORMULA_PARSERS: dict[str, Callable[[Mapping[str, Any], str, str, int, _Guard], Formula]] = {
    "const": _formula_const,
    "and": _formula_and,
    "or": _formula_or,
    "not": _formula_not,
    "is_true": _formula_is_true,
    "is_null": _formula_is_null,
    "is_empty": _formula_is_empty,
    "eq": _formula_equality(negated=False),
    "ne": _formula_equality(negated=True),
    "lt": _formula_ordering("lt"),
    "le": _formula_ordering("le"),
    "gt": _formula_ordering("gt"),
    "ge": _formula_ordering("ge"),
    "in_set": _formula_in_set,
    "covers": _formula_covers,
    "unchanged": _formula_unchanged,
}


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedContract:
    """A contract that survived parsing, not yet bound to a configuration."""

    spec: Mapping[str, Any]
    precondition: Formula
    postcondition: Formula
    node_count: int
    name: str
    notes: str = ""

    @property
    def canonical_json(self) -> str:
        """Stable rendering, used when showing a contract back to the model."""
        return json.dumps(self.spec, ensure_ascii=False, indent=2, sort_keys=True)

    def bind(self, config: PriceShoppingConfig, name: str | None = None) -> Contract:
        """Produce the evaluator-facing :class:`Contract` for one sandbox."""
        precondition = self.precondition
        postcondition = self.postcondition
        return Contract(
            name=name or self.name,
            action=PriceShoppingAction.place_order(),
            precondition=lambda state: precondition(Binding(state, state, config)),
            postcondition=lambda before, after: postcondition(Binding(before, after, config)),
            description=self.notes or "Model-proposed contract in the JSON DSL.",
        )


def parse_contract(spec: Any, *, name: str = "model.place_order") -> ParsedContract:
    """Compile a decoded JSON contract, raising :class:`DslError` if unsafe."""
    spec = _object(spec, "contract")
    _reject_unknown_keys(spec, "contract", CONTRACT_KEYS)
    skill = spec.get("skill", SKILL)
    if skill != SKILL:
        raise DslError(
            f"contract: unsupported skill {skill!r}; only {SKILL!r} has a reference specification"
        )
    missing = [key for key in REQUIRED_CONTRACT_KEYS if key not in spec]
    if missing:
        raise DslError(f"contract: missing required keys {missing}")
    notes = spec.get("notes", "")
    if not isinstance(notes, str):
        raise DslError("contract: 'notes' must be a string")
    declared = spec.get("name", name)
    if not isinstance(declared, str) or not declared:
        raise DslError("contract: 'name' must be a non-empty string")

    guard = _Guard()
    precondition = parse_formula(spec["precondition"], SCOPE_PRE, SCOPE_PRE, 0, guard)
    postcondition = parse_formula(spec["postcondition"], SCOPE_POST, SCOPE_POST, 0, guard)
    return ParsedContract(
        spec=json.loads(json.dumps(spec)),  # a plain, owned copy of the data
        precondition=precondition,
        postcondition=postcondition,
        node_count=guard.nodes,
        name=name,
        notes=notes,
    )


def extract_json_object(text: str) -> str:
    """Return the first balanced ``{...}`` block in ``text``.

    Models wrap JSON in prose or code fences; this pulls the object out without
    interpreting anything around it. String literals and escapes are tracked so
    that a brace inside a string does not close the object.
    """
    if not isinstance(text, str):
        raise DslError("response content was not text")
    if len(text) > MAX_SOURCE_CHARS:
        raise DslError(f"response is longer than {MAX_SOURCE_CHARS} characters")
    start = text.find("{")
    if start < 0:
        raise DslError("response contains no JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise DslError("response contains an unterminated JSON object")


def parse_contract_text(text: str, *, name: str = "model.place_order") -> ParsedContract:
    """Extract, decode, and compile a contract from raw model output."""
    source = extract_json_object(text)
    try:
        decoded = json.loads(source)
    except json.JSONDecodeError as exc:
        raise DslError(f"response is not valid JSON: {exc}") from exc
    return parse_contract(decoded, name=name)


def contract_json_schema() -> dict[str, Any]:
    """Return a JSON Schema dict matching the contract DSL grammar."""
    def operator(
        op: str,
        properties: Mapping[str, Any],
        required: Sequence[str],
    ) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"op": {"const": op}, **properties},
            "required": ["op", *required],
            "additionalProperties": False,
        }

    term_ref = {"$ref": "#/$defs/term"}
    formula_ref = {"$ref": "#/$defs/formula"}
    binary_term_ops = [
        operator(op, {"left": term_ref, "right": term_ref}, ("left", "right"))
        for op in ("difference", "subtract")
    ]
    equality_ops = [
        operator(op, {"left": term_ref, "right": term_ref}, ("left", "right"))
        for op in ("eq", "ne", "lt", "le", "gt", "ge")
    ]
    return {
        "type": "object",
        "properties": {
            "skill": {"type": "string"},
            "name": {"type": "string"},
            "notes": {"type": "string"},
            "precondition": {"$ref": "#/$defs/formula"},
            "postcondition": {"$ref": "#/$defs/formula"},
        },
        "required": ["precondition", "postcondition"],
        "additionalProperties": False,
        "$defs": {
            "formula": {
                "anyOf": [
                    operator("const", {"value": {"type": "boolean"}}, ("value",)),
                    operator(
                        "and",
                        {"args": {"type": "array", "items": formula_ref}},
                        ("args",),
                    ),
                    operator(
                        "or",
                        {"args": {"type": "array", "items": formula_ref}},
                        ("args",),
                    ),
                    operator("not", {"arg": formula_ref}, ("arg",)),
                    *[
                        operator(op, {"arg": term_ref}, ("arg",))
                        for op in ("is_true", "is_null", "is_empty")
                    ],
                    *equality_ops,
                    operator(
                        "in_set",
                        {
                            "value": term_ref,
                            "set": {"type": "string", "enum": list(CONFIG_SETS)},
                        },
                        ("value", "set"),
                    ),
                    operator(
                        "covers",
                        {"available": term_ref, "required": term_ref},
                        ("available", "required"),
                    ),
                    operator(
                        "unchanged",
                        {
                            "vars": {
                                "type": "array",
                                "items": {"type": "string", "enum": list(STATE_VARS)},
                            }
                        },
                        ("vars",),
                    ),
                ]
            },
            "term": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "var": {"type": "string", "enum": list(STATE_VARS)},
                            "when": {"type": "string", "enum": ["before", "after"]},
                        },
                        "required": ["var"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "const": {
                                "anyOf": [
                                    {"type": "boolean"},
                                    {"type": "integer"},
                                    {"type": "string"},
                                    {"type": "null"},
                                ]
                            }
                        },
                        "required": ["const"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "multiset": {
                                "type": "object",
                                "additionalProperties": {"type": "integer"},
                            }
                        },
                        "required": ["multiset"],
                        "additionalProperties": False,
                    },
                    operator(
                        "count",
                        {"of": term_ref, "item": {"type": "string"}},
                        ("of", "item"),
                    ),
                    operator("total", {"of": term_ref}, ("of",)),
                    *binary_term_ops,
                ]
            },
        }
    }
