#!/usr/bin/env python3
"""A safe, declarative JSON DSL for candidate ``deploy_service`` contracts.

Model output is data, never code. Parsed into small closures without eval/exec.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import DeploymentConfig
from .contracts import Contract
from .state import (
    DeploymentAction,
    DeploymentState,
    DeploymentStatus,
    counts_covers,
    counts_from_mapping,
    counts_get,
    counts_total,
)

SKILL = "deploy_service"

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
    __slots__ = ()

    def __repr__(self) -> str:
        return "<invalid>"


INVALID = _Invalid()


@dataclass(frozen=True)
class Binding:
    before: DeploymentState
    after: DeploymentState
    config: DeploymentConfig


Term = Callable[[Binding], Any]
Formula = Callable[[Binding], bool]


_VAR_READERS: dict[str, Callable[[DeploymentState], Any]] = {
    "authenticated": lambda state: state.authenticated,
    "allocated_resources": lambda state: state.allocated_resources,
    "available_quota": lambda state: state.available_quota,
    "active_deployment": lambda state: state.active_deployment,
    "target_region": lambda state: state.target_region,
    "cluster_tier": lambda state: state.cluster_tier,
    "deployment_status": lambda state: state.deployment_status.value,
    "allocation_size": lambda state: state.allocated_total,
    "quota_size": lambda state: counts_total(state.available_quota),
    "active_size": lambda state: counts_total(state.active_deployment),
}

_VAR_KINDS: dict[str, str] = {
    "authenticated": "bool",
    "allocated_resources": "multiset",
    "available_quota": "multiset",
    "active_deployment": "multiset",
    "target_region": "option",
    "cluster_tier": "option",
    "deployment_status": "text",
    "allocation_size": "number",
    "quota_size": "number",
    "active_size": "number",
}

STATE_VARS: tuple[str, ...] = tuple(sorted(_VAR_READERS))
VAR_KINDS: Mapping[str, str] = dict(_VAR_KINDS)

DEPLOYMENT_STATUS_VALUES: tuple[str, ...] = tuple(status.value for status in DeploymentStatus)

CONFIG_SETS: tuple[str, ...] = (
    "region_options",
    "resource_types",
    "tier_options",
    "rejected_regions",
    "rejected_cluster_tiers",
    "valid_regions",
    "valid_cluster_tiers",
)

_TEXTUAL = frozenset({"text", "option", "null"})

FORMULA_OPS: dict[str, str] = {
    "const": '{"op": "const", "value": true} — a literal truth value.',
    "and": '{"op": "and", "args": [f, ...]} — every listed formula holds.',
    "or": '{"op": "or", "args": [f, ...]} — at least one listed formula holds.',
    "not": '{"op": "not", "arg": f} — the formula does not hold.',
    "is_true": '{"op": "is_true", "arg": t} — a boolean term is true.',
    "is_null": '{"op": "is_null", "arg": t} — an optional term is unset (null).',
    "is_empty": '{"op": "is_empty", "arg": t} — a resource multiset has no units.',
    "eq": '{"op": "eq", "left": t, "right": t} — two terms of the same kind are equal.',
    "ne": '{"op": "ne", "left": t, "right": t} — two terms of the same kind differ.',
    "lt": '{"op": "lt", "left": t, "right": t} — a number is strictly smaller.',
    "le": '{"op": "le", "left": t, "right": t} — a number is smaller or equal.',
    "gt": '{"op": "gt", "left": t, "right": t} — a number is strictly greater.',
    "ge": '{"op": "ge", "left": t, "right": t} — a number is greater or equal.',
    "in_set": '{"op": "in_set", "value": t, "set": "<config set>"} — member of set.',
    "covers": '{"op": "covers", "available": t, "required": t} — multiset covers.',
    "unchanged": '{"op": "unchanged", "vars": ["..."]} — postcondition variables unchanged.',
}

TERM_OPS: dict[str, str] = {
    "var": '{"var": "<name>", "when": "before"|"after"} — read a state variable.',
    "const": '{"const": ...} — literal number, string, null, or multiset.',
    "difference": '{"op": "difference", "left": t, "right": t} — multiset difference.',
    "union": '{"op": "union", "left": t, "right": t} — multiset sum.',
    "get": '{"op": "get", "multiset": t, "resource": "<resource>"} — resource count.',
    "total": '{"op": "total", "multiset": t} — total units in a multiset.',
}

# Served models frequently emit ``add`` where the grammar spells the operator
# ``union``. The two are semantically identical for multiset operands, so the
# alias is accepted there and only there; every other use of ``add`` (notably
# numeric addition, which this grammar does not have) stays a parse error.
UNION_COMPAT_ALIAS = "add"

NUMERIC_ADD_MESSAGE = (
    "'add' is accepted only as an alias for 'union' over two multiset terms, "
    "got {left} and {right}. This grammar has no numeric arithmetic: compare "
    "allocation_size, quota_size, active_size, or a 'total'/'get' of a multiset "
    "directly with eq/ne/lt/le/gt/ge instead."
)

NUMERIC_MULTISET_MESSAGE = (
    "{op} requires a multiset term, got {kind}; allocation_size, quota_size and "
    "active_size are already numbers, so use them directly instead of wrapping "
    "them in '{op}'."
)


@dataclass(frozen=True)
class _CompiledTerm:
    kind: str
    func: Term


@dataclass(frozen=True)
class ParsedContract:
    skill: str
    name: str
    notes: str
    precondition: Formula
    postcondition: Formula
    raw_json: str
    canonical_json: str
    spec: dict[str, Any]
    node_count: int
    compat_rewrites: tuple[str, ...] = ()

    def bind(self, config: DeploymentConfig) -> Contract:
        dummy_state = DeploymentState(
            authenticated=False,
            allocated_resources=(),
            available_quota=(),
            target_region=None,
            cluster_tier=None,
            deployment_status=DeploymentStatus.IDLE,
            active_deployment=(),
        )

        def pre(state: DeploymentState) -> bool:
            return self.precondition(Binding(before=state, after=dummy_state, config=config))

        def post(before: DeploymentState, after: DeploymentState) -> bool:
            return self.postcondition(Binding(before=before, after=after, config=config))

        return Contract(
            name=self.name,
            action=DeploymentAction.deploy_service(),
            precondition=pre,
            postcondition=post,
            description=self.notes or f"DSL contract for {self.skill}",
        )


class _Parser:
    def __init__(self, spec: Mapping[str, Any], name: str = "model.deploy_service") -> None:
        self.spec = dict(spec)
        self.name = name
        self.nodes = 0
        self.compat_rewrites: list[str] = []

    def parse(self) -> ParsedContract:
        for key in REQUIRED_CONTRACT_KEYS:
            if key not in self.spec:
                raise DslError(f"contract is missing required key {key!r}")
        unknown = sorted(set(self.spec) - set(CONTRACT_KEYS))
        if unknown:
            raise DslError(f"unknown top-level contract keys: {unknown}")

        skill = self.spec.get("skill", SKILL)
        if skill != SKILL:
            raise DslError(f"expected skill {SKILL!r}, got {skill!r}")

        pre_fn = self._parse_formula(self.spec["precondition"], scope=SCOPE_PRE, depth=0)
        post_fn = self._parse_formula(self.spec["postcondition"], scope=SCOPE_POST, depth=0)

        canonical = json.dumps(
            {
                "skill": SKILL,
                "notes": str(self.spec.get("notes", "")),
                "precondition": self.spec["precondition"],
                "postcondition": self.spec["postcondition"],
            },
            sort_keys=True,
        )

        return ParsedContract(
            skill=SKILL,
            name=self.name,
            notes=str(self.spec.get("notes", "")),
            precondition=pre_fn,
            postcondition=post_fn,
            raw_json=json.dumps(self.spec),
            canonical_json=canonical,
            spec=dict(self.spec),
            node_count=self.nodes,
            compat_rewrites=tuple(self.compat_rewrites),
        )

    def _tick(self, depth: int) -> None:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise DslError(f"contract exceeds the node cap of {MAX_NODES}")
        if depth > MAX_DEPTH:
            raise DslError(f"contract exceeds the nesting cap of {MAX_DEPTH}")

    def _parse_formula(self, node: Any, scope: str, depth: int) -> Formula:
        self._tick(depth)
        if not isinstance(node, Mapping):
            raise DslError(f"expected formula object, got {type(node).__name__}")
        op = node.get("op")
        if not isinstance(op, str):
            raise DslError(f"formula is missing string op: {node!r}")

        if op == "const":
            val = node.get("value")
            if not isinstance(val, bool):
                raise DslError(f"formula const requires boolean value, got {val!r}")
            return (lambda b, v=val: v)

        if op == "and":
            args = self._parse_formula_list(node.get("args"), scope, depth + 1, "and")
            return (lambda b, funcs=args: all(f(b) for f in funcs))

        if op == "or":
            args = self._parse_formula_list(node.get("args"), scope, depth + 1, "or")
            return (lambda b, funcs=args: any(f(b) for f in funcs))

        if op == "not":
            arg = self._parse_formula(node.get("arg"), scope, depth + 1)
            return (lambda b, f=arg: not f(b))

        if op == "is_true":
            t = self._parse_term(node.get("arg"), scope, depth + 1)
            if t.kind != "bool":
                raise DslError(f"is_true expects a boolean term, got {t.kind!r}")
            return (lambda b, fn=t.func: fn(b) is True)

        if op == "is_null":
            t = self._parse_term(node.get("arg"), scope, depth + 1)
            if t.kind not in _TEXTUAL:
                raise DslError(f"is_null expects an option/text term, got {t.kind!r}")
            return (lambda b, fn=t.func: fn(b) is None or fn(b) is INVALID)

        if op == "is_empty":
            t = self._parse_term(node.get("arg"), scope, depth + 1)
            if t.kind != "multiset":
                raise DslError(f"is_empty expects a multiset term, got {t.kind!r}")
            return (lambda b, fn=t.func: counts_total(fn(b)) == 0 if fn(b) is not INVALID else False)

        if op in ("eq", "ne"):
            left = self._parse_term(node.get("left"), scope, depth + 1)
            right = self._parse_term(node.get("right"), scope, depth + 1)
            if not _kinds_compatible(left.kind, right.kind):
                raise DslError(f"{op} cannot compare {left.kind} with {right.kind}")
            is_eq = (op == "eq")
            return (
                lambda b, lf=left.func, rf=right.func, eq=is_eq:
                _compare_eq(lf(b), rf(b), eq)
            )

        if op in ("lt", "le", "gt", "ge"):
            left = self._parse_term(node.get("left"), scope, depth + 1)
            right = self._parse_term(node.get("right"), scope, depth + 1)
            if left.kind != "number" or right.kind != "number":
                raise DslError(f"{op} requires number terms, got {left.kind} and {right.kind}")
            return (
                lambda b, lf=left.func, rf=right.func, o=op:
                _compare_num(lf(b), rf(b), o)
            )

        if op == "in_set":
            val = self._parse_term(node.get("value"), scope, depth + 1)
            if val.kind not in _TEXTUAL:
                raise DslError(f"in_set value must be text/option, got {val.kind!r}")
            set_name = node.get("set")
            if set_name not in CONFIG_SETS:
                raise DslError(f"unknown config set {set_name!r}; known: {CONFIG_SETS}")
            return (
                lambda b, vf=val.func, sn=set_name:
                _check_in_set(vf(b), getattr(b.config, sn))
            )

        if op == "covers":
            avail = self._parse_term(node.get("available"), scope, depth + 1)
            req = self._parse_term(node.get("required"), scope, depth + 1)
            if avail.kind != "multiset" or req.kind != "multiset":
                raise DslError(f"covers requires multiset terms, got {avail.kind} and {req.kind}")
            return (
                lambda b, af=avail.func, rf=req.func:
                _covers(af(b), rf(b))
            )

        if op == "unchanged":
            if scope != SCOPE_POST:
                raise DslError("unchanged is only valid in postconditions")
            vars_list = node.get("vars")
            if not isinstance(vars_list, Sequence) or isinstance(vars_list, str) or not vars_list:
                raise DslError("unchanged requires a non-empty list of variable names")
            for var_name in vars_list:
                if var_name not in STATE_VARS:
                    raise DslError(f"unknown state variable {var_name!r}")
            readers = [_VAR_READERS[var_name] for var_name in vars_list]
            return (lambda b, rds=readers: all(r(b.before) == r(b.after) for r in rds))

        raise DslError(f"unknown formula op {op!r}")

    def _parse_formula_list(self, args: Any, scope: str, depth: int, parent_op: str) -> list[Formula]:
        if not isinstance(args, Sequence) or isinstance(args, str):
            raise DslError(f"{parent_op} args must be a list")
        if not args:
            raise DslError(f"{parent_op} args cannot be empty")
        return [self._parse_formula(arg, scope, depth) for arg in args]

    def _parse_term(self, node: Any, scope: str, depth: int) -> _CompiledTerm:
        self._tick(depth)
        if not isinstance(node, Mapping):
            raise DslError(f"expected term object, got {type(node).__name__}")

        if "var" in node:
            var_name = node["var"]
            if var_name not in STATE_VARS:
                raise DslError(f"unknown state variable {var_name!r}")
            when = node.get("when", "before")
            if when not in ("before", "after"):
                raise DslError(f"when must be before or after, got {when!r}")
            if when == "after" and scope == SCOPE_PRE:
                raise DslError("cannot reference after state in a precondition")
            reader = _VAR_READERS[var_name]
            kind = _VAR_KINDS[var_name]
            if when == "before":
                return _CompiledTerm(kind=kind, func=lambda b, r=reader: r(b.before))
            return _CompiledTerm(kind=kind, func=lambda b, r=reader: r(b.after))

        if "const" in node:
            val = node["const"]
            if isinstance(val, bool):
                return _CompiledTerm(kind="bool", func=lambda b, v=val: v)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return _CompiledTerm(kind="number", func=lambda b, v=int(val): v)
            if isinstance(val, str):
                return _CompiledTerm(kind="text", func=lambda b, v=val: v)
            if val is None:
                return _CompiledTerm(kind="null", func=lambda b: None)
            if isinstance(val, Mapping):
                try:
                    multiset = counts_from_mapping(val)
                except Exception as exc:
                    raise DslError(f"invalid multiset const: {exc}") from exc
                return _CompiledTerm(kind="multiset", func=lambda b, m=multiset: m)
            raise DslError(f"unsupported const type {type(val).__name__}")

        op = node.get("op")
        if not isinstance(op, str):
            raise DslError(f"term object is missing var, const, or string op: {node!r}")

        if op in ("difference", "union", UNION_COMPAT_ALIAS):
            left = self._parse_term(node.get("left"), scope, depth + 1)
            right = self._parse_term(node.get("right"), scope, depth + 1)
            if left.kind != "multiset" or right.kind != "multiset":
                if op == UNION_COMPAT_ALIAS:
                    raise DslError(NUMERIC_ADD_MESSAGE.format(left=left.kind, right=right.kind))
                raise DslError(f"{op} requires multiset terms, got {left.kind} and {right.kind}")
            sign = -1 if op == "difference" else 1
            if op == UNION_COMPAT_ALIAS:
                self.compat_rewrites.append("add->union")
            return _CompiledTerm(
                kind="multiset",
                func=lambda b, lf=left.func, rf=right.func, s=sign: _combine(lf(b), rf(b), s),
            )

        if op == "get":
            ms = self._parse_term(node.get("multiset"), scope, depth + 1)
            if ms.kind != "multiset":
                raise DslError(NUMERIC_MULTISET_MESSAGE.format(op="get", kind=ms.kind))
            res = node.get("resource")
            if not isinstance(res, str):
                raise DslError(f"get requires string resource, got {res!r}")
            return _CompiledTerm(
                kind="number",
                func=lambda b, mf=ms.func, r=res: counts_get(mf(b), r) if mf(b) is not INVALID else 0,
            )

        if op == "total":
            ms = self._parse_term(node.get("multiset"), scope, depth + 1)
            if ms.kind != "multiset":
                raise DslError(NUMERIC_MULTISET_MESSAGE.format(op="total", kind=ms.kind))
            return _CompiledTerm(
                kind="number",
                func=lambda b, mf=ms.func: counts_total(mf(b)) if mf(b) is not INVALID else 0,
            )

        raise DslError(f"unknown term op {op!r}")


def _kinds_compatible(k1: str, k2: str) -> bool:
    if k1 == k2:
        return True
    return k1 in _TEXTUAL and k2 in _TEXTUAL


def _compare_eq(left: Any, right: Any, is_eq: bool) -> bool:
    if left is INVALID or right is INVALID:
        return False
    return (left == right) if is_eq else (left != right)


def _compare_num(left: Any, right: Any, op: str) -> bool:
    if left is INVALID or right is INVALID or not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    if op == "lt":
        return left < right
    if op == "le":
        return left <= right
    if op == "gt":
        return left > right
    if op == "ge":
        return left >= right
    return False


def _check_in_set(val: Any, collection: Iterable[str]) -> bool:
    if val is INVALID or val is None:
        return False
    return str(val) in collection


def _covers(available: Any, required: Any) -> bool:
    if available is INVALID or required is INVALID:
        return False
    return counts_covers(available, required)


def _combine(left: Any, right: Any, sign: int) -> Any:
    if left is INVALID or right is INVALID:
        return INVALID
    try:
        from .state import counts_combine
        return counts_combine(left, right, sign=sign)
    except Exception:
        return INVALID


def parse_contract_dict(spec: Mapping[str, Any], name: str = "model.deploy_service") -> ParsedContract:
    return _Parser(spec, name=name).parse()


def extract_json_object(text: str) -> str:
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
    raise DslError(
        "response contains an unterminated JSON object; the answer was most likely "
        "cut off by the token limit, so produce a shorter contract with fewer clauses"
    )


def parse_contract_text(text: str, name: str = "model.deploy_service") -> ParsedContract:
    source = extract_json_object(text)
    try:
        spec = json.loads(source)
    except Exception as exc:
        raise DslError(f"JSON syntax error: {exc}") from exc
    if not isinstance(spec, Mapping):
        raise DslError("JSON root must be an object")
    return parse_contract_dict(spec, name=name)


def contract_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "enum": [SKILL]},
            "name": {"type": "string"},
            "notes": {"type": "string"},
            "precondition": {"type": "object"},
            "postcondition": {"type": "object"},
        },
        "required": ["precondition", "postcondition"],
        "additionalProperties": False,
    }
