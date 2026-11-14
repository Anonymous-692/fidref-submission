#!/usr/bin/env python3
"""Restricted patch algebra for the contract-repair control condition (B).

A patch is a short list of local edits against a parsed contract's JSON tree. The point of
the restriction is that the model cannot rewrite the contract: it must name a location and a
small edit. Two independent limits enforce that, because either alone is evadable:

  * ``patch_cap``      how many primitive edits one patch may contain;
  * ``insert_node_cap`` how large any inserted or replacing formula may be, so a single
                        ``replace_conjunct`` cannot smuggle in a whole rewritten scope.

Nothing here is tuned to the reference contract. ``replace_conjunct`` applies to any legal
top-level conjunct index, and an inserted formula may be anything the DSL type checker
accepts. The caps come from measured reference-clause sizes (see PLAN.txt 2.1), not from
knowledge of which clause is the right one.

This module performs no scoring and never sees the reference contract.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import dsl

SCOPES = (dsl.SCOPE_PRE, dsl.SCOPE_POST)

ADD_CONJUNCT = "add_conjunct"
DROP_CONJUNCT = "drop_conjunct"
REPLACE_CONJUNCT = "replace_conjunct"
RETARGET = "retarget"
SWAP_OPERATOR = "swap_operator"
NEGATE = "negate"

OPERATIONS = (ADD_CONJUNCT, DROP_CONJUNCT, REPLACE_CONJUNCT, RETARGET, SWAP_OPERATOR, NEGATE)

# Operators that may be swapped for one another in place: same arity, same operand kinds.
SWAP_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"eq", "ne"}),
    frozenset({"lt", "le", "gt", "ge"}),
    frozenset({"and", "or"}),
)

# Leaf fields ``retarget`` may rewrite. Anything else is a structural change and needs an
# explicit conjunct-level operation instead.
# ``vars`` is included so that an over-strong frame condition (an ``unchanged`` list naming a
# variable that must change) can be corrected precisely instead of only deleted. The DSL parser
# validates the names, and the list is bounded by the state vocabulary.
RETARGET_FIELDS = ("var", "when", "const", "set", "resource", "vars")

DEFAULT_PATCH_CAP = 3
DEFAULT_INSERT_NODE_CAP = 7


class PatchError(ValueError):
    """A patch that the restriction rejects. Never raised for a merely unhelpful patch."""


@dataclass
class AppliedPatch:
    """Result of applying one patch: the new source plus what it actually cost."""

    source: str
    spec: dict[str, Any]
    edits: int
    operations: tuple[str, ...]
    inserted_nodes: int
    delta: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def count_nodes(formula: Any) -> int:
    """Nodes in a formula or term subtree, counting operator and leaf nodes once each.

    Keys that carry no subtree (``op``, ``set``, ``vars``, ``when``, ``resource``) are not
    counted separately; they belong to the node that holds them.
    """
    if isinstance(formula, dict):
        skip = ("op", "set", "vars", "when", "resource")
        return 1 + sum(count_nodes(v) for k, v in formula.items() if k not in skip)
    if isinstance(formula, list):
        return sum(count_nodes(x) for x in formula)
    return 0


def _conjuncts(spec: dict[str, Any], scope: str) -> list[Any]:
    """Top-level conjunct list of one scope. A non-``and`` scope is a single conjunct."""
    node = spec.get(scope)
    if isinstance(node, dict) and node.get("op") == "and" and isinstance(node.get("args"), list):
        return node["args"]
    return [node]


def _set_conjuncts(spec: dict[str, Any], scope: str, args: list[Any]) -> None:
    if len(args) == 1:
        spec[scope] = args[0]
    else:
        spec[scope] = {"op": "and", "args": args}


def _walk(node: Any, path: Sequence[Any]) -> Any:
    """Follow a path of dict keys and list indices. Raises PatchError on a bad path."""
    cur = node
    for step in path:
        # Guided decoding emits every path step as a string; a digit string addresses a list.
        if isinstance(step, str) and step.lstrip("-").isdigit() and isinstance(cur, list):
            step = int(step)
        try:
            cur = cur[step]
        except (KeyError, IndexError, TypeError) as exc:
            raise PatchError(f"path {list(path)!r} does not resolve: {exc}") from exc
    return cur


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise PatchError(message)


def _check_scope(edit: dict[str, Any]) -> str:
    scope = edit.get("scope")
    _require(scope in SCOPES, f"scope must be one of {SCOPES}, got {scope!r}")
    return scope


def _check_insert(formula: Any, cap: int) -> int:
    _require(isinstance(formula, dict), "inserted formula must be a JSON object")
    size = count_nodes(formula)
    _require(size <= cap, f"inserted formula has {size} nodes, cap is {cap}")
    return size


def apply_patch(
    source: str,
    patch: Any,
    *,
    patch_cap: int = DEFAULT_PATCH_CAP,
    insert_node_cap: int = DEFAULT_INSERT_NODE_CAP,
) -> AppliedPatch:
    """Apply a patch to ``source`` and return the new contract source.

    Raises :class:`PatchError` if the patch is malformed, names an illegal operation or
    location, exceeds a cap, or produces a contract the DSL rejects. The caller charges the
    attempt to the budget either way; a rejected patch leaves the incumbent untouched.
    """
    if isinstance(patch, str):
        try:
            patch = json.loads(dsl.extract_json_object(patch))
        except Exception as exc:  # noqa: BLE001
            raise PatchError(f"patch is not JSON: {exc}") from exc
    _require(isinstance(patch, dict), "patch must be a JSON object")
    edits = patch.get("edits")
    _require(isinstance(edits, list) and edits, "patch must carry a non-empty 'edits' list")
    _require(len(edits) <= patch_cap, f"patch has {len(edits)} edits, cap is {patch_cap}")

    try:
        spec = json.loads(dsl.extract_json_object(source))
    except Exception as exc:  # noqa: BLE001
        raise PatchError(f"incumbent source is not JSON: {exc}") from exc
    spec = copy.deepcopy(spec)

    ops: list[str] = []
    inserted = 0
    delta: list[dict[str, Any]] = []

    for raw in edits:
        _require(isinstance(raw, dict), "each edit must be a JSON object")
        op = raw.get("op")
        _require(op in OPERATIONS, f"unknown patch operation {op!r}; allowed: {OPERATIONS}")
        scope = _check_scope(raw)
        args = _conjuncts(spec, scope)
        before = copy.deepcopy(args)

        if op == ADD_CONJUNCT:
            inserted += _check_insert(raw.get("formula"), insert_node_cap)
            args = args + [raw["formula"]]

        elif op == DROP_CONJUNCT:
            index = raw.get("index")
            _require(isinstance(index, int) and 0 <= index < len(args),
                     f"drop_conjunct index {index!r} out of range 0..{len(args) - 1}")
            _require(len(args) > 1, "refusing to drop the only conjunct of a scope")
            args = args[:index] + args[index + 1:]

        elif op == REPLACE_CONJUNCT:
            index = raw.get("index")
            _require(isinstance(index, int) and 0 <= index < len(args),
                     f"replace_conjunct index {index!r} out of range 0..{len(args) - 1}")
            inserted += _check_insert(raw.get("formula"), insert_node_cap)
            args = args[:index] + [raw["formula"]] + args[index + 1:]

        elif op == NEGATE:
            index = raw.get("index")
            _require(isinstance(index, int) and 0 <= index < len(args),
                     f"negate index {index!r} out of range 0..{len(args) - 1}")
            target = args[index]
            if isinstance(target, dict) and target.get("op") == "not":
                args = args[:index] + [target["arg"]] + args[index + 1:]
            else:
                args = args[:index] + [{"op": "not", "arg": target}] + args[index + 1:]

        elif op in (RETARGET, SWAP_OPERATOR):
            path = raw.get("path")
            _require(isinstance(path, list), "retarget/swap_operator need a 'path' list")
            _set_conjuncts(spec, scope, args)
            node = _walk(spec[scope], path)
            _require(isinstance(node, dict), f"path {path!r} does not reach an object node")
            if op == RETARGET:
                field_name = raw.get("field")
                _require(field_name in RETARGET_FIELDS,
                         f"retarget field must be one of {RETARGET_FIELDS}, got {field_name!r}")
                _require(field_name in node,
                         f"node at {path!r} has no field {field_name!r} to retarget")
                _require("value" in raw, "retarget needs a 'value'")
                _require(count_nodes(raw["value"]) <= insert_node_cap,
                         "retarget value exceeds the insert node cap")
                node[field_name] = raw["value"]
            else:
                new_op = raw.get("op_to")
                old_op = node.get("op")
                group = next((g for g in SWAP_GROUPS if old_op in g), None)
                _require(group is not None, f"operator {old_op!r} is not swappable")
                _require(new_op in group,
                         f"cannot swap {old_op!r} to {new_op!r}; allowed: {sorted(group)}")
                node["op"] = new_op
            args = _conjuncts(spec, scope)

        _set_conjuncts(spec, scope, args)
        ops.append(op)
        delta.append({"op": op, "scope": scope,
                      "before": json.dumps(before, sort_keys=True)[:400],
                      "after": json.dumps(_conjuncts(spec, scope), sort_keys=True)[:400]})

    new_source = json.dumps(spec, sort_keys=True)
    try:
        dsl.parse_contract_text(new_source)
    except dsl.DslError as exc:
        raise PatchError(f"patched contract does not type-check: {exc}") from exc

    return AppliedPatch(source=new_source, spec=spec, edits=len(edits),
                        operations=tuple(ops), inserted_nodes=inserted, delta=tuple(delta))


def patch_json_schema(patch_cap: int = DEFAULT_PATCH_CAP) -> dict[str, Any]:
    """Guided-decoding schema for a patch.

    Condition B asks the model for an output shape it has never been trained to produce, so
    without a schema a smaller model loses on formatting rather than on the mechanism under
    test. This constrains the envelope (operation names, scope names, field names) at decoding
    time; the semantic checks in :func:`apply_patch` still run afterwards, because a schema
    cannot express path validity, index bounds, or the node cap.

    Deliberately free of ``minItems``/``maxItems``/``minimum``. vLLM's xgrammar backend does
    not implement item limits or property bounds and silently falls back to ``outlines``, whose
    FSM compilation on this schema does not finish: a measured probe on the serving box
    returned in 3.4 s without those keywords and timed out at 150 s with them, with the GPU
    idle the whole time. The edit count and index bounds are enforced in :func:`apply_patch`,
    which is where a violation can be reported to the model anyway. ``patch_cap`` is kept in
    the signature because the prompt quotes it and the caller passes it through.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["edits"],
        "properties": {
            "rationale": {"type": "string"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["op", "scope"],
                    "properties": {
                        "op": {"type": "string", "enum": list(OPERATIONS)},
                        "scope": {"type": "string", "enum": list(SCOPES)},
                        "index": {"type": "integer"},
                        "formula": {"type": "object"},
                        # Path steps are strings even when they index a list. A union type
                        # ("string" or "integer") is rejected outright by guided decoding, so
                        # digit strings are coerced in ``_walk`` instead.
                        "path": {"type": "array", "items": {"type": "string"}},
                        "field": {"type": "string", "enum": list(RETARGET_FIELDS)},
                        "value": {},
                        "op_to": {"type": "string"},
                    },
                },
            },
        },
    }
