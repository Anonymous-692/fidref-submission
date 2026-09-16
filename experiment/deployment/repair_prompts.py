#!/usr/bin/env python3
"""Prompts for the contract-repair comparison (conditions A and B).

Both conditions see the same evidence in the same wording: the same fixed pool, the same
observed FA/FR/PV counts, and the same counterexamples. They differ only in what the model is
asked to return -- a whole contract (A) or a restricted patch (B). That difference in response
format is part of the treatment and is disclosed as such.

Nothing here references the reference contract.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .patch import OPERATIONS, RETARGET_FIELDS, SCOPES


def _format_counterexamples(counterexamples: Sequence[Mapping[str, Any]]) -> str:
    """Same rendering the domain already uses in ``prompts.counterexample_prompt``."""
    lines = []
    for idx, ce in enumerate(counterexamples, 1):
        before = ce.get("before_state", ce.get("state_fields", ce.get("state")))
        after = ce.get("after_state")
        lines.append(
            f"Counterexample {idx} ({ce.get('symptom')}):\n"
            f"  Before: {json.dumps(before, ensure_ascii=False, sort_keys=True)}\n"
            f"  After: {json.dumps(after, ensure_ascii=False, sort_keys=True) if after is not None else 'null (sandbox rejected the action)'}\n"
            f"  Changed fields: {json.dumps(ce.get('changed_fields', {}), ensure_ascii=False, sort_keys=True)}\n"
            f"  Unchanged fields: {json.dumps(ce.get('unchanged_fields', []), ensure_ascii=False)}\n"
            f"  Detail: {ce.get('detail')}"
        )
    return "\n".join(lines) if lines else "(none recorded)"


def _evidence_block(metrics: Mapping[str, Any], counterexamples: Sequence[Any]) -> str:
    return (
        "Testing your contract against the fixed observed pool found:\n"
        f"  false accepts:            {metrics['false_accepts']}\n"
        f"  false rejects:            {metrics['false_rejects']}\n"
        f"  postcondition violations: {metrics['postcondition_violations']}\n"
        f"  pool states checked:      {metrics['states_checked']}\n\n"
        "Observed violations:\n" + _format_counterexamples(counterexamples)
    )


def free_rewrite_prompt(
    contract_source: str, metrics: Mapping[str, Any], counterexamples: Sequence[Any]
) -> str:
    """Condition A: rewrite the whole contract from the observed errors."""
    return "\n\n".join([
        "Here is the contract you proposed:",
        contract_source,
        _evidence_block(metrics, counterexamples),
        "Repair the contract so that these observed violations are resolved without breaking "
        "valid cases. Change as little as possible. Return the complete contract as JSON.",
    ])


def _patch_instructions(patch_cap: int, insert_node_cap: int) -> str:
    return (
        "Return a PATCH, not a contract. A patch is a JSON object "
        '{"edits": [ ... ]} with at most '
        f"{patch_cap} edits. Each edit names one operation and where it applies.\n\n"
        f'  scope is "{SCOPES[0]}" or "{SCOPES[1]}".\n'
        '  {"op": "add_conjunct", "scope": s, "formula": F}\n'
        '      append F to that scope\'s top-level conjunction.\n'
        '  {"op": "drop_conjunct", "scope": s, "index": i}\n'
        "      remove top-level conjunct i.\n"
        '  {"op": "replace_conjunct", "scope": s, "index": i, "formula": F}\n'
        "      replace top-level conjunct i with F.\n"
        '  {"op": "negate", "scope": s, "index": i}\n'
        '      wrap conjunct i in "not", or remove an existing "not".\n'
        '  {"op": "retarget", "scope": s, "path": p, "field": f, "value": v}\n'
        f"      replace one leaf field; field is one of {list(RETARGET_FIELDS)}.\n"
        '  {"op": "swap_operator", "scope": s, "path": p, "op_to": o}\n'
        "      swap a comparison operator in place (eq/ne, or lt/le/gt/ge, or and/or).\n\n"
        "A path is a list of keys and indices from the scope root, e.g. "
        '["args", 2, "left"].\n'
        f"Any formula you insert may have at most {insert_node_cap} nodes, so fix one clause at "
        "a time rather than rewriting a whole scope.\n"
        "Indices refer to the numbered conjuncts listed above."
    )


def _numbered_conjuncts(source: str) -> str:
    """Show the model the indices it must cite. Purely a rendering of its own contract."""
    try:
        spec = json.loads(source)
    except Exception:  # noqa: BLE001
        return source
    out = []
    for scope in SCOPES:
        node = spec.get(scope)
        args = node["args"] if isinstance(node, dict) and node.get("op") == "and" else [node]
        out.append(f"{scope} conjuncts:")
        for i, a in enumerate(args):
            out.append(f"  [{i}] {json.dumps(a, sort_keys=True)}")
    return "\n".join(out)


def patch_prompt(
    contract_source: str,
    metrics: Mapping[str, Any],
    counterexamples: Sequence[Any],
    *,
    patch_cap: int,
    insert_node_cap: int,
) -> str:
    """Condition B: propose a restricted patch against the incumbent."""
    # The numbered conjuncts already carry the whole contract; printing the raw source as well
    # made this prompt about twice the size of the free-rewrite prompt, which is a difference in
    # context pressure rather than in the mechanism under test.
    return "\n\n".join([
        "Here is the current contract, one top-level conjunct per line:",
        _numbered_conjuncts(contract_source),
        _evidence_block(metrics, counterexamples),
        _patch_instructions(patch_cap, insert_node_cap),
        "Propose the smallest patch that removes observed violations without breaking valid "
        "cases. The patch is applied only if it strictly reduces the observed violation count.",
    ])


def patch_repair_prompt(error: str, *, patch_cap: int, insert_node_cap: int) -> str:
    """The patch was rejected by the validator. One retry, charged to the budget."""
    return "\n\n".join([
        f"That patch was rejected: {error}",
        _patch_instructions(patch_cap, insert_node_cap),
        "Return one corrected patch as JSON.",
    ])
