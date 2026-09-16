#!/usr/bin/env python3
"""Prompt templates for synthesizing ``deploy_service`` contracts."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .config import DeploymentConfig
from .dsl import (
    CONFIG_SETS,
    DEPLOYMENT_STATUS_VALUES,
    FORMULA_OPS,
    STATE_VARS,
    TERM_OPS,
    VAR_KINDS,
)

# A complete, in-grammar contract used purely to show the JSON envelope.  Keep
# it semantically neutral: realistic-looking thresholds and update clauses
# caused the model to copy syntax examples into candidate specifications.
EXAMPLE_CONTRACT: dict[str, Any] = {
    "skill": "deploy_service",
    "notes": "JSON envelope demonstration only - not a candidate contract",
    "precondition": {"op": "const", "value": True},
    "postcondition": {"op": "const", "value": True},
}

OPERATOR_RULES: tuple[str, ...] = (
    "Use ONLY the formula and term operators listed above. Any other operator name "
    "is a parse error and the whole answer is thrown away.",
    "There is NO numeric arithmetic in this grammar. `add`, `sub`, `plus`, `minus`, "
    "`sum`, `mul` and `div` over numbers are forbidden. `add` is tolerated only as a "
    "spelling of `union` when BOTH operands are multisets; prefer `union` and never "
    "write `add` over numbers.",
    "`total` and `get` take a multiset. `allocation_size`, `quota_size` and "
    "`active_size` are ALREADY numbers, so never wrap them in `total` or `get` — "
    "compare them directly with eq/ne/lt/le/gt/ge.",
    "Derived variable totals (`allocation_size`, `quota_size`, `active_size`) are numbers. "
    "Do NOT write arithmetic on them. Express state transitions directly on the underlying multisets "
    "(`allocated_resources`, `available_quota`, `active_deployment`) using `difference` or `union`.",
    "`union` and `difference` take two multisets and return a multiset; they may not "
    "be applied to numbers, strings or booleans.",
    "`unchanged` is valid only inside the postcondition, and only over the state "
    "variable names listed above.",
    "`when: \"after\"` is valid only inside the postcondition; the precondition may "
    "only read `when: \"before\"`.",
    "Comparisons must join two terms of the same kind: number with number, multiset "
    "with multiset, text/option with text/option. Wrap literal values in term objects "
    'like `{"const": "deployed"}` or `{"const": 0}`, never raw scalar values.',
    "Output ONE JSON object and nothing else: no markdown fences, no comments, no "
    "explanation before or after. Keep the contract compact so it is never truncated.",
)

POSTCONDITION_PATTERNS: tuple[str, ...] = (
    "- State assignment: compare a variable's after-state to a constant:\n"
    '    {"op": "eq", "left": {"var": "deployment_status", "when": "after"}, "right": {"const": "deployed"}}',
    "- Multiset transfer: set an after-state multiset equal to a before-state multiset:\n"
    '    {"op": "eq", "left": {"var": "active_deployment", "when": "after"}, "right": {"var": "allocated_resources", "when": "before"}}',
    "- Multiset difference: set after-state multiset equal to difference between two before-state multisets:\n"
    '    {"op": "eq", "left": {"var": "available_quota", "when": "after"}, "right": {"op": "difference", "left": {"var": "available_quota", "when": "before"}, "right": {"var": "allocated_resources", "when": "before"}}}',
    "- Empty multiset: assert that an after-state multiset has no units:\n"
    '    {"op": "is_empty", "arg": {"var": "allocated_resources", "when": "after"}}',
    "- Frame conditions: specify which variables remain unchanged:\n"
    '    {"op": "unchanged", "vars": ["authenticated", "target_region", "cluster_tier"]}',
)


# A compact-context turn replaces the transcript with a single user message, so
# the answer that failed to parse has to travel inside the repair prompt itself.
# The echo is clipped so a runaway answer can never crowd out the grammar
# reminders or the task configuration that follow it.
PARSE_REPAIR_ECHO_CHARS = 1200


def example_contract_json() -> str:
    return json.dumps(EXAMPLE_CONTRACT, indent=2, ensure_ascii=False)


def bounded_echo(text: str, limit: int = PARSE_REPAIR_ECHO_CHARS) -> str:
    """Quote the model back to itself within a fixed character budget."""
    if not text:
        return "(the previous answer was empty)"
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...[truncated: {len(text) - limit} more characters]"


def system_prompt() -> str:
    lines = [
        "You are an expert formal verification engineer specializing in cloud deployment contracts.",
        "Your task is to write declarative contracts in JSON for the `deploy_service` operation.",
        "You must output valid JSON matching the specified grammar, without surrounding prose.",
        "",
        "State variables:",
    ]
    for var_name in STATE_VARS:
        lines.append(f"- `{var_name}` ({VAR_KINDS[var_name]})")
    lines.append("")
    lines.append("Deployment status values: " + ", ".join(repr(v) for v in DEPLOYMENT_STATUS_VALUES))
    lines.append("Configuration sets: " + ", ".join(repr(s) for s in CONFIG_SETS))
    lines.append("")
    lines.append("Formula operators:")
    for op, desc in FORMULA_OPS.items():
        lines.append(f"- `{op}`: {desc}")
    lines.append("")
    lines.append("Term operators:")
    for op, desc in TERM_OPS.items():
        lines.append(f"- `{op}`: {desc}")
    lines.append("")
    lines.append("Rules (violating any of these makes the answer unparseable):")
    for rule in OPERATOR_RULES:
        lines.append(f"- {rule}")
    lines.append("")
    lines.append("Common valid postcondition structural patterns:")
    for pattern in POSTCONDITION_PATTERNS:
        lines.append(pattern)
    lines.append("")
    lines.append(
        "The object below is a COMPLETE, VALID example of the required JSON shape. "
        "It exists only to show the syntax: it is deliberately incomplete and is NOT "
        "the correct contract for `deploy_service`. Copy its structure, not its clauses."
    )
    lines.append("")
    lines.append("```json")
    lines.append(example_contract_json())
    lines.append("```")
    return "\n".join(lines)


def direct_prompt(config: DeploymentConfig) -> str:
    return (
        "Synthesize the complete formal contract for `deploy_service` under the configuration:\n"
        f"- Resource types: {list(config.resource_types)}\n"
        f"- Valid regions: {list(config.valid_regions)}\n"
        f"- Rejected regions: {list(config.rejected_regions)}\n"
        f"- Valid cluster tiers: {list(config.valid_cluster_tiers)}\n"
        f"- Rejected cluster tiers: {list(config.rejected_cluster_tiers)}\n"
        f"- Max quota: {config.max_quota}\n"
        f"- Max allocation limit: {config.max_allocation_limit}\n\n"
        "The two maxima above only bound generated sandbox states. Do NOT copy them "
        "into the contract as allocation_size or quota_size threshold clauses unless "
        "the observations independently require such a clause.\n\n"
        "Output ONLY the JSON object with keys skill (deploy_service), precondition, and postcondition.\n"
        "Keep it compact and remember the operator rules: no numeric arithmetic, and never "
        "apply `total` or `get` to allocation_size, quota_size or active_size."
    )


def critique_prompt(previous_contract: str) -> str:
    return (
        "Critique and revise the following proposed contract for `deploy_service`:\n\n"
        f"```json\n{previous_contract}\n```\n\n"
        "Check all edge conditions (authentication, resource quota coverage, serviceable regions, cluster tiers) "
        "and effect clauses (quota decrement, active deployment update, field frame conditions).\n"
        "Output ONLY the revised JSON contract."
    )


def probe_prompt(config: DeploymentConfig, observations: Sequence[Mapping[str, Any]]) -> str:
    obs_lines = []
    for idx, obs in enumerate(observations, 1):
        status_str = "ACCEPTED" if obs.get("ok") else f"REFUSED: {obs.get('error')}"
        obs_lines.append(f"Observation {idx}:\n  Before: {obs.get('before')}\n  Result: {status_str}")
        if obs.get("ok"):
            obs_lines.append(f"  After:  {obs.get('after')}")
    obs_text = "\n".join(obs_lines)
    return (
        "Based on the following execution observations of `deploy_service`:\n\n"
        f"{obs_text}\n\n"
        "Synthesize the complete declarative JSON contract for `deploy_service`."
    )


def counterexample_prompt(
    previous_contract: str,
    metrics: Mapping[str, Any],
    counterexamples: Sequence[Mapping[str, Any]],
) -> str:
    ce_lines = []
    for idx, ce in enumerate(counterexamples, 1):
        before = ce.get("before_state", ce.get("state_fields", ce.get("state")))
        after = ce.get("after_state")
        ce_lines.append(
            f"Counterexample {idx} ({ce.get('symptom')}):\n"
            f"  Before: {json.dumps(before, ensure_ascii=False, sort_keys=True)}\n"
            f"  After: {json.dumps(after, ensure_ascii=False, sort_keys=True) if after is not None else 'null (sandbox rejected the action)'}\n"
            f"  Changed fields: {json.dumps(ce.get('changed_fields', {}), ensure_ascii=False, sort_keys=True)}\n"
            f"  Unchanged fields: {json.dumps(ce.get('unchanged_fields', []), ensure_ascii=False)}\n"
            f"  Detail: {ce.get('detail')}"
        )
    ce_text = "\n".join(ce_lines)
    return (
        "The proposed contract is inexact:\n\n"
        f"```json\n{previous_contract}\n```\n\n"
        f"Checker metrics: false accepts={metrics.get('false_accepts')}, "
        f"false rejects={metrics.get('false_rejects')}, "
        f"postcondition violations={metrics.get('postcondition_violations')}\n\n"
        f"Concrete counterexamples:\n{ce_text}\n\n"
        "Repair every clause contradicted by changed or unchanged observed fields, then output ONLY the corrected JSON contract."
    )


def specific_error_hint(error: str) -> str:
    if "no numeric arithmetic" in error or "'add' is accepted only as an alias" in error:
        return (
            "\nSpecific Hint: Do NOT use `add` or numeric arithmetic on numbers. "
            "To express resource/quota changes, use `{\"op\": \"difference\", ...}` or `{\"op\": \"union\", ...}` on multiset variables.\n"
        )
    if "expected term object, got" in error:
        return (
            "\nSpecific Hint: Wrap literal values inside term objects like `{\"const\": \"value\"}` rather than passing raw strings/numbers directly.\n"
        )
    return ""


def parse_failure_note(error: str) -> str:
    reminders = "\n".join(f"- {rule}" for rule in OPERATOR_RULES)
    hint = specific_error_hint(error)
    return (
        "Your previous response could not be parsed as a valid contract DSL object:\n"
        f"Error: {error}\n"
        f"{hint}\n"
        "Reminders:\n"
        f"{reminders}\n\n"
        "Please fix the syntax/grammar and output ONLY a well-formed JSON contract object."
    )


def parse_repair_prompt(
    config: DeploymentConfig,
    *,
    error: str,
    previous_output: str,
    echo_limit: int = PARSE_REPAIR_ECHO_CHARS,
) -> str:
    """A self-contained repair turn: the rejected answer, the error, and the task.

    Used when the runner sends only ``[system, user]``. Without the echo the model
    is asked to fix an answer it can no longer see, which is how a compact-context
    run ends up re-emitting the same unparseable text.
    """
    reminders = "\n".join(f"- {rule}" for rule in OPERATOR_RULES)
    hint = specific_error_hint(error)
    return (
        "Your previous response could not be parsed as a valid contract DSL object.\n"
        "It is quoted below exactly as you produced it, so that you can repair it. "
        "Do not send it back unchanged.\n\n"
        "Previous response:\n"
        f"```\n{bounded_echo(previous_output, echo_limit)}\n```\n\n"
        f"Parse error: {error}\n"
        f"{hint}\n"
        "Reminders:\n"
        f"{reminders}\n\n"
        f"{direct_prompt(config)}"
    )


def vacuity_revision_prompt(
    config: DeploymentConfig,
    *,
    previous_contract: str,
    reachable_states: int,
) -> str:
    """Ask for a revision of a precondition that admits nothing.

    Everything reported here is derived from the candidate alone: how many of the
    enumerated sandbox states its own precondition admits. No execution outcome,
    no reference-contract label and no closure score is disclosed.
    """
    return (
        "Your previous contract is vacuous. Its precondition admits none of the "
        f"{reachable_states} reachable sandbox states, so `deploy_service` would be "
        "forbidden everywhere and the contract says nothing about the operation. "
        "This is a property of the contract text you wrote; no execution results and "
        "no scores are being disclosed to you.\n\n"
        f"```json\n{previous_contract}\n```\n\n"
        "Revise it so the precondition admits exactly the states in which "
        "`deploy_service` is legitimately permitted, and keep the postcondition "
        "consistent with that precondition.\n\n"
        f"{direct_prompt(config)}"
    )
