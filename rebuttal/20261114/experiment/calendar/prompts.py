#!/usr/bin/env python3
"""Prompt templates for synthesizing ``schedule_meeting`` contracts."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .config import CalendarConfig
from .contracts import ContractReport, Symptom
from .dsl import (
    CONFIG_SETS,
    FORMULA_OPS,
    STATE_VARS,
    TERM_OPS,
    VAR_KINDS,
)

# A complete, in-grammar contract used purely to show the JSON envelope.
# Keep it semantically neutral so the model does not copy syntax examples as answers.
EXAMPLE_CONTRACT: dict[str, Any] = {
    "skill": "schedule_meeting",
    "notes": "Demonstration schema envelope only - not a candidate contract",
    "precondition": {"op": "const", "value": True},
    "postcondition": {"op": "const", "value": True},
}

OPERATOR_RULES: tuple[str, ...] = (
    "Use ONLY the formula and term operators listed above. Any other operator name "
    "is a parse error and the whole answer is rejected.",
    '`when: "after"` is valid only inside the postcondition; the precondition may '
    'only read `when: "before"`.',
    "`unchanged` is valid only inside the postcondition, and only over known state variables.",
    "Comparisons must join two terms of the same kind: number with number, option with option, "
    "attendees with attendees, bool with bool. Wrap literal values in term objects like "
    '{"const": "morning"} or {"const": 0}.',
    "Output ONE JSON object and nothing else: no markdown fences, no comments, no explanation "
    "before or after. Keep the contract compact.",
)

POSTCONDITION_PATTERNS: tuple[str, ...] = (
    "- Snapshot assignment: assert that the scheduled active meeting matches the draft parameters:\n"
    '    {"op": "scheduled_snapshot_matches", "active_room": {"var": "active_room", "when": "after"}, "active_slot": {"var": "active_slot", "when": "after"}, "active_type": {"var": "active_type", "when": "after"}, "active_attendees": {"var": "active_attendees", "when": "after"}, "draft_room": {"var": "draft_room", "when": "before"}, "draft_slot": {"var": "draft_slot", "when": "before"}, "draft_type": {"var": "draft_type", "when": "before"}, "draft_attendees": {"var": "draft_attendees", "when": "before"}}',
    "- Clearing draft fields: assert that draft parameters are cleared in after-state:\n"
    '    {"op": "is_empty", "arg": {"var": "draft_attendees", "when": "after"}}',
    '    {"op": "is_null", "arg": {"var": "draft_slot", "when": "after"}}',
    '    {"op": "is_null", "arg": {"var": "draft_room", "when": "after"}}',
    '    {"op": "is_null", "arg": {"var": "draft_type", "when": "after"}}',
    "- Frame conditions: specify which variables remain unchanged:\n"
    '    {"op": "unchanged", "vars": ["authenticated", "external_booking"]}',
)

PARSE_REPAIR_ECHO_CHARS = 1200
V3_PARSE_REPAIR_ECHO_CHARS = 6000


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
        "You are an expert formal verification engineer specializing in calendar workspace scheduling contracts.",
        "Your task is to write declarative contracts in JSON for the `schedule_meeting` operation.",
        "You must output valid JSON matching the specified grammar, without surrounding prose.",
        "",
        "State variables:",
    ]
    for var_name in STATE_VARS:
        lines.append(f"- `{var_name}` ({VAR_KINDS[var_name]})")
    lines.append("")
    lines.append("Configuration sets:")
    for set_name in CONFIG_SETS:
        lines.append(f"- `{set_name}`")
    lines.append("")
    lines.append("Formula operators:")
    for op, desc in FORMULA_OPS.items():
        lines.append(f"- `{op}`: {desc}")
    lines.append("")
    lines.append("Term operators:")
    for op, desc in TERM_OPS.items():
        lines.append(f"- `{op}`: {desc}")
    lines.append("")
    lines.append("Rules:")
    for rule in OPERATOR_RULES:
        lines.append(f"- {rule}")
    return "\n".join(lines)


def direct_prompt(config: CalendarConfig) -> str:
    lines = [
        "Synthesize a declarative contract for `schedule_meeting` in the calendar workspace domain.",
        "",
        "Task configuration:",
        f"- Valid slots: {list(config.valid_slots)}",
        f"- Rejected slots: {list(config.rejected_slots)}",
        f"- Valid rooms: {list(config.valid_rooms)}",
        f"- Rejected rooms: {list(config.rejected_rooms)}",
        f"- Valid types: {list(config.valid_types)}",
        f"- Rejected types: {list(config.rejected_types)}",
        f"- Registered attendees: {list(config.attendees)}",
        f"- Max attendees: {config.max_attendees}",
        "",
        "The contract must specify:",
        "1. `precondition`: when `schedule_meeting` is allowed to execute.",
        "2. `postcondition`: the relationship between the before-state and after-state upon execution.",
        "",
        "Typical postcondition patterns:",
    ]
    lines.extend(POSTCONDITION_PATTERNS)
    lines.append("")
    lines.append("Example JSON schema envelope:")
    lines.append(example_contract_json())
    lines.append("")
    lines.append("Output ONLY the JSON object:")
    return "\n".join(lines)


def counterexample_prompt(
    report: ContractReport,
    config: CalendarConfig,
    counterexample_limit: int = 3,
) -> str:
    lines = [
        f"Your previous contract candidate had {report.failures} defects out of {report.states_checked} evaluated states.",
        f"Primary defect category: {report.primary_defect.value}",
        "",
        "Counterexample states where your contract disagreed with the sandbox:",
    ]
    cx_shown = list(report.counterexamples[:counterexample_limit])
    for idx, cx in enumerate(cx_shown, start=1):
        payload = cx.to_dict()
        lines.append(f"Counterexample {idx} ({cx.symptom.value}):")
        lines.append(f"- Detail: {cx.detail}")
        lines.append(f"- Before: {json.dumps(cx.state.to_dict(), indent=2, ensure_ascii=False)}")
        lines.append(
            "- After: "
            + (
                json.dumps(cx.after_state.to_dict(), indent=2, ensure_ascii=False)
                if cx.after_state is not None
                else "null (sandbox rejected the action)"
            )
        )
        lines.append(f"- Changed fields: {json.dumps(payload['changed_fields'], indent=2, ensure_ascii=False)}")
        lines.append(f"- Unchanged fields: {json.dumps(payload['unchanged_fields'], ensure_ascii=False)}")
        lines.append("")

    lines.append("Refine every clause contradicted by the observed changed or unchanged fields.")
    lines.append("Output ONLY the revised JSON object:")
    return "\n".join(lines)


def counterexample_prompt_v3(
    previous_contract: str,
    report: ContractReport,
    config: CalendarConfig,
    counterexample_limit: int = 3,
) -> str:
    """Self-contained refinement prompt for compact-context Calendar runs."""
    lines = [
        "Refine the following valid contract candidate for `schedule_meeting`.",
        "Preserve every correct clause and change only what the checker evidence requires.",
        "",
        "Previous valid contract:",
        previous_contract,
        "",
        "Checker metrics:",
        f"- False accepts: {report.false_accepts}",
        f"- False rejects: {report.false_rejects}",
        f"- Postcondition violations: {report.postcondition_violations}",
        f"- States checked: {report.states_checked}",
        "",
        "Counterexample states where the contract disagreed with the sandbox:",
    ]
    cx_shown = list(report.counterexamples[:counterexample_limit])
    for idx, cx in enumerate(cx_shown, start=1):
        payload = cx.to_dict()
        lines.append(f"Counterexample {idx} ({cx.symptom.value}):")
        lines.append(f"- Detail: {cx.detail}")
        lines.append(f"- Before: {json.dumps(cx.state.to_dict(), indent=2, ensure_ascii=False)}")
        lines.append(
            "- After: "
            + (
                json.dumps(cx.after_state.to_dict(), indent=2, ensure_ascii=False)
                if cx.after_state is not None
                else "null (sandbox rejected the action)"
            )
        )
        lines.append(f"- Changed fields: {json.dumps(payload['changed_fields'], indent=2, ensure_ascii=False)}")
        lines.append(f"- Unchanged fields: {json.dumps(payload['unchanged_fields'], ensure_ascii=False)}")
        lines.append("")

    lines.append("Revise every effect clause contradicted by changed or unchanged observed fields.")
    lines.append("Output ONLY the complete revised JSON object:")
    return "\n".join(lines)


def parse_repair_prompt(raw_text: str, error: str, config: CalendarConfig) -> str:
    lines = [
        "Your previous response could not be parsed into a valid contract.",
        f"Parser error: {error}",
        "",
        "Your previous response was:",
        bounded_echo(raw_text),
        "",
        "Fix the formatting and operator errors. Output ONLY a valid JSON object matching the contract grammar:",
    ]
    return "\n".join(lines)


def parse_repair_prompt_v3(raw_text: str, error: str, config: CalendarConfig) -> str:
    """Self-contained syntax repair prompt with enough room for a full candidate."""
    lines = [
        "Your previous response could not be parsed into a valid `schedule_meeting` contract.",
        f"Parser error: {error}",
        "",
        "Previous response:",
        bounded_echo(raw_text, V3_PARSE_REPAIR_ECHO_CHARS),
        "",
        "Repair only the JSON or DSL error while preserving the intended contract semantics.",
        "Output ONLY the complete valid JSON object:",
    ]
    return "\n".join(lines)


def vacuity_revision_prompt(config: CalendarConfig) -> str:
    lines = [
        "Your previous contract had a vacuous precondition that rejected all reachable states.",
        "A valid contract must accept reachable states that meet all domain requirements.",
        "Synthesize a non-vacuous contract for `schedule_meeting`.",
        "",
        "Output ONLY the revised JSON object:",
    ]
    return "\n".join(lines)


def vacuity_revision_prompt_v3(previous_contract: str, config: CalendarConfig) -> str:
    """Self-contained vacuity repair prompt for compact-context Calendar runs."""
    lines = [
        "The following valid contract has a vacuous precondition that rejects every reachable state:",
        "",
        previous_contract,
        "",
        "Revise its precondition so that legitimate `schedule_meeting` states are admitted.",
        "Preserve the existing postcondition unless the revision logically requires a change.",
        "Output ONLY the complete revised JSON object:",
    ]
    return "\n".join(lines)


def self_refine_prompt_v3(previous_contract: str) -> str:
    """Self-contained self-refinement prompt for compact-context Calendar runs."""
    return (
        "Critique and refine the following valid `schedule_meeting` contract. "
        "Preserve correct clauses and output ONLY the complete revised JSON object:\n\n"
        f"{previous_contract}"
    )
