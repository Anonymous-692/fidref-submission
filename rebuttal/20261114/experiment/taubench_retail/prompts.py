#!/usr/bin/env python3
"""Prompt templates for synthesizing exchange_delivered_order_items contracts."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .config import RetailConfig
from .contracts import ContractReport, Symptom
from .dsl import (
    CONFIG_SETS,
    FORMULA_OPS,
    TERM_OPS,
    VOCABULARY_LEGACY,
    VOCABULARY_ORDER_NEUTRAL,
    VOCABULARY_SEMANTIC_FIELDS,
    state_var_kinds,
)

# A complete, in-grammar contract used purely to show the JSON envelope.
# Keep it semantically neutral so the model does not copy syntax examples as answers.
EXAMPLE_CONTRACT: dict[str, Any] = {
    "skill": "exchange_delivered_order_items",
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
    "items with items, bool with bool. Wrap literal values in term objects like "
    '{"const": "#W1"} or {"const": 0}.',
    "`in_set` accepts either an option/text term (single element membership) or an items "
    "list term (asserts all items belong to the set, vacuously true if empty).",
    "Output ONE JSON object and nothing else: no markdown fences, no comments, no explanation "
    "before or after. Keep the contract compact.",
)

POSTCONDITION_PATTERNS: tuple[str, ...] = (
    "- Asserting order status change upon exchange:\n"
    '    {"op": "eq", "left": {"var": "order_w1_status", "when": "after"}, "right": {"const": "exchange requested"}}',
    "- Clearing draft fields in after-state:\n"
    '    {"op": "is_null", "arg": {"var": "order_id", "when": "after"}}',
    '    {"op": "is_empty", "arg": {"var": "item_ids", "when": "after"}}',
    '    {"op": "is_empty", "arg": {"var": "new_item_ids", "when": "after"}}',
    '    {"op": "is_null", "arg": {"var": "payment_method_id", "when": "after"}}',
    "- Frame conditions specifying unchanged variables:\n"
    '    {"op": "unchanged", "vars": ["order_w2_status", "order_w2_cancel_reason", "gift_card_balance"]}',
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


def system_prompt(vocabulary_protocol: str = VOCABULARY_LEGACY) -> str:
    lines = [
        "You are an expert formal verification engineer specializing in online retail service contracts.",
        "Your task is to write declarative contracts in JSON for the `exchange_delivered_order_items` operation.",
        "You must output valid JSON matching the specified grammar, without surrounding prose.",
        "",
        "State variables:",
    ]
    kinds = state_var_kinds(vocabulary_protocol)
    for var_name in sorted(kinds):
        lines.append(f"- `{var_name}` ({kinds[var_name]})")
    lines.append("")
    lines.append("Configuration sets:")
    for set_name in CONFIG_SETS:
        lines.append(f"- `{set_name}`")
    lines.append("")
    lines.append("Formula operators:")
    for op, desc in FORMULA_OPS.items():
        if vocabulary_protocol == VOCABULARY_SEMANTIC_FIELDS and op == "exchange_snapshot_matches":
            desc += " In this W1 vocabulary, also checks preservation of return items, return payment method and authentication."
        if vocabulary_protocol == VOCABULARY_ORDER_NEUTRAL and op == "exchange_snapshot_matches":
            desc = ('{"op": "exchange_snapshot_matches"} — checks the observable exchange '
                    'snapshot for the order selected by the BEFORE draft, including draft reset '
                    'and preserved unrelated fields. Only state fields exposed by this sandbox '
                    'are checked; unrepresented exchange payloads are not certified.')
        lines.append(f"- `{op}`: {desc}")
    lines.append("")
    lines.append("Term operators:")
    for op, desc in TERM_OPS.items():
        lines.append(f"- `{op}`: {desc}")
    lines.append("")
    lines.append("Rules:")
    for rule in OPERATOR_RULES:
        if vocabulary_protocol == VOCABULARY_ORDER_NEUTRAL:
            rule = rule.replace('{"const": "#W1"}', '{"const": "literal_text"}')
        lines.append(f"- {rule}")
    return "\n".join(lines)


def direct_prompt(config: RetailConfig, vocabulary_protocol: str = VOCABULARY_LEGACY) -> str:
    lines = [
        "Synthesize a declarative contract for `exchange_delivered_order_items` in the retail domain.",
        "",
        "Task configuration:",
        f"- Valid orders: {list(config.valid_orders)}",
        f"- Rejected orders: {list(config.rejected_orders)}",
        f"- Valid items: {list(config.valid_items)}",
        f"- Rejected items: {list(config.rejected_items)}",
        f"- Valid new items: {list(config.valid_new_items)}",
        f"- Rejected new items: {list(config.rejected_new_items)}",
        f"- Valid payment methods: {list(config.valid_payment_methods)}",
        f"- Rejected payment methods: {list(config.rejected_payment_methods)}",
        f"- Max items: {config.max_items}",
        f"- Max new items: {config.max_new_items}",
        "",
        "The contract must specify:",
        "1. `precondition`: when `exchange_delivered_order_items` is allowed to execute.",
        "2. `postcondition`: the relationship between the before-state and after-state upon execution.",
        "",
        "Typical postcondition patterns:",
    ]
    if vocabulary_protocol == VOCABULARY_ORDER_NEUTRAL:
        # Syntax only: no target-order status or non-target-order frame examples.
        lines.append('Use before/after terms to describe observed changes and `unchanged` '
                     'for observed frame conditions. No order-specific effect is assumed.')
    else:
        lines.extend(POSTCONDITION_PATTERNS)
    lines.append("")
    lines.append("Example JSON schema envelope:")
    lines.append(example_contract_json())
    lines.append("")
    lines.append("Output ONLY the JSON object:")
    return "\n".join(lines)


def counterexample_prompt(
    report: ContractReport,
    config: RetailConfig,
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
        lines.append(f"- Before: {json.dumps(payload['before_state'], indent=2, ensure_ascii=False)}")
        lines.append(
            "- After: "
            + (
                json.dumps(payload['after_state'], indent=2, ensure_ascii=False)
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
    config: RetailConfig,
    counterexample_limit: int = 3,
) -> str:
    """Self-contained refinement prompt for compact-context retail runs."""
    lines = [
        "Refine the following valid contract candidate for `exchange_delivered_order_items`.",
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
        lines.append(f"- Before: {json.dumps(payload['before_state'], indent=2, ensure_ascii=False)}")
        lines.append(
            "- After: "
            + (
                json.dumps(payload['after_state'], indent=2, ensure_ascii=False)
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


def parse_repair_prompt(raw_text: str, error: str, config: RetailConfig) -> str:
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


def parse_repair_prompt_v3(raw_text: str, error: str, config: RetailConfig) -> str:
    """Self-contained syntax repair prompt with enough room for a full candidate."""
    lines = [
        "Your previous response could not be parsed into a valid `exchange_delivered_order_items` contract.",
        f"Parser error: {error}",
        "",
        "Previous response:",
        bounded_echo(raw_text, V3_PARSE_REPAIR_ECHO_CHARS),
        "",
        "Repair only the JSON or DSL error while preserving the intended contract semantics.",
        "Output ONLY the complete valid JSON object:",
    ]
    return "\n".join(lines)


def vacuity_revision_prompt(config: RetailConfig) -> str:
    lines = [
        "Your previous contract had a vacuous precondition that rejected all reachable states.",
        "A valid contract must accept reachable states that meet all domain requirements.",
        "Synthesize a non-vacuous contract for `exchange_delivered_order_items`.",
        "",
        "Output ONLY the revised JSON object:",
    ]
    return "\n".join(lines)


def vacuity_revision_prompt_v3(previous_contract: str, config: RetailConfig) -> str:
    """Self-contained vacuity repair prompt for compact-context retail runs."""
    lines = [
        "The following valid contract has a vacuous precondition that rejects every reachable state:",
        "",
        previous_contract,
        "",
        "Revise its precondition so that legitimate `exchange_delivered_order_items` states are admitted.",
        "Preserve the existing postcondition unless the revision logically requires a change.",
        "Output ONLY the complete revised JSON object:",
    ]
    return "\n".join(lines)


def self_refine_prompt_v3(previous_contract: str) -> str:
    """Self-contained self-refinement prompt for compact-context retail runs."""
    return (
        "Critique and refine the following valid `exchange_delivered_order_items` contract. "
        "Preserve correct clauses and output ONLY the complete revised JSON object:\n\n"
        f"{previous_contract}"
    )
