#!/usr/bin/env python3
"""Prompts for price-aware Shopping contract synthesis."""

from __future__ import annotations

import json
from typing import Any

from .config import PriceShoppingConfig
from .contracts import ContractReport
from .dsl import CONFIG_SETS, FORMULA_OPS, STATE_VARS, TERM_LITERALS, TERM_OPS, VAR_KINDS

PARSE_REPAIR_ECHO_CHARS = 6000

EXAMPLE_CONTRACT: dict[str, Any] = {
    "skill": "place_order",
    "notes": "schema envelope only; not a candidate contract",
    "precondition": {"op": "const", "value": True},
    "postcondition": {"op": "const", "value": True},
}


def _bounded(text: str, limit: int = PARSE_REPAIR_ECHO_CHARS) -> str:
    if not text:
        return "(empty response)"
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def system_prompt() -> str:
    lines = [
        "You are a formal verification engineer.",
        "Write one declarative JSON contract for the `place_order` operation.",
        "Output JSON only, without markdown or explanatory prose.",
        "All monetary values are integer cents.",
        "",
        "State variables:",
    ]
    lines.extend(f"- `{name}` ({VAR_KINDS[name]})" for name in STATE_VARS)
    lines.append("")
    lines.append("Public configuration sets:")
    lines.extend(f"- `{name}`" for name in CONFIG_SETS)
    lines.append("")
    lines.append("Term literals:")
    lines.extend(f"- {description}" for description in TERM_LITERALS.values())
    lines.append("")
    lines.append("Term operators:")
    lines.extend(f"- `{name}`: {description}" for name, description in TERM_OPS.items())
    lines.append("")
    lines.append("Formula operators:")
    lines.extend(f"- `{name}`: {description}" for name, description in FORMULA_OPS.items())
    lines.extend(
        [
            "",
            "Rules:",
            "- The precondition may read only before-state variables.",
            "- The postcondition may compare before and after variables.",
            "- `unchanged` is valid only in the postcondition.",
            "- Use only the listed operators and state variables.",
            "- Return exactly one complete JSON object.",
        ]
    )
    return "\n".join(lines)


def direct_prompt(config: PriceShoppingConfig) -> str:
    public_prices = {item: config.item_price(item) for item in config.items}
    lines = [
        "Synthesize the exact applicability and effect contract for `place_order`.",
        "The storefront is deterministic, but not every policy boundary is stated explicitly.",
        "Counterexamples, when supplied later, are observations from the storefront.",
        "",
        "Agent-visible configuration:",
        f"- Items and unit prices: {public_prices}",
        f"- Cart capacity: {config.cart_capacity}",
        f"- Selectable payment methods: {list(config.payment_methods)}",
        f"- Selectable addresses: {list(config.addresses)}",
        f"- Selectable coupon codes: {list(config.coupon_codes)}",
        f"- Initial wallet balance: {config.initial_wallet_balance}",
        f"- Card limit: {config.card_limit}",
        "",
        "The state exposes `cart_subtotal` and `checkout_total` as storefront-computed values.",
        "Selecting a payment method, address, or coupon does not prove that checkout will accept it.",
        "A successful order records payment, populates the order slot, and may update checkout-related state.",
        "Infer the exact conditions and frame/effect relations.",
        "",
        "Neutral JSON envelope:",
        json.dumps(EXAMPLE_CONTRACT, indent=2),
        "",
        "Output only the complete JSON contract:",
    ]
    return "\n".join(lines)


def counterexample_prompt(
    report: ContractReport,
    config: PriceShoppingConfig,
    counterexample_limit: int = 3,
) -> str:
    lines = [
        "Revise the previous contract using the checker evidence below.",
        f"False accepts: {report.false_accepts}",
        f"False rejects: {report.false_rejects}",
        f"Postcondition violations: {report.postcondition_violations}",
        f"States checked: {report.states_checked}",
        "",
    ]
    for index, counterexample in enumerate(report.counterexamples[:counterexample_limit], 1):
        payload = counterexample.to_dict()
        lines.extend(
            [
                f"Counterexample {index} ({counterexample.symptom.value}):",
                f"- Detail: {counterexample.detail}",
                f"- Before: {json.dumps(counterexample.state.to_dict(), ensure_ascii=False, sort_keys=True)}",
                "- After: "
                + (
                    json.dumps(counterexample.after_state.to_dict(), ensure_ascii=False, sort_keys=True)
                    if counterexample.after_state is not None
                    else "null (sandbox rejected the action)"
                ),
                "- Changed fields: " + json.dumps(payload["changed_fields"], ensure_ascii=False, sort_keys=True),
                "- Unchanged fields: " + json.dumps(payload["unchanged_fields"], ensure_ascii=False),
                "",
            ]
        )
    lines.append("Revise every clause contradicted by the observed changed or unchanged fields.")
    lines.append("Output only the complete revised JSON contract:")
    return "\n".join(lines)


def counterexample_prompt_v3(
    previous_contract: str,
    report: ContractReport,
    config: PriceShoppingConfig,
    counterexample_limit: int = 3,
) -> str:
    return "\n".join(
        [
            "Refine this valid `place_order` contract using the observed transition fields.",
            "",
            "Previous contract:",
            previous_contract,
            "",
            counterexample_prompt(report, config, counterexample_limit),
        ]
    )


def parse_repair_prompt(raw_text: str, error: str, config: PriceShoppingConfig) -> str:
    return "\n".join(
        [
            "The previous response was not a valid contract.",
            f"Parser error: {error}",
            "Previous response:",
            _bounded(raw_text),
            "Repair only the JSON or DSL error and output the complete JSON contract:",
        ]
    )


def parse_repair_prompt_v3(raw_text: str, error: str, config: PriceShoppingConfig) -> str:
    return parse_repair_prompt(raw_text, error, config)


def vacuity_revision_prompt(config: PriceShoppingConfig) -> str:
    return (
        "The previous precondition rejected every reachable state. "
        "Revise it to admit legitimate orders and output only the complete JSON contract:"
    )


def vacuity_revision_prompt_v3(previous_contract: str, config: PriceShoppingConfig) -> str:
    return "\n\n".join(
        [
            "This contract rejects every reachable state:",
            previous_contract,
            "Revise its precondition while preserving correct effect clauses. Output only the complete JSON contract:",
        ]
    )


def self_refine_prompt_v3(previous_contract: str) -> str:
    return "\n\n".join(
        [
            "Critique and refine this `place_order` contract without new observations:",
            previous_contract,
            "Output only the complete revised JSON contract:",
        ]
    )
