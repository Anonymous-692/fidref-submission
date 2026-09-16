#!/usr/bin/env python3
"""Prompt construction for the four contract-proposal methods.

Every prompt is built from the agent-facing view of the sandbox: the state
variables, the configured catalog, and — for the probing methods — observations
the caller actually collected by attempting the action. Nothing here imports the
reference specification, so no prompt can leak the answer to the model.

The DSL reference embedded in the system prompt is generated from the operator
tables in :mod:`.dsl`, so the grammar the model is taught cannot drift away from
the grammar the parser accepts.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ..environment import EnvConfig, ShoppingState
from . import dsl

SKILL = dsl.SKILL

_EXAMPLE_CONTRACT: dict[str, Any] = {
    "skill": SKILL,
    "notes": "Example only — the clauses below are illustrative, not correct.",
    "precondition": {
        "op": "and",
        "args": [
            {"op": "is_true", "arg": {"var": "logged_in"}},
            {"op": "not", "arg": {"op": "is_empty", "arg": {"var": "cart"}}},
            {"op": "in_set", "value": {"var": "payment_method"}, "set": "valid_payment_methods"},
        ],
    },
    "postcondition": {
        "op": "and",
        "args": [
            {
                "op": "eq",
                "left": {"var": "order_status", "when": "after"},
                "right": {"const": "placed"},
            },
            {"op": "is_empty", "arg": {"var": "cart", "when": "after"}},
            {"op": "unchanged", "vars": ["logged_in"]},
        ],
    },
}


def dsl_reference() -> str:
    """The grammar, rendered from the parser's own operator tables."""
    lines = [
        "A contract is a JSON object with these keys:",
        '  "skill"          the skill being specified; must be "%s".' % SKILL,
        '  "precondition"   a formula over the state BEFORE the action.',
        '  "postcondition"  a formula over the states BEFORE and AFTER the action.',
        '  "name", "notes"  optional free text (ignored by the scorer).',
        "",
        "Terms (values):",
    ]
    lines.extend(f"  {text}" for text in dsl.TERM_LITERALS.values())
    lines.extend(f"  {text}" for text in dsl.TERM_OPS.values())
    lines.extend(["", "Formulas (truth values):"])
    lines.extend(f"  {text}" for text in dsl.FORMULA_OPS.values())
    lines.extend(
        [
            "",
            "State variables and their kinds:",
            "  "
            + ", ".join(f"{name} ({dsl.VAR_KINDS[name]})" for name in dsl.STATE_VARS),
            f"  order_status is one of: {', '.join(dsl.ORDER_STATUS_VALUES)}",
            "",
            "Configured sets usable with in_set:",
            "  " + ", ".join(dsl.CONFIG_SETS),
            "",
            "Rules:",
            "  - Output the JSON object only. No prose, no code fences, no Python.",
            "  - Use only the operators listed above; anything else is rejected.",
            '  - A precondition may not read "after" variables, and only a '
            "postcondition may use unchanged.",
            f"  - At most {dsl.MAX_NODES} nodes and {dsl.MAX_DEPTH} levels of nesting.",
            "",
            "Shape of a well-formed answer:",
            json.dumps(_EXAMPLE_CONTRACT, ensure_ascii=False, indent=2),
        ]
    )
    return "\n".join(lines)


def system_prompt() -> str:
    return (
        "You specify the behaviour of one skill in a small storefront simulator.\n"
        "You answer with a single JSON object in the contract DSL described below,\n"
        "and with nothing else.\n\n" + dsl_reference()
    )


def sandbox_briefing(config: EnvConfig) -> str:
    """Describe the world the way an agent driving it can see it."""
    return "\n".join(
        [
            "Sandbox description (agent-facing):",
            f"  items: {list(config.items)}",
            f"  cart capacity: {config.cart_capacity} units in total",
            f"  opening stock per item: at most {config.max_stock} units",
            f"  selectable payment methods: {list(config.payment_options)}",
            f"    of these, the store accepts: {list(config.valid_payment_methods)}",
            f"  selectable addresses: {list(config.address_options)}",
            f"    of these, the store ships to: {list(config.valid_addresses)}",
            "",
            "The shopper can log in and out, add or remove cart items, clear the cart,",
            "set or clear a payment method, set or clear a shipping address, and place,",
            "confirm, cancel, or clear a single order. Selecting a payment method or an",
            "address always succeeds, even for options the store does not accept.",
            "",
            "There is one order slot per shopper, so order_status tracks that slot and",
            "order_items holds what is in it.",
        ]
    )


def _task(instruction: str) -> str:
    return (
        f"Task: write a contract for the {SKILL} action.\n"
        f"{instruction}\n"
        "Reply with the JSON contract object and nothing else."
    )


def direct_prompt(config: EnvConfig) -> str:
    """One-pass proposal with no observations and no feedback."""
    return "\n\n".join(
        [
            sandbox_briefing(config),
            _task(
                "The precondition must hold exactly where the simulator accepts "
                f"{SKILL}, and the postcondition must describe exactly how the state "
                "changes when it is accepted there."
            ),
        ]
    )


def critique_prompt(contract_source: str) -> str:
    """The single self-refinement pass: critique, then revise, no oracle."""
    return "\n\n".join(
        [
            "Here is the contract you proposed:",
            contract_source,
            (
                "Critique it silently: look for clauses that admit states the store "
                "would refuse, clauses that refuse states the store would accept, and "
                "effect claims that are too strong or too weak. You have no new "
                "observations, so rely on the sandbox description alone."
            ),
            _task("Output your revised contract."),
        ]
    )


def format_observations(observations: Sequence[Mapping[str, Any]]) -> str:
    """Render probe outcomes as a compact, deterministic transcript."""
    lines = []
    for index, observation in enumerate(observations):
        verdict = "ACCEPTED" if observation["ok"] else f"REFUSED ({observation['error']})"
        lines.append(f"[{index}] before: {observation['before']}")
        lines.append(f"     {SKILL} -> {verdict}")
        if observation["ok"]:
            lines.append(f"     after:  {observation['after']}")
    return "\n".join(lines)


def probe_prompt(
    config: EnvConfig, observations: Sequence[Mapping[str, Any]]
) -> str:
    """Proposal informed by randomly sampled attempts at the action."""
    accepted = sum(1 for observation in observations if observation["ok"])
    return "\n\n".join(
        [
            sandbox_briefing(config),
            (
                f"You probed {len(observations)} randomly chosen reachable states by "
                f"attempting {SKILL} in each. {accepted} were accepted. The transcript "
                "below is the whole of your evidence; the states were sampled at "
                "random, so they need not cover every case."
            ),
            format_observations(observations),
            _task(
                "Generalise from the transcript: the precondition should separate the "
                "accepted attempts from the refused ones, and the postcondition should "
                "describe the observed before/after changes."
            ),
        ]
    )


def format_counterexamples(counterexamples: Sequence[Mapping[str, Any]]) -> str:
    """Render evaluator counterexamples without naming the reference clauses."""
    lines = []
    for index, counterexample in enumerate(counterexamples):
        lines.append(f"[{index}] symptom: {counterexample['symptom']}")
        before = counterexample.get("before_state", counterexample.get("state_fields"))
        after = counterexample.get("after_state")
        lines.append(f"     before:  {json.dumps(before, ensure_ascii=False, sort_keys=True)}")
        lines.append(
            "     after:   "
            + (
                json.dumps(after, ensure_ascii=False, sort_keys=True)
                if after is not None
                else "null (sandbox rejected the action)"
            )
        )
        if after is not None:
            lines.append(
                "     changed_fields:   "
                + json.dumps(counterexample.get("changed_fields", {}), ensure_ascii=False, sort_keys=True)
            )
            lines.append(
                "     unchanged_fields: "
                + json.dumps(counterexample.get("unchanged_fields", []), ensure_ascii=False)
            )
        lines.append(f"     detail:  {counterexample['detail']}")
    return "\n".join(lines)


SYMPTOM_GLOSSARY = (
    "  false_accept             your precondition admits a state the store refuses.\n"
    "  false_reject             your precondition refuses a state the store accepts.\n"
    "  postcondition_violation  the action succeeded in a state you claimed, but the\n"
    "                           observed effect breaks your effect claim."
)


def counterexample_prompt(
    contract_source: str,
    metrics: Mapping[str, Any],
    counterexamples: Sequence[Mapping[str, Any]],
) -> str:
    """Refinement driven by concrete disagreements found by the checker."""
    return "\n\n".join(
        [
            "Here is the contract you proposed:",
            contract_source,
            (
                "A checker ran it against the simulator and found these disagreements:\n"
                f"  false accepts:            {metrics['false_accepts']}\n"
                f"  false rejects:            {metrics['false_rejects']}\n"
                f"  postcondition violations: {metrics['postcondition_violations']}\n"
                f"  states checked:           {metrics['states_checked']}\n\n"
                "What the symptoms mean:\n" + SYMPTOM_GLOSSARY
            ),
            "Concrete counterexamples:\n" + format_counterexamples(counterexamples),
            _task(
                "Repair the contract so that these counterexamples are resolved. Before "
                "preserving an effect clause, compare it with changed_fields and "
                "unchanged_fields; revise every clause contradicted by an observed transition."
            ),
        ]
    )


def parse_failure_note(error: str) -> str:
    """Appended to a follow-up prompt when the previous answer did not parse."""
    return (
        "Your previous answer could not be parsed as a contract:\n"
        f"  {error}\n"
        "Answer again with a single well-formed JSON object in the DSL, and nothing else."
    )


def asi_prompt(config: EnvConfig, observation: Mapping[str, Any]) -> str:
    """Proposal informed by a single successful place_order execution (Agent Skill Induction / ASI adaptation)."""
    return "\n\n".join(
        [
            sandbox_briefing(config),
            (
                f"You are provided with exactly one historical successful execution demonstration of {SKILL}. "
                "This single positive demonstration shows that the action succeeded "
                "in the observed state and produced the resulting state below."
            ),
            format_observations([observation]),
            _task(
                "Induce a general contract for the action from this single successful demonstration: "
                "the precondition must specify when the action is applicable, and the "
                "postcondition must describe all state changes and invariant frame conditions."
            ),
        ]
    )


def skillcommit_proposal_prompt(
    config: EnvConfig, observation: Mapping[str, Any]
) -> str:
    """Initial proposal prompt for SkillCommit from a single historical successful instance."""
    return "\n\n".join(
        [
            sandbox_briefing(config),
            (
                f"You are provided with one historical successful execution demonstration of {SKILL} (positive example). "
                "This demonstration shows that the action succeeded in the initial state and produced the resulting state below."
            ),
            format_observations([observation]),
            _task(
                "Propose an initial contract for the action based on this demonstration. "
                "The precondition should specify when the action is applicable, and the "
                "postcondition should describe state transitions and invariant frame conditions. "
                "Your proposed contract will subsequently undergo cross-instance validation against other distinct successful executions."
            ),
        ]
    )


def skillcommit_prompt(
    config: EnvConfig, observations: Sequence[Mapping[str, Any]] | Mapping[str, Any]
) -> str:
    """Proposal informed by historical successful execution instances (positive examples only)."""
    if isinstance(observations, Mapping):
        obs = observations
    elif observations:
        obs = observations[0]
    else:
        obs = {}
    return skillcommit_proposal_prompt(config, obs)


def skillcommit_revision_prompt(
    contract_source: str,
    metrics: Mapping[str, Any],
    incompatibilities: Sequence[Mapping[str, Any]],
) -> str:
    """Revision prompt for SkillCommit when candidate contract conflicts with distinct successful replay instances."""
    return "\n\n".join(
        [
            "Here is the contract you proposed:",
            contract_source,
            (
                "Cross-instance validation against additional distinct historical successful executions "
                "found the following behavioral incompatibilities:\n"
                f"  false rejects (rejected valid successful executions): {metrics.get('false_rejects', 0)}\n"
                f"  postcondition violations (incorrect transition claims): {metrics.get('postcondition_violations', 0)}\n"
                f"  distinct successful instances tested: {metrics.get('replay_states_checked', 0)}\n\n"
                "All tested instances are verified successful historical executions (positive examples only). "
                "No negative/refusal traces or full-closure oracle counterexamples are included."
            ),
            "Observed incompatibilities on historical successful instances:\n"
            + format_counterexamples(incompatibilities),
            _task(
                "Revise your contract to resolve these incompatibilities so that it covers all validated successful "
                "executions without making false claims about transitions. Change as little as possible."
            ),
        ]
    )


def contractskill_repair_prompt(
    contract_source: str,
    metrics: Mapping[str, Any],
    counterexamples: Sequence[Mapping[str, Any]],
) -> str:
    """Repair driven by violations observed within the budget-limited replay test set."""
    return "\n\n".join(
        [
            "Here is the contract you proposed:",
            contract_source,
            (
                "Testing against the budget-limited observed replay set found these violations:\n"
                f"  false accepts:            {metrics['false_accepts']}\n"
                f"  false rejects:            {metrics['false_rejects']}\n"
                f"  postcondition violations: {metrics['postcondition_violations']}\n"
                f"  replay states checked:    {metrics['states_checked']}\n\n"
                "What the symptoms mean:\n" + SYMPTOM_GLOSSARY
            ),
            "Observed replay violations:\n" + format_counterexamples(counterexamples),
            _task(
                "Repair the contract so that these observed replay violations are resolved without "
                "breaking valid cases. Change as little as possible."
            ),
        ]
    )


def contractskill_full_oracle_prompt(
    contract_source: str,
    metrics: Mapping[str, Any],
    counterexamples: Sequence[Mapping[str, Any]],
) -> str:
    """ContractSkill-style repair driven by full-closure violations."""
    return "\n\n".join(
        [
            "Here is the contract you proposed:",
            contract_source,
            (
                "Testing against the full reachable-state closure found these violations:\n"
                f"  false accepts:            {metrics['false_accepts']}\n"
                f"  false rejects:            {metrics['false_rejects']}\n"
                f"  postcondition violations: {metrics['postcondition_violations']}\n"
                f"  closure states checked:   {metrics['states_checked']}\n\n"
                "What the symptoms mean:\n" + SYMPTOM_GLOSSARY
            ),
            "Full-closure violations:\n" + format_counterexamples(counterexamples),
            _task(
                "Apply the same minimal ContractSkill-style repair discipline to resolve "
                "these full-closure violations without breaking valid cases. Change as "
                "little as possible."
            ),
        ]
    )


def describe_state(state: ShoppingState) -> str:
    """The one-line state rendering used in every transcript."""
    return state.describe()
