#!/usr/bin/env python3
"""Execution harness for synthesizing and evaluating retail contracts.

Integrates active CEGIS, ablation selectors, guards, and Phase 2 context clamping.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..modeling.artifacts import (
    OUTCOME_EXACT,
    OUTCOME_INEXACT,
    OUTCOME_NO_CONTRACT,
    OUTCOME_PARSE_FAILURE,
    STATUS_MISSING,
    STATUS_PARSED,
    STATUS_PARSE_FAILURE,
    COUNTEREXAMPLE_FORMAT_VERSION,
    ContractRecord,
    EvaluationRecord,
    Interaction,
    RunArtifact,
    sha256_json,
    sha256_text,
)
from ..modeling.client import ChatClient, RawResponse, TransportError, Usage
from ..modeling.runner import (
    ACTIVE_CEGIS,
    ACTIVE_CEGIS_NO_BALANCE,
    ACTIVE_CEGIS_NO_COVERAGE,
    ACTIVE_CEGIS_UNIFORM,
    DIRECT,
    RANDOM_PROBE,
    SAMPLED_CEGIS,
    SAMPLED_CEGIS_FIXED,
    SELF_REFINE,
    METHODS as BASE_METHODS,
    Budgets,
    Decoding,
    RunSpec,
)
from .config import DEFAULT_RETAIL_CONFIG, RetailConfig
from .contracts import Contract, ContractReport, Symptom, evaluate_contract
from .dsl import (
    DslError, contract_json_schema, parse_contract,
    VOCABULARY_LEGACY, VOCABULARY_PROTOCOLS,
)
from .enumeration import EnumerationResult, enumerate_reachable
from .env import ActionKind, RetailAction, apply_action
from .ground_truth import (
    ITEM_AVAILABLE,
    ITEM_PRICES,
    ITEM_TO_PRODUCT,
    ORDER_ITEMS,
    calculate_price_difference,
    exchange_items_postcondition,
    exchange_items_precondition,
)
from .prompts import (
    counterexample_prompt,
    counterexample_prompt_v3,
    direct_prompt,
    parse_repair_prompt,
    parse_repair_prompt_v3,
    self_refine_prompt_v3,
    system_prompt,
    vacuity_revision_prompt,
    vacuity_revision_prompt_v3,
)
from .state import RetailState, SEMANTIC_FIELDS_PROTOCOL, SEMANTIC_FIELDS_SCOPE

logger = logging.getLogger(__name__)

STOP_QUERY_BUDGET = "query_budget_exhausted"
STOP_TOKEN_BUDGET = "token_budget_exhausted"
STOP_STATE_BUDGET = "state_budget_exhausted"
STOP_CONTEXT_BUDGET = "context_budget_exhausted"
STOP_SAMPLE_SATISFIED = "sampled_oracle_satisfied"
STOP_TRANSPORT = "transport_error"
STOP_METHOD_COMPLETE = "method_complete"
STOP_DUPLICATE_OUTPUT = "duplicate_model_output"
STOP_VACUOUS_CANDIDATE = "vacuous_candidate_unrepaired"

RETAIL_SELECTOR_VERSION = 1
METHODS = BASE_METHODS + (SAMPLED_CEGIS_FIXED,)
REFINEMENT_PROTOCOL_LEGACY_V2 = "legacy_v2"
REFINEMENT_PROTOCOL_SELF_CONTAINED_V3 = "self_contained_v3"
REFINEMENT_PROTOCOLS = (
    REFINEMENT_PROTOCOL_LEGACY_V2,
    REFINEMENT_PROTOCOL_SELF_CONTAINED_V3,
)


def state_relational_features(state: RetailState, config: RetailConfig) -> tuple[tuple[str, Any], ...]:
    """Non-oracle relational feature projection for retail state."""
    ord_id = state.draft_order_id
    ord_exists = ord_id in config.valid_orders
    ord_delivered = (ord_id == "#W1" and state.order_w1_status == "delivered")

    it_count = len(state.draft_item_ids)
    nit_count = len(state.draft_new_item_ids)
    count_match = (it_count == nit_count)

    ref_order_items = ORDER_ITEMS.get(ord_id or "#W1", ())
    items_in_order = all(state.draft_item_ids.count(it) <= ref_order_items.count(it) for it in state.draft_item_ids)

    valid_variants = True
    for old_it, new_it in zip(state.draft_item_ids, state.draft_new_item_ids):
        p1 = ITEM_TO_PRODUCT.get(old_it)
        p2 = ITEM_TO_PRODUCT.get(new_it)
        if p1 is None or p2 is None or p1 != p2:
            valid_variants = False
            break

    all_available = all(ITEM_AVAILABLE.get(x, False) for x in state.draft_new_item_ids)

    pm = state.draft_payment_method_id
    valid_pm = pm in config.valid_payment_methods
    diff = calculate_price_difference(state.draft_item_ids, state.draft_new_item_ids)
    bal_sign = (state.user_gift_card_balance >= diff) if pm == "gift_card_0" else True

    return (
        ("order_id", ord_id or ""),
        ("order_exists", ord_exists),
        ("order_delivered", ord_delivered),
        ("w1_status", state.order_w1_status),
        ("w2_status", state.order_w2_status),
        ("it_count", it_count),
        ("nit_count", nit_count),
        ("count_match", count_match),
        ("items_in_order", items_in_order),
        ("valid_variants", valid_variants),
        ("all_available", all_available),
        ("pm", pm or ""),
        ("valid_pm", valid_pm),
        ("diff_positive", diff > 0),
        ("diff_zero", diff == 0),
        ("balance_sufficient", bal_sign),
    )


def sample_relational_coverage(
    states: Sequence[RetailState],
    count: int,
    tie_seed: str,
    config: RetailConfig,
    feature_fn=state_relational_features,
) -> tuple[RetailState, ...]:
    """Greedy relational feature coverage selector for retail domain."""
    if len(states) <= count:
        return tuple(states)
    pool = list(states)
    random.Random(tie_seed).shuffle(pool)
    selected: list[RetailState] = []
    seen_features: Counter[tuple[str, Any]] = Counter()

    while pool and len(selected) < count:
        best_idx = 0
        best_score = (-1, -1.0)
        for idx, s in enumerate(pool):
            feats = feature_fn(s, config)
            novel = sum(1 for f in feats if seen_features[f] == 0)
            rare = sum(1.0 / (seen_features[f] + 1) for f in feats)
            score = (novel, rare)
            if score > best_score:
                best_idx = idx
                best_score = score

        chosen = pool.pop(best_idx)
        selected.append(chosen)
        for f in feature_fn(chosen, config):
            seen_features[f] += 1

    return tuple(selected)


class _Ledger:
    def __init__(self, budgets: Budgets) -> None:
        self.budgets = budgets
        self.calls = 0
        self.usage = Usage()
        self.states_observed = 0
        self.oracle_feedback_queries = 0
        self.sampled_feedback_queries = 0
        self.sampled_states_checked = 0

    @property
    def remaining_calls(self) -> int:
        return self.budgets.query_budget - self.calls

    @property
    def remaining_tokens(self) -> int:
        return self.budgets.token_budget - self.usage.total_tokens

    @property
    def remaining_states(self) -> int:
        return self.budgets.state_budget - self.states_observed

    def stop_reason(self) -> str:
        if self.remaining_calls <= 0:
            return STOP_QUERY_BUDGET
        return STOP_TOKEN_BUDGET

    def charge_call(self, usage: Usage) -> None:
        self.calls += 1
        self.usage = self.usage + usage

    def observe_states(self, count: int) -> None:
        self.states_observed += count

    def spend(self) -> dict[str, Any]:
        return {
            "model_calls": self.calls,
            "states_observed": self.states_observed,
            "oracle_feedback_queries": self.oracle_feedback_queries,
            "sampled_feedback_queries": self.sampled_feedback_queries,
            "sampled_states_checked": self.sampled_states_checked,
            "remaining_calls": max(self.remaining_calls, 0),
            "remaining_tokens": max(self.remaining_tokens, 0),
            "remaining_states": max(self.remaining_states, 0),
        }


@dataclass
class _Session:
    spec: RunSpec
    runner: RetailExperimentRunner
    interactions: list[Interaction] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    ledger: _Ledger = field(init=False)
    messages: list[dict[str, str]] = field(default_factory=list)
    contract: ContractRecord = field(default_factory=lambda: ContractRecord(STATUS_MISSING))
    parsed: Contract | None = None
    stopped_because: str = STOP_QUERY_BUDGET
    parse_failures: int = 0
    seen_responses: set[str] = field(default_factory=set)
    vacuity_revisions_used: int = 0

    def __post_init__(self) -> None:
        self.ledger = _Ledger(self.spec.budgets)
        policy = self.runner.information_policy
        self.messages = [{"role": "system", "content": policy.system_prompt() if policy else system_prompt(self.runner.vocabulary_protocol)}]

    def ask(self, role: str, user_text: str) -> str | None:
        journal = getattr(self.runner, "journal", None)
        if journal is not None and not (not self.interactions and self.runner.initial_candidate is not None):
            def execute():
                text = self._ask(role, user_text)
                return self.interactions[-1] if text is not None else None
            interaction = journal.invoke(self, role, user_text, execute)
            return interaction.response_text if interaction is not None else None
        return self._ask(role, user_text)

    def _ask(self, role: str, user_text: str) -> str | None:
        policy = self.runner.information_policy
        if policy:
            if user_text == direct_prompt(self.runner.config, self.runner.vocabulary_protocol):
                user_text = policy.direct_prompt()
            else:
                user_text = policy.render(user_text)
        if self.ledger.remaining_calls <= 0:
            self.stopped_because = STOP_QUERY_BUDGET
            return None
        if self.ledger.remaining_tokens < 1:
            self.stopped_because = self.ledger.stop_reason()
            return None

        requested_max_tokens = self.spec.decoding.max_tokens
        token_budget_allowance = self.ledger.remaining_tokens
        budget_clamped = token_budget_allowance < requested_max_tokens

        user_message = {"role": "user", "content": user_text}
        if self.runner.compact_context:
            request_messages = [self.messages[0], user_message]
        else:
            self.messages.append(user_message)
            request_messages = self.messages
        sent = tuple(dict(m) for m in request_messages)
        index = len(self.interactions)

        context_window = self.runner.context_token_limit
        safety_margin = self.runner.context_margin if context_window is not None else None
        prompt_tokens: int | None = None
        clamp_reason: str

        if context_window is not None:
            try:
                prompt_tokens = self.runner.client.count_chat_tokens(request_messages)
                if policy and (type(prompt_tokens) is not int or prompt_tokens < 0):
                    raise TransportError("invalid prompt token count")
            except TransportError as exc:
                if not policy:
                    self.ledger.charge_call(Usage())
                interaction = Interaction(
                    index=index,
                    role=role,
                    messages=sent,
                    error=str(exc),
                    prompt_tokens=None,
                    requested_max_tokens=requested_max_tokens,
                    effective_max_tokens=None,
                    clamp_reason="not_evaluated",
                    context_window=context_window,
                    safety_margin=safety_margin,
                )
                self.interactions.append(interaction)
                self.stopped_because = STOP_TRANSPORT
                return None

            context_allowance = context_window - self.runner.context_margin - prompt_tokens
            context_clamped = context_allowance < requested_max_tokens
            if hasattr(self.runner, 'g2_runtime') or policy:
                # Opt-in partial-pool replay: reserve actual input before output.
                token_budget_allowance -= prompt_tokens
                budget_clamped = token_budget_allowance < requested_max_tokens

            if context_clamped and budget_clamped:
                clamp_reason = "both"
            elif context_clamped:
                clamp_reason = "context_window"
            elif budget_clamped:
                clamp_reason = "token_budget"
            else:
                clamp_reason = "none"

            effective_max_tokens = min(requested_max_tokens, token_budget_allowance, context_allowance)

            if context_allowance < 1:
                interaction = Interaction(
                    index=index,
                    role=role,
                    messages=sent,
                    error=f"prompt uses {prompt_tokens} tokens and leaves no output room in {context_window} limit",
                    prompt_tokens=prompt_tokens,
                    requested_max_tokens=requested_max_tokens,
                    effective_max_tokens=effective_max_tokens,
                    clamp_reason=clamp_reason,
                    context_window=context_window,
                    safety_margin=safety_margin,
                )
                self.interactions.append(interaction)
                self.stopped_because = STOP_CONTEXT_BUDGET
                return None
        else:
            prompt_tokens = None
            clamp_reason = "token_budget" if budget_clamped else "none"
            effective_max_tokens = min(requested_max_tokens, token_budget_allowance)

        if effective_max_tokens < 1:
            self.stopped_because = STOP_TOKEN_BUDGET
            return None

        if policy and index == 0 and self.runner.initial_candidate is not None:
            payload = self.runner.initial_candidate
            stored = payload["interaction"]
            if stored is None:
                self.stopped_because = payload["stopped_because"]
                return None
            interaction = Interaction.from_dict(stored)
            if interaction.messages != sent or interaction.effective_max_tokens != effective_max_tokens:
                raise ValueError("shared initial request mismatch")
            from dataclasses import replace
            interaction = replace(interaction, role=role)
            self.interactions.append(interaction)
            usage = Usage(**{k: v for k, v in payload["usage"].items()
                            if k in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")})
            if payload["calls"]:
                self.ledger.charge_call(usage)
            if payload["stopped_because"] is not None:
                self.stopped_because = payload["stopped_because"]
                return None
            if not self.runner.compact_context:
                self.messages.append({"role": "assistant", "content": interaction.response_text})
            return interaction.response_text

        try:
            response = self.runner.client.complete(
                request_messages,
                temperature=self.spec.decoding.temperature,
                top_p=self.spec.decoding.top_p,
                max_tokens=effective_max_tokens,
                seed=self.spec.seed,
                guided_json=(policy.schema() if policy else contract_json_schema(self.runner.vocabulary_protocol)) if self.spec.use_guided_json else None,
            )
        except TransportError as exc:
            self.ledger.charge_call(Usage())
            interaction = Interaction(
                index=index,
                role=role,
                messages=sent,
                error=str(exc),
                prompt_tokens=prompt_tokens,
                requested_max_tokens=requested_max_tokens,
                effective_max_tokens=effective_max_tokens,
                clamp_reason=clamp_reason,
                context_window=context_window,
                safety_margin=safety_margin,
            )
            self.interactions.append(interaction)
            self.stopped_because = STOP_TRANSPORT
            return None

        self.ledger.charge_call(response.usage)
        interaction = Interaction(
            index=index,
            role=role,
            messages=sent,
            request=response.request,
            response_text=response.text,
            raw_response=response.raw,
            usage=response.usage.to_dict(),
            latency_s=response.latency_s,
            finish_reason=response.finish_reason,
            prompt_tokens=prompt_tokens,
            requested_max_tokens=requested_max_tokens,
            effective_max_tokens=effective_max_tokens,
            clamp_reason=clamp_reason,
            context_window=context_window,
            safety_margin=safety_margin,
        )
        self.interactions.append(interaction)
        if hasattr(self.runner, 'g2_runtime') or policy:
            from ..modeling.g2_runtime import usage_valid
            if not usage_valid((response.raw or {}).get('usage')):
                self.stopped_because = 'usage_unavailable'
                return None
            if self.ledger.remaining_tokens < 0:
                self.stopped_because = 'token_budget_overrun'
                return None
        if not self.runner.compact_context:
            self.messages.append({"role": "assistant", "content": response.text})
        return response.text

    def parse_with_repair(self, initial_text: str | None, role: str) -> Contract | None:
        if initial_text is None:
            return None
        current_text = initial_text
        index = len(self.interactions) - 1
        for attempt in range(2):
            try:
                parsed_c = (self.runner.information_policy.parse(current_text)
                            if self.runner.information_policy else
                            parse_contract(current_text, self.runner.config,
                                           vocabulary_protocol=self.runner.vocabulary_protocol))
                contract = parsed_c.bind(self.runner.config)
                self.contract = ContractRecord(
                    status=STATUS_PARSED,
                    source=current_text,
                    interaction_index=index,
                )
                self.parsed = contract
                return contract
            except DslError as exc:
                self.parse_failures += 1
                self.contract = ContractRecord(
                    status=STATUS_PARSE_FAILURE,
                    source=current_text,
                    error=str(exc),
                    interaction_index=index,
                )
                self.parsed = None
                if attempt == 0 and self.ledger.remaining_calls > 0 and self.ledger.remaining_tokens > 0:
                    if self.runner.refinement_protocol == REFINEMENT_PROTOCOL_SELF_CONTAINED_V3:
                        repair_prompt = parse_repair_prompt_v3(current_text, str(exc), self.runner.config)
                    else:
                        repair_prompt = parse_repair_prompt(current_text, str(exc), self.runner.config)
                    repaired = self.ask(f"{role}_parse_repair", repair_prompt)
                    if repaired is None:
                        return None
                    current_text = repaired
                    index = len(self.interactions) - 1
                else:
                    return None
        return None


def compute_protocol_sha256(
    *,
    method: str,
    requested_max_tokens: int,
    clamp_policy: str = "context_clamp_v2",
    clamp_version: int = 2,
    context_token_limit: int | None,
    context_margin: int,
    compact_context: bool,
    counterexample_limit: int,
    guided_json: bool,
    selector_policy: str,
    selector_version: int,
    prompt_sha256: str,
    dsl_schema_sha256: str,
    loop_guards: Mapping[str, Any],
    refinement_protocol: str | None = None,
    counterexample_format: str = COUNTEREXAMPLE_FORMAT_VERSION,
    vocabulary_protocol: str | None = None,
) -> str:
    payload = {
        "clamp_policy": clamp_policy,
        "clamp_version": clamp_version,
        "compact_context": compact_context,
        "context_margin": context_margin,
        "context_token_limit": context_token_limit,
        "counterexample_limit": counterexample_limit,
        "counterexample_format": counterexample_format,
        "dsl_schema_sha256": dsl_schema_sha256,
        "guided_json": guided_json,
        "loop_guards": dict(loop_guards),
        "method": method,
        "prompt_sha256": prompt_sha256,
        "requested_max_tokens": requested_max_tokens,
        "selector_policy": selector_policy,
        "selector_version": selector_version,
    }
    if refinement_protocol is not None:
        payload["refinement_protocol"] = refinement_protocol
    if vocabulary_protocol is not None:
        payload["vocabulary_protocol"] = vocabulary_protocol
    return sha256_json(payload)


class RetailExperimentRunner:
    """Manages synthesis experiments over the tau-bench retail domain."""

    def __init__(
        self,
        client: ChatClient,
        config: RetailConfig | None = None,
        config_path: str = "experiment/configs/taubench_retail_default.json",
        states: Sequence[RetailState] | None = None,
        enumeration: EnumerationResult | None = None,
        max_depth: int = 20,
        max_states: int = 5000,
        now_fn: Callable[[], datetime.datetime] | None = None,
        context_token_limit: int | None = None,
        context_margin: int = 256,
        compact_context: bool = False,
        counterexample_limit: int = 3,
        vacuity_revision_limit: int = 1,
        refinement_protocol: str = REFINEMENT_PROTOCOL_LEGACY_V2,
        serving_metadata: dict[str, Any] | None = None,
        vocabulary_protocol: str = VOCABULARY_LEGACY,
        information_condition: str | None = None,
        initial_candidate: Mapping[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.config = config or DEFAULT_RETAIL_CONFIG
        self.config_path = str(config_path)
        self.max_depth = max_depth
        self.max_states = max_states
        self._now = now_fn or (lambda: datetime.datetime.now(datetime.timezone.utc))
        self.context_token_limit = context_token_limit
        self.context_margin = context_margin
        self.compact_context = compact_context
        self.counterexample_limit = counterexample_limit
        self.vacuity_revision_limit = vacuity_revision_limit
        if refinement_protocol not in REFINEMENT_PROTOCOLS:
            raise ValueError(f"unknown refinement protocol: {refinement_protocol!r}")
        self.refinement_protocol = refinement_protocol
        if vocabulary_protocol not in VOCABULARY_PROTOCOLS:
            raise ValueError(f"unknown vocabulary protocol: {vocabulary_protocol!r}")
        self.vocabulary_protocol = vocabulary_protocol
        self.information_policy = None
        self.initial_candidate = json.loads(json.dumps(initial_candidate)) if initial_candidate is not None else None
        if information_condition is not None:
            from .information_policy import InformationPolicy
            if vocabulary_protocol != SEMANTIC_FIELDS_PROTOCOL or context_token_limit is None:
                raise ValueError("Retail information conditions require W1 semantic fields and explicit context limit")
            if not compact_context or refinement_protocol != REFINEMENT_PROTOCOL_SELF_CONTAINED_V3:
                raise ValueError("Retail information conditions require compact self-contained refinement")
            if RetailConfig.from_json(config_path) != self.config:
                raise ValueError("Retail fixture and config path disagree")
            self.information_policy = InformationPolicy(information_condition, self.config)
        elif initial_candidate is not None:
            raise ValueError("shared initial requires an information condition")
        if vocabulary_protocol == SEMANTIC_FIELDS_PROTOCOL and set(self.config.deliverable_orders) - {"#W1"}:
            raise ValueError("w1_semantic_fields_v1 does not support delivered W2 payloads")
        self.serving_metadata = dict(serving_metadata) if serving_metadata is not None else None

        if enumeration is not None:
            self.enumeration = enumeration
        else:
            self.enumeration = enumerate_reachable(self.config, max_depth=max_depth, max_states=max_states)
        self.states = tuple(states) if states is not None else self.enumeration.states

    def score(self, contract: Contract, subset: Sequence[RetailState] | None = None) -> ContractReport:
        evaluation_states = subset if subset is not None else self.states
        return evaluate_contract(
            contract,
            evaluation_states,
            self.config,
            max_counterexamples=self.counterexample_limit,
            vocabulary_protocol=self.vocabulary_protocol,
        )

    def run(self, spec: RunSpec) -> RunArtifact:
        if self.information_policy:
            if spec.method not in (DIRECT, SAMPLED_CEGIS_FIXED, ACTIVE_CEGIS):
                raise ValueError("V-only preparation supports direct, sampled_cegis_fixed and active_cegis")
            if self.initial_candidate is not None:
                self.validate_initial(spec, self.initial_candidate)
        if spec.method not in METHODS:
            raise ValueError(f"unknown method {spec.method}; choose from {METHODS}")

        session = _Session(spec, self)
        dispatch = {
            DIRECT: self._run_direct,
            SELF_REFINE: self._run_self_refine,
            RANDOM_PROBE: self._run_random_probe,
            SAMPLED_CEGIS: self._run_sampled_cegis,
            SAMPLED_CEGIS_FIXED: self._run_sampled_cegis_fixed,
            ACTIVE_CEGIS: lambda s: self._run_active_cegis(s, balance=True, coverage=True),
            ACTIVE_CEGIS_NO_BALANCE: lambda s: self._run_active_cegis(s, balance=False, coverage=True),
            ACTIVE_CEGIS_NO_COVERAGE: lambda s: self._run_active_cegis(s, balance=True, coverage=False),
            ACTIVE_CEGIS_UNIFORM: lambda s: self._run_active_cegis(s, balance=False, coverage=False),
        }
        dispatch[spec.method](session)
        return self._finish(session)

    run_method = run

    def _run_direct(self, session: _Session) -> None:
        raw = session.ask("synthesis", direct_prompt(self.config, self.vocabulary_protocol))
        if raw is None:
            return
        contract = session.parse_with_repair(raw, "synthesis")
        if self.information_policy and contract is None:
            return
        session.stopped_because = STOP_METHOD_COMPLETE

    def _run_self_refine(self, session: _Session) -> None:
        raw = session.ask("synthesis", direct_prompt(self.config, self.vocabulary_protocol))
        if raw is None:
            return
        contract = session.parse_with_repair(raw, "synthesis")
        if contract is None:
            session.stopped_because = STOP_METHOD_COMPLETE
            return

        round_index = 1
        while session.ledger.remaining_calls > 0:
            if self.refinement_protocol == REFINEMENT_PROTOCOL_SELF_CONTAINED_V3:
                prompt = self_refine_prompt_v3(session.contract.source or raw)
            else:
                prompt = (
                    "Critique and refine your previous contract. "
                    "Output ONLY the revised JSON object:"
                )
            raw = session.ask(f"self_refine_round_{round_index}", prompt)
            if raw is None:
                return
            contract = session.parse_with_repair(raw, f"self_refine_round_{round_index}")
            if contract is None:
                break
            round_index += 1
        session.stopped_because = session.ledger.stop_reason()

    def _run_random_probe(self, session: _Session) -> None:
        self._run_cegis_loop(
            session,
            batch_size=8,
            balance=False,
            coverage=False,
            policy="unpartitioned_uniform",
        )

    def _run_sampled_cegis(self, session: _Session) -> None:
        self._run_cegis_loop(
            session,
            batch_size=8,
            balance=True,
            coverage=False,
            policy="candidate_partitioned_uniform",
        )

    def _run_sampled_cegis_fixed(self, session: _Session) -> None:
        """Fixed-pool control: draw B states once and re-score that same pool every round.

        This is the arm the other three domains call ``sampled_cegis``. The retail
        ``sampled_cegis`` above runs the candidate-partitioned loop instead, so it is not a
        fixed-pool control; see the 2026-09-05 protocol audit. Existing artifacts are kept
        as-is and this method is registered under a new name so its protocol hash differs.
        """
        sample_pool = self._deterministic_sample(
            session.spec.budgets.state_budget, f"sampled_cegis_fixed|{session.spec.seed}"
        )
        session.ledger.observe_states(len(sample_pool))
        prompt = direct_prompt(self.config, self.vocabulary_protocol)
        for round_idx in range(session.spec.budgets.query_budget):
            raw = session.ask(f"round_{round_idx}", prompt)
            if raw is None:
                return
            contract = session.parse_with_repair(raw, f"round_{round_idx}")
            if contract is None:
                if not self.information_policy:
                    session.stopped_because = STOP_METHOD_COMPLETE
                return
            report = self.score(contract, sample_pool)
            session.ledger.sampled_feedback_queries += 1
            session.ledger.sampled_states_checked += len(sample_pool)
            session.rounds.append({
                "round": round_idx,
                "role": "synthesis" if round_idx == 0 else f"refinement_round_{round_idx}",
                "adaptation": session.spec.method,
                "evidence_policy": "fixed_pool_unpartitioned",
                "contract_status": session.contract.status,
                "fresh_states": len(sample_pool) if round_idx == 0 else 0,
                "cumulative_states": len(sample_pool),
                "report": report.to_dict(),
            })
            if report.exact:
                session.stopped_because = STOP_SAMPLE_SATISFIED
                return
            prompt = self._counterexample_prompt(session, report)
        session.stopped_because = STOP_QUERY_BUDGET

    def _run_active_cegis(
        self,
        session: _Session,
        balance: bool = True,
        coverage: bool = True,
    ) -> None:
        policy = _active_evidence_policy(balance, coverage)
        self._run_cegis_loop(
            session,
            batch_size=8,
            balance=balance,
            coverage=coverage,
            policy=policy,
        )

    def _run_cegis_loop(
        self,
        session: _Session,
        batch_size: int,
        balance: bool,
        coverage: bool,
        policy: str,
    ) -> None:
        prompt = direct_prompt(self.config, self.vocabulary_protocol)
        round_index = 0
        audit_index = 0
        audited_states: list[RetailState] = []
        audited_set: set[RetailState] = set()

        while True:
            role = "synthesis" if round_index == 0 else f"refinement_round_{round_index}"
            raw = session.ask(role, prompt)
            if raw is None:
                return

            contract = session.parse_with_repair(raw, role)
            if contract is None:
                if not self.information_policy:
                    session.stopped_because = STOP_METHOD_COMPLETE
                return

            if self.refinement_protocol == REFINEMENT_PROTOCOL_SELF_CONTAINED_V3:
                candidate_text = (session.contract.source or raw).strip()
            else:
                candidate_text = raw.strip()
            if candidate_text in session.seen_responses:
                session.rounds.append({
                    "round": round_index,
                    "role": role,
                    "adaptation": session.spec.method,
                    "evidence_policy": policy,
                    "contract_status": session.contract.status,
                    "guard": "duplicate_output",
                    "duplicate_of_round": round_index - 1,
                    "cumulative_states": len(audited_states),
                })
                session.stopped_because = STOP_DUPLICATE_OUTPUT
                return
            session.seen_responses.add(candidate_text)

            predicted_accepts = sum(1 for s in self.states if contract.holds_in(s))
            if predicted_accepts == 0 and session.vacuity_revisions_used < self.vacuity_revision_limit:
                session.vacuity_revisions_used += 1
                session.rounds.append({
                    "round": round_index,
                    "role": role,
                    "adaptation": session.spec.method,
                    "evidence_policy": policy,
                    "contract_status": session.contract.status,
                    "guard": "vacuous_candidate",
                    "vacuity": {
                        "predicted_accepts": predicted_accepts,
                        "reachable_states": len(self.states),
                        "evidence": "candidate_prediction_only",
                    },
                    "vacuity_revision": session.vacuity_revisions_used,
                    "cumulative_states": len(audited_states),
                })
                if self.refinement_protocol == REFINEMENT_PROTOCOL_SELF_CONTAINED_V3:
                    prompt = vacuity_revision_prompt_v3(
                        session.contract.source or raw,
                        self.config,
                    )
                else:
                    prompt = vacuity_revision_prompt(self.config)
                round_index += 1
                continue

            allowance = min(batch_size, session.ledger.remaining_states)
            tie_seed = f"{session.spec.seed}|round_{round_index}|audit_{audit_index}|{contract.name}"
            fresh_states = self._select_active_sample(
                contract,
                allowance,
                tie_seed,
                balance=balance,
                coverage=coverage,
                excluded=audited_set,
            )

            if fresh_states:
                for s in fresh_states:
                    if s not in audited_set:
                        audited_set.add(s)
                        audited_states.append(s)
                session.ledger.observe_states(len(fresh_states))

            if not audited_states:
                session.stopped_because = STOP_STATE_BUDGET
                return

            fresh_set = set(fresh_states)
            audit_states = tuple(fresh_states) + tuple(s for s in audited_states if s not in fresh_set)
            report = self.score(contract, audit_states)
            session.ledger.sampled_feedback_queries += 1
            session.ledger.sampled_states_checked += len(audit_states)

            session.rounds.append({
                "round": round_index,
                "audit": audit_index,
                "role": role,
                "adaptation": session.spec.method,
                "evidence_policy": policy,
                "contract_status": session.contract.status,
                "fresh_states": len(fresh_states),
                "reachable_predicted_accepts": predicted_accepts,
                "cumulative_states": len(audited_states),
                "report": report.to_dict(),
            })
            audit_index += 1

            if report.exact:
                session.stopped_because = (
                    STOP_VACUOUS_CANDIDATE if predicted_accepts == 0 else STOP_SAMPLE_SATISFIED
                )
                return

            if session.ledger.remaining_calls <= 0:
                session.stopped_because = STOP_QUERY_BUDGET
                return

            prompt = self._counterexample_prompt(session, report)
            round_index += 1

    def _counterexample_prompt(self, session: _Session, report: ContractReport) -> str:
        if self.refinement_protocol == REFINEMENT_PROTOCOL_SELF_CONTAINED_V3:
            return counterexample_prompt_v3(
                session.contract.source or "",
                report,
                self.config,
                self.counterexample_limit,
            )
        return counterexample_prompt(report, self.config, self.counterexample_limit)

    def _select_active_sample(
        self,
        contract: Contract,
        budget: int,
        tie_seed: str,
        balance: bool = True,
        coverage: bool = True,
        excluded: set[RetailState] | None = None,
    ) -> tuple[RetailState, ...]:
        if budget <= 0:
            return ()
        ex = excluded or set()
        available = [s for s in self.states if s not in ex]
        feature_fn = self.information_policy.features if self.information_policy else state_relational_features
        if not available:
            return ()
        if len(available) <= budget:
            return tuple(available)

        if not balance:
            pool = available
            if not coverage:
                rng = random.Random(tie_seed)
                return tuple(rng.sample(pool, budget))
            return sample_relational_coverage(pool, budget, tie_seed, self.config, feature_fn)

        pos_pool = [s for s in available if contract.holds_in(s)]
        neg_pool = [s for s in available if not contract.holds_in(s)]
        pos_target = budget // 2
        neg_target = budget - pos_target

        if not pos_pool:
            if not coverage:
                rng = random.Random(tie_seed)
                return tuple(rng.sample(neg_pool, min(len(neg_pool), budget)))
            return sample_relational_coverage(neg_pool, min(len(neg_pool), budget), tie_seed, self.config, feature_fn)

        if not neg_pool:
            if not coverage:
                rng = random.Random(tie_seed)
                return tuple(rng.sample(pos_pool, min(len(pos_pool), budget)))
            return sample_relational_coverage(pos_pool, min(len(pos_pool), budget), tie_seed, self.config, feature_fn)

        if not coverage:
            rng = random.Random(tie_seed)
            pos_sel = rng.sample(pos_pool, min(len(pos_pool), pos_target))
            neg_sel = rng.sample(neg_pool, min(len(neg_pool), neg_target))
            return tuple(pos_sel + neg_sel)

        pos_sel = sample_relational_coverage(pos_pool, min(len(pos_pool), pos_target), f"{tie_seed}|pos", self.config, feature_fn)
        neg_sel = sample_relational_coverage(neg_pool, min(len(neg_pool), neg_target), f"{tie_seed}|neg", self.config, feature_fn)
        return tuple(pos_sel + neg_sel)

    def _deterministic_sample(self, count: int, tie_seed: str) -> tuple[RetailState, ...]:
        if len(self.states) <= count:
            return self.states
        rng = random.Random(tie_seed)
        return tuple(rng.sample(self.states, count))

    def _finish(self, session: _Session) -> RunArtifact:
        spec = session.spec
        evaluation: EvaluationRecord | None = None
        if session.parsed is not None:
            evaluation = EvaluationRecord.from_report(self.score(session.parsed), self.enumeration_summary())

        if session.contract.status == STATUS_PARSED and evaluation is not None:
            outcome = OUTCOME_EXACT if evaluation.exact else OUTCOME_INEXACT
        elif session.contract.status == STATUS_PARSE_FAILURE:
            outcome = OUTCOME_PARSE_FAILURE
        else:
            outcome = OUTCOME_NO_CONTRACT

        latencies = [i.latency_s for i in session.interactions if i.latency_s is not None]
        usage = session.ledger.usage.to_dict()
        usage["calls"] = session.ledger.calls
        spend = session.ledger.spend()
        spend["parse_failures"] = session.parse_failures
        if self.information_policy:
            allocated = self.initial_candidate["calls"] if self.initial_candidate is not None else 0
            spend["initial_calls_allocated"] = allocated
            spend["live_model_calls"] = session.ledger.calls - allocated
            journal = getattr(self, "journal", None)
            if journal is not None:
                spend["checkpoint_replayed_calls"] = journal.replayed_calls
                spend["live_model_calls"] -= journal.replayed_calls

        protocol_hash = self.compute_protocol_hash(spec)

        hashes = {
            "sandbox_config_sha256": sha256_json(json.loads(Path(self.config_path).read_text())),
            "decoding_sha256": sha256_json(spec.decoding.to_dict(spec.seed)),
            "budgets_sha256": sha256_json(spec.budgets.to_dict()),
            "protocol_sha256": protocol_hash,
        }
        reasoning_config = getattr(self.client, "reasoning_config", {})
        if reasoning_config:
            hashes["reasoning_sha256"] = sha256_json(reasoning_config)
        if self.information_policy:
            hashes["information_policy_sha256"] = sha256_json(self.information_policy.metadata())
            if self.initial_candidate is not None:
                hashes["initial_candidate_sha256"] = self.initial_candidate["sha256"]

        return RunArtifact(
            method=spec.method,
            model=self.client.model,
            endpoint=self.client.endpoint,
            seed=spec.seed,
            decoding=spec.decoding.to_dict(spec.seed),
            budgets=spec.budgets.to_dict(),
            sandbox=self.sandbox_context(),
            interactions=tuple(session.interactions),
            rounds=tuple(session.rounds),
            contract=session.contract,
            usage=usage,
            latency={"total_s": sum(latencies), "per_call_s": latencies},
            spend=spend,
            evaluation=evaluation,
            outcome=outcome,
            stopped_because=session.stopped_because,
            created_at=self._now().isoformat(),
            serving=(self.serving_metadata or {}) if self.information_policy else self.serving_metadata,
            hashes=hashes,
        )

    def compute_protocol_hash(self, spec: RunSpec) -> str:
        prompt_sha256 = sha256_json({
            "system": system_prompt(self.vocabulary_protocol),
            "direct": direct_prompt(self.config, self.vocabulary_protocol),
        })
        dsl_schema_sha256 = sha256_json(contract_json_schema(self.vocabulary_protocol))
        loop_guards = {
            "duplicate_output": True,
            "vacuity_revision_limit": self.vacuity_revision_limit,
            "parse_repair": True,
        }
        if spec.method == SAMPLED_CEGIS_FIXED:
            # Fixed-pool control: no candidate partitioning, no coverage. Recorded explicitly so
            # the hash input matches the loop that actually runs (2026-09-05 protocol audit).
            selector_policy = "fixed_pool_unpartitioned"
        else:
            selector_policy = _active_evidence_policy(
                spec.method in (ACTIVE_CEGIS, ACTIVE_CEGIS_NO_COVERAGE),
                spec.method in (ACTIVE_CEGIS, ACTIVE_CEGIS_NO_BALANCE),
            )
        if self.vocabulary_protocol != VOCABULARY_LEGACY:
            # Correct metadata only in the new epoch; preserve legacy hashes.
            iterative = spec.method not in (DIRECT, SELF_REFINE, SAMPLED_CEGIS_FIXED)
            loop_guards["duplicate_output"] = iterative
            loop_guards["vacuity_revision_limit"] = self.vacuity_revision_limit if iterative else 0
            if spec.method in (DIRECT, SELF_REFINE):
                selector_policy = "none"
            elif spec.method == SAMPLED_CEGIS:
                selector_policy = "candidate_partitioned_uniform"
        base_hash = compute_protocol_sha256(
            method=spec.method,
            requested_max_tokens=spec.decoding.max_tokens,
            clamp_policy="context_clamp_v2",
            clamp_version=2,
            context_token_limit=self.context_token_limit,
            context_margin=self.context_margin,
            compact_context=self.compact_context,
            counterexample_limit=self.counterexample_limit,
            guided_json=spec.use_guided_json,
            selector_policy=selector_policy,
            selector_version=RETAIL_SELECTOR_VERSION,
            prompt_sha256=prompt_sha256,
            dsl_schema_sha256=dsl_schema_sha256,
            loop_guards=loop_guards,
            vocabulary_protocol=(
                self.vocabulary_protocol if self.vocabulary_protocol != VOCABULARY_LEGACY else None
            ),
            refinement_protocol=(
                self.refinement_protocol
                if self.refinement_protocol != REFINEMENT_PROTOCOL_LEGACY_V2
                else None
            ),
        )
        if self.information_policy:
            context = self.initial_context(spec)
            context.pop("seed")
            context["decoding"] = {k: v for k, v in context["decoding"].items() if k != "seed"}
            return sha256_json({"version": "retail_information_runner_v1", "legacy_loop_hash": base_hash,
                                "initial_context_without_seed": context, "method": spec.method})
        return base_hash

    def initial_context(self, spec: RunSpec) -> dict[str, Any]:
        if self.information_policy is None:
            raise ValueError("information policy required")
        from .information_policy import source_hashes
        policy = self.information_policy
        return {"policy": policy.metadata(), "source_sha256": source_hashes(),
                "fixture": self.config.to_dict(), "enumeration": self.enumeration_summary(),
                "states_sha256": sha256_json([s.to_dict(vocabulary_protocol=self.vocabulary_protocol) for s in self.states]),
                "seed": spec.seed, "model": self.client.model, "endpoint": self.client.endpoint,
                "serving": self.serving_metadata, "decoding": spec.decoding.to_dict(spec.seed),
                "reasoning": getattr(self.client, "reasoning_config", {}),
                "structured_output_style": getattr(self.client, "structured_output_style", None),
                "budgets": spec.budgets.to_dict(), "guided_json": spec.use_guided_json,
                "context_token_limit": self.context_token_limit, "context_margin": self.context_margin,
                "counterexample_limit": self.counterexample_limit, "vacuity_revision_limit": self.vacuity_revision_limit,
                "compact_context": self.compact_context, "refinement_protocol": self.refinement_protocol,
                "schema": policy.schema(), "system": policy.system_prompt(), "direct": policy.direct_prompt()}

    def generate_initial(self, spec: RunSpec) -> dict[str, Any]:
        if self.initial_candidate is not None:
            raise ValueError("cannot generate over an attached initial")
        context = self.initial_context(spec)
        session = _Session(spec, self)
        text = session.ask("initial", direct_prompt(self.config, self.vocabulary_protocol))
        payload = {"version": "retail_initial_response_v1", "context_sha256": sha256_json(context),
                   "context": context, "interaction": session.interactions[0].to_dict() if session.interactions else None,
                   "calls": session.ledger.calls, "usage": session.ledger.usage.to_dict(),
                   "stopped_because": session.stopped_because if text is None else None}
        payload["sha256"] = sha256_json(payload)
        return payload

    def validate_initial(self, spec: RunSpec, payload: Mapping[str, Any]) -> None:
        contents = dict(payload)
        digest = contents.pop("sha256", None)
        if sha256_json(contents) != digest or payload.get("version") != "retail_initial_response_v1":
            raise ValueError("initial content hash or version mismatch")
        if payload["context_sha256"] != sha256_json(self.initial_context(spec)):
            raise ValueError("initial belongs to different condition, seed, fixture, source or serving")
        if sha256_json(payload["context"]) != payload["context_sha256"]:
            raise ValueError("initial context content mismatch")
        if type(payload["calls"]) is not int or payload["calls"] not in (0, 1):
            raise ValueError("invalid initial call count")
        usage = payload["usage"]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if type(usage.get(key)) is not int or usage[key] < 0:
                raise ValueError("invalid initial token count")
        if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
            raise ValueError("invalid initial usage total")
        if payload["interaction"] is not None:
            interaction = Interaction.from_dict(payload["interaction"])
            if interaction.index != 0 or interaction.role != "initial":
                raise ValueError("invalid initial interaction or usage")
            if any(interaction.usage.get(k, 0) != usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
                raise ValueError("initial usage disagrees with interaction")
            if interaction.error is None and (payload["calls"] != 1 or interaction.response_text is None):
                # Context-budget rejection records an interaction without a completion.
                if payload["stopped_because"] is None:
                    raise ValueError("invalid initial response")
            if interaction.error is not None and payload["stopped_because"] is None:
                raise ValueError("initial error must be terminal")
        elif payload["calls"] != 0 or usage["total_tokens"] != 0 or payload["stopped_because"] is None:
            raise ValueError("invalid absent initial response")

    def enumeration_summary(self) -> dict[str, Any]:
        return {
            "states": len(self.enumeration.states),
            "transitions": len(self.enumeration.transitions),
            "truncated": self.enumeration.truncated,
            "max_depth": self.max_depth,
            "max_states": self.max_states,
            "deepest_level": self.enumeration.deepest_level,
        }

    def sandbox_context(self) -> dict[str, Any]:
        return {
            **({"information_policy": self.information_policy.metadata()} if self.information_policy else {}),
            **({"observation_scope": SEMANTIC_FIELDS_SCOPE,
                "postcondition_protocol": SEMANTIC_FIELDS_PROTOCOL}
               if self.vocabulary_protocol == SEMANTIC_FIELDS_PROTOCOL else {}),
            **({"vocabulary_protocol": self.vocabulary_protocol}
               if self.vocabulary_protocol != VOCABULARY_LEGACY else {}),
            "domain": "taubench_retail",
            "config_path": self.config_path,
            "config": json.loads(Path(self.config_path).read_text()),
            "max_depth": self.max_depth,
            "max_states": self.max_states,
            "enumeration": self.enumeration_summary(),
            "evaluation_states": len(self.states),
            "oracle_contract": "ground_truth.exchange_delivered_order_items",
        }


def _active_evidence_policy(balance: bool, coverage: bool) -> str:
    return {
        (True, True): "candidate_partitioned_relational_coverage",
        (True, False): "candidate_partitioned_uniform",
        (False, True): "unpartitioned_relational_coverage",
        (False, False): "unpartitioned_uniform",
    }[(balance, coverage)]
