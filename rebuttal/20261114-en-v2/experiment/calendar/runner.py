#!/usr/bin/env python3
"""Execution harness for synthesizing and evaluating calendar contracts.

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
    SELF_REFINE,
    METHODS,
    Budgets,
    Decoding,
    RunSpec,
)
from .config import DEFAULT_CALENDAR_CONFIG, CalendarConfig
from .contracts import Contract, ContractReport, Symptom, evaluate_contract
from .dsl import DslError, contract_json_schema, parse_contract
from .enumeration import EnumerationResult, enumerate_reachable
from .env import ActionKind, CalendarAction, apply_action
from .ground_truth import schedule_meeting_postcondition, schedule_meeting_precondition
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
from .state import CalendarState

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

CALENDAR_SELECTOR_VERSION = 2
REFINEMENT_PROTOCOL_LEGACY_V2 = "legacy_v2"
REFINEMENT_PROTOCOL_SELF_CONTAINED_V3 = "self_contained_v3"
REFINEMENT_PROTOCOLS = (
    REFINEMENT_PROTOCOL_LEGACY_V2,
    REFINEMENT_PROTOCOL_SELF_CONTAINED_V3,
)


def state_relational_features(state: CalendarState, config: CalendarConfig) -> tuple[tuple[str, Any], ...]:
    """Non-oracle relational feature projection from raw state + config tables."""
    ext = state.external_booking
    d_room = state.draft_room
    d_slot = state.draft_slot
    d_type = state.draft_type
    d_att = state.draft_attendees

    cap = config.room_capacity(d_room) if d_room else None
    cap_exceeded = bool(cap is not None and len(d_att) > cap)
    video_mismatch = bool(d_type == "video_conf" and d_room is not None and not config.room_supports_video(d_room))

    same_slot = bool(ext and d_slot and ext.slot == d_slot)
    same_room = bool(ext and d_room and ext.room == d_room)
    same_room_slot = bool(same_slot and same_room)
    att_overlap = bool(ext and d_att and (set(ext.attendees) & set(d_att)))
    same_slot_att_overlap = bool(same_slot and att_overlap)

    valid_r = d_room in config.valid_rooms if d_room else False
    valid_s = d_slot in config.valid_slots if d_slot else False
    valid_t = d_type in config.valid_types if d_type else False

    draft_complete = bool(state.authenticated and state.active_meeting is None and d_att and d_slot and d_room and d_type)

    return (
        ("auth", state.authenticated),
        ("active_meet", state.active_meeting is not None),
        ("att_count", len(d_att)),
        ("slot", d_slot or ""),
        ("room", d_room or ""),
        ("type", d_type or ""),
        ("has_ext", ext is not None),
        ("cap_exceeded", cap_exceeded),
        ("video_mismatch", video_mismatch),
        ("same_room_slot", same_room_slot),
        ("same_slot_att_overlap", same_slot_att_overlap),
        ("valid_r", valid_r),
        ("valid_s", valid_s),
        ("valid_t", valid_t),
        ("draft_complete", draft_complete),
        ("complete_cap_exceeded", draft_complete and cap_exceeded),
        ("complete_video_mismatch", draft_complete and video_mismatch),
        ("complete_room_slot_conflict", draft_complete and same_room_slot),
        ("complete_att_slot_conflict", draft_complete and same_slot_att_overlap),
    )


def sample_relational_coverage(
    states: Sequence[CalendarState],
    count: int,
    tie_seed: str,
    config: CalendarConfig,
) -> tuple[CalendarState, ...]:
    """Greedy relational feature coverage selector for calendar domain."""
    if len(states) <= count:
        return tuple(states)
    pool = list(states)
    random.Random(tie_seed).shuffle(pool)
    selected: list[CalendarState] = []
    seen_features: Counter[tuple[str, Any]] = Counter()

    while pool and len(selected) < count:
        best_idx = 0
        best_score = (-1, -1.0)
        for idx, s in enumerate(pool):
            feats = state_relational_features(s, config)
            novel = sum(1 for f in feats if seen_features[f] == 0)
            rare = sum(1.0 / (seen_features[f] + 1) for f in feats)
            score = (novel, rare)
            if score > best_score:
                best_idx = idx
                best_score = score

        chosen = pool.pop(best_idx)
        selected.append(chosen)
        for f in state_relational_features(chosen, config):
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
    runner: CalendarExperimentRunner
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
        self.messages = [{"role": "system", "content": system_prompt()}]

    def ask(self, role: str, user_text: str) -> str | None:
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
            except TransportError as exc:
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

        try:
            response = self.runner.client.complete(
                request_messages,
                temperature=self.spec.decoding.temperature,
                top_p=self.spec.decoding.top_p,
                max_tokens=effective_max_tokens,
                seed=self.spec.seed,
                guided_json=contract_json_schema() if self.spec.use_guided_json else None,
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
        if not self.runner.compact_context:
            self.messages.append({"role": "assistant", "content": response.text})
        return response.text

    def parse_with_repair(self, raw_text: str | None, role: str) -> Contract | None:
        if raw_text is None:
            return None
        current_text = raw_text
        index = len(self.interactions) - 1
        for attempt in range(2):
            try:
                contract = parse_contract(current_text, self.runner.config)
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
                if attempt == 0:
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
    return sha256_json(payload)


class CalendarExperimentRunner:
    """Manages synthesis experiments over the Calendar workspace domain."""

    def __init__(
        self,
        client: ChatClient,
        config: CalendarConfig | None = None,
        config_path: str = "experiment/configs/calendar_default.json",
        max_depth: int = 20,
        max_states: int = 15000,
        compact_context: bool = True,
        context_token_limit: int | None = 8192,
        context_margin: int = 256,
        counterexample_limit: int = 3,
        vacuity_revision_limit: int = 1,
        refinement_protocol: str = REFINEMENT_PROTOCOL_LEGACY_V2,
        now_fn: Callable[[], datetime.datetime] | None = None,
        serving_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.config = config or DEFAULT_CALENDAR_CONFIG
        self.config_path = config_path
        self.max_depth = max_depth
        self.max_states = max_states
        self.compact_context = compact_context
        self.context_token_limit = context_token_limit
        self.context_margin = context_margin
        self.counterexample_limit = counterexample_limit
        self.vacuity_revision_limit = vacuity_revision_limit
        if refinement_protocol not in REFINEMENT_PROTOCOLS:
            raise ValueError(
                f"unsupported Calendar refinement protocol: {refinement_protocol!r}; "
                f"choose from {REFINEMENT_PROTOCOLS}"
            )
        self.refinement_protocol = refinement_protocol
        self._now = now_fn or (lambda: datetime.datetime.now(datetime.timezone.utc))
        self.serving_metadata = dict(serving_metadata or {})

        self.enumeration: EnumerationResult = enumerate_reachable(self.config, max_depth=max_depth, max_states=max_states)
        self.states = tuple(self.enumeration.states)

    def score(self, contract: Contract, states: Sequence[CalendarState] | None = None) -> ContractReport:
        eval_states = self.states if states is None else states
        return evaluate_contract(contract, eval_states, self.config, self.counterexample_limit)

    def run_method(self, spec: RunSpec) -> RunArtifact:
        if spec.method not in METHODS:
            raise ValueError(f"unsupported method: {spec.method!r}; choose from {METHODS}")
        session = _Session(spec, self)
        if spec.method == DIRECT:
            self._direct(session)
        elif spec.method == SELF_REFINE:
            self._self_refine(session)
        elif spec.method == RANDOM_PROBE:
            self._random_probe(session)
        elif spec.method == SAMPLED_CEGIS:
            self._sampled_cegis(session)
        elif spec.method in (ACTIVE_CEGIS, ACTIVE_CEGIS_NO_COVERAGE, ACTIVE_CEGIS_NO_BALANCE, ACTIVE_CEGIS_UNIFORM):
            balance = spec.method not in (ACTIVE_CEGIS_NO_BALANCE, ACTIVE_CEGIS_UNIFORM)
            coverage = spec.method not in (ACTIVE_CEGIS_NO_COVERAGE, ACTIVE_CEGIS_UNIFORM)
            self._active_cegis_strategy(session, balance=balance, coverage=coverage)
        else:
            raise ValueError(f"unhandled method: {spec.method}")
        return self._finish(session)

    def _direct(self, session: _Session) -> None:
        prompt = direct_prompt(self.config)
        raw = session.ask("direct", prompt)
        session.parse_with_repair(raw, "direct")
        session.stopped_because = STOP_METHOD_COMPLETE

    def _self_refine(self, session: _Session) -> None:
        prompt = direct_prompt(self.config)
        raw = session.ask("direct", prompt)
        contract = session.parse_with_repair(raw, "direct")
        if contract is None or session.stopped_because in (STOP_TRANSPORT, STOP_CONTEXT_BUDGET):
            return
        if self.refinement_protocol == REFINEMENT_PROTOCOL_SELF_CONTAINED_V3:
            critique_prompt = self_refine_prompt_v3(session.contract.source or raw)
        else:
            critique_prompt = "Critique and refine the previous contract for schedule_meeting. Output ONLY the improved JSON object:"
        refined_raw = session.ask("self_refine", critique_prompt)
        session.parse_with_repair(refined_raw, "self_refine")
        session.stopped_because = STOP_METHOD_COMPLETE

    def _random_probe(self, session: _Session) -> None:
        sampled = self._deterministic_sample(session.spec.budgets.state_budget, f"random_probe|{session.spec.seed}")
        session.ledger.observe_states(len(sampled))
        observations = []
        target = CalendarAction.make(ActionKind.SCHEDULE_MEETING)
        for s in sampled:
            res = apply_action(s, target, self.config)
            observations.append(f"State: {s.key} -> Ok: {res.ok} (Error: {res.error})")
        prompt = (
            f"{direct_prompt(self.config)}\n\n"
            f"Observed sandbox executions on random states:\n" + "\n".join(observations[:15]) + "\n\n"
            "Output ONLY the JSON object:"
        )
        raw = session.ask("random_probe", prompt)
        session.parse_with_repair(raw, "random_probe")
        session.stopped_because = STOP_METHOD_COMPLETE

    def _sampled_cegis(self, session: _Session) -> None:
        sample_pool = self._deterministic_sample(session.spec.budgets.state_budget, f"sampled_cegis|{session.spec.seed}")
        session.ledger.observe_states(len(sample_pool))
        prompt = direct_prompt(self.config)
        for round_idx in range(session.spec.budgets.query_budget):
            raw = session.ask(f"round_{round_idx}", prompt)
            if raw is None:
                return
            contract = session.parse_with_repair(raw, f"round_{round_idx}")
            if contract is None:
                return
            report = self.score(contract, sample_pool)
            session.ledger.sampled_feedback_queries += 1
            session.ledger.sampled_states_checked += len(sample_pool)
            session.rounds.append({"round": round_idx, "report": report.to_dict()})
            if report.exact:
                session.stopped_because = STOP_SAMPLE_SATISFIED
                return
            prompt = self._counterexample_prompt(session, report)

    def _active_cegis_strategy(self, session: _Session, balance: bool, coverage: bool) -> None:
        total_budget = session.ledger.remaining_states
        if total_budget <= 0:
            session.stopped_because = STOP_STATE_BUDGET
            return

        audit_batches = max(session.spec.budgets.query_budget - 1, 1)
        batch_size = max((total_budget + audit_batches - 1) // audit_batches, 1)
        audited_states: list[CalendarState] = []
        audited_set: set[CalendarState] = set()
        audit_index = 0
        round_index = 0
        policy = _active_evidence_policy(balance, coverage)
        prompt = direct_prompt(self.config)

        while True:
            role = "propose" if round_index == 0 else "revise"
            raw = session.ask(role, prompt)
            if raw is None:
                return
            contract = session.parse_with_repair(raw, role)
            if contract is None:
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
        excluded: set[CalendarState] | None = None,
    ) -> tuple[CalendarState, ...]:
        if budget <= 0:
            return ()
        ex = excluded or set()
        available = [s for s in self.states if s not in ex]
        if not available:
            return ()
        if len(available) <= budget:
            return tuple(available)

        if not balance:
            pool = available
            if not coverage:
                rng = random.Random(tie_seed)
                return tuple(rng.sample(pool, budget))
            return sample_relational_coverage(pool, budget, tie_seed, self.config)

        pos_pool = [s for s in available if contract.holds_in(s)]
        neg_pool = [s for s in available if not contract.holds_in(s)]
        pos_target = budget // 2
        neg_target = budget - pos_target

        if not pos_pool:
            if not coverage:
                rng = random.Random(tie_seed)
                return tuple(rng.sample(neg_pool, min(len(neg_pool), budget)))
            return sample_relational_coverage(neg_pool, min(len(neg_pool), budget), tie_seed, self.config)

        if not neg_pool:
            if not coverage:
                rng = random.Random(tie_seed)
                return tuple(rng.sample(pos_pool, min(len(pos_pool), budget)))
            return sample_relational_coverage(pos_pool, min(len(pos_pool), budget), tie_seed, self.config)

        if not coverage:
            rng = random.Random(tie_seed)
            pos_sel = rng.sample(pos_pool, min(len(pos_pool), pos_target))
            neg_sel = rng.sample(neg_pool, min(len(neg_pool), neg_target))
            return tuple(pos_sel + neg_sel)

        pos_sel = sample_relational_coverage(pos_pool, min(len(pos_pool), pos_target), f"{tie_seed}|pos", self.config)
        neg_sel = sample_relational_coverage(neg_pool, min(len(neg_pool), neg_target), f"{tie_seed}|neg", self.config)
        return tuple(pos_sel + neg_sel)

    def _deterministic_sample(self, count: int, tie_seed: str) -> tuple[CalendarState, ...]:
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

        prompt_sha256 = sha256_json({
            "system": system_prompt(),
            "direct": direct_prompt(self.config),
        })
        dsl_schema_sha256 = sha256_json(contract_json_schema())
        loop_guards = {
            "duplicate_output": True,
            "vacuity_revision_limit": self.vacuity_revision_limit,
            "parse_repair": True,
        }
        selector_policy = _active_evidence_policy(
            spec.method in (ACTIVE_CEGIS, ACTIVE_CEGIS_NO_COVERAGE),
            spec.method in (ACTIVE_CEGIS, ACTIVE_CEGIS_NO_BALANCE),
        )

        protocol_hash = compute_protocol_sha256(
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
            selector_version=CALENDAR_SELECTOR_VERSION,
            prompt_sha256=prompt_sha256,
            dsl_schema_sha256=dsl_schema_sha256,
            loop_guards=loop_guards,
            refinement_protocol=(
                self.refinement_protocol
                if self.refinement_protocol != REFINEMENT_PROTOCOL_LEGACY_V2
                else None
            ),
        )

        hashes = {
            "sandbox_config_sha256": sha256_json(json.loads(Path(self.config_path).read_text())),
            "decoding_sha256": sha256_json(spec.decoding.to_dict(spec.seed)),
            "budgets_sha256": sha256_json(spec.budgets.to_dict()),
            "protocol_sha256": protocol_hash,
        }
        reasoning_config = getattr(self.client, "reasoning_config", {})
        if reasoning_config:
            hashes["reasoning_sha256"] = sha256_json(reasoning_config)

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
            serving=self.serving_metadata,
            hashes=hashes,
        )


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
            "domain": "calendar",
            "config_path": self.config_path,
            "config": json.loads(Path(self.config_path).read_text()),
            "max_depth": self.max_depth,
            "max_states": self.max_states,
            "enumeration": self.enumeration_summary(),
            "evaluation_states": len(self.states),
            "oracle_contract": "ground_truth.schedule_meeting",
        }


def _active_evidence_policy(balance: bool, coverage: bool) -> str:
    return {
        (True, True): "candidate_partitioned_relational_coverage",
        (True, False): "candidate_partitioned_uniform",
        (False, True): "unpartitioned_relational_coverage",
        (False, False): "unpartitioned_uniform",
    }[(balance, coverage)]
