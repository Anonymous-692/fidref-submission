#!/usr/bin/env python3
"""Experiment runner for the cloud deployment domain."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

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
    counterexample_record,
    report_metrics,
    sha256_json,
)
from ..modeling.client import ChatClient, TransportError, Usage
from ..modeling.runner import (
    ACTIVE_CEGIS,
    ACTIVE_CEGIS_NO_BALANCE,
    ACTIVE_CEGIS_NO_COVERAGE,
    ACTIVE_CEGIS_UNIFORM,
    COUNTEREXAMPLE_GUIDED,
    DIRECT,
    METHODS,
    RANDOM_PROBE,
    SAMPLED_CEGIS,
    SELF_REFINE,
    STOP_COMPLETE,
    STOP_EXACT,
    STOP_QUERY_BUDGET,
    STOP_SAMPLE_SATISFIED,
    STOP_STATE_BUDGET,
    STOP_TOKEN_BUDGET,
    STOP_TRANSPORT,
    Budgets,
    Decoding,
    RunSpec,
)
from . import dsl, prompts
from .config import DeploymentConfig
from .contracts import ContractReport, evaluate_contract
from .contracts import MAX_RECORDED_COUNTEREXAMPLES
from .enumeration import enumerate_reachable
from .env import apply_action
from .ground_truth import ground_truth_contract
from .state import DeploymentAction, DeploymentState

MAX_SOURCE_ECHO_CHARS = 4000
STOP_CONTEXT_BUDGET = "context_budget_exhausted"

# Distinct stop reasons for the active-CEGIS loop guards. They are deliberately
# not `sampled_oracle_satisfied` and not a budget exhaustion: a run that ends for
# one of these reasons produced no usable agreement and must not be reported as
# though the sampled oracle had been satisfied.
STOP_DUPLICATE_OUTPUT = "duplicate_model_output"
STOP_VACUOUS_CANDIDATE = "vacuous_candidate_unrepaired"

DEFAULT_VACUITY_REVISION_LIMIT = 2


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

    def allowance(self, max_tokens: int) -> int | None:
        if self.remaining_calls <= 0:
            return None
        allowed = min(max_tokens, self.remaining_tokens)
        return allowed if allowed >= 1 else None

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


class _Session:
    def __init__(self, runner: "DeploymentExperimentRunner", spec: RunSpec) -> None:
        self.runner = runner
        self.spec = spec
        self.ledger = _Ledger(spec.budgets)
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": prompts.system_prompt()}
        ]
        self.interactions: list[Interaction] = []
        self.rounds: list[dict[str, Any]] = []
        self.contract = ContractRecord(status=STATUS_MISSING, error="no answer was produced")
        self.parsed: dsl.ParsedContract | None = None
        self.parse_failures = 0
        self.stopped_because: str | None = None
        self.previous_output: str | None = None

    def is_repeat(self, text: str) -> bool:
        """True when this answer is byte-identical to the answer before it."""
        repeated = self.previous_output is not None and text == self.previous_output
        self.previous_output = text
        return repeated

    def ask(self, role: str, user_text: str) -> Interaction | None:
        if self.ledger.remaining_calls <= 0:
            self.stopped_because = self.ledger.stop_reason()
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
        sent = tuple(dict(message) for message in request_messages)
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
            if hasattr(self.runner, 'g2_runtime'):
                # Opt-in G2 reserves this request's input as well as its completion.
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
                    error=(
                        f"prompt uses {prompt_tokens} tokens and leaves no output room "
                        f"inside the {context_window}-token context limit "
                        f"with margin {self.runner.context_margin}"
                    ),
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
            if hasattr(self.runner, 'g2_runtime'):
                self.interactions.append(Interaction(
                    index=index, role=role, messages=sent,
                    error='input reservation leaves no completion token budget',
                    prompt_tokens=prompt_tokens, requested_max_tokens=requested_max_tokens,
                    effective_max_tokens=effective_max_tokens, clamp_reason=clamp_reason,
                    context_window=context_window, safety_margin=safety_margin))
            self.stopped_because = STOP_TOKEN_BUDGET
            return None

        try:
            response = self.runner.client.complete(
                request_messages,
                temperature=self.spec.decoding.temperature,
                top_p=self.spec.decoding.top_p,
                max_tokens=effective_max_tokens,
                seed=self.spec.seed,
                guided_json=dsl.contract_json_schema() if self.spec.use_guided_json else None,
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
        if not self.runner.compact_context:
            self.messages.append({"role": "assistant", "content": response.text})
        actual_prompt_tokens = prompt_tokens if prompt_tokens is not None else response.usage.prompt_tokens
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
            prompt_tokens=actual_prompt_tokens,
            requested_max_tokens=requested_max_tokens,
            effective_max_tokens=effective_max_tokens,
            clamp_reason=clamp_reason,
            context_window=context_window,
            safety_margin=safety_margin,
        )
        self.interactions.append(interaction)
        if hasattr(self.runner, 'g2_runtime'):
            from ..modeling.g2_runtime import usage_valid
            if not usage_valid((response.raw or {}).get('usage')):
                self.stopped_because = 'usage_unavailable'
                return None
            if self.ledger.remaining_tokens < 0:
                # Preserve actual usage even if server token counts exceed our estimate.
                self.stopped_because = 'token_budget_overrun'
                return None
        return interaction

    def adopt(self, interaction: Interaction) -> ContractRecord:
        text = interaction.response_text or ""
        try:
            parsed = dsl.parse_contract_text(text, name=f"model.{self.spec.method}")
            self.runner.validate(parsed)
        except dsl.DslError as exc:
            self.parse_failures += 1
            record = ContractRecord(
                status=STATUS_PARSE_FAILURE,
                source=_truncate(text),
                error=str(exc),
                interaction_index=interaction.index,
            )
            if self.parsed is None:
                self.contract = record
            return record
        record = ContractRecord(
            status=STATUS_PARSED,
            source=parsed.canonical_json,
            spec=parsed.spec,
            node_count=parsed.node_count,
            interaction_index=interaction.index,
        )
        self.parsed = parsed
        self.contract = record
        return record

    def latest_source(self, record: ContractRecord) -> str:
        return record.source or "(the previous answer was empty)"


class DeploymentExperimentRunner:
    """Runs methods against the deployment sandbox configuration and model endpoint."""

    def __init__(
        self,
        config: DeploymentConfig,
        client: ChatClient,
        *,
        max_depth: int,
        max_states: int,
        config_path: str | None = None,
        serving_metadata: Mapping[str, Any] | None = None,
        compact_context: bool = False,
        context_token_limit: int | None = None,
        context_margin: int = 256,
        counterexample_limit: int = MAX_RECORDED_COUNTEREXAMPLES,
        vacuity_revision_limit: int = DEFAULT_VACUITY_REVISION_LIMIT,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.max_depth = max_depth
        self.max_states = max_states
        self.config_path = config_path
        self.serving_metadata = dict(serving_metadata or {})
        if context_token_limit is not None and context_token_limit < 1:
            raise ValueError("context_token_limit must be positive when set")
        if context_margin < 0:
            raise ValueError("context_margin must not be negative")
        if counterexample_limit < 1:
            raise ValueError("counterexample_limit must be positive")
        if vacuity_revision_limit < 0:
            raise ValueError("vacuity_revision_limit must not be negative")

        serving_max_len = self.serving_metadata.get("max_model_len")
        if serving_max_len is not None and isinstance(serving_max_len, int):
            if context_token_limit is not None and context_token_limit > serving_max_len:
                raise ValueError(
                    f"context_token_limit ({context_token_limit}) exceeds serving max_model_len ({serving_max_len})"
                )

        self.compact_context = compact_context
        self.context_token_limit = context_token_limit
        self.context_margin = context_margin
        self.counterexample_limit = counterexample_limit
        self.vacuity_revision_limit = vacuity_revision_limit
        self._now = now if now is not None else (lambda: datetime.now(timezone.utc))

        enumeration = enumerate_reachable(config, max_depth=max_depth, max_states=max_states)
        self.states: tuple[DeploymentState, ...] = enumeration.states
        self.enumeration_summary = enumeration.summary()
        self.oracle = ground_truth_contract(config)

    def validate(self, parsed: dsl.ParsedContract) -> None:
        contract = parsed.bind(self.config)
        probe = self.states[:2] if self.states else ()
        try:
            for state in probe:
                outcome = apply_action(state, DeploymentAction.deploy_service(), self.config)
                contract.holds_in(state)
                contract.transition_holds(state, outcome.state)
        except Exception as exc:
            raise dsl.DslError(f"contract raised while being validated: {exc!r}") from exc

    def probe_states(self, seed: int, count: int) -> tuple[DeploymentState, ...]:
        if count <= 0 or not self.states:
            return ()
        rng = random.Random(f"deployment_probe|{seed}|{self.config.fingerprint!r}")
        pool = list(self.states)
        if count >= len(pool):
            return tuple(pool)
        return tuple(rng.sample(pool, count))

    def candidate_aware_states(
        self,
        parsed: dsl.ParsedContract,
        *,
        seed: int,
        count: int,
        audit_index: int = 0,
        excluded: Sequence[DeploymentState] = (),
        balance: bool = True,
        coverage: bool = True,
    ) -> tuple[DeploymentState, ...]:
        if count <= 0:
            return ()
        excluded_set = set(excluded)
        pool = [state for state in self.states if state not in excluded_set]
        if not pool:
            return ()

        def select(
            candidates: Sequence[DeploymentState], quota: int, *, partition: str
        ) -> tuple[DeploymentState, ...]:
            tie_seed = (
                f"deployment_active|{partition}|{seed}|{audit_index}|"
                f"{self.config.fingerprint!r}"
            )
            if coverage:
                return _greedy_raw_coverage(candidates, quota, tie_seed=tie_seed)
            shuffled = list(candidates)
            random.Random(tie_seed).shuffle(shuffled)
            return tuple(shuffled[:quota])

        if not balance:
            return select(pool, min(count, len(pool)), partition="all")

        contract = parsed.bind(self.config)
        predicted_accept = [state for state in pool if contract.holds_in(state)]
        predicted_reject = [state for state in pool if not contract.holds_in(state)]
        accept_quota = min(len(predicted_accept), (count + 1) // 2)
        reject_quota = min(len(predicted_reject), count - accept_quota)

        selected = list(select(predicted_accept, accept_quota, partition="accept"))
        selected.extend(select(predicted_reject, reject_quota, partition="reject"))

        remaining = min(count, len(pool)) - len(selected)
        if remaining > 0:
            chosen = set(selected)
            selected.extend(
                select(
                    [state for state in pool if state not in chosen],
                    remaining,
                    partition="fill",
                )
            )
        return tuple(selected)

    def predicted_accept_count(self, parsed: dsl.ParsedContract) -> int:
        """How many enumerated reachable states the candidate's own precondition admits.

        Candidate prediction only, exactly like ``candidate_aware_states``: the
        sandbox action is never applied and the reference contract is never
        consulted, so this costs no oracle query and no state budget.
        """
        contract = parsed.bind(self.config)
        return sum(1 for state in self.states if contract.holds_in(state))

    def repair_prompt(self, record: ContractRecord, raw_text: str) -> str:
        """The follow-up turn after an unparseable answer.

        With a full transcript the offending answer is already in the messages, so
        the note alone is enough. In compact mode the transcript is dropped, so the
        answer is echoed back in bounded form beside the error and the task config.
        """
        if not self.compact_context:
            return (
                prompts.parse_failure_note(record.error or "")
                + "\n\n"
                + prompts.direct_prompt(self.config)
            )
        return prompts.parse_repair_prompt(
            self.config,
            error=record.error or "",
            previous_output=record.source or raw_text,
        )

    def observe(self, states: Sequence[DeploymentState]) -> tuple[dict[str, Any], ...]:
        action = DeploymentAction.deploy_service()
        observations = []
        for state in states:
            outcome = apply_action(state, action, self.config)
            observations.append(
                {
                    "before": state.describe(),
                    "before_fields": state.to_dict(),
                    "ok": outcome.ok,
                    "error": outcome.error,
                    "after": outcome.state.describe() if outcome.ok else None,
                    "after_fields": outcome.state.to_dict() if outcome.ok else None,
                }
            )
        return tuple(observations)

    def score(
        self, parsed: dsl.ParsedContract, *, max_counterexamples: int = MAX_RECORDED_COUNTEREXAMPLES
    ) -> ContractReport:
        return evaluate_contract(
            parsed.bind(self.config),
            self.states,
            self.config,
            max_counterexamples=max_counterexamples,
        )

    def run(self, spec: RunSpec) -> RunArtifact:
        session = _Session(self, spec)
        _METHOD_BODIES[spec.method](self, session)
        return self._finish(session)

    def _direct(self, session: _Session) -> None:
        interaction = session.ask("propose", prompts.direct_prompt(self.config))
        if interaction is None:
            return
        session.adopt(interaction)
        session.stopped_because = session.stopped_because or STOP_COMPLETE

    def _self_refine(self, session: _Session) -> None:
        first = session.ask("propose", prompts.direct_prompt(self.config))
        if first is None:
            return
        initial = session.adopt(first)
        session.rounds.append(
            {"round": 0, "role": "propose", "contract_status": initial.status, "parse_error": initial.error}
        )

        follow_up = prompts.critique_prompt(session.latest_source(initial))
        if not initial.parsed:
            follow_up = prompts.parse_failure_note(initial.error or "") + "\n\n" + follow_up
        second = session.ask("revise", follow_up)
        if second is None:
            return
        revised = session.adopt(second)
        session.rounds.append(
            {"round": 1, "role": "revise", "contract_status": revised.status, "parse_error": revised.error}
        )
        session.stopped_because = STOP_COMPLETE

    def _random_probe(self, session: _Session) -> None:
        budget = session.ledger.remaining_states
        states = self.probe_states(session.spec.seed, budget)
        observations = self.observe(states)
        session.ledger.observe_states(len(observations))
        session.rounds.append(
            {
                "round": 0,
                "role": "probe",
                "states_probed": len(observations),
                "accepted": sum(1 for entry in observations if entry["ok"]),
                "observations": list(observations),
            }
        )
        if not observations:
            session.stopped_because = STOP_STATE_BUDGET
            return
        interaction = session.ask("propose", prompts.probe_prompt(self.config, observations))
        if interaction is None:
            return
        session.adopt(interaction)
        session.stopped_because = session.stopped_because or STOP_COMPLETE

    def _counterexample_guided(self, session: _Session) -> None:
        prompt = prompts.direct_prompt(self.config)
        round_index = 0
        while True:
            role = "propose" if round_index == 0 else "revise"
            interaction = session.ask(role, prompt)
            if interaction is None:
                return
            record = session.adopt(interaction)

            if not record.parsed:
                session.rounds.append(
                    {"round": round_index, "role": role, "contract_status": record.status, "parse_error": record.error}
                )
                prompt = (
                    prompts.parse_failure_note(record.error or "")
                    + "\n\n"
                    + prompts.direct_prompt(self.config)
                )
                round_index += 1
                continue

            allowance = min(session.ledger.remaining_states, MAX_RECORDED_COUNTEREXAMPLES)
            report = self.score(session.parsed, max_counterexamples=max(allowance, 0))
            session.ledger.oracle_feedback_queries += 1
            metrics = report_metrics(report)
            counterexamples = [counterexample_record(entry) for entry in report.counterexamples]
            session.ledger.observe_states(len(counterexamples))
            session.rounds.append(
                {
                    "round": round_index,
                    "role": role,
                    "contract_status": record.status,
                    "metrics": metrics,
                    "counterexamples": counterexamples,
                }
            )

            if report.is_exact:
                session.stopped_because = STOP_EXACT
                return
            if not counterexamples:
                session.stopped_because = (
                    STOP_STATE_BUDGET if session.ledger.remaining_states <= 0 else STOP_COMPLETE
                )
                return
            prompt = prompts.counterexample_prompt(
                session.latest_source(record), metrics, counterexamples
            )
            round_index += 1

    def _sampled_cegis(self, session: _Session) -> None:
        sampled_states = self.probe_states(session.spec.seed, session.ledger.remaining_states)
        if not sampled_states:
            session.stopped_because = STOP_STATE_BUDGET
            return
        session.ledger.observe_states(len(sampled_states))

        prompt = prompts.direct_prompt(self.config)
        round_index = 0
        while True:
            role = "propose" if round_index == 0 else "revise"
            interaction = session.ask(role, prompt)
            if interaction is None:
                return
            record = session.adopt(interaction)

            if not record.parsed:
                session.rounds.append(
                    {
                        "round": round_index,
                        "role": role,
                        "adaptation": SAMPLED_CEGIS,
                        "evidence_policy": "fixed_budget_sampled_equivalence_oracle",
                        "contract_status": record.status,
                        "parse_error": record.error,
                        "sampled_oracle_size": len(sampled_states),
                    }
                )
                prompt = (
                    prompts.parse_failure_note(record.error or "")
                    + "\n\n"
                    + prompts.direct_prompt(self.config)
                )
                round_index += 1
                continue

            report = evaluate_contract(
                session.parsed.bind(self.config),
                sampled_states,
                self.config,
                max_counterexamples=MAX_RECORDED_COUNTEREXAMPLES,
            )
            session.ledger.sampled_feedback_queries += 1
            session.ledger.sampled_states_checked += len(sampled_states)
            metrics = report_metrics(report)
            counterexamples = [counterexample_record(entry) for entry in report.counterexamples]
            session.rounds.append(
                {
                    "round": round_index,
                    "role": role,
                    "adaptation": SAMPLED_CEGIS,
                    "evidence_policy": "fixed_budget_sampled_equivalence_oracle",
                    "contract_status": record.status,
                    "sampled_metrics": metrics,
                    "counterexamples": counterexamples,
                    "sampled_oracle_size": len(sampled_states),
                }
            )

            if report.is_exact:
                session.stopped_because = STOP_SAMPLE_SATISFIED
                return
            prompt = prompts.counterexample_prompt(
                session.latest_source(record), metrics, counterexamples
            )
            round_index += 1

    def _active_cegis(self, session: _Session) -> None:
        self._active_cegis_strategy(session, method=ACTIVE_CEGIS, balance=True, coverage=True)

    def _active_cegis_no_balance(self, session: _Session) -> None:
        self._active_cegis_strategy(session, method=ACTIVE_CEGIS_NO_BALANCE, balance=False, coverage=True)

    def _active_cegis_no_coverage(self, session: _Session) -> None:
        self._active_cegis_strategy(session, method=ACTIVE_CEGIS_NO_COVERAGE, balance=True, coverage=False)

    def _active_cegis_uniform(self, session: _Session) -> None:
        self._active_cegis_strategy(session, method=ACTIVE_CEGIS_UNIFORM, balance=False, coverage=False)

    def _active_cegis_strategy(
        self,
        session: _Session,
        *,
        method: str,
        balance: bool,
        coverage: bool,
    ) -> None:
        total_budget = session.ledger.remaining_states
        if total_budget <= 0:
            session.stopped_because = STOP_STATE_BUDGET
            return
        audit_batches = max(session.spec.budgets.query_budget - 1, 1)
        batch_size = max((total_budget + audit_batches - 1) // audit_batches, 1)
        audited_states: list[DeploymentState] = []
        audit_index = 0
        round_index = 0
        vacuity_revisions = 0
        policy = _active_evidence_policy(balance, coverage)
        prompt = prompts.direct_prompt(self.config)

        while True:
            role = "propose" if round_index == 0 else "revise"
            interaction = session.ask(role, prompt)
            if interaction is None:
                return
            record = session.adopt(interaction)
            repeated = session.is_repeat(interaction.response_text or "")

            if repeated:
                # A byte-identical answer means the feedback turn changed nothing.
                # Asking again would burn the remaining query budget on the same
                # text, so stop here and say so.
                session.rounds.append(
                    {
                        "round": round_index,
                        "role": role,
                        "adaptation": method,
                        "evidence_policy": policy,
                        "contract_status": record.status,
                        "guard": "duplicate_output",
                        "duplicate_of_round": round_index - 1,
                        "cumulative_states": len(audited_states),
                    }
                )
                session.stopped_because = STOP_DUPLICATE_OUTPUT
                return

            if not record.parsed:
                session.rounds.append(
                    {
                        "round": round_index,
                        "role": role,
                        "adaptation": method,
                        "evidence_policy": policy,
                        "contract_status": record.status,
                        "parse_error": record.error,
                        "states_observed": len(audited_states),
                    }
                )
                prompt = self.repair_prompt(record, interaction.response_text or "")
                round_index += 1
                continue

            reachable_accepts = self.predicted_accept_count(session.parsed)
            if reachable_accepts == 0 and vacuity_revisions < self.vacuity_revision_limit:
                # The precondition admits nothing, so auditing it can only confirm
                # that it refuses states; spend a bounded number of turns asking for
                # a non-vacuous candidate instead, and disclose nothing but the
                # candidate's own accept count.
                vacuity_revisions += 1
                session.rounds.append(
                    {
                        "round": round_index,
                        "role": role,
                        "adaptation": method,
                        "evidence_policy": policy,
                        "contract_status": record.status,
                        "guard": "vacuous_candidate",
                        "vacuity": {
                            "predicted_accepts": reachable_accepts,
                            "reachable_states": len(self.states),
                            "evidence": "candidate_prediction_only",
                        },
                        "vacuity_revision": vacuity_revisions,
                        "cumulative_states": len(audited_states),
                    }
                )
                prompt = prompts.vacuity_revision_prompt(
                    self.config,
                    previous_contract=session.latest_source(record),
                    reachable_states=len(self.states),
                )
                round_index += 1
                continue

            while True:
                allowance = min(batch_size, session.ledger.remaining_states)
                fresh_states = self.candidate_aware_states(
                    session.parsed,
                    seed=session.spec.seed,
                    count=allowance,
                    audit_index=audit_index,
                    excluded=audited_states,
                    balance=balance,
                    coverage=coverage,
                )
                if fresh_states:
                    audited_states.extend(fresh_states)
                    session.ledger.observe_states(len(fresh_states))
                if not audited_states:
                    session.stopped_because = STOP_STATE_BUDGET
                    return

                fresh_set = set(fresh_states)
                audit_states = tuple(fresh_states) + tuple(
                    state for state in audited_states if state not in fresh_set
                )
                contract = session.parsed.bind(self.config)
                report = evaluate_contract(
                    contract,
                    audit_states,
                    self.config,
                    max_counterexamples=self.counterexample_limit,
                )
                session.ledger.sampled_feedback_queries += 1
                session.ledger.sampled_states_checked += len(audit_states)
                metrics = report_metrics(report)
                counterexamples = [
                    counterexample_record(entry) for entry in report.counterexamples
                ]
                predicted_accepts = sum(contract.holds_in(state) for state in fresh_states)
                session.rounds.append(
                    {
                        "round": round_index,
                        "audit": audit_index,
                        "role": role,
                        "adaptation": method,
                        "evidence_policy": policy,
                        "contract_status": record.status,
                        "fresh_states": len(fresh_states),
                        "reachable_predicted_accepts": reachable_accepts,
                        "fresh_predicted_accepts": predicted_accepts,
                        "cumulative_states": len(audited_states),
                        "audit_metrics": metrics,
                        "counterexamples": counterexamples,
                    }
                )
                audit_index += 1

                if counterexamples:
                    prompt = prompts.counterexample_prompt(
                        session.latest_source(record), metrics, counterexamples
                    )
                    round_index += 1
                    break
                if session.ledger.remaining_states <= 0 or not fresh_states:
                    # A contract that admits no reachable state agrees with the
                    # sampled oracle only because it never claims anything, and the
                    # bounded revision turns have already been spent on it.
                    session.stopped_because = (
                        STOP_VACUOUS_CANDIDATE
                        if reachable_accepts == 0
                        else STOP_SAMPLE_SATISFIED
                    )
                    return

    def _finish(self, session: _Session) -> RunArtifact:
        spec = session.spec
        evaluation: EvaluationRecord | None = None
        if session.parsed is not None:
            evaluation = EvaluationRecord.from_report(self.score(session.parsed), self.enumeration_summary)

        if session.contract.status == STATUS_PARSED and evaluation is not None:
            outcome = OUTCOME_EXACT if evaluation.exact else OUTCOME_INEXACT
        elif session.contract.status == STATUS_PARSE_FAILURE:
            outcome = OUTCOME_PARSE_FAILURE
        else:
            outcome = OUTCOME_NO_CONTRACT

        latencies = [
            interaction.latency_s
            for interaction in session.interactions
            if interaction.latency_s is not None
        ]
        usage = session.ledger.usage.to_dict()
        usage["calls"] = session.ledger.calls
        spend = session.ledger.spend()
        spend["parse_failures"] = session.parse_failures

        prompt_sha256 = sha256_json({
            "system": prompts.system_prompt(),
            "direct": prompts.direct_prompt(self.config),
        })
        dsl_schema_sha256 = sha256_json(dsl.contract_json_schema())
        loop_guards = {
            "duplicate_output": True,
            "vacuity_revision_limit": self.vacuity_revision_limit,
            "parse_repair": True,
        }
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
            prompt_sha256=prompt_sha256,
            dsl_schema_sha256=dsl_schema_sha256,
            loop_guards=loop_guards,
        )

        hashes = {
            "sandbox_config_sha256": sha256_json(self.config.to_dict()),
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

    def sandbox_context(self) -> dict[str, Any]:
        return {
            "domain": "deployment",
            "config_path": self.config_path,
            "config": self.config.to_dict(),
            "max_depth": self.max_depth,
            "max_states": self.max_states,
            "enumeration": self.enumeration_summary,
            "evaluation_states": len(self.states),
            "oracle_contract": self.oracle.name,
        }


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
    prompt_sha256: str,
    dsl_schema_sha256: str,
    loop_guards: Mapping[str, Any],
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
    }
    return sha256_json(payload)


def _active_evidence_policy(balance: bool, coverage: bool) -> str:
    return {
        (True, True): "candidate_partitioned_progressive_coverage",
        (False, True): "progressive_coverage_without_candidate_balance",
        (True, False): "candidate_partitioned_progressive_random",
        (False, False): "progressive_uniform_random",
    }[(balance, coverage)]


def _greedy_raw_coverage(
    states: Sequence[DeploymentState], count: int, *, tie_seed: str
) -> tuple[DeploymentState, ...]:
    if count <= 0 or not states:
        return ()
    pool = list(states)
    random.Random(tie_seed).shuffle(pool)
    selected: list[DeploymentState] = []
    seen_singletons: set[tuple[int, object]] = set()
    seen_pairs: set[tuple[int, int, object, object]] = set()

    while pool and len(selected) < count:
        best_index = 0
        best_score: tuple[int, int] | None = None
        for index, state in enumerate(pool):
            features = state.key
            new_singletons = sum(
                (feature_index, value) not in seen_singletons
                for feature_index, value in enumerate(features)
            )
            new_pairs = 0
            for left in range(len(features)):
                for right in range(left + 1, len(features)):
                    if (left, right, features[left], features[right]) not in seen_pairs:
                        new_pairs += 1
            score = (new_singletons, new_pairs)
            if best_score is None or score > best_score:
                best_index = index
                best_score = score

        chosen = pool.pop(best_index)
        selected.append(chosen)
        features = chosen.key
        seen_singletons.update((index, value) for index, value in enumerate(features))
        for left in range(len(features)):
            for right in range(left + 1, len(features)):
                seen_pairs.add((left, right, features[left], features[right]))
    return tuple(selected)


_METHOD_BODIES: dict[str, Callable[[DeploymentExperimentRunner, _Session], None]] = {
    DIRECT: DeploymentExperimentRunner._direct,
    SELF_REFINE: DeploymentExperimentRunner._self_refine,
    RANDOM_PROBE: DeploymentExperimentRunner._random_probe,
    COUNTEREXAMPLE_GUIDED: DeploymentExperimentRunner._counterexample_guided,
    SAMPLED_CEGIS: DeploymentExperimentRunner._sampled_cegis,
    ACTIVE_CEGIS: DeploymentExperimentRunner._active_cegis,
    ACTIVE_CEGIS_NO_BALANCE: DeploymentExperimentRunner._active_cegis_no_balance,
    ACTIVE_CEGIS_NO_COVERAGE: DeploymentExperimentRunner._active_cegis_no_coverage,
    ACTIVE_CEGIS_UNIFORM: DeploymentExperimentRunner._active_cegis_uniform,
}


def _truncate(text: str, limit: int = MAX_SOURCE_ECHO_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


__all__ = [
    "DeploymentExperimentRunner",
    "STOP_CONTEXT_BUDGET",
    "STOP_DUPLICATE_OUTPUT",
    "STOP_VACUOUS_CANDIDATE",
    "compute_protocol_sha256",
]
