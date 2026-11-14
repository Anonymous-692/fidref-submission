#!/usr/bin/env python3
"""The four contract-proposal methods and the budgets they share.

Every method answers the same question — what is the contract for
``place_order``? — under the same configurable model, decoding parameters, state
budget, query budget, and token budget. They differ only in what they are allowed
to look at:

``direct``                one call, no observations, no feedback.
``self_refine``           one proposal, then one critique/revision pass over its
                          own answer. Still no observations and no feedback.
``random_probe``          attempts the action in randomly sampled reachable
                          states, then proposes once from that transcript.
``counterexample_guided`` proposes, is shown concrete disagreements found by the
                          checker, and revises; repeated until the contract is
                          exact or a budget runs out.
``sampled_cegis``         follows the same repair loop, but the checker may only
                          inspect one fixed, budget-limited sample of states.

Three budgets bound a run. The *query* budget caps model calls; the *token*
budget caps prompt-plus-completion tokens and also clamps each call's
``max_tokens`` to what is left; the *state* budget caps how many sandbox states a
method may observe — probe attempts for ``random_probe``, counterexamples for
``counterexample_guided``, and nothing at all for the two blind methods. The
oracle's own scoring pass is never charged against a method budget, because it is
measurement rather than part of the method.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from ..environment import (
    EnvConfig,
    ShoppingAction,
    ShoppingState,
    apply_action,
    enumerate_reachable,
)
from ..evaluation import (
    ContractReport,
    evaluate_contract,
    ground_truth_contract,
)
from ..evaluation.contracts import MAX_RECORDED_COUNTEREXAMPLES
from . import dsl, prompts
from .artifacts import (
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
from .client import ChatClient, TransportError, Usage

DIRECT = "direct"
SELF_REFINE = "self_refine"
RANDOM_PROBE = "random_probe"
COUNTEREXAMPLE_GUIDED = "counterexample_guided"
SAMPLED_CEGIS = "sampled_cegis"
SAMPLED_CEGIS_FIXED = "sampled_cegis_fixed"
# B16 selection x termination factorial (Deployment and tau-bench; shared control loop).
ST2X2_PARTITIONED_STOP = "st2x2_partitioned_stop"
ST2X2_UNIFORM_STOP = "st2x2_uniform_stop"
ST2X2_PARTITIONED_EXHAUST = "st2x2_partitioned_exhaust"
ST2X2_UNIFORM_EXHAUST = "st2x2_uniform_exhaust"
ST2X2_METHODS = (ST2X2_PARTITIONED_STOP, ST2X2_UNIFORM_STOP, ST2X2_PARTITIONED_EXHAUST, ST2X2_UNIFORM_EXHAUST)
# Contract-repair comparison on Deployment (restricted patch + improvement-based adoption).
REPAIR_FREE_POOL = "repair_free_pool"        # A: fixed pool, whole-contract rewrite
REPAIR_PATCH_POOL = "repair_patch_pool"      # B: fixed pool, restricted patch, adopt on improvement
REPAIR_FREE_ACTIVE = "repair_free_active"    # C: candidate-partitioned, whole-contract rewrite
REPAIR_FREE_WITNESS = "repair_free_witness"  # A': free rewrite on witnessed fixed evidence
REPAIR_PATCH_WITNESS = "repair_patch_witness"  # B': restricted patch on the same evidence
REPAIR_PATCH_TIE = "repair_patch_tie"  # B'': same as B' but adopts on a non-increasing D_obs
REPAIR_CORE_METHODS = (REPAIR_FREE_POOL, REPAIR_PATCH_POOL, REPAIR_FREE_ACTIVE,
                       REPAIR_FREE_WITNESS, REPAIR_PATCH_WITNESS, REPAIR_PATCH_TIE)
ACTIVE_CEGIS = "active_cegis"
ACTIVE_CEGIS_NO_BALANCE = "active_cegis_no_balance"
ACTIVE_CEGIS_NO_COVERAGE = "active_cegis_no_coverage"
ACTIVE_CEGIS_UNIFORM = "active_cegis_uniform"
ASI_REPLAY = "asi_replay"
SKILLCOMMIT_REPLAY = "skillcommit_replay"
CONTRACTSKILL_REPAIR = "contractskill_repair"
CONTRACTSKILL_FULL_ORACLE = "contractskill_full_oracle"

METHODS: tuple[str, ...] = (
    DIRECT,
    SELF_REFINE,
    RANDOM_PROBE,
    COUNTEREXAMPLE_GUIDED,
    SAMPLED_CEGIS,
    ACTIVE_CEGIS,
    ACTIVE_CEGIS_NO_BALANCE,
    ACTIVE_CEGIS_NO_COVERAGE,
    ACTIVE_CEGIS_UNIFORM,
    ASI_REPLAY,
    SKILLCOMMIT_REPLAY,
    CONTRACTSKILL_REPAIR,
    CONTRACTSKILL_FULL_ORACLE,
)

# Reasons a method stopped before it was finished.
STOP_QUERY_BUDGET = "query_budget_exhausted"
STOP_TOKEN_BUDGET = "token_budget_exhausted"
STOP_STATE_BUDGET = "state_budget_exhausted"
STOP_TRANSPORT = "transport_error"
STOP_EXACT = "contract_is_exact"
STOP_SAMPLE_SATISFIED = "sampled_oracle_satisfied"
STOP_COMPLETE = "method_complete"

MAX_SOURCE_ECHO_CHARS = 4000


@dataclass(frozen=True)
class Decoding:
    """Sampling parameters shared by every method."""

    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 1024
    repetition_penalty: float | None = None

    def to_dict(self, seed: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": seed,
        }
        # 설정하지 않은 실행의 decoding_sha256 을 보존하려면 키 자체가 없어야 한다.
        if self.repetition_penalty is not None:
            payload["repetition_penalty"] = self.repetition_penalty
        return payload


@dataclass(frozen=True)
class Budgets:
    """The three allowances every method runs under."""

    state_budget: int = 24
    query_budget: int = 4
    token_budget: int = 16000

    def __post_init__(self) -> None:
        for name in ("state_budget", "query_budget", "token_budget"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")

    def to_dict(self) -> dict[str, int]:
        return {
            "state_budget": self.state_budget,
            "query_budget": self.query_budget,
            "token_budget": self.token_budget,
        }


@dataclass(frozen=True)
class RunSpec:
    """One (method, seed) run under one set of budgets."""

    method: str
    seed: int = 0
    decoding: Decoding = field(default_factory=Decoding)
    budgets: Budgets = field(default_factory=Budgets)
    use_guided_json: bool = False

    def __post_init__(self) -> None:
        # RunSpec is shared metadata; fixed-pool retail is not a Shopping method.
        if self.method not in METHODS + (SAMPLED_CEGIS_FIXED,) + ST2X2_METHODS + REPAIR_CORE_METHODS:
            raise ValueError(f"unknown method {self.method!r}; known methods: {list(METHODS)}")


class _Ledger:
    """Tracks spending against the query, token, and state budgets."""

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
        """Tokens this call may generate, or ``None`` if it may not happen.

        The token budget covers prompt *and* completion tokens, so clamping
        ``max_tokens`` to what is left is conservative: a call can still overrun
        slightly on prompt tokens, which the next check then catches.
        """
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
    """Mutable bookkeeping for one run: conversation, records, spending."""

    def __init__(self, runner: "ExperimentRunner", spec: RunSpec) -> None:
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

    # -- model calls -------------------------------------------------------

    def ask(self, role: str, user_text: str) -> Interaction | None:
        """Send one user turn, or return ``None`` when a budget forbids it."""
        allowance = self.ledger.allowance(self.spec.decoding.max_tokens)
        if allowance is None:
            self.stopped_because = self.ledger.stop_reason()
            return None

        self.messages.append({"role": "user", "content": user_text})
        sent = tuple(dict(message) for message in self.messages)
        index = len(self.interactions)
        try:
            response = self.runner.client.complete(
                self.messages,
                temperature=self.spec.decoding.temperature,
                top_p=self.spec.decoding.top_p,
                max_tokens=allowance,
                seed=self.spec.seed,
                guided_json=dsl.contract_json_schema() if self.spec.use_guided_json else None,
            )
        except TransportError as exc:
            self.ledger.charge_call(Usage())
            interaction = Interaction(index=index, role=role, messages=sent, error=str(exc))
            self.interactions.append(interaction)
            self.stopped_because = STOP_TRANSPORT
            return None

        self.ledger.charge_call(response.usage)
        self.messages.append({"role": "assistant", "content": response.text})
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
        )
        self.interactions.append(interaction)
        return interaction

    # -- contracts ---------------------------------------------------------

    def adopt(self, interaction: Interaction) -> ContractRecord:
        """Parse one answer, keeping it only if it is a well-formed contract.

        A malformed or out-of-grammar answer becomes a recorded parse failure and
        is never executed. A contract adopted earlier survives a later failure, so
        a refinement pass that produces garbage cannot destroy a working answer;
        when nothing has parsed yet, the failure itself becomes the run's result.
        """
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
        """What to echo back to the model: its contract, or its raw answer."""
        return record.source or "(the previous answer was empty)"


class ExperimentRunner:
    """Runs methods against one sandbox configuration and one model endpoint."""

    def __init__(
        self,
        config: EnvConfig,
        client: ChatClient,
        *,
        max_depth: int,
        max_states: int,
        config_path: str | None = None,
        serving_metadata: Mapping[str, Any] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.max_depth = max_depth
        self.max_states = max_states
        self.config_path = config_path
        self.serving_metadata = dict(serving_metadata or {})
        self._now = now if now is not None else (lambda: datetime.now(timezone.utc))

        enumeration = enumerate_reachable(config, max_depth=max_depth, max_states=max_states)
        self.states: tuple[ShoppingState, ...] = enumeration.states
        self.enumeration_summary = enumeration.summary()
        # The oracle. It scores every method's answer and never appears in a
        # prompt; only ``counterexample_guided`` consumes it as feedback.
        self.oracle = ground_truth_contract(config)

    # -- sandbox access ----------------------------------------------------

    def validate(self, parsed: dsl.ParsedContract) -> None:
        """Smoke-run a freshly parsed contract so scoring cannot blow up later.

        Compiled formulas are total by construction; this checks that promise on
        real states, and turns any surprise into a parse failure rather than an
        exception in the middle of an evaluation sweep.
        """
        contract = parsed.bind(self.config)
        probe = self.states[:2] if self.states else ()
        try:
            for state in probe:
                outcome = apply_action(state, ShoppingAction.place_order(), self.config)
                contract.holds_in(state)
                contract.transition_holds(state, outcome.state)
        except Exception as exc:  # pragma: no cover - defensive
            raise dsl.DslError(f"contract raised while being validated: {exc!r}") from exc

    def probe_states(self, seed: int, count: int) -> tuple[ShoppingState, ...]:
        """Sample states to attempt the action in, reproducibly for one seed."""
        if count <= 0 or not self.states:
            return ()
        rng = random.Random(f"probe|{seed}|{self.config.fingerprint!r}")
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
        excluded: Sequence[ShoppingState] = (),
        balance: bool = True,
        coverage: bool = True,
    ) -> tuple[ShoppingState, ...]:
        """Select diverse unseen states from both sides of a candidate.

        Selection uses only the candidate's prediction and raw state fields. It
        never executes ``place_order`` or consults the ground-truth contract.
        """
        if count <= 0:
            return ()
        excluded_set = set(excluded)
        pool = [state for state in self.states if state not in excluded_set]
        if not pool:
            return ()

        def select(
            candidates: Sequence[ShoppingState], quota: int, *, partition: str
        ) -> tuple[ShoppingState, ...]:
            tie_seed = (
                f"active|{partition}|{seed}|{audit_index}|"
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

        selected = list(
            select(predicted_accept, accept_quota, partition="accept")
        )
        selected.extend(
            select(predicted_reject, reject_quota, partition="reject")
        )

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

    def successful_states(self, seed: int, count: int) -> tuple[ShoppingState, ...]:
        """Sample states where place_order succeeds, reproducibly for one seed.

        Fairness disclosure: Selecting successful historical instances is treated as
        construction of an offline experience corpus; its pool-construction scan is
        not an online method query and is disclosed as a prototype simplification.
        """
        if count <= 0 or not self.states:
            return ()
        action = ShoppingAction.place_order()
        pool = [state for state in self.states if apply_action(state, action, self.config).ok]
        if not pool:
            return ()
        if count >= len(pool):
            return tuple(pool)
        rng = random.Random(f"success|{seed}|{self.config.fingerprint!r}")
        return tuple(rng.sample(pool, count))

    def observe(self, states: Sequence[ShoppingState]) -> tuple[dict[str, Any], ...]:
        """Attempt ``place_order`` in each state through the agent-facing API."""
        action = ShoppingAction.place_order()
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
        """Run the oracle evaluator over the whole enumerated state set.

        Fairness disclosure: Each CEGIS equivalence query scans the full reachable closure
        even though oracle_feedback_queries counts queries rather than individual state checks.
        """
        return evaluate_contract(
            parsed.bind(self.config),
            self.states,
            self.config,
            max_counterexamples=max_counterexamples,
        )

    # -- methods -----------------------------------------------------------

    def run(self, spec: RunSpec) -> RunArtifact:
        """Execute one method and return its artifact."""
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
            {"round": 0, "role": "propose", "contract_status": initial.status,
             "parse_error": initial.error}
        )

        # Exactly one critique/revision pass, and no evaluator feedback: the
        # model may only reconsider what it already said.
        follow_up = prompts.critique_prompt(session.latest_source(initial))
        if not initial.parsed:
            follow_up = prompts.parse_failure_note(initial.error or "") + "\n\n" + follow_up
        second = session.ask("revise", follow_up)
        if second is None:
            return
        revised = session.adopt(second)
        session.rounds.append(
            {"round": 1, "role": "revise", "contract_status": revised.status,
             "parse_error": revised.error}
        )
        # ``adopt`` only replaces the kept contract when the new answer parsed,
        # so a broken revision falls back to the initial proposal.
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
                    {"round": round_index, "role": role, "contract_status": record.status,
                     "parse_error": record.error}
                )
                prompt = prompts.parse_failure_note(record.error or "")
                round_index += 1
                continue

            # Feedback costs an oracle query, and every counterexample shown is a
            # state the method got to observe.
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
                # Nothing concrete left to show, either because the state budget
                # is spent or because the disagreement was not recorded.
                session.stopped_because = (
                    STOP_STATE_BUDGET if session.ledger.remaining_states <= 0 else STOP_COMPLETE
                )
                return
            prompt = prompts.counterexample_prompt(
                session.latest_source(record), metrics, counterexamples
            )
            round_index += 1

    def _sampled_cegis(self, session: _Session) -> None:
        """CEGIS over one fixed state sample instead of the full closure.

        The sample is identical to the one used by ``random_probe`` and
        ``contractskill_repair`` for the same seed and state budget. Reusing it
        across rounds keeps the information set fixed; final scoring still uses
        the full reachable closure in ``_finish``.
        """
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
                prompt = prompts.parse_failure_note(record.error or "")
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
        """Candidate-aware CEGIS under a cumulative environment-query budget."""
        self._active_cegis_strategy(
            session, method=ACTIVE_CEGIS, balance=True, coverage=True
        )

    def _active_cegis_no_balance(self, session: _Session) -> None:
        self._active_cegis_strategy(
            session, method=ACTIVE_CEGIS_NO_BALANCE, balance=False, coverage=True
        )

    def _active_cegis_no_coverage(self, session: _Session) -> None:
        self._active_cegis_strategy(
            session, method=ACTIVE_CEGIS_NO_COVERAGE, balance=True, coverage=False
        )

    def _active_cegis_uniform(self, session: _Session) -> None:
        self._active_cegis_strategy(
            session, method=ACTIVE_CEGIS_UNIFORM, balance=False, coverage=False
        )

    def _active_cegis_strategy(
        self,
        session: _Session,
        *,
        method: str,
        balance: bool,
        coverage: bool,
    ) -> None:
        """Run progressive CEGIS with one state-selection strategy."""
        total_budget = session.ledger.remaining_states
        if total_budget <= 0:
            session.stopped_because = STOP_STATE_BUDGET
            return
        audit_batches = max(session.spec.budgets.query_budget - 1, 1)
        batch_size = max((total_budget + audit_batches - 1) // audit_batches, 1)
        audited_states: list[ShoppingState] = []
        audit_index = 0
        round_index = 0
        prompt = prompts.direct_prompt(self.config)

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
                        "adaptation": method,
                        "evidence_policy": _active_evidence_policy(balance, coverage),
                        "contract_status": record.status,
                        "parse_error": record.error,
                        "states_observed": len(audited_states),
                    }
                )
                prompt = prompts.parse_failure_note(record.error or "")
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

                # Fresh states come first so newly exposed regions receive
                # priority when the counterexample record is capped.
                fresh_set = set(fresh_states)
                audit_states = tuple(fresh_states) + tuple(
                    state for state in audited_states if state not in fresh_set
                )
                contract = session.parsed.bind(self.config)
                report = evaluate_contract(
                    contract,
                    audit_states,
                    self.config,
                    max_counterexamples=MAX_RECORDED_COUNTEREXAMPLES,
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
                        "evidence_policy": _active_evidence_policy(balance, coverage),
                        "contract_status": record.status,
                        "fresh_states": len(fresh_states),
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
                    session.stopped_because = STOP_SAMPLE_SATISFIED
                    return

    def _asi_replay(self, session: _Session) -> None:
        if session.ledger.remaining_states < 1:
            session.stopped_because = STOP_STATE_BUDGET
            return
        states = self.successful_states(session.spec.seed, 1)
        if not states:
            session.stopped_because = STOP_STATE_BUDGET
            return
        observations = self.observe(states)
        session.ledger.observe_states(len(observations))
        session.rounds.append(
            {
                "round": 0,
                "role": "propose",
                "adaptation": "asi_replay",
                "evidence_policy": "single_successful_observation",
                "states_observed": len(observations),
                "accepted": len(observations),
                "observations": list(observations),
            }
        )
        interaction = session.ask("propose", prompts.asi_prompt(self.config, observations[0]))
        if interaction is None:
            return
        record = session.adopt(interaction)
        session.rounds[0]["contract_status"] = record.status
        if record.error:
            session.rounds[0]["parse_error"] = record.error
        session.stopped_because = session.stopped_because or STOP_COMPLETE

    def _skillcommit_replay(self, session: _Session) -> None:
        budget = session.ledger.remaining_states
        if budget <= 0:
            session.stopped_because = STOP_STATE_BUDGET
            return
        states = self.successful_states(session.spec.seed, budget)
        if not states:
            session.stopped_because = STOP_STATE_BUDGET
            return

        # Charge initial and replay instances once against shared state_budget.
        session.ledger.observe_states(len(states))
        initial_state = states[0]
        replay_states = states[1:]
        initial_obs = self.observe([initial_state])[0]

        # Round 0: Proposal from single initial historical successful instance.
        session.rounds.append(
            {
                "round": 0,
                "role": "propose",
                "adaptation": "skillcommit_replay",
                "evidence_policy": "proposal_from_single_successful_instance",
                "states_observed": len(states),
                "initial_instance": initial_obs,
                "replay_set_size": len(replay_states),
            }
        )
        interaction = session.ask(
            "propose", prompts.skillcommit_proposal_prompt(self.config, initial_obs)
        )
        if interaction is None:
            return
        record = session.adopt(interaction)
        session.rounds[0]["contract_status"] = record.status
        if record.error:
            session.rounds[0]["parse_error"] = record.error

        if not record.parsed:
            if session.ledger.remaining_calls <= 0:
                session.stopped_because = session.stopped_because or STOP_QUERY_BUDGET
                return
            revise_interaction = session.ask(
                "revise", prompts.parse_failure_note(record.error or "")
            )
            if revise_interaction is None:
                return
            record = session.adopt(revise_interaction)
            session.rounds.append(
                {
                    "round": 1,
                    "role": "revise",
                    "adaptation": "skillcommit_replay",
                    "evidence_policy": "parse_failure_recovery",
                    "contract_status": record.status,
                    "parse_error": record.error if record.error else None,
                }
            )
            if not record.parsed or not replay_states:
                session.stopped_because = session.stopped_because or STOP_COMPLETE
                return

        if not replay_states:
            session.stopped_because = session.stopped_because or STOP_COMPLETE
            return

        # Cross-instance validation against additional distinct historical successful executions.
        # Positive-only instances: oracle_feedback_queries remains 0.
        replay_report = evaluate_contract(
            session.parsed.bind(self.config),
            replay_states,
            self.config,
            max_counterexamples=MAX_RECORDED_COUNTEREXAMPLES,
        )
        has_incompatibilities = (
            replay_report.false_rejects > 0 or replay_report.postcondition_violations > 0
        )

        if not has_incompatibilities:
            # All historical replay instances compatible: early stop.
            session.rounds.append(
                {
                    "round": len(session.rounds),
                    "role": "validate",
                    "adaptation": "skillcommit_replay",
                    "evidence_policy": "cross_instance_replay_validation",
                    "contract_status": record.status,
                    "replay_states_checked": len(replay_states),
                    "incompatibilities": 0,
                    "compatible": True,
                }
            )
            session.stopped_because = STOP_COMPLETE
            return

        # Candidate has incompatibilities with distinct replay instances: revise within query budget.
        if session.ledger.remaining_calls <= 0:
            session.stopped_because = STOP_QUERY_BUDGET
            return

        incompatibilities = [
            counterexample_record(entry) for entry in replay_report.counterexamples
        ]
        metrics = {
            "false_rejects": replay_report.false_rejects,
            "postcondition_violations": replay_report.postcondition_violations,
            "replay_states_checked": len(replay_states),
        }
        revision_prompt = prompts.skillcommit_revision_prompt(
            session.latest_source(record), metrics, incompatibilities
        )
        revise_interaction = session.ask("revise", revision_prompt)
        if revise_interaction is None:
            return
        revised_record = session.adopt(revise_interaction)
        session.rounds.append(
            {
                "round": len(session.rounds),
                "role": "revise",
                "adaptation": "skillcommit_replay",
                "evidence_policy": "cross_instance_replay_revision",
                "contract_status": revised_record.status,
                "parse_error": revised_record.error if revised_record.error else None,
                "replay_states_checked": len(replay_states),
                "incompatibilities": (
                    replay_report.false_rejects + replay_report.postcondition_violations
                ),
                "incompatibility_details": incompatibilities,
            }
        )
        session.stopped_because = session.stopped_because or STOP_COMPLETE

    def _contractskill_repair(self, session: _Session) -> None:
        budget = session.ledger.remaining_states
        if budget <= 0:
            session.stopped_because = STOP_STATE_BUDGET
            return
        replay_states = self.probe_states(session.spec.seed, budget)
        if not replay_states:
            session.stopped_because = STOP_STATE_BUDGET
            return
        session.ledger.observe_states(len(replay_states))

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
                        "adaptation": "contractskill_repair",
                        "evidence_policy": "budget_limited_observed_replay_repair",
                        "contract_status": record.status,
                        "parse_error": record.error,
                        "replay_set_size": len(replay_states),
                    }
                )
                prompt = prompts.parse_failure_note(record.error or "")
                round_index += 1
                continue

            # Evaluate strictly against the budget-limited observed replay set.
            # This does NOT query the full-closure equivalence oracle.
            replay_report = evaluate_contract(
                session.parsed.bind(self.config),
                replay_states,
                self.config,
                max_counterexamples=MAX_RECORDED_COUNTEREXAMPLES,
            )
            replay_metrics = report_metrics(replay_report)
            replay_counterexamples = [
                counterexample_record(entry) for entry in replay_report.counterexamples
            ]
            session.rounds.append(
                {
                    "round": round_index,
                    "role": role,
                    "adaptation": "contractskill_repair",
                    "evidence_policy": "budget_limited_observed_replay_repair",
                    "contract_status": record.status,
                    "replay_metrics": replay_metrics,
                    "replay_counterexamples": replay_counterexamples,
                    "replay_set_size": len(replay_states),
                }
            )

            has_violations = (
                replay_metrics["false_accepts"] > 0
                or replay_metrics["false_rejects"] > 0
                or replay_metrics["postcondition_violations"] > 0
            )
            if not has_violations:
                session.stopped_because = STOP_COMPLETE
                return

            prompt = prompts.contractskill_repair_prompt(
                session.latest_source(record), replay_metrics, replay_counterexamples
            )
            round_index += 1

    def _contractskill_full_oracle(self, session: _Session) -> None:
        """ContractSkill-style repair with the complete equivalence oracle.

        This completes the oracle-by-repair-loop ablation: the prompt and
        minimal-repair framing follow ``contractskill_repair``, while candidate
        checking and state accounting follow full-closure CEGIS.
        """
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
                        "adaptation": CONTRACTSKILL_FULL_ORACLE,
                        "evidence_policy": "full_closure_contractskill_repair",
                        "contract_status": record.status,
                        "parse_error": record.error,
                    }
                )
                prompt = prompts.parse_failure_note(record.error or "")
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
                    "adaptation": CONTRACTSKILL_FULL_ORACLE,
                    "evidence_policy": "full_closure_contractskill_repair",
                    "contract_status": record.status,
                    "full_closure_metrics": metrics,
                    "full_closure_counterexamples": counterexamples,
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
            prompt = prompts.contractskill_full_oracle_prompt(
                session.latest_source(record), metrics, counterexamples
            )
            round_index += 1

    # -- artifact ----------------------------------------------------------

    def _finish(self, session: _Session) -> RunArtifact:
        spec = session.spec
        evaluation: EvaluationRecord | None = None
        if session.parsed is not None:
            # The final scoring pass is measurement, not method spending, so it
            # is deliberately not charged to any budget.
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

        hashes = {
            "sandbox_config_sha256": sha256_json(self.config.to_dict()),
            "decoding_sha256": sha256_json(spec.decoding.to_dict(spec.seed)),
            "budgets_sha256": sha256_json(spec.budgets.to_dict()),
            "protocol_sha256": sha256_json(
                {
                    "method": spec.method,
                    "system_prompt": prompts.system_prompt(),
                    "direct_prompt": prompts.direct_prompt(self.config),
                    "counterexample_format": COUNTEREXAMPLE_FORMAT_VERSION,
                }
            ),
        }

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
        """The scoring setup, recorded in every artifact."""
        return {
            "config_path": self.config_path,
            "config": self.config.to_dict(),
            "max_depth": self.max_depth,
            "max_states": self.max_states,
            "enumeration": self.enumeration_summary,
            "evaluation_states": len(self.states),
            "oracle_contract": self.oracle.name,
        }


def _active_evidence_policy(balance: bool, coverage: bool) -> str:
    return {
        (True, True): "candidate_partitioned_progressive_coverage",
        (False, True): "progressive_coverage_without_candidate_balance",
        (True, False): "candidate_partitioned_progressive_random",
        (False, False): "progressive_uniform_random",
    }[(balance, coverage)]


def _greedy_raw_coverage(
    states: Sequence[ShoppingState], count: int, *, tie_seed: str
) -> tuple[ShoppingState, ...]:
    """Greedily cover raw field values and pairs without outcome labels."""
    if count <= 0 or not states:
        return ()
    pool = list(states)
    random.Random(tie_seed).shuffle(pool)
    selected: list[ShoppingState] = []
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


_METHOD_BODIES: dict[str, Callable[[ExperimentRunner, _Session], None]] = {
    DIRECT: ExperimentRunner._direct,
    SELF_REFINE: ExperimentRunner._self_refine,
    RANDOM_PROBE: ExperimentRunner._random_probe,
    COUNTEREXAMPLE_GUIDED: ExperimentRunner._counterexample_guided,
    SAMPLED_CEGIS: ExperimentRunner._sampled_cegis,
    ACTIVE_CEGIS: ExperimentRunner._active_cegis,
    ACTIVE_CEGIS_NO_BALANCE: ExperimentRunner._active_cegis_no_balance,
    ACTIVE_CEGIS_NO_COVERAGE: ExperimentRunner._active_cegis_no_coverage,
    ACTIVE_CEGIS_UNIFORM: ExperimentRunner._active_cegis_uniform,
    ASI_REPLAY: ExperimentRunner._asi_replay,
    SKILLCOMMIT_REPLAY: ExperimentRunner._skillcommit_replay,
    CONTRACTSKILL_REPAIR: ExperimentRunner._contractskill_repair,
    CONTRACTSKILL_FULL_ORACLE: ExperimentRunner._contractskill_full_oracle,
}


def _truncate(text: str, limit: int = MAX_SOURCE_ECHO_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


__all__ = [
    "ACTIVE_CEGIS",
    "ACTIVE_CEGIS_NO_BALANCE",
    "ACTIVE_CEGIS_NO_COVERAGE",
    "ACTIVE_CEGIS_UNIFORM",
    "ASI_REPLAY",
    "CONTRACTSKILL_REPAIR",
    "CONTRACTSKILL_FULL_ORACLE",
    "COUNTEREXAMPLE_GUIDED",
    "DIRECT",
    "METHODS",
    "SAMPLED_CEGIS_FIXED",
    "ST2X2_METHODS",
    "ST2X2_PARTITIONED_STOP",
    "ST2X2_UNIFORM_STOP",
    "ST2X2_PARTITIONED_EXHAUST",
    "ST2X2_UNIFORM_EXHAUST",
    "REPAIR_CORE_METHODS",
    "REPAIR_FREE_POOL",
    "REPAIR_PATCH_POOL",
    "REPAIR_FREE_ACTIVE",
    "REPAIR_FREE_WITNESS",
    "REPAIR_PATCH_WITNESS",
    "REPAIR_PATCH_TIE",
    "RANDOM_PROBE",
    "SAMPLED_CEGIS",
    "SELF_REFINE",
    "SKILLCOMMIT_REPLAY",
    "Budgets",
    "Decoding",
    "ExperimentRunner",
    "RunSpec",
]
