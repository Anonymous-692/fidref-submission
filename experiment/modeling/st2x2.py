#!/usr/bin/env python3
"""Shared control loop for the B16 selection x termination factorial.

Four conditions, all with the coverage term OFF:

    selection   in {candidate-partitioned (balance=True), global uniform (balance=False)}
    termination in {stop on a counterexample-free batch, keep auditing until the
                    state budget is exhausted}

Everything else is held fixed by construction, because all four conditions run through
this one function: batch size, cumulative re-scoring, counterexample count, guards,
revision prompt, and budgets. Domains plug in through :class:`DomainAdapter`, so the
control flow cannot drift between Deployment and tau-bench the way the per-domain
loops did (2026-09-05 audit: Deployment already exhausted the budget, tau-bench stopped
on the first satisfied batch).

The initial contract is supplied from outside (``initial_source``) so that the four
conditions start from the identical round-0 candidate for a given (model, domain, seed);
its generation is still charged as one model call in every condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

SELECTION_PARTITIONED = "candidate_partitioned"
SELECTION_UNIFORM = "global_uniform"
TERMINATION_STOP = "stop_on_satisfied_batch"
TERMINATION_EXHAUST = "audit_until_state_budget"

STOP_SAMPLE_SATISFIED = "sampled_oracle_satisfied"
STOP_STATE_BUDGET = "state_budget_exhausted"
STOP_QUERY_BUDGET = "query_budget_exhausted"
STOP_DUPLICATE = "duplicate_model_output"
STOP_PARSE_FAILURE = "parse_failure_unrepaired"
STOP_VACUOUS = "vacuous_candidate_unrepaired"
STOP_TRANSPORT = "transport_error"


@dataclass
class Report:
    """Normalised scoring result over a set of states."""

    states_checked: int
    false_accepts: int
    false_rejects: int
    postcondition_violations: int
    counterexamples: list[Any] = field(default_factory=list)
    raw: Any = None  # the domain's own ContractReport, for prompt builders

    @property
    def defects(self) -> int:
        return self.false_accepts + self.false_rejects + self.postcondition_violations

    @property
    def clean(self) -> bool:
        return self.defects == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "states_checked": self.states_checked,
            "false_accepts": self.false_accepts,
            "false_rejects": self.false_rejects,
            "postcondition_violations": self.postcondition_violations,
            "counterexample_count": len(self.counterexamples),
        }


@dataclass
class DomainAdapter:
    """Everything the control loop needs from a domain, and nothing else."""

    name: str
    states: Sequence[Any]
    direct_prompt: Callable[[], str]
    parse: Callable[[str], Any]
    predicted_accepts: Callable[[Any], int]
    score_subset: Callable[[Any, Sequence[Any]], Report]
    score_closure: Callable[[Any], Report]
    select: Callable[..., Sequence[Any]]  # (contract, count, tie_seed, balance, excluded)
    counterexample_prompt: Callable[[str, Report], str]
    parse_repair_prompt: Callable[[str, str], str]
    vacuity_prompt: Callable[[str], str]


def batch_size_for(state_budget: int, query_budget: int) -> int:
    """One rule for every condition and domain: split the states over the audit rounds."""
    audit_batches = max(query_budget - 1, 1)
    return max(-(-state_budget // audit_batches), 1)  # ceil without importing math


def run_st2x2(
    adapter: DomainAdapter,
    *,
    ask: Callable[[str, str], str | None],
    seed: int,
    state_budget: int,
    query_budget: int,
    balance: bool,
    exhaust: bool,
    counterexample_limit: int = 3,
    vacuity_revision_limit: int = 1,
    initial_source: str | None = None,
) -> dict[str, Any]:
    """Run one condition. ``ask(role, prompt)`` performs one model call, or replays one.

    Returns a record with the rounds, the final contract source, and the ledger. The
    caller owns artifact writing, hashing, and closure evaluation of the final contract.
    """

    batch = batch_size_for(state_budget, query_budget)
    selection = SELECTION_PARTITIONED if balance else SELECTION_UNIFORM
    termination = TERMINATION_EXHAUST if exhaust else TERMINATION_STOP

    rounds: list[dict[str, Any]] = []
    audited: list[Any] = []
    audited_set: set[Any] = set()
    seen_sources: set[str] = set()
    calls = 0
    vacuity_used = 0
    parsed = None
    source: str | None = None
    stopped = None

    def remaining_states() -> int:
        return max(state_budget - len(audited), 0)

    def remaining_calls() -> int:
        return max(query_budget - calls, 0)

    # ---- round 0: the shared initial candidate -------------------------------------
    if initial_source is None:
        raw = ask("synthesis", adapter.direct_prompt())
        if raw is None:
            return _finish(rounds, None, None, STOP_TRANSPORT, calls, audited, batch, selection, termination)
        source = raw
    else:
        source = initial_source
    calls += 1  # charged identically whether generated here or replayed

    round_index = 0
    while True:
        try:
            parsed = adapter.parse(source)
        except Exception as exc:  # noqa: BLE001 - the domain raises its own DslError
            repair_prompt = adapter.parse_repair_prompt(source, str(exc))
            if remaining_calls() <= 0:
                stopped = STOP_PARSE_FAILURE
                break
            raw = ask(f"parse_repair_{round_index}", repair_prompt)
            calls += 1
            if raw is None:
                stopped = STOP_TRANSPORT
                break
            source = raw
            try:
                parsed = adapter.parse(source)
            except Exception:  # noqa: BLE001
                stopped = STOP_PARSE_FAILURE
                break

        if source in seen_sources:
            rounds.append({
                "round": round_index,
                "selection": selection,
                "termination": termination,
                "guard": "duplicate_output",
                "cumulative_states": len(audited),
            })
            stopped = STOP_DUPLICATE
            break
        seen_sources.add(source)

        accepts = adapter.predicted_accepts(parsed)
        if accepts == 0 and vacuity_used < vacuity_revision_limit and remaining_calls() > 0:
            vacuity_used += 1
            rounds.append({
                "round": round_index,
                "selection": selection,
                "termination": termination,
                "guard": "vacuous_candidate",
                "reachable_predicted_accepts": 0,
                "cumulative_states": len(audited),
            })
            raw = ask(f"vacuity_revision_{round_index}", adapter.vacuity_prompt(source))
            calls += 1
            if raw is None:
                stopped = STOP_TRANSPORT
                break
            source = raw
            round_index += 1
            continue

        # ---- audit phase: one batch, or many if this condition exhausts ------------
        report = None
        audit_index = 0
        while True:
            allowance = min(batch, remaining_states())
            fresh = tuple(
                adapter.select(
                    parsed,
                    allowance,
                    f"{seed}|{round_index}|{audit_index}",
                    balance,
                    tuple(audited),
                )
            ) if allowance > 0 else ()
            fresh = tuple(s for s in fresh if s not in audited_set)
            for s in fresh:
                audited_set.add(s)
                audited.append(s)

            if not audited:
                stopped = STOP_STATE_BUDGET
                break

            report = adapter.score_subset(parsed, tuple(audited))
            rounds.append({
                "round": round_index,
                "audit": audit_index,
                "selection": selection,
                "termination": termination,
                "fresh_states": len(fresh),
                "cumulative_states": len(audited),
                "reachable_predicted_accepts": accepts,
                "report": report.to_dict(),
            })
            audit_index += 1

            if not report.clean:
                break
            if not exhaust:
                stopped = STOP_VACUOUS if accepts == 0 else STOP_SAMPLE_SATISFIED
                break
            if remaining_states() <= 0 or not fresh:
                stopped = STOP_VACUOUS if accepts == 0 else STOP_SAMPLE_SATISFIED
                break
            # exhaust mode: keep auditing the same candidate, no model call

        if stopped is not None:
            break
        if remaining_calls() <= 0:
            stopped = STOP_QUERY_BUDGET
            break

        raw = ask(f"revision_{round_index}", adapter.counterexample_prompt(source, report))
        calls += 1
        if raw is None:
            stopped = STOP_TRANSPORT
            break
        source = raw
        round_index += 1

    return _finish(rounds, parsed, source, stopped, calls, audited, batch, selection, termination)


def _finish(rounds, parsed, source, stopped, calls, audited, batch, selection, termination):
    return {
        "rounds": rounds,
        "parsed": parsed,
        "source": source,
        "stopped_because": stopped,
        "model_calls": calls,
        "states_observed": len(audited),
        "batch_size": batch,
        "selection": selection,
        "termination": termination,
    }
