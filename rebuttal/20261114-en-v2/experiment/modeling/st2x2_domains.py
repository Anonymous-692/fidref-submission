#!/usr/bin/env python3
"""Domain adapters for the B16 selection x termination factorial.

Each adapter exposes exactly the primitives :mod:`experiment.modeling.st2x2` needs, so the
control flow lives in one place and cannot differ between domains. The adapters reuse the
existing runners' enumeration, DSL, scoring, selector, and prompts; nothing here changes
how any existing method behaves.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from .st2x2 import DomainAdapter, Report


def _report_from(rep: Any, states_checked: int, limit: int) -> Report:
    """Normalise a domain ContractReport into the loop's Report."""
    ces = list(getattr(rep, "counterexamples", []) or [])[:limit]
    return Report(
        states_checked=states_checked,
        false_accepts=int(rep.false_accepts),
        false_rejects=int(rep.false_rejects),
        postcondition_violations=int(rep.postcondition_violations),
        counterexamples=ces,
        raw=rep,
    )


def taubench_adapter(runner: Any, counterexample_limit: int = 3) -> DomainAdapter:
    from ..taubench_retail import prompts as p
    from ..taubench_retail.dsl import parse_contract

    cfg = runner.config

    def parse(text: str):
        return parse_contract(text, cfg, vocabulary_protocol=runner.vocabulary_protocol)

    def predicted_accepts(contract) -> int:
        return sum(1 for s in runner.states if contract.holds_in(s))

    def score_subset(contract, subset: Sequence[Any]) -> Report:
        rep = runner.score(contract, tuple(subset))
        return _report_from(rep, len(subset), counterexample_limit)

    def score_closure(contract) -> Report:
        rep = runner.score(contract)
        return _report_from(rep, len(runner.states), counterexample_limit)

    def select(contract, count: int, tie_seed: str, balance: bool, excluded: Sequence[Any]):
        return runner._select_active_sample(
            contract, count, tie_seed, balance=balance, coverage=False, excluded=set(excluded)
        )

    def counterexample_prompt(source: str, report: Report) -> str:
        return p.counterexample_prompt_v3(source, report.raw, cfg, counterexample_limit)

    return DomainAdapter(
        name="taubench_retail",
        states=runner.states,
        direct_prompt=lambda: p.direct_prompt(cfg, runner.vocabulary_protocol),
        parse=parse,
        predicted_accepts=predicted_accepts,
        score_subset=score_subset,
        score_closure=score_closure,
        select=select,
        counterexample_prompt=counterexample_prompt,
        parse_repair_prompt=lambda src, err: p.parse_repair_prompt_v3(src, err, cfg),
        vacuity_prompt=lambda src: p.vacuity_revision_prompt_v3(src, cfg),
    )


def deployment_adapter(runner: Any, counterexample_limit: int = 3) -> DomainAdapter:
    from ..deployment import prompts as p
    from ..deployment import dsl
    from ..deployment.contracts import evaluate_contract

    cfg = runner.config

    def parse(text: str):
        return dsl.parse_contract_text(text)

    def predicted_accepts(parsed) -> int:
        bound = parsed.bind(cfg)
        return sum(1 for s in runner.states if bound.holds_in(s))

    def score_subset(parsed, subset: Sequence[Any]) -> Report:
        rep = evaluate_contract(
            parsed.bind(cfg), tuple(subset), cfg, max_counterexamples=counterexample_limit
        )
        return _report_from(rep, len(subset), counterexample_limit)

    def score_closure(parsed) -> Report:
        rep = evaluate_contract(
            parsed.bind(cfg), runner.states, cfg, max_counterexamples=counterexample_limit
        )
        return _report_from(rep, len(runner.states), counterexample_limit)

    def select(parsed, count: int, tie_seed: str, balance: bool, excluded: Sequence[Any]):
        # The deployment selector derives its own tie seed from (seed, audit_index); we pass
        # the loop's round/audit counters through those fields so the stream stays deterministic.
        seed_part, round_part, audit_part = tie_seed.split("|")
        # Deterministic: the selector's tie seed must not depend on process-level hashing.
        return runner.candidate_aware_states(
            parsed,
            seed=int(seed_part),
            count=count,
            audit_index=int(round_part) * 1000 + int(audit_part),
            excluded=tuple(excluded),
            balance=balance,
            coverage=False,
        )

    def counterexample_prompt(source: str, report: Report) -> str:
        metrics = {
            "false_accepts": report.false_accepts,
            "false_rejects": report.false_rejects,
            "postcondition_violations": report.postcondition_violations,
            "states_checked": report.states_checked,
        }
        ces = [c.to_dict() if hasattr(c, "to_dict") else c for c in report.counterexamples]
        return p.counterexample_prompt(source, metrics, ces)

    return DomainAdapter(
        name="deployment",
        states=runner.states,
        direct_prompt=lambda: p.direct_prompt(cfg),
        parse=parse,
        predicted_accepts=predicted_accepts,
        score_subset=score_subset,
        score_closure=score_closure,
        select=select,
        counterexample_prompt=counterexample_prompt,
        parse_repair_prompt=lambda src, err: p.parse_repair_prompt(cfg, error=err, previous_output=src),
        vacuity_prompt=lambda src: p.vacuity_revision_prompt(
            cfg, previous_contract=src, reachable_states=len(runner.states)
        ),
    )
