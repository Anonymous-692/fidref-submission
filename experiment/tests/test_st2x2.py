#!/usr/bin/env python3
"""Regression tests for the B16 shared control loop (selection x termination).

These cover the paths the offline replay check could not reach: a clean first batch
followed by a counterexample, revision accounting, evidence retention across revisions,
parse failure, transport error, and budget exhaustion.
"""

from __future__ import annotations

import unittest

from ..modeling.st2x2 import (
    STOP_DUPLICATE,
    STOP_PARSE_FAILURE,
    STOP_QUERY_BUDGET,
    STOP_SAMPLE_SATISFIED,
    STOP_TRANSPORT,
    DomainAdapter,
    Report,
    run_st2x2,
)

STATES = list(range(60))
ACCEPTING = set(range(10))          # the reference accepts 0..9
LATE = {10, 11}                     # offending states the selector shows last


class _Candidate:
    """Accepts every state below ``k``."""

    def __init__(self, k: int) -> None:
        self.k = k

    def holds(self, s: int) -> bool:
        return s < self.k


def _parse(source: str) -> _Candidate:
    if not source.isdigit():
        raise ValueError(f"not a contract: {source!r}")
    return _Candidate(int(source))


def _score(c: _Candidate, subset) -> Report:
    fa = sum(1 for s in subset if c.holds(s) and s not in ACCEPTING)
    fr = sum(1 for s in subset if s in ACCEPTING and not c.holds(s))
    ces = [s for s in subset if c.holds(s) != (s in ACCEPTING)][:3]
    return Report(len(subset), fa, fr, 0, ces)


def _select(c, count, tie_seed, balance, excluded=()):
    """Deterministic, and it shows the offending states only after the first batch."""
    ex = set(excluded)
    # Order: clean states first, then the withheld offenders, then the rest. This makes the
    # first batch counterexample-free and the second batch decisive.
    def rank(s: int) -> tuple[int, int]:
        if s in LATE:
            return (1, s)
        return (0, s) if s < 18 else (2, s)

    pool = sorted((s for s in STATES if s not in ex), key=rank)
    if not balance:
        return pool[:count]
    pos = [s for s in pool if c.holds(s)]
    neg = [s for s in pool if not c.holds(s)]
    half = count // 2
    picked = pos[:half] + neg[: count - half]
    return picked or pool[:count]


def _adapter() -> DomainAdapter:
    return DomainAdapter(
        name="mock",
        states=STATES,
        direct_prompt=lambda: "DIRECT",
        parse=_parse,
        predicted_accepts=lambda c: sum(1 for s in STATES if c.holds(s)),
        score_subset=_score,
        score_closure=lambda c: _score(c, STATES),
        select=_select,
        counterexample_prompt=lambda src, rep: f"CE:{src}",
        parse_repair_prompt=lambda src, err: f"REPAIR:{src}",
        vacuity_prompt=lambda src: f"VACUITY:{src}",
    )


def _asker(responses):
    log = []

    def ask(role, prompt):
        log.append(role)
        if not responses:
            return "10"
        return responses.pop(0)

    return ask, log


class St2x2LoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = _adapter()

    def test_clean_batch_then_counterexample_costs_exactly_one_revision(self) -> None:
        ask, log = _asker(["10"])
        result = run_st2x2(
            self.adapter, ask=ask, seed=0, state_budget=48, query_budget=4,
            balance=False, exhaust=True, initial_source="12",
        )
        audits = [r for r in result["rounds"] if "audit" in r]
        first_round = [r for r in audits if r["round"] == 0]
        self.assertGreaterEqual(len(first_round), 2, "exhaust mode must audit more than once")
        self.assertEqual(first_round[0]["report"]["false_accepts"], 0, "first batch is clean")
        self.assertTrue(
            any(r["report"]["false_accepts"] > 0 for r in first_round[1:]),
            "a later batch must expose the withheld counterexamples",
        )
        self.assertEqual(log, ["revision_0"], "exactly one model call after the counterexample")
        self.assertEqual(result["model_calls"], 2, "round-0 replay plus one revision")

    def test_no_model_call_while_batches_stay_clean(self) -> None:
        ask, log = _asker([])
        result = run_st2x2(
            self.adapter, ask=ask, seed=0, state_budget=48, query_budget=4,
            balance=False, exhaust=True, initial_source="10",
        )
        self.assertEqual(log, [], "a clean audit must never call the model")
        self.assertEqual(result["stopped_because"], STOP_SAMPLE_SATISFIED)
        self.assertEqual(result["states_observed"], 48, "exhaust spends the whole state budget")

    def test_evidence_is_retained_across_a_revision(self) -> None:
        ask, _ = _asker(["10"])
        result = run_st2x2(
            self.adapter, ask=ask, seed=0, state_budget=48, query_budget=4,
            balance=False, exhaust=True, initial_source="12",
        )
        cumulative = [r["cumulative_states"] for r in result["rounds"] if "audit" in r]
        self.assertEqual(cumulative, sorted(cumulative), "observed evidence must never shrink")
        rounds = {r["round"] for r in result["rounds"] if "audit" in r}
        self.assertIn(1, rounds, "the revised candidate is audited too")
        after = [r for r in result["rounds"] if r.get("round") == 1 and "audit" in r]
        self.assertGreaterEqual(
            after[0]["cumulative_states"], max(
                r["cumulative_states"] for r in result["rounds"] if r.get("round") == 0 and "audit" in r
            ),
            "the revised candidate re-scores the evidence gathered before it",
        )

    def test_stop_mode_stops_on_the_first_clean_batch(self) -> None:
        ask, log = _asker([])
        result = run_st2x2(
            self.adapter, ask=ask, seed=0, state_budget=48, query_budget=4,
            balance=False, exhaust=False, initial_source="10",
        )
        audits = [r for r in result["rounds"] if "audit" in r]
        self.assertEqual(len(audits), 1)
        self.assertEqual(result["states_observed"], 16, "one batch only")
        self.assertEqual(log, [])

    def test_selection_changes_which_states_are_audited(self) -> None:
        """The two selection policies must draw different states from the same pool."""
        candidate = _parse("40")
        partitioned = set(_select(candidate, 16, "0|0|0", True, ()))
        uniform = set(_select(candidate, 16, "0|0|0", False, ()))
        self.assertNotEqual(
            partitioned, uniform, "partitioned and uniform selection must not audit the same set"
        )
        accepted_by_candidate = {s for s in partitioned if candidate.holds(s)}
        rejected_by_candidate = partitioned - accepted_by_candidate
        self.assertTrue(
            accepted_by_candidate and rejected_by_candidate,
            "the partitioned batch must contain both predicted accepts and predicted rejects",
        )

    def test_parse_failure_is_repaired_then_recorded(self) -> None:
        ask, log = _asker(["still bad"])
        result = run_st2x2(
            self.adapter, ask=ask, seed=0, state_budget=48, query_budget=4,
            balance=True, exhaust=False, initial_source="not a contract",
        )
        self.assertEqual(result["stopped_because"], STOP_PARSE_FAILURE)
        self.assertEqual(log, ["parse_repair_0"], "one repair attempt, then stop")

    def test_transport_error_terminates_the_run(self) -> None:
        def ask(role, prompt):
            return None

        result = run_st2x2(
            self.adapter, ask=ask, seed=0, state_budget=48, query_budget=4,
            balance=False, exhaust=False, initial_source="30",
        )
        self.assertEqual(result["stopped_because"], STOP_TRANSPORT)

    def test_query_budget_exhaustion_is_recorded(self) -> None:
        ask, log = _asker(["31", "32", "33", "34", "35"])
        result = run_st2x2(
            self.adapter, ask=ask, seed=0, state_budget=48, query_budget=2,
            balance=False, exhaust=False, initial_source="30",
        )
        self.assertEqual(result["stopped_because"], STOP_QUERY_BUDGET)
        self.assertLessEqual(result["model_calls"], 2)

    def test_duplicate_candidate_is_guarded(self) -> None:
        ask, _ = _asker(["30"])  # the model repeats the candidate it was asked to fix
        result = run_st2x2(
            self.adapter, ask=ask, seed=0, state_budget=48, query_budget=4,
            balance=False, exhaust=False, initial_source="30",
        )
        self.assertEqual(result["stopped_because"], STOP_DUPLICATE)

    def test_per_audit_counts_are_recorded(self) -> None:
        ask, _ = _asker([])
        result = run_st2x2(
            self.adapter, ask=ask, seed=0, state_budget=48, query_budget=4,
            balance=True, exhaust=True, initial_source="10",
        )
        audits = [r for r in result["rounds"] if "audit" in r]
        for entry in audits:
            self.assertIn("fresh_states", entry)
            self.assertIn("cumulative_states", entry)
        self.assertEqual(
            sum(e["fresh_states"] for e in audits), audits[-1]["cumulative_states"],
            "fresh counts must add up to the cumulative count",
        )


if __name__ == "__main__":
    unittest.main()
