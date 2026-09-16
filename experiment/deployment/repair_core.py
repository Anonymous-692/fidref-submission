#!/usr/bin/env python3
"""Shared control loop for the contract-repair comparison (conditions A, B, C).

One loop serves all three conditions so that the control flow cannot drift between them the
way the per-method loops did. What differs is declared up front:

    A  free_pool    fixed pool, whole-contract rewrite, always replace the incumbent
    B  patch_pool   fixed pool, restricted patch, adopt only on strict D_obs decrease
    C  free_active  candidate-partitioned re-selection, whole-contract rewrite, always replace

Adoption in B uses ``D_obs = FA + FR + PV`` measured on the *same* evidence set as the
incumbent. There is no weighted objective: cost and edit counts are recorded, never traded
against observed errors (PLAN.txt 3). Ties and regressions are rejected and the incumbent
stands. A rejected patch still costs its call.

The loop never sees the reference contract; closure scoring happens after the run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from . import dsl
from .patch import DEFAULT_INSERT_NODE_CAP, DEFAULT_PATCH_CAP, PatchError, apply_patch

COND_FREE_POOL = "repair_free_pool"            # A  uniform fixed pool, free rewrite
COND_PATCH_POOL = "repair_patch_pool"          # B  uniform fixed pool, restricted patch
COND_FREE_ACTIVE = "repair_free_active"        # C  candidate-partitioned, free rewrite
# A' and B': fixed evidence that is known to witness at least one observed error for the shared
# initial candidate. Same axis as A/B on "evidence is fixed", different on "evidence is
# informative", so that free rewrite and restricted patch can be compared where repair actually
# runs. These are NOT a uniform fixed pool and must not be reported as one.
COND_FREE_WITNESS = "repair_free_witness"      # A'
COND_PATCH_WITNESS = "repair_patch_witness"    # B'
# B'': identical to B' except that the adoption test admits observed ties. The offline audit
# of the 14B B' runs found that 9 of 12 uniquely rejected proposals improved the full closure
# while scoring an observed delta of exactly 0, so the strict rule's misses are all ties.
# Nothing else changes: same initial contract, same evidence pool, same patch grammar, same
# budget. Only the comparison operator differs.
COND_PATCH_TIE = "repair_patch_tie"            # B''
CONDITIONS = (COND_FREE_POOL, COND_PATCH_POOL, COND_FREE_ACTIVE,
              COND_FREE_WITNESS, COND_PATCH_WITNESS, COND_PATCH_TIE)
PATCH_CONDITIONS = (COND_PATCH_POOL, COND_PATCH_WITNESS, COND_PATCH_TIE)
# Conditions whose evidence is supplied by the caller (witness search).
WITNESS_CONDITIONS = (COND_FREE_WITNESS, COND_PATCH_WITNESS, COND_PATCH_TIE)
# Conditions that accept an observed tie.
TIE_TOLERANT_CONDITIONS = (COND_PATCH_TIE,)
ACTIVE_CONDITIONS = (COND_FREE_ACTIVE,)

STOP_POOL_CLEAN = "observed_pool_clean"
STOP_QUERY_BUDGET = "query_budget_exhausted"
STOP_STATE_BUDGET = "state_budget_exhausted"
STOP_NO_PROGRESS = "no_adopted_patch"
STOP_PARSE_FAILURE = "parse_failure_unrepaired"
STOP_TRANSPORT = "transport_error"
STOP_DUPLICATE = "duplicate_model_output"


@dataclass
class Observation:
    """Scoring of one contract over one evidence set. Observed evidence only."""

    states_checked: int
    false_accepts: int
    false_rejects: int
    postcondition_violations: int
    counterexamples: list[Any] = field(default_factory=list)

    @property
    def d_obs(self) -> int:
        return self.false_accepts + self.false_rejects + self.postcondition_violations

    def metrics(self) -> dict[str, int]:
        return {
            "states_checked": self.states_checked,
            "false_accepts": self.false_accepts,
            "false_rejects": self.false_rejects,
            "postcondition_violations": self.postcondition_violations,
            "d_obs": self.d_obs,
        }


@dataclass
class RepairAdapter:
    """Everything the loop needs from the domain."""

    parse: Callable[[str], Any]
    score_subset: Callable[[Any, Sequence[Any]], Observation]
    fixed_pool: Callable[[int, int], Sequence[Any]]           # (seed, count)
    select_active: Callable[..., Sequence[Any]]               # candidate-partitioned
    free_rewrite_prompt: Callable[[str, dict, Sequence[Any]], str]
    patch_prompt: Callable[[str, dict, Sequence[Any]], str]
    patch_repair_prompt: Callable[[str], str]
    parse_repair_prompt: Callable[[str, str], str]


def run_repair(
    adapter: RepairAdapter,
    *,
    ask: Callable[[str, str, str | None], str | None],
    condition: str,
    seed: int,
    state_budget: int,
    query_budget: int,
    initial_source: str,
    patch_cap: int = DEFAULT_PATCH_CAP,
    insert_node_cap: int = DEFAULT_INSERT_NODE_CAP,
    evidence: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Run one condition from a shared initial candidate.

    ``ask(role, prompt, schema_kind)`` performs one model call. ``schema_kind`` is
    ``"contract"``, ``"patch"`` or ``None`` so the caller can pick the guided-decoding schema.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    is_patch = condition in PATCH_CONDITIONS
    is_active = condition in ACTIVE_CONDITIONS
    allow_tie = condition in TIE_TOLERANT_CONDITIONS

    rounds: list[dict[str, Any]] = []
    calls = 0
    stopped: str | None = None
    seen_sources: set[str] = set()
    seen_patches: set[str] = set()

    audited: list[Any] = []
    audited_keys: set[Any] = set()

    def take(states: Sequence[Any]) -> int:
        fresh = 0
        for s in states:
            key = getattr(s, "key", s)
            if key in audited_keys:
                continue
            audited_keys.add(key)
            audited.append(s)
            fresh += 1
        return fresh

    # ---- evidence ---------------------------------------------------------------
    # A and B draw one fixed pool, candidate-independent, once. A' and B' receive a prepared
    # fixed evidence set from the caller. C re-selects against the current candidate each round.
    if evidence is not None:
        take(evidence)
    elif not is_active:
        take(adapter.fixed_pool(seed, state_budget))
        if not audited:
            return _finish(rounds, None, initial_source, STOP_STATE_BUDGET, calls, audited, condition)

    # ---- round 0: the shared initial candidate ----------------------------------
    incumbent_source = initial_source
    calls += 1  # charged identically in every condition
    try:
        incumbent = adapter.parse(incumbent_source)
    except dsl.DslError as exc:
        if query_budget - calls <= 0:
            return _finish(rounds, None, incumbent_source, STOP_PARSE_FAILURE, calls, audited, condition)
        raw = ask("parse_repair_0", adapter.parse_repair_prompt(incumbent_source, str(exc)), "contract")
        calls += 1
        if raw is None:
            return _finish(rounds, None, incumbent_source, STOP_TRANSPORT, calls, audited, condition)
        incumbent_source = raw
        try:
            incumbent = adapter.parse(incumbent_source)
        except dsl.DslError:
            return _finish(rounds, None, incumbent_source, STOP_PARSE_FAILURE, calls, audited, condition)
    seen_sources.add(incumbent_source)

    if is_active:
        take(adapter.select_active(incumbent, min(state_budget, _batch(state_budget, query_budget)),
                                   seed=seed, audit_index=0, excluded=()))

    incumbent_obs = adapter.score_subset(incumbent, tuple(audited))
    rounds.append({
        "round": 0, "role": "initial", "condition": condition,
        "observed": incumbent_obs.metrics(), "cumulative_states": len(audited),
        "incumbent_sha256": _sha(incumbent_source), "adopted": True,
    })

    round_index = 1
    candidate_source: str | None = None
    while True:
        if incumbent_obs.d_obs == 0:
            stopped = STOP_POOL_CLEAN
            break
        if query_budget - calls <= 0:
            stopped = STOP_QUERY_BUDGET
            break

        metrics = incumbent_obs.metrics()
        if is_patch:
            prompt = adapter.patch_prompt(incumbent_source, metrics, incumbent_obs.counterexamples)
            raw = ask(f"patch_{round_index}", prompt, "patch")
        else:
            prompt = adapter.free_rewrite_prompt(incumbent_source, metrics, incumbent_obs.counterexamples)
            raw = ask(f"revise_{round_index}", prompt, "contract")
        calls += 1
        if raw is None:
            stopped = STOP_TRANSPORT
            break

        entry: dict[str, Any] = {"round": round_index, "condition": condition,
                                 "incumbent_sha256": _sha(incumbent_source)}

        # ---- turn the response into a candidate --------------------------------
        candidate_source = None
        if is_patch:
            try:
                applied = apply_patch(incumbent_source, raw,
                                      patch_cap=patch_cap, insert_node_cap=insert_node_cap)
                candidate_source = applied.source
                entry.update({"patch_valid": True, "edits": applied.edits,
                              "operations": list(applied.operations),
                              "inserted_nodes": applied.inserted_nodes,
                              "patch_raw": raw[:2000] if isinstance(raw, str) else raw,
                              "ast_delta": list(applied.delta)})
            except PatchError as exc:
                entry.update({"patch_valid": False, "patch_error": str(exc),
                              "patch_raw": raw[:2000] if isinstance(raw, str) else raw,
                              "adopted": False})
                if query_budget - calls > 0:
                    retry = ask(f"patch_repair_{round_index}",
                                adapter.patch_repair_prompt(str(exc)), "patch")
                    calls += 1
                    entry["repair_attempted"] = True
                    if retry is None:
                        rounds.append(entry); stopped = STOP_TRANSPORT; break
                    try:
                        applied = apply_patch(incumbent_source, retry,
                                              patch_cap=patch_cap, insert_node_cap=insert_node_cap)
                        candidate_source = applied.source
                        entry.update({"patch_valid_after_repair": True, "edits": applied.edits,
                                      "operations": list(applied.operations),
                                      "inserted_nodes": applied.inserted_nodes,
                                      "ast_delta": list(applied.delta)})
                    except PatchError as exc2:
                        entry["patch_error_after_repair"] = str(exc2)
                if candidate_source is None:
                    rounds.append(entry)
                    round_index += 1
                    continue  # incumbent untouched; the attempt was still charged
        else:
            candidate_source = raw

        # ---- duplicate guard ----------------------------------------------------
        # The guard compares the *resulting contract*, not the patch text: the same patch
        # applied to a different incumbent yields a different contract and is not a repeat.
        # Both identities are recorded so the two can be told apart when reading artifacts.
        entry["patch_sha256"] = _norm_patch_sha(raw) if is_patch else None
        entry["applied_to_sha256"] = _sha(incumbent_source)
        entry["result_sha256"] = _sha(candidate_source)
        entry["patch_text_seen_before"] = (
            entry["patch_sha256"] in seen_patches if is_patch else None)
        if is_patch and entry["patch_sha256"] is not None:
            seen_patches.add(entry["patch_sha256"])
        if candidate_source in seen_sources:
            entry.update({"guard": "duplicate_output", "adopted": False,
                          "duplicate_basis": "identical resulting contract"})
            rounds.append(entry)
            stopped = STOP_DUPLICATE
            break
        seen_sources.add(candidate_source)

        # ---- parse the candidate -------------------------------------------------
        try:
            candidate = adapter.parse(candidate_source)
        except dsl.DslError as exc:
            entry.update({"candidate_parse_error": str(exc), "adopted": False})
            rounds.append(entry)
            round_index += 1
            continue  # incumbent stands; a broken proposal never becomes the final contract

        # ---- C reveals fresh states before scoring -------------------------------
        if is_active:
            allowance = min(_batch(state_budget, query_budget), max(state_budget - len(audited), 0))
            if allowance > 0:
                take(adapter.select_active(candidate, allowance, seed=seed,
                                           audit_index=round_index, excluded=tuple(audited)))

        # ---- score candidate and incumbent on the SAME evidence -------------------
        evidence = tuple(audited)
        candidate_obs = adapter.score_subset(candidate, evidence)
        incumbent_obs_here = (incumbent_obs if not is_active
                              else adapter.score_subset(incumbent, evidence))
        entry.update({"observed_candidate": candidate_obs.metrics(),
                      "observed_incumbent": incumbent_obs_here.metrics(),
                      "cumulative_states": len(audited)})

        # ---- adoption -------------------------------------------------------------
        if is_patch:
            if allow_tie:
                adopt = candidate_obs.d_obs <= incumbent_obs_here.d_obs
                entry["adoption_rule"] = "non_increasing_d_obs"
                entry["tie_adopted"] = (candidate_obs.d_obs == incumbent_obs_here.d_obs)
            else:
                adopt = candidate_obs.d_obs < incumbent_obs_here.d_obs   # strict decrease only
                entry["adoption_rule"] = "strict_d_obs_decrease"
        else:
            adopt = True                                             # A and C always replace
            entry["adoption_rule"] = "always_replace"
        entry["adopted"] = adopt
        entry["d_obs_delta"] = candidate_obs.d_obs - incumbent_obs_here.d_obs
        if adopt:
            incumbent_source, incumbent, incumbent_obs = candidate_source, candidate, candidate_obs
        else:
            incumbent_obs = incumbent_obs_here
        entry["incumbent_sha256_after"] = _sha(incumbent_source)
        rounds.append(entry)
        round_index += 1

    if stopped is None:
        stopped = STOP_QUERY_BUDGET
    return _finish(rounds, incumbent, incumbent_source, stopped, calls, audited, condition,
                   last_candidate=candidate_source)


def _batch(state_budget: int, query_budget: int) -> int:
    return max(-(-state_budget // max(query_budget - 1, 1)), 1)


def _norm_patch_sha(raw: Any) -> str | None:
    """Hash of the patch text normalised as JSON, so key order does not create false novelty."""
    import hashlib
    import json as _json
    if raw is None:
        return None
    try:
        canon = _json.dumps(_json.loads(raw), sort_keys=True)
    except Exception:  # noqa: BLE001 - unparsable patches hash as their raw text
        canon = str(raw)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _sha(text: str | None) -> str | None:
    if text is None:
        return None
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _finish(rounds, parsed, source, stopped, calls, audited, condition, last_candidate=None):
    adopted = sum(1 for r in rounds if r.get("adopted") and r.get("round", 0) > 0)
    proposed = sum(1 for r in rounds if r.get("round", 0) > 0)
    return {
        "rounds": rounds,
        "parsed": parsed,
        "source": source,                     # the incumbent; this is what gets scored
        "last_candidate_source": last_candidate,
        "stopped_because": stopped,
        "model_calls": calls,
        "states_observed": len(audited),
        "condition": condition,
        "patches_proposed": proposed,
        "patches_adopted": adopted,
        "patches_rejected": proposed - adopted,
        "total_edits": sum(r.get("edits", 0) for r in rounds),
    }
