#!/usr/bin/env python3
"""Offline audit of sampled-oracle satisfaction against full closure.

The audit reads archived final contracts only.  It never calls a model or an
external service; ``ChatClient`` is present solely because the deployment
runner owns the deterministic candidate-aware state selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..modeling.client import ChatClient
from ..modeling.runner import ACTIVE_CEGIS_NO_COVERAGE, STOP_SAMPLE_SATISFIED
from . import dsl
from .config import DeploymentConfig
from .contracts import evaluate_contract
from .runner import DeploymentExperimentRunner

DEFAULT_INPUTS = (
    "results/deployment_active_cegis_sb48_s20_32b",
    "results/deployment_32b_run2_webhook/active_cegis",
    "results/deployment_32b_run2_webhook/active_cegis_no_coverage",
    "results/deployment_active_cegis_contextfix_pilot_32b/active_cegis",
)


def artifact_paths(inputs: Sequence[Path]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for entry in inputs:
        if entry.is_file():
            paths.append(entry)
        elif entry.is_dir():
            paths.extend(entry.glob("*__seed*.json"))
    return tuple(sorted(set(paths), key=str))


def progressive_sample(
    runner: DeploymentExperimentRunner,
    parsed: dsl.ParsedContract,
    *,
    seed: int,
    state_budget: int,
    query_budget: int,
    coverage: bool,
) -> tuple[Any, ...]:
    """Recreate the fixed-candidate progressive state selection schedule."""
    audit_batches = max(query_budget - 1, 1)
    batch_size = max((state_budget + audit_batches - 1) // audit_batches, 1)
    audited: list[Any] = []
    for audit_index in range(audit_batches):
        allowance = min(batch_size, max(state_budget - len(audited), 0))
        if allowance <= 0:
            break
        fresh = runner.candidate_aware_states(
            parsed,
            seed=seed,
            count=allowance,
            audit_index=audit_index,
            excluded=audited,
            balance=True,
            coverage=coverage,
        )
        if not fresh:
            break
        audited.extend(fresh)
    return tuple(audited)


def audit_artifact(
    path: Path,
    runner_cache: dict[tuple[str, int, int], DeploymentExperimentRunner] | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract_record = payload.get("contract") or {}
    if contract_record.get("status") != "parsed" or not isinstance(contract_record.get("spec"), Mapping):
        return {
            "artifact": str(path),
            "suite": str(path.parent),
            "method": payload.get("method"),
            "seed": payload.get("seed"),
            "contract_status": contract_record.get("status"),
            "audited": False,
        }

    sandbox = payload["sandbox"]
    config = DeploymentConfig.from_dict(sandbox["config"])
    max_depth = int(sandbox.get("max_depth", 20))
    max_states = int(sandbox.get("max_states", 15000))
    cache = runner_cache if runner_cache is not None else {}
    cache_key = (json.dumps(config.to_dict(), sort_keys=True), max_depth, max_states)
    runner = cache.get(cache_key)
    if runner is None:
        runner = DeploymentExperimentRunner(
            config,
            ChatClient(model="offline-oracle-audit"),
            max_depth=max_depth,
            max_states=max_states,
        )
        cache[cache_key] = runner
    parsed = dsl.parse_contract_dict(contract_record["spec"], name="audit.replayed")
    budgets = payload.get("budgets") or {}
    state_budget = int(budgets.get("state_budget", 48))
    query_budget = int(budgets.get("query_budget", 4))
    coverage = payload.get("method") != ACTIVE_CEGIS_NO_COVERAGE
    sampled_states = progressive_sample(
        runner,
        parsed,
        seed=int(payload["seed"]),
        state_budget=state_budget,
        query_budget=query_budget,
        coverage=coverage,
    )
    bound = parsed.bind(config)
    sampled = evaluate_contract(bound, sampled_states, config, max_counterexamples=0)
    full = evaluate_contract(bound, runner.states, config, max_counterexamples=0)
    spec_hash = hashlib.sha256(
        json.dumps(contract_record["spec"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    reported_satisfied = payload.get("stopped_because") == STOP_SAMPLE_SATISFIED
    return {
        "artifact": str(path),
        "suite": str(path.parent),
        "method": payload.get("method"),
        "seed": payload.get("seed"),
        "contract_status": "parsed",
        "contract_sha256": spec_hash,
        "audited": True,
        "coverage": coverage,
        "sample_states": len(sampled_states),
        "sample_oracle_successes": sampled.successes,
        "sample_exact": sampled.is_exact,
        "full_states": len(runner.states),
        "full_false_accepts": full.false_accepts,
        "full_false_rejects": full.false_rejects,
        "full_postcondition_violations": full.postcondition_violations,
        "full_exact": full.is_exact,
        "replay_false_satisfaction": sampled.is_exact and not full.is_exact,
        "reported_sampled_oracle_satisfied": reported_satisfied,
        "reported_false_satisfaction": reported_satisfied and not full.is_exact,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audited = [row for row in rows if row.get("audited")]
    unique = {row["contract_sha256"] for row in audited}
    by_suite: dict[str, Any] = {}
    for suite in sorted({str(row.get("suite")) for row in rows}):
        members = [row for row in rows if row.get("suite") == suite]
        parsed = [row for row in members if row.get("audited")]
        by_suite[suite] = {
            "artifacts": len(members),
            "parsed_final_contracts": len(parsed),
            "sample_exact": sum(bool(row.get("sample_exact")) for row in parsed),
            "replay_false_satisfaction": sum(
                bool(row.get("replay_false_satisfaction")) for row in parsed
            ),
            "reported_sampled_oracle_satisfied": sum(
                bool(row.get("reported_sampled_oracle_satisfied")) for row in parsed
            ),
            "reported_false_satisfaction": sum(
                bool(row.get("reported_false_satisfaction")) for row in parsed
            ),
        }
    return {
        "artifacts": len(rows),
        "contract_statuses": dict(Counter(str(row.get("contract_status")) for row in rows)),
        "parsed_final_contracts": len(audited),
        "unique_parsed_contracts": len(unique),
        "full_exact": sum(bool(row.get("full_exact")) for row in audited),
        "sample_exact": sum(bool(row.get("sample_exact")) for row in audited),
        "replay_false_satisfaction": sum(
            bool(row.get("replay_false_satisfaction")) for row in audited
        ),
        "reported_sampled_oracle_satisfied": sum(
            bool(row.get("reported_sampled_oracle_satisfied")) for row in audited
        ),
        "reported_false_satisfaction": sum(
            bool(row.get("reported_false_satisfaction")) for row in audited
        ),
        "by_suite": by_suite,
    }


def run(inputs: Sequence[Path]) -> dict[str, Any]:
    runner_cache: dict[tuple[str, int, int], DeploymentExperimentRunner] = {}
    rows = [audit_artifact(path, runner_cache) for path in artifact_paths(inputs)]
    return {"summary": summarize(rows), "rows": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=[Path(p) for p in DEFAULT_INPUTS])
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args.inputs)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
