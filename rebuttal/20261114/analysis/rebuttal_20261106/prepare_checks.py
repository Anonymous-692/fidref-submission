"""Offline preparation only: saved-result aggregation and evaluator-only DSL checks.

No network clients, credentials, model calls, or experiment-runner mutation.
The reference below must never be supplied to a synthesis prompt.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def table9():
    rows = []
    for directory in (
        "full_closure_default_s20_32b", "full_closure_default_s20",
        "full_closure_literature_s20_32b", "full_closure_contractskill_full_oracle_s20_32b",
    ):
        groups = defaultdict(list)
        for path in sorted((ROOT / "results" / directory).glob("*__seed*.json")):
            record = json.loads(path.read_text())
            groups[record["method"]].append((path, record))
        assert groups, directory
        for method, pairs in sorted(groups.items()):
            assert sorted(r["seed"] for _, r in pairs) == list(range(20))
            rows.append({
                "directory": directory, "method": method, "n": len(pairs),
                "exact": sum(bool((r.get("evaluation") or {}).get("exact")) for _, r in pairs),
                "mean_calls": statistics.mean(r["spend"]["model_calls"] for _, r in pairs),
                "oracle_queries": dict(Counter(str(r["spend"]["oracle_feedback_queries"]) for _, r in pairs)),
                "sources": [{"path": str(p.relative_to(ROOT)),
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p, _ in pairs],
            })
    return rows


def var(name, when="before"):
    return {"var": name, "when": when}


def eq(left, right):
    return {"op": "eq", "left": left, "right": right}


def reference_spec(config):
    """Evaluator-only reference using literals, never policy-set handles."""
    pre = [
        {"op": "is_true", "arg": var("authenticated")},
        eq(var("deployment_status"), {"const": "idle"}),
        {"op": "not", "arg": {"op": "is_empty", "arg": var("allocated_resources")}},
        {"op": "or", "args": [eq(var("target_region"), {"const": x}) for x in config.valid_regions]},
        {"op": "or", "args": [eq(var("cluster_tier"), {"const": x}) for x in config.valid_cluster_tiers]},
        {"op": "covers", "available": var("available_quota"), "required": var("allocated_resources")},
    ]
    post = [
        eq(var("deployment_status", "after"), {"const": "deployed"}),
        eq(var("active_deployment", "after"), var("allocated_resources")),
        eq(var("available_quota", "after"), {"op": "difference", "left": var("available_quota"),
                                               "right": var("allocated_resources")}),
        {"op": "is_empty", "arg": var("allocated_resources", "after")},
        {"op": "unchanged", "vars": ["authenticated", "target_region", "cluster_tier"]},
    ]
    return {"skill": "deploy_service", "precondition": {"op": "and", "args": pre},
            "postcondition": {"op": "and", "args": post}}


class NoPolicyAccess:
    def __getattr__(self, name):
        raise AssertionError(f"reference expression accessed config: {name}")


def expression_gate(*, restricted=False):
    from experiment.deployment.config import DeploymentConfig
    from experiment.deployment.contracts import evaluate_contract
    from experiment.deployment.dsl import parse_contract_dict
    from experiment.deployment.enumeration import enumerate_reachable
    from experiment.deployment.env import apply_action
    from experiment.deployment.information_policy import InformationPolicy

    parse_options = {"information_policy": InformationPolicy("restricted_vp", False, False)} if restricted else {}

    original = DeploymentConfig.from_json_file(ROOT / "experiment/configs/deployment_default.json")
    neutral = replace(original, valid_regions=("region_a", "region_c"), rejected_regions=("region_b",),
                      valid_cluster_tiers=("tier_b", "tier_c"), rejected_cluster_tiers=("tier_a",))
    results = {}
    for name, config in (("original", original), ("neutral_candidate_v1", neutral)):
        closure = enumerate_reachable(config, max_depth=20, max_states=15000)
        assert not closure.truncated
        spec = reference_spec(config)
        encoded = json.dumps(spec, sort_keys=True)
        assert '"in_set"' not in encoded and '"set"' not in encoded
        parsed = parse_contract_dict(spec, **parse_options)
        contract = parsed.bind(NoPolicyAccess())
        report = evaluate_contract(contract, closure.states, config)
        assert report.is_exact
        clauses = [parse_contract_dict({**spec, "precondition": c}, **parse_options).bind(NoPolicyAccess())
                   for c in spec["precondition"]["args"]]
        witnesses = [0] * len(clauses)
        for state in closure.states:
            values = [c.holds_in(state) for c in clauses]
            if values.count(False) == 1:
                witnesses[values.index(False)] += 1
        assert all(witnesses), witnesses
        vacuous = copy.deepcopy(spec)
        vacuous["postcondition"] = {"op": "const", "value": True}
        vacuous_contract = parse_contract_dict(vacuous, **parse_options).bind(NoPolicyAccess())
        assert evaluate_contract(vacuous_contract, closure.states, config).is_exact
        faults = rejected = vacuous_rejected = 0
        for state in closure.states:
            if not contract.holds_in(state):
                continue
            outcome = apply_action(state, contract.action, config)
            assert outcome.ok
            wrong = outcome.state.evolve(authenticated=not outcome.state.authenticated)
            faults += 1
            rejected += not contract.transition_holds(state, wrong)
            vacuous_rejected += not vacuous_contract.transition_holds(state, wrong)
        assert faults > 0 and rejected == faults and vacuous_rejected == 0
        results[name] = {"closure": closure.summary(), "evaluation": report.summary(),
                         "clause_witness_counts": witnesses, "dsl_nodes": parsed.node_count,
                         "reference_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                         "single_field_fault_smoke": {"n": faults, "reference_rejected": rejected,
                                                      "true_post_rejected": vacuous_rejected}}
    assert results["original"]["clause_witness_counts"] == results["neutral_candidate_v1"]["clause_witness_counts"]
    assert results["original"]["closure"]["states"] == results["neutral_candidate_v1"]["closure"]["states"]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restricted", action="store_true", help="exercise the implemented restricted parser")
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve previous audit output; choose a new path."
    output = {"scope": "offline preparation, not a model experiment or complete launch gate",
              "restricted_parser": args.restricted,
              "table9": table9(), "expression_gate": expression_gate(restricted=args.restricted),
              "pending": ["complete bounded fault audit", "serving and protocol freeze",
                          "live-server smoke test after configuration freeze"]}
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"table9_cells": len(output["table9"]), "expression_gate": output["expression_gate"],
                      "pending": output["pending"]}, indent=2))


if __name__ == "__main__":
    main()
