#!/usr/bin/env python3
"""Read-only verification entry point for the anonymous latest-policy release."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

def release_root():
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "RELEASE_MANIFEST.json").is_file():
            return candidate
    # Source-tree development fallback; the release manifest appears only after export.
    if len(here.parents) > 3:
        return here.parents[3]
    raise RuntimeError("cannot locate release root containing RELEASE_MANIFEST.json")


ROOT = release_root()


def read(rel):
    return json.loads((ROOT / rel).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, tol=1e-10):
    return math.isclose(float(a), float(b), abs_tol=tol)


def manifest_gate():
    manifest = read("RELEASE_MANIFEST.json")
    raw = manifest["files"]
    entries = ({"path": p, **v} for p, v in raw.items()) if isinstance(raw, dict) else raw
    by_path = {}
    for item in entries:
        path = item["path"]
        assert path not in by_path, path
        target = ROOT / path
        assert target.is_file(), path
        actual = sha(target)
        expected = item.get("release_sha256") or item.get("sha256")
        assert actual == expected, (path, actual, expected)
        if item.get("bytes") is not None:
            assert target.stat().st_size == item["bytes"], path
        assert item.get("source_sha256"), f"missing source_sha256: {path}"
        by_path[path] = item
    return by_path


def artifact_metrics(path):
    d = read(path)
    metrics = (d.get("evaluation") or {}).get("metrics")
    valid = metrics is not None and (d.get("contract") or {}).get("status") == "parsed"
    if not valid:
        return False, False, Fraction(), Fraction()
    a = metrics["successes"]
    fa, fr, pv = (metrics[k] for k in ("false_accepts", "false_rejects", "postcondition_violations"))
    assert a > 0 and 0 <= pv <= a - fr
    iou = Fraction(100 * (a - fr), a + fa)
    cfr = Fraction(100 * (a - fr - pv), a + fa)
    exact = fa == fr == pv == 0
    assert exact == bool(metrics["exact"])
    return True, exact, iou, cfr


def main58_gate(files):
    policy = read("analysis/truncation_retry_20260918/integrated_summary.json")
    result = read("analysis/retry_manuscript_20260921/results.json")
    assert policy["verified_runs"] == policy["total"]["n"] == len(policy["runs"]) == 58
    mapping = {r["source"]: r["retry"] for r in policy["runs"]}
    assert len(mapping) == 58
    seen = set()
    assert len(result["cells"]) == 136
    for cell in result["cells"]:
        runs = cell["runs"]
        assert len(runs) == cell["n"] == 20
        assert sorted(r["seed"] for r in runs) == list(range(20))
        exact = 0
        iou = cfr = Fraction()
        for run in runs:
            source, selected = run["source_path"], run["path"]
            assert source in files and selected in files
            expected = mapping.get(source, source)
            assert selected == expected and run["replaced"] == (source in mapping)
            valid, ex, iv, cv = artifact_metrics(selected)
            assert valid == run["valid"] and ex == run["exact"]
            assert close(iv, run["iou"]) and close(cv, run["cfr"])
            exact += ex; iou += iv; cfr += cv
            if run["replaced"]: seen.add(source)
        assert exact == cell["exact"]
        assert close(iou / 20, cell["iou_mean"]) and close(cfr / 20, cell["cfr_mean"])
    assert seen == set(mapping)
    return {"cells": 136, "runs": 2720, "replacements": 58}


def pvalue(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n) if n else 1.0


def ablation52_gate(files):
    manifest = read("analysis/ablation_truncation_retry_20260921/manifest.json")
    result = read("analysis/ablation_retry_integration_20260922/results.json")
    jobs = {j["source"]: j for j in manifest["jobs"]}
    assert len(jobs) == len(result["selected"]) == result["summary"]["runs"] == 52
    for source, job in jobs.items():
        assert source in files
        assert files[source]["source_sha256"] == job["source_sha256"], source
    assert len(result["rows"]) == 18
    total = 0
    for row in result["rows"]:
        for arm, cell in row["cells"].items():
            runs = cell["runs"]
            assert len(runs) == 20 and sorted(r["seed"] for r in runs) == list(range(20))
            exact = 0; iou = cfr = Fraction()
            for run in runs:
                assert run["path"] in files and run["source"] in files
                valid, ex, iv, cv = artifact_metrics(run["path"])
                assert valid == run["valid"] and ex == run["exact"]
                exact += ex; iou += iv; cfr += cv; total += 1
            assert exact == cell["exact"]
            assert close(iou / 20, cell["iou_mean"]) and close(cfr / 20, cell["cfr_mean"])
        active = row["cells"]["active_cegis"]["runs"]
        for arm, stat in row["comparisons"].items():
            other = row["cells"][arm]["runs"]
            b = sum(a["exact"] and not o["exact"] for a, o in zip(active, other))
            c = sum(not a["exact"] and o["exact"] for a, o in zip(active, other))
            assert (b, c) == (stat["active_only"], stat["other_only"])
            assert close(pvalue(b, c), stat["raw_p"])
        ordered = sorted(row["comparisons"], key=lambda a: row["comparisons"][a]["raw_p"])
        running = 0.0
        for i, arm in enumerate(ordered):
            running = max(running, min(1.0, (3 - i) * row["comparisons"][arm]["raw_p"]))
            assert close(running, row["comparisons"][arm]["holm_p"])
    assert total == 1440
    return {"cells": 72, "runs": total, "replacements": 52, "comparisons": 54}


def evidence_gate(files):
    audit = read("analysis/completed_evidence_20260921/verification.json")
    api_audit = read("analysis/completed_evidence_20260921/hosted_api_raw_audit.json")
    expected = {"partial_pool_raw": 560, "api_raw": 1600, "calendar14b_raw": 40}
    counted = Counter()
    outcomes = Counter()
    partial = Counter()
    calendar_exact = Counter()
    api_rows = {name: [] for name in api_audit["suite_summaries"]}
    api_dirs = {
        "openai_json_mode_v1_20260916": "original",
        "openai_nano_json_mode_v1_20260916": "nano_original",
        "openai_deployment_extension_v1_20260916": "deployment_extension",
        "openai_terra_flex_v1_20260917": "terra_flex",
        "openai_mini_all_bench_v1_20260917": "mini_other",
        "openai_nano_all_bench_v1_20260917": "nano_other",
        "openai_mini_shopping_reasoning_v1_20260917": "reasoning",
        "openai_common_initial_v1_20260916": "common_initial",
    }
    for path, item in files.items():
        role = item.get("role")
        if role not in expected:
            continue
        d = read(path)
        counted[role] += 1
        outcomes[(role, d.get("outcome"))] += 1
        metrics = (d.get("evaluation") or {}).get("metrics") or {}
        stored_exact = bool(metrics.get("exact", False))
        assert stored_exact == (d.get("outcome") == "exact")
        if role == "partial_pool_raw":
            assert d["spend"]["oracle_feedback_queries"] == 0
            assert d["method"].startswith("st2x2_")
            assert d["seed"] in range(20)
            model = "gemma" if "/gemma/" in f"/{path}" else "qwen38"
            key = ("retail" if "_tau_" in path else "deployment", model,
                   d["partial_pool"]["size"], d["method"])
            partial[key, "n"] += 1
            partial[key, "exact"] += stored_exact
            partial[key, f"status:{d['contract']['status']}"] += 1
        elif role == "calendar14b_raw":
            assert d["method"] in ("random_probe", "sampled_cegis")
            assert d["seed"] in range(20)
            calendar_exact[d["method"]] += stored_exact
        else:
            assert d.get("outcome") is not None
            matches = [suite for dirname, suite in api_dirs.items() if f"/{dirname}/" in f"/{path}"]
            assert len(matches) == 1, path
            api_rows[matches[0]].append(d)
    assert dict(counted) == expected, (counted, expected)
    assert audit["retail_partial_pool"]["artifacts"] + audit["deployment_partial_pool"]["artifacts"] == 560
    assert audit["openai_api"]["total_result_files"] == audit["openai_api"]["expected_total"] == 1600
    assert audit["qwen14b_calendar_controls"]["artifacts"] == 40
    assert all(audit[k]["all_checks_passed"] for k in
               ("retail_partial_pool", "deployment_partial_pool", "openai_api", "qwen14b_calendar_controls"))
    for domain, section in (("retail", "retail_partial_pool"), ("deployment", "deployment_partial_pool")):
        for row in audit[section]["summaries"]:
            key = (domain, row["model"], row["size"], row["method"])
            assert partial[key, "n"] == row["n"]
            assert partial[key, "exact"] == row["exact"]
            assert {k.split(":", 1)[1]: v for (group, k), v in partial.items()
                    if group == key and k.startswith("status:")} == row["parse_status"]
    stored_calendar = audit["qwen14b_calendar_controls"]["outcomes"]
    actual_calendar = Counter(k[1] for k, v in outcomes.items() if k[0] == "calendar14b_raw" for _ in range(v))
    assert dict(actual_calendar) == stored_calendar
    assert dict(calendar_exact) == audit["qwen14b_calendar_controls"]["exact_by_method"]
    labels = {"direct": "Direct", "sampled_cegis": "Fixed-pool",
              "sampled_cegis_fixed": "Fixed-pool", "fixed_pool": "Fixed-pool",
              "active_cegis": "Active", "active_cegis_no_balance": "No-Balance"}
    expected_successes = {"deployment": 88, "taubench_retail": 14,
                          "calendar": 35, "shopping_price": 8}
    for suite, rows in api_rows.items():
        grouped = {}
        for d in rows:
            domain = "deployment" if suite == "common_initial" else d["sandbox"]["domain"]
            key = (d["model"], domain, labels.get(d["method"], d["method"]))
            grouped.setdefault(key, []).append(d)
        actual = []
        for (model, domain, method), cohort in sorted(grouped.items()):
            ious = []; cfrs = []; valid = 0; states = set(); successes = set()
            for d in cohort:
                met = (d.get("evaluation") or {}).get("metrics")
                parsed = (d.get("contract") or {}).get("status") == "parsed"
                if parsed:
                    assert met; valid += 1
                    a = met["successes"]; fa = met["false_accepts"]; fr = met["false_rejects"]
                    pv = met["postcondition_violations"]
                    assert a == expected_successes[domain] and 0 <= pv <= a - fr
                    states.add(met["states_checked"]); successes.add(a)
                    ious.append(float(Fraction(100 * (a - fr), a + fa)))
                    cfrs.append(float(Fraction(100 * (a - fr - pv), a + fa)))
                else:
                    assert d["outcome"] == "parse_failure"
                    ious.append(0.0); cfrs.append(0.0)
            actual.append({"model": model, "domain": domain, "method": method, "n": len(cohort),
                "exact": sum(d["outcome"] == "exact" for d in cohort),
                "parse_failure": sum(d["outcome"] == "parse_failure" for d in cohort),
                "valid_evaluation_n": valid, "states_checked_values": sorted(states),
                "successes_values": sorted(successes), "iou_mean": round(sum(ious) / len(cohort), 10),
                "cfr_mean": round(sum(cfrs) / len(cohort), 10),
                "outcomes": dict(Counter(d["outcome"] for d in cohort)),
                "api_seed_values": sorted({str((d.get("hosted_api") or {}).get("api_seed")) for d in cohort})})
        assert actual == api_audit["suite_summaries"][suite], suite
    return {**expected, "fresh_closure_replay": False}


def main():
    files = manifest_gate()
    report = {"release_files": len(files), "main_policy": main58_gate(files),
              "ablation_policy": ablation52_gate(files), "evidence": evidence_gate(files),
              "network_or_model_calls": False}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"release verification failed: {exc}", file=sys.stderr)
        raise
