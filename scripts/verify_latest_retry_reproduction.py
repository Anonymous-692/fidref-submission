#!/usr/bin/env python3
"""Verify the explicit 58+52 retry lineage without reading paper tables."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-results", required=True)
    args = parser.parse_args()
    rebuilt = read(ROOT / args.main_results)
    canonical = read(ROOT / "analysis/retry_manuscript_20260921/results.json")
    manifest58 = read(ROOT / "analysis/truncation_retry_20260918/integrated_summary.json")
    manifest52 = read(ROOT / "analysis/ablation_truncation_retry_20260921/manifest.json")
    target52 = ROOT / "analysis/retry_manuscript_20260921/ablation_table5_length_counts.json"

    assert rebuilt == canonical, "58-run regenerated aggregation differs from the canonical latest result"
    assert manifest58["verified_runs"] == manifest58["total"]["n"] == len(manifest58["runs"]) == 58
    assert len({row["source"] for row in manifest58["runs"]}) == 58
    assert len(manifest52["jobs"]) == 52
    assert manifest52["target_sha256"] == sha(target52)
    assert len({job["source"] for job in manifest52["jobs"]}) == 52
    print(json.dumps({
        "main_policy_runs": 58,
        "ablation_policy_runs": 52,
        "main_cells": len(rebuilt["cells"]),
        "main_results_match": True,
        "ablation_target_sha256": sha(target52),
    }, indent=2))


if __name__ == "__main__":
    main()
