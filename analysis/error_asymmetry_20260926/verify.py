"""Recompute the manuscript's seven descriptive FA>FR counts; no model calls.

Run from the repository or restored anonymous release with --root PATH.
The two cohorts overlap: their denominators must never be added together.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

MAIN = "analysis/retry_manuscript_20260921/results.json"
ABLATION = "analysis/ablation_retry_integration_20260922/results.json"
RETRY = "analysis/truncation_retry_20260918/integrated_summary.json"
EXPECTED = {
    "main": {"Fixed-pool": [270, 391], "Random": [187, 278], "Active": [89, 280]},
    "ablation": {"No-Balance": [265, 321], "Uniform": [259, 307],
                 "Active": [52, 127], "No-Coverage": [57, 128]},
}
LABELS = {"sampled_cegis": "Fixed-pool", "sampled_cegis_fixed": "Fixed-pool",
          "random_probe": "Random", "active_cegis": "Active",
          "active_cegis_no_balance": "No-Balance",
          "active_cegis_no_coverage": "No-Coverage", "active_cegis_uniform": "Uniform"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def classify(valid, fa=None, fr=None, pv=None):
    if not valid:
        return "invalid"
    if fa == fr == pv == 0:
        return "exact"
    return "fa_gt_fr" if fa > fr else "fa_lt_fr" if fa < fr else "fa_eq_fr"


def verify(root):
    release = root / "RELEASE_MANIFEST.json"
    files = json.loads(release.read_text())["files"] if release.exists() else None

    def read(rel, expected_sha=None):
        path = Path(rel)
        assert not path.is_absolute() and ".." not in path.parts, rel
        raw = (root / path).read_bytes()
        actual = sha(raw)
        source_sha = actual
        if files is not None:
            entry = files[rel]
            assert actual == entry["sha256"] and len(raw) == entry["bytes"], rel
            source_sha = entry["source_sha256"]
        if expected_sha:
            assert source_sha == expected_sha, (rel, "source hash mismatch")
        return json.loads(raw), source_sha

    main, main_sha = read(MAIN)
    abl, abl_sha = read(ABLATION)
    retry, retry_sha = read(RETRY)
    replacements = {r["source"]: r["retry"] for r in retry["runs"]}
    assert len(replacements) == 58
    ab_replacements = {r["old"]["source"]: r["new"]["path"] for r in abl["selected"]}
    assert len(ab_replacements) == 52
    assert not (set(replacements) & set(ab_replacements))
    assert len(main["cells"]) == 136 and len(abl["rows"]) == 18
    stats = defaultdict(Counter)
    records, cell_summary = [], []
    seen = set()

    def cell(cohort, model, domain, method, value, mapping):
        runs = value["runs"]
        assert len(runs) == value["n"] == 20
        assert sorted(r["seed"] for r in runs) == list(range(20))
        # Historical Retail sampled_cegis was partitioned, not a fixed pool.
        assert not (domain == "taubench_retail" and method == "sampled_cegis")
        assert not (cohort == "main" and domain == "calendar"
                    and method in {"random_probe", "sampled_cegis", "sampled_cegis_fixed"})
        counts = Counter()
        for run in runs:
            rel = run["path"]
            assert (cohort, rel) not in seen, (cohort, rel)
            seen.add((cohort, rel))
            source = run.get("source_path", run.get("source"))
            assert rel == mapping.get(source, source), (source, rel)
            raw, original_sha = read(rel, run["sha256"])
            assert raw["seed"] == run["seed"]
            metrics = (raw.get("evaluation") or {}).get("metrics")
            valid = metrics is not None and (raw.get("contract") or {}).get("status") == "parsed"
            assert valid == run["valid"], rel
            fa = fr = pv = None
            if valid:
                fa, fr, pv = [metrics[k] for k in
                              ("false_accepts", "false_rejects", "postcondition_violations")]
                assert all(type(v) is int and v >= 0 for v in (fa, fr, pv))
                assert pv <= metrics["successes"] - fr
                assert bool(metrics["exact"]) == (fa == fr == pv == 0)
                if cohort == "main":
                    assert [run[k] for k in ("FA", "FR", "PV")] == [fa, fr, pv]
            category = classify(valid, fa, fr, pv)
            assert run["exact"] == (category == "exact"), rel
            counts[category] += 1
            label = LABELS.get(method, method)
            stats[(cohort, label)][category] += 1
            records.append(dict(cohort=cohort, model=model, domain=domain, method=method,
                                seed=run["seed"], source=source, path=rel,
                                source_sha256=original_sha, valid=valid, FA=fa, FR=fr, PV=pv,
                                category=category))
        assert counts["exact"] == value["exact"]
        cell_summary.append(dict(cohort=cohort, model=model, domain=domain, method=method,
                                 counts=dict(sorted(counts.items()))))

    for c in main["cells"]:
        cell("main", c["model"], c["domain"], c["method"], c, replacements)
    for row in abl["rows"]:
        for method, c in row["cells"].items():
            cell("ablation", row["model"], row["domain"], method, c,
                 replacements | ab_replacements)
    assert len(records) == 4160 and len(cell_summary) == 208
    claims = {}
    for cohort, expected in EXPECTED.items():
        claims[cohort] = {}
        for method, pair in expected.items():
            counts = stats[(cohort, method)]
            num = counts["fa_gt_fr"]
            den = sum(counts[k] for k in ("fa_gt_fr", "fa_lt_fr", "fa_eq_fr"))
            assert [num, den] == pair, (cohort, method, num, den, pair)
            claims[cohort][method] = dict(numerator=num, denominator=den,
                                         counts=dict(sorted(counts.items())))
    return dict(status="pass", model_calls=0, raw_artifacts_modified=False,
                definition="FA > FR among parsed, evaluated, inexact final contracts",
                scope="Two overlapping descriptive cohorts; no pooled total or new hypothesis tests",
                exclusions="Invalid and exact contracts excluded from the ratio; FA=FR stays in denominator",
                inputs={MAIN: main_sha, ABLATION: abl_sha, RETRY: retry_sha},
                reviewed_cells=len(cell_summary), reviewed_run_occurrences=len(records),
                distinct_artifact_paths=len({r["path"] for r in records}),
                claims=claims, cells=cell_summary, records=records)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    # Boundary checks: invalid, exact, PV-only failure, FR-only failure, FA-only failure.
    assert [classify(False), classify(True, 0, 0, 0), classify(True, 0, 0, 1),
            classify(True, 0, 1, 0), classify(True, 1, 0, 0)] == [
                "invalid", "exact", "fa_eq_fr", "fa_lt_fr", "fa_gt_fr"]
    report = verify(args.root.resolve())
    if args.check:
        assert json.loads(args.check.read_text()) == report, "recorded audit differs"
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"records", "cells"}}, indent=2))


if __name__ == "__main__":
    main()
