#!/usr/bin/env python3
"""Extensional reference-clause recovery over full reachable closures.

A clause is recovered when a parsed candidate rejects every state in that
clause's independent witness set.  Empty witness sets are reported as
untestable and are excluded from denominators.  Result artifacts are read only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_MANIFEST = ROOT / "analysis/cell_manifest_2026-09-04.json"
DEFAULT_MD = ROOT / "analysis/clause_recovery_2026-09-04.txt"
DEFAULT_JSON = ROOT / "analysis/clause_recovery_2026-09-04.json"

GROUP_ORDER = ("composed", "set_named", "predicate", "hidden")

TARGETS = (
    ("Shopping (price)", "relaxed", "Gemma-4-31B"),
    ("Shopping (price)", "relaxed", "Qwen3.8-27B"),
    ("Deployment", "full", "Gemma-4-31B"),
    ("Deployment", "full", "Qwen3.8-27B"),
    ("Calendar", "full", "Gemma-4-31B"),
    ("Calendar", "full", "Qwen3.8-27B"),
    ("tau-bench Retail", "full", "Gemma-4-31B"),
    ("tau-bench Retail", "full", "Qwen3.8-27B"),
    ("Shopping (price)", "identifiable", "Gemma-4-31B"),
)


def build_domain(domain: str, variant: str) -> dict[str, Any]:
    if domain == "Shopping (price)":
        from experiment.shopping_price.config import PriceShoppingConfig
        from experiment.shopping_price.enumeration import enumerate_reachable
        from experiment.shopping_price.ground_truth import CLAUSE_NAMES, clause_values
        from experiment.shopping_price.dsl import parse_contract_text
        cfg_file = "shopping_price_easy_v4.json" if variant == "relaxed" else "shopping_price_easy_v5.json"
        cfg = PriceShoppingConfig.from_json(ROOT / "experiment/configs" / cfg_file)
        states = tuple(enumerate_reachable(cfg).states)
        parser = lambda text: parse_contract_text(text).bind(cfg)
        groups = {name: ("composed" if i in (0, 1, 2, 5) else "hidden")
                  for i, name in enumerate(CLAUSE_NAMES)}
        values = [clause_values(state, cfg) for state in states]
        return {"states": states, "clauses": list(CLAUSE_NAMES), "values": values,
                "parser": parser, "groups": groups}

    if domain == "Deployment":
        from experiment.deployment.config import DeploymentConfig
        from experiment.deployment.enumeration import enumerate_reachable
        from experiment.deployment.ground_truth import GROUND_TRUTH_CLAUSES
        from experiment.deployment.state import DeploymentStatus, counts_covers
        from experiment.deployment.dsl import parse_contract_text
        cfg = DeploymentConfig.from_json_file(ROOT / "experiment/configs/deployment_default.json")
        states = tuple(enumerate_reachable(cfg).states)
        def clause_values(state):
            return (
                bool(state.authenticated), state.deployment_status is DeploymentStatus.IDLE,
                not state.is_empty_allocation, cfg.is_valid_region(state.target_region),
                cfg.is_valid_tier(state.cluster_tier),
                counts_covers(state.available_quota, state.allocated_resources),
            )
        parser = lambda text: parse_contract_text(text).bind(cfg)
        clauses = list(GROUND_TRUTH_CLAUSES)
        groups = {name: ("set_named" if i in (3, 4) else "composed")
                  for i, name in enumerate(clauses)}
        return {"states": states, "clauses": clauses,
                "values": [clause_values(state) for state in states],
                "parser": parser, "groups": groups}

    if domain == "Calendar":
        from experiment.calendar.config import CalendarConfig
        from experiment.calendar.enumeration import enumerate_reachable
        from experiment.calendar.ground_truth import CLAUSE_NAMES, clause_values
        from experiment.calendar.dsl import parse_contract
        cfg = CalendarConfig.from_json(ROOT / "experiment/configs/calendar_default.json")
        states = tuple(enumerate_reachable(cfg).states)
        parser = lambda text: parse_contract(text, cfg)
        groups = {}
        for i, name in enumerate(CLAUSE_NAMES):
            groups[name] = "set_named" if i in (3, 4, 5) else "predicate" if i in (6, 7, 8, 9) else "composed"
        return {"states": states, "clauses": list(CLAUSE_NAMES),
                "values": [clause_values(state, cfg) for state in states],
                "parser": parser, "groups": groups}

    if domain == "tau-bench Retail":
        from experiment.taubench_retail.enumeration import enumerate_reachable, RetailConfig
        from experiment.taubench_retail.ground_truth import CLAUSE_NAMES, clause_values
        from experiment.taubench_retail.dsl import parse_contract
        raw = json.loads((ROOT / "experiment/configs/taubench_retail_default.json").read_text())
        cfg = RetailConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
        states = tuple(enumerate_reachable(cfg).states)
        parser = lambda text: parse_contract(text, cfg).bind(cfg)
        groups = {}
        for i, name in enumerate(CLAUSE_NAMES):
            groups[name] = "set_named" if i in (0, 6) else "predicate" if i in (2, 4, 5, 7) else "composed"
        return {"states": states, "clauses": list(CLAUSE_NAMES),
                "values": [clause_values(state, cfg) for state in states],
                "parser": parser, "groups": groups}

    raise ValueError(f"unsupported domain: {domain}")


def witnesses(values: list[tuple[bool, ...]], clauses: list[str], states: tuple[Any, ...]) -> dict[str, tuple[Any, ...]]:
    out: dict[str, tuple[Any, ...]] = {}
    for k, name in enumerate(clauses):
        out[name] = tuple(state for state, row in zip(states, values)
                          if not row[k] and all(value for j, value in enumerate(row) if j != k))
    return out


def manifest_cell(manifest: dict[str, Any], domain: str, variant: str,
                  model: str, method: str) -> dict[str, Any] | None:
    matches = [cell for cell in manifest["cells"].values()
               if cell["domain"] == domain and cell["variant"] == variant
               and cell["model"] == model and cell["method"] == method]
    # A cell may occur in both the main and ablation tables with the same source.
    unique = {(json.dumps(cell["dir"], sort_keys=True), cell["exact"], cell["n_runs"]): cell
              for cell in matches}
    if len(unique) > 1:
        raise RuntimeError(f"ambiguous manifest cells for {(domain, variant, model, method)}")
    return next(iter(unique.values())) if unique else None


def load_artifacts(cell: dict[str, Any], method: str) -> list[tuple[Path, dict[str, Any]]]:
    dirs = cell["dir"] if isinstance(cell["dir"], list) else [cell["dir"]]
    rows = []
    seen = set()
    for rel in dirs:
        for path in sorted((ROOT / "results" / rel).glob("*seed*.json")):
            if path.name.endswith("summary.json"):
                continue
            obj = json.loads(path.read_text(encoding="utf-8"))
            if obj.get("method") != method:
                continue
            seed = int(obj["seed"])
            if seed in seen:
                raise RuntimeError(f"duplicate seed {seed}: {path}")
            seen.add(seed)
            rows.append((path, obj))
    return sorted(rows, key=lambda row: int(row[1]["seed"]))


def score_cell(cell: dict[str, Any], method: str, domain_data: dict[str, Any],
               witness_sets: dict[str, tuple[Any, ...]]) -> dict[str, Any]:
    rows = load_artifacts(cell, method)
    run_details = []
    parsed = 0
    recovered = defaultdict(int)
    reparse_errors = []
    for path, obj in rows:
        record = obj.get("contract") or {}
        detail: dict[str, Any] = {"seed": int(obj["seed"]), "artifact": str(path.relative_to(ROOT)),
                                  "artifact_status": record.get("status"), "recovered": {}}
        if record.get("status") != "parsed":
            detail["parse_failure"] = True
            run_details.append(detail)
            continue
        source = record.get("source")
        try:
            contract = domain_data["parser"](source)
        except Exception as exc:
            detail["reparse_error"] = f"{type(exc).__name__}: {exc}"
            reparse_errors.append(detail["reparse_error"])
            run_details.append(detail)
            continue
        parsed += 1
        for clause in domain_data["clauses"]:
            ws = witness_sets[clause]
            value = None if not ws else all(not contract.holds_in(state) for state in ws)
            detail["recovered"][clause] = value
            if value:
                recovered[clause] += 1
        run_details.append(detail)
    clause_rows = []
    for clause in domain_data["clauses"]:
        ws = witness_sets[clause]
        clause_rows.append({"clause": clause, "group": domain_data["groups"][clause],
                            "n_witnesses": len(ws), "testable": bool(ws),
                            "recovered": recovered[clause] if ws else None,
                            "denominator": parsed if ws else None})
    return {"method": method, "n_runs": len(rows), "parsed": parsed,
            "parse_failures": len(rows) - parsed - len(reparse_errors),
            "reparse_errors": reparse_errors, "clauses": clause_rows, "runs": run_details}


def aggregate_groups(cell: dict[str, Any]) -> dict[str, dict[str, int] | None]:
    out = {}
    for group in GROUP_ORDER:
        rows = [row for row in cell["clauses"] if row["group"] == group and row["testable"]]
        out[group] = None if not rows else {
            "recovered": sum(int(row["recovered"]) for row in rows),
            "denominator": sum(int(row["denominator"]) for row in rows),
            "n_clauses": len(rows),
        }
    return out


def build(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    analyses = []
    for domain, variant, model in TARGETS:
        data = build_domain(domain, variant)
        ws = witnesses(data["values"], data["clauses"], data["states"])
        direct = manifest_cell(manifest, domain, variant, model, "direct")
        active = manifest_cell(manifest, domain, variant, model, "active_cegis")
        method = "active_cegis" if active else "active_cegis_no_coverage"
        method_cell = active or manifest_cell(manifest, domain, variant, model, method)
        if direct is None or method_cell is None:
            raise RuntimeError(f"missing required cells for {(domain, variant, model)}")
        direct_score = score_cell(direct, "direct", data, ws)
        method_score = score_cell(method_cell, method, data, ws)
        direct_score["groups"] = aggregate_groups(direct_score)
        method_score["groups"] = aggregate_groups(method_score)
        analyses.append({
            "domain": domain, "variant": variant, "model": model,
            "closure_states": len(data["states"]),
            "witness_counts": {name: len(ws[name]) for name in data["clauses"]},
            "direct": direct_score, "method": method_score,
        })
    return {"schema_version": 1, "definition": "candidate rejects every state in the clause's independent witness set",
            "generated_by": "scripts/clause_recovery.py", "analyses": analyses}


def ratio(group: dict[str, int] | None) -> str:
    return "--" if group is None else f"{group['recovered']}/{group['denominator']}"


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Clause recovery audit (2026-09-04)", "",
             "A clause is recovered iff the candidate rejects every state in its independent witness set. "
             "Untestable clauses have an empty witness set and are excluded. Parse failures are excluded and reported.", "",
             "## Grouped summary", "",
             "Each cell is `Direct -> counterexample-guided method`; denominators are parsed runs times testable clauses in that group.", "",
             "| Domain / variant | Model | Method | Composed | Set named | Predicate | Hidden | Parsed (D/M) |",
             "| :-- | :-- | :-- | --: | --: | --: | --: | --: |"]
    for row in report["analyses"]:
        d, m = row["direct"], row["method"]
        cells = [f"{ratio(d['groups'][g])} -> {ratio(m['groups'][g])}" if d['groups'][g] or m['groups'][g] else "--"
                 for g in GROUP_ORDER]
        lines.append(f"| {row['domain']} ({row['variant']}) | {row['model']} | {m['method']} | "
                     + " | ".join(cells) + f" | {d['parsed']}/{m['parsed']} |")

    lines += ["", "## Clause-level totals", ""]
    for row in report["analyses"]:
        lines += [f"### {row['domain']} ({row['variant']}) — {row['model']}", "",
                  f"Closure: {row['closure_states']:,} states. Direct parsed {row['direct']['parsed']}/{row['direct']['n_runs']}; "
                  f"{row['method']['method']} parsed {row['method']['parsed']}/{row['method']['n_runs']}.", "",
                  "| Clause | Group | Witnesses | Direct | Method |", "| :-- | :-- | --: | --: | --: |"]
        direct_by = {r["clause"]: r for r in row["direct"]["clauses"]}
        method_by = {r["clause"]: r for r in row["method"]["clauses"]}
        for clause, n_witness in row["witness_counts"].items():
            d, m = direct_by[clause], method_by[clause]
            dval = "untestable" if not d["testable"] else f"{d['recovered']}/{d['denominator']}"
            mval = "untestable" if not m["testable"] else f"{m['recovered']}/{m['denominator']}"
            lines.append(f"| `{clause}` | {d['group']} | {n_witness} | {dval} | {mval} |")
        lines.append("")

    lines += ["## Run-level detail", ""]
    for row in report["analyses"]:
        lines.append(f"### {row['domain']} ({row['variant']}) — {row['model']}")
        for label in ("direct", "method"):
            cell = row[label]
            lines += ["", f"#### {cell['method']}", "",
                      "| Seed | Parsed | Recovered clauses | Artifact |", "| --: | :--: | :-- | :-- |"]
            for run in cell["runs"]:
                good = [name for name, recovered in run.get("recovered", {}).items() if recovered is True]
                parsed = "yes" if run.get("artifact_status") == "parsed" and "reparse_error" not in run else "no"
                lines.append(f"| {run['seed']} | {parsed} | {', '.join(good) if good else '--'} | `{run['artifact']}` |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()
    report = build(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown(report), encoding="utf-8")
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.json_out}")
    errors = [err for row in report["analyses"] for side in ("direct", "method")
              for err in row[side]["reparse_errors"]]
    print(f"analyses={len(report['analyses'])} reparse_errors={len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
