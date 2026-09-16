#!/usr/bin/env python3
"""Generate the S17 exact-rate and paired-test appendix additions.

All outcome vectors come from ``analysis/cell_manifest_2026-09-04.json``.
Holm correction is applied separately to the four core and three selector
comparisons for each domain/model row.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "analysis/cell_manifest_2026-09-04.json"
JSON_OUT = ROOT / "analysis/s17_stats_2026-09-04.json"
MD_OUT = ROOT / "analysis/s17_stats_2026-09-04.txt"
TEX_OUT = ROOT / "analysis/s17_stats_2026-09-04.tex"

METHODS = ("direct", "self_refine", "random_probe", "sampled_cegis", "active_cegis",
           "active_cegis_no_balance", "active_cegis_no_coverage", "active_cegis_uniform")
SHORT = {"direct": "Direct", "self_refine": "Self-Refine", "random_probe": "Random",
         "sampled_cegis": "Sampled", "active_cegis": "Active",
         "active_cegis_no_balance": "No-Balance",
         "active_cegis_no_coverage": "No-Coverage", "active_cegis_uniform": "Uniform"}
CORE = METHODS[:4]
SELECTOR = METHODS[5:]

EXACT_ROWS = (
    ("Shopping (price)", "relaxed", "Gemma-4-31B"),
    ("Shopping (price)", "relaxed", "Qwen3.8-27B"),
    ("Shopping (price)", "relaxed", "Qwen3.6-35B-A3B"),
    ("Shopping (price)", "relaxed", "Qwen2.5-32B"),
    ("Shopping (price)", "relaxed", "Qwen2.5-14B"),
    ("Shopping (price)", "relaxed", "Gemma-4-26B-A4B"),
    ("Deployment", "full", "Gemma-4-26B-A4B"),
    ("Calendar", "full", "Gemma-4-26B-A4B"),
    ("tau-bench Retail", "full", "Gemma-4-31B"),
    ("tau-bench Retail", "full", "Qwen3.8-27B"),
    ("tau-bench Retail", "full", "Qwen2.5-32B"),
    ("tau-bench Retail", "full", "Qwen2.5-14B"),
    ("tau-bench Retail", "full", "Gemma-4-26B-A4B"),
)

NEW_TEST_ROWS = (
    ("Shopping (price)", "relaxed", "Gemma-4-26B-A4B"),
    ("Deployment", "full", "Gemma-4-26B-A4B"),
    ("Calendar", "full", "Gemma-4-26B-A4B"),
    ("tau-bench Retail", "full", "Gemma-4-31B"),
    ("tau-bench Retail", "full", "Qwen3.8-27B"),
    ("tau-bench Retail", "full", "Qwen2.5-32B"),
    ("tau-bench Retail", "full", "Qwen2.5-14B"),
    ("tau-bench Retail", "full", "Gemma-4-26B-A4B"),
)

STATES = {("Shopping (price)", "relaxed"): 288, ("Deployment", "full"): 11712,
          ("Calendar", "full"): 1676, ("tau-bench Retail", "full"): 1294}


def cell(manifest: dict[str, Any], domain: str, variant: str, model: str, method: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["cells"].values()
               if entry["domain"] == domain and entry["variant"] == variant
               and entry["model"] == model and entry["method"] == method]
    unique = {(json.dumps(entry["dir"], sort_keys=True), entry["n_runs"], entry["exact"]): entry
              for entry in matches}
    if len(unique) != 1:
        raise RuntimeError(f"expected one source for {(domain, variant, model, method)}, got {len(unique)}")
    return next(iter(unique.values()))


def load_outcomes(entry: dict[str, Any], method: str) -> dict[int, bool]:
    dirs = entry["dir"] if isinstance(entry["dir"], list) else [entry["dir"]]
    out: dict[int, bool] = {}
    for rel in dirs:
        for path in sorted((ROOT / "results" / rel).glob("*seed*.json")):
            if path.name.endswith("summary.json"):
                continue
            obj = json.loads(path.read_text(encoding="utf-8"))
            if obj.get("method") != method:
                continue
            seed = int(obj["seed"])
            if seed in out:
                raise RuntimeError(f"duplicate {(method, seed)} in {dirs}")
            out[seed] = bool(obj.get("outcome") == "exact" or
                             (obj.get("evaluation") or {}).get("metrics", {}).get("exact"))
    if len(out) != entry["n_runs"]:
        raise RuntimeError(f"manifest/run disagreement for {method}")
    return out


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def mcnemar(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm(raw: dict[str, float | None]) -> dict[str, float | None]:
    # Retain zero-discordance comparisons in the declared family (p=1 for
    # bookkeeping), while preserving their dash/None display, not equivalence.
    valid = sorted(((k, 1.0 if p is None else p) for k, p in raw.items()), key=lambda x: x[1])
    adjusted: dict[str, float | None] = {k: None for k in raw}
    running = 0.0
    for i, (key, p) in enumerate(valid):
        running = max(running, min(1.0, (len(valid) - i) * p))
        if raw[key] is not None:
            adjusted[key] = running
    return adjusted


def build() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    exact_rows = []
    for domain, variant, model in EXACT_ROWS:
        methods = {}
        for method in METHODS:
            entry = cell(manifest, domain, variant, model, method)
            outcomes = load_outcomes(entry, method)
            k, n = sum(outcomes.values()), len(outcomes)
            lo, hi = wilson(k, n)
            methods[method] = {"exact": k, "n": n, "wilson95": [lo, hi]}
        exact_rows.append({"domain": domain, "variant": variant, "states": STATES[(domain, variant)],
                           "model": model, "methods": methods})

    tests = []
    for domain, variant, model in NEW_TEST_ROWS:
        vectors = {m: load_outcomes(cell(manifest, domain, variant, model, m), m) for m in METHODS}
        active = vectors["active_cegis"]
        families = {}
        for family, others in (("core", CORE), ("selector", SELECTOR)):
            raw = {}
            details = {}
            for other in others:
                common = sorted(set(active) & set(vectors[other]))
                b = sum(active[s] and not vectors[other][s] for s in common)
                c = sum(not active[s] and vectors[other][s] for s in common)
                raw[other] = mcnemar(b, c)
                details[other] = {"n": len(common), "active_only": b, "other_only": c,
                                  "raw_p": raw[other]}
            adjusted = holm(raw)
            for other in others:
                details[other]["holm_p"] = adjusted[other]
            families[family] = details
        tests.append({"domain": domain, "variant": variant, "model": model, "families": families})
    return {"generated_by": "scripts/build_s17_stats.py", "exact_rows": exact_rows, "tests": tests}


def fmt_p(p: float | None) -> str:
    return "--" if p is None else f"{p:.3g}"


def markdown(report: dict[str, Any]) -> str:
    lines = ["# S17 appendix statistics", "", "## Exact recovery", "",
             "| Domain | Model | " + " | ".join(SHORT[m] for m in METHODS) + " |",
             "| :-- | :-- |" + " --: |" * len(METHODS)]
    for row in report["exact_rows"]:
        vals = []
        for method in METHODS:
            x = row["methods"][method]
            vals.append(f"{x['exact']}/{x['n']} [{x['wilson95'][0]:.2f},{x['wilson95'][1]:.2f}]")
        lines.append(f"| {row['domain']} ({row['states']:,}) | {row['model']} | " + " | ".join(vals) + " |")
    for family in ("core", "selector"):
        lines += ["", f"## {family.title()} paired tests", "",
                  "Each cell is `Active-only:other-only / raw p / Holm p`.", "",
                  "| Domain | Model | " + " | ".join(SHORT[m] for m in (CORE if family == "core" else SELECTOR)) + " |",
                  "| :-- | :-- |" + " --: |" * len(CORE if family == "core" else SELECTOR)]
        for row in report["tests"]:
            cells = []
            for method in (CORE if family == "core" else SELECTOR):
                x = row["families"][family][method]
                cells.append(f"{x['active_only']}:{x['other_only']} / {fmt_p(x['raw_p'])} / {fmt_p(x['holm_p'])}")
            lines.append(f"| {row['domain']} | {row['model']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def tex_domain(domain: str) -> str:
    return {"Shopping (price)": "Shopping (relaxed)", "tau-bench Retail": "$\\tau$-bench"}.get(domain, domain)


def tex_p(p: float | None) -> str:
    return "--" if p is None else f"{p:.2g}"


def latex(report: dict[str, Any]) -> str:
    lines = [
        "% Generated by scripts/build_s17_stats.py from the cell manifest.",
        "% Existing relaxed-Shopping McNemar tables remain in appendix_stats_tables.tex.",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Exact successes out of 20 seeds, with Wilson 95\\,\\% intervals. Closure sizes are 288 (relaxed Shopping), 11{,}712 (Deployment), 1{,}676 (Calendar), and 1{,}294 ($\\tau$-bench).}",
        "\\label{tab:app-exact}", "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccccc}", "\\toprule",
        "Domain & Model & Direct & Self-Refine & Random & Sampled & Active & No-Balance & No-Coverage & Uniform \\\\",
        "\\midrule",
    ]
    for row in report["exact_rows"]:
        vals = []
        for method in METHODS:
            x = row["methods"][method]
            vals.append(f"{x['exact']}/{x['n']} \\tiny[{x['wilson95'][0]:.2f},{x['wilson95'][1]:.2f}]")
        lines.append(f"{tex_domain(row['domain'])} & {row['model']} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    for family, methods, label in (("core", CORE, "core"), ("selector", SELECTOR, "selector")):
        colspec = "ll" + "c" * len(methods)
        lines += ["\\begin{table}[t]", "\\centering",
                  f"\\caption{{Additional {family} paired tests against Active CEGIS. Each cell is Active-only:other-only / raw $p$ / Holm $p$; -- denotes no discordant pairs.}}",
                  f"\\label{{tab:app-mcnemar-s17-{label}}}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
                  "\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
                  "Domain & Model & " + " & ".join(SHORT[m] for m in methods) + " \\\\", "\\midrule"]
        for row in report["tests"]:
            cells = []
            for method in methods:
                x = row["families"][family][method]
                cells.append(f"{x['active_only']}:{x['other_only']} / {tex_p(x['raw_p'])} / {tex_p(x['holm_p'])}")
            lines.append(f"{tex_domain(row['domain'])} & {row['model']} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=MD_OUT)
    parser.add_argument("--tex-out", type=Path, default=TEX_OUT)
    args = parser.parse_args()
    report = build()
    for path in (args.json_out, args.md_out, args.tex_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(markdown(report), encoding="utf-8")
    args.tex_out.write_text(latex(report), encoding="utf-8")
    print(f"wrote {args.json_out}, {args.md_out}, and {args.tex_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
