#!/usr/bin/env python3
"""S20 Full-Closure Trajectory Recomputation.

Re-evaluates round-by-round candidate contracts on the full domain closures:
  - Shopping (price, relaxed): 288 closure states (|R+| = 8, |R-| = 280)
  - tau-bench Retail (full): 1,294 closure states (|R+| = 14, |R-| = 1,280)
  - Deployment (full): 11,712 closure states (|R+| = 88, |R-| = 11,624)

Outputs:
  - analysis/closure_trajectory_2026-09-04.json
  - analysis/closure_trajectory_2026-09-04.txt
  - outputs/figures/convergence_D.pdf
  - outputs/fragments/s20_trajectory.tex
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST_PATH = ROOT / "analysis/cell_manifest_2026-09-04.json"
JSON_OUT = ROOT / "analysis/closure_trajectory_2026-09-04.json"
MD_OUT = ROOT / "analysis/closure_trajectory_2026-09-04.txt"
# 원고가 include 하는 정본 그림. S19(`build_s19_graded_metrics.py`)는 이 파일을 쓰지
# 않고 convergence_D_s19.pdf 로 분리한다(R6). 재생성 순서와 무관하게 정본이 유지된다.
PDF_OUT = ROOT / "outputs/figures/convergence_D.pdf"
TEX_OUT = ROOT / "outputs/fragments/s20_trajectory.tex"

TARGET_COMBINATIONS = [
    ("Shopping (price)", "relaxed", "Gemma-4-31B"),
    ("Shopping (price)", "relaxed", "Qwen2.5-32B"),
    ("Shopping (price)", "relaxed", "Qwen3.8-27B"),
    ("tau-bench Retail", "full", "Gemma-4-31B"),
    ("tau-bench Retail", "full", "Qwen3.8-27B"),
    ("tau-bench Retail", "full", "Qwen2.5-32B"),
    ("Deployment", "full", "Gemma-4-31B"),
    ("Deployment", "full", "Qwen3.8-27B"),
    ("Deployment", "full", "Qwen2.5-32B"),
]

METHODS = ["sampled_cegis", "active_cegis", "active_cegis_no_balance"]

METHOD_NAMES = {
    "sampled_cegis": "Sampled",
    "active_cegis": "Active",
    "active_cegis_no_balance": "No-Balance",
}

DOMAIN_SPECS = {
    "Shopping (price)": {"pos": 8, "neg": 280, "total": 288},
    "tau-bench Retail": {"pos": 14, "neg": 1280, "total": 1294},
    "Deployment": {"pos": 88, "neg": 11624, "total": 11712},
}


def extract_candidate_texts(interactions: list[dict[str, Any]]) -> list[str]:
    """Extract candidate contract strings per round attempt from interactions.
    
    Handles initial attempts followed by parse repair interactions.
    """
    candidates = []
    i = 0
    while i < len(interactions):
        inter = interactions[i]
        role = inter.get("role", "")
        # If unexpected standalone parse repair, advance
        if "parse_repair" in role:
            i += 1
            continue
        # Check if immediately followed by parse repair
        if i + 1 < len(interactions) and "parse_repair" in interactions[i + 1].get("role", ""):
            candidates.append(interactions[i + 1].get("response_text", ""))
            i += 2
        else:
            candidates.append(inter.get("response_text", ""))
            i += 1
    return candidates


def compute_all_trajectories() -> dict[str, Any]:
    print("Enumerating closures...")
    from experiment.shopping_price.config import PriceShoppingConfig
    from experiment.shopping_price.dsl import parse_contract_text as parse_shop
    from experiment.shopping_price.enumeration import enumerate_reachable as enum_shop
    from experiment.shopping_price.contracts import evaluate_contract as eval_shop

    from experiment.deployment.config import DeploymentConfig
    from experiment.deployment.dsl import parse_contract_text as parse_dep
    from experiment.deployment.enumeration import enumerate_reachable as enum_dep
    from experiment.deployment.contracts import evaluate_contract as eval_dep

    from experiment.taubench_retail.enumeration import enumerate_reachable as enum_tau, RetailConfig
    from experiment.taubench_retail.dsl import parse_contract as parse_tau
    from experiment.taubench_retail.contracts import evaluate_contract as eval_tau

    cfg_shop = PriceShoppingConfig.from_json(ROOT / "experiment/configs/shopping_price_easy_v4.json")
    states_shop = tuple(enum_shop(cfg_shop).states)

    cfg_dep = DeploymentConfig.from_json_file(ROOT / "experiment/configs/deployment_default.json")
    states_dep = tuple(enum_dep(cfg_dep).states)

    raw_tau = json.loads((ROOT / "experiment/configs/taubench_retail_default.json").read_text())
    cfg_tau = RetailConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw_tau.items()})
    states_tau = tuple(enum_tau(cfg_tau).states)

    print(f"Closures enumerated: Shop={len(states_shop)}, Dep={len(states_dep)}, Tau={len(states_tau)}")

    def parse_and_eval(dom: str, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        if dom == "Shopping (price)":
            parsed = parse_shop(text).bind(cfg_shop)
            rep = eval_shop(parsed, states_shop, cfg_shop)
        elif dom == "Deployment":
            parsed = parse_dep(text).bind(cfg_dep)
            rep = eval_dep(parsed, states_dep, cfg_dep)
        elif dom == "tau-bench Retail":
            parsed = parse_tau(text, cfg_tau)
            rep = eval_tau(parsed, states_tau, cfg_tau)
        else:
            raise ValueError(f"Unknown domain {dom}")

        d = rep.false_accepts + rep.false_rejects + rep.postcondition_violations
        pos = DOMAIN_SPECS[dom]["pos"]
        neg = DOMAIN_SPECS[dom]["neg"]
        far = rep.false_accepts / neg
        frr = rep.false_rejects / pos
        n_checked = pos - rep.false_rejects
        pvr = (rep.postcondition_violations / n_checked) if n_checked > 0 else 0.0
        return {
            "D": d,
            "FA": rep.false_accepts,
            "FR": rep.false_rejects,
            "PV": rep.postcondition_violations,
            "FAR": far,
            "FRR": frr,
            "PVR": pvr,
            "n_checked": n_checked,
            "exact": bool(d == 0),
        }

    manifest = json.loads(MANIFEST_PATH.read_text())
    trajectories = {}

    for dom, var, model in TARGET_COMBINATIONS:
        group_key = f"{dom}|{var}|{model}"
        trajectories[group_key] = {}

        for method in METHODS:
            cell = None
            for entry in manifest["cells"].values():
                if (entry["domain"] == dom and entry["variant"] == var and
                        entry["model"] == model and entry["method"] == method):
                    cell = entry
                    break
            if not cell:
                print(f"Warning: cell not found for {group_key} {method}")
                continue

            d_dirs = cell["dir"] if isinstance(cell["dir"], list) else [cell["dir"]]
            files = []
            for d in d_dirs:
                files.extend(sorted((ROOT / "results" / d).glob(f"{method}__seed*.json"),
                                    key=lambda x: int(x.stem.rsplit("seed", 1)[1]) if "seed" in x.stem else 0))

            runs_data = []
            for p in files:
                data = json.loads(p.read_text())
                seed = data.get("seed")
                stopped_because = data.get("stopped_because")
                cands = extract_candidate_texts(data.get("interactions", []))

                # Evaluate candidate per round attempt
                evaluated_rounds = []
                for c_text in cands:
                    try:
                        ev = parse_and_eval(dom, c_text)
                        evaluated_rounds.append(ev)
                    except Exception:
                        evaluated_rounds.append("parse_failure")

                # Carry-forward logic across rounds 0..3:
                # - If evaluated_rounds[r] is valid: alive
                # - If evaluated_rounds[r] failed parsing: parse_failure (missing at round r, not carried)
                # - If r >= len(evaluated_rounds): carry forward last valid evaluation (carried)
                # - If no valid evaluation exists: unparsed
                round_records = []
                last_valid = None
                for r in range(4):
                    if r < len(evaluated_rounds):
                        ev = evaluated_rounds[r]
                        if ev == "parse_failure":
                            round_records.append({"status": "parse_failure", "metrics": None})
                        else:
                            round_records.append({"status": "alive", "metrics": ev})
                            last_valid = ev
                    else:
                        if last_valid is not None:
                            round_records.append({"status": "carried", "metrics": last_valid})
                        else:
                            round_records.append({"status": "unparsed", "metrics": None})

                runs_data.append({
                    "seed": seed,
                    "stopped_because": stopped_because,
                    "num_attempts": len(cands),
                    "rounds": round_records,
                })

            # Compute statistics across all evaluated runs (alive + carried) for each round
            round_stats = []
            for r in range(4):
                eval_metrics = [
                    rd["rounds"][r]["metrics"]
                    for rd in runs_data
                    if rd["rounds"][r]["metrics"] is not None
                ]
                n_alive = sum(1 for rd in runs_data if rd["rounds"][r]["status"] == "alive")
                n_carried = sum(1 for rd in runs_data if rd["rounds"][r]["status"] == "carried")
                n_pf = sum(1 for rd in runs_data if rd["rounds"][r]["status"] == "parse_failure")
                n_unparsed = sum(1 for rd in runs_data if rd["rounds"][r]["status"] == "unparsed")

                if eval_metrics:
                    d_vals = [m["D"] for m in eval_metrics]
                    far_vals = [m["FAR"] for m in eval_metrics]
                    frr_vals = [m["FRR"] for m in eval_metrics]
                    pvr_vals = [m["PVR"] for m in eval_metrics]
                    exact_count = sum(1 for m in eval_metrics if m["exact"])

                    med_d = float(np.median(d_vals))
                    q25_d = float(np.percentile(d_vals, 25))
                    q75_d = float(np.percentile(d_vals, 75))

                    med_far = float(np.median(far_vals))
                    med_frr = float(np.median(frr_vals))
                    med_pvr = float(np.median(pvr_vals))
                else:
                    med_d, q25_d, q75_d = None, None, None
                    med_far, med_frr, med_pvr = None, None, None
                    exact_count = 0

                round_stats.append({
                    "round": r,
                    "n_alive": n_alive,
                    "n_carried": n_carried,
                    "n_parse_failure": n_pf,
                    "n_unparsed": n_unparsed,
                    "n_eval": len(eval_metrics),
                    "exact_count": exact_count,
                    "median_D": med_d,
                    "iqr_D": [q25_d, q75_d],
                    "mean_D": float(np.mean([m["D"] for m in eval_metrics])) if eval_metrics else None,
                    "median_FAR": med_far,
                    "median_FRR": med_frr,
                    "median_PVR": med_pvr,
                })

            trajectories[group_key][method] = {
                "n_runs": len(files),
                "round_stats": round_stats,
                "runs": runs_data,
            }
            summary_str = " -> ".join(
                f"{s['median_D']:.1f} (a={s['n_alive']},c={s['n_carried']})"
                if s['median_D'] is not None else f"-- (a={s['n_alive']},c={s['n_carried']})"
                for s in round_stats
            )
            print(f"{dom[:12]} | {model[:11]} | {method[:15]:<15} : {summary_str}")

    return trajectories


def generate_plot(trajectories: dict[str, Any], out_pdf: Path) -> None:
    """Generate publication-ready 2-panel convergence plot on full closure."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)

    panels = [
        ("Shopping (price)|relaxed|Gemma-4-31B", "Gemma-4-31B", axes[0], (-1.0, 30.0)),
        ("Shopping (price)|relaxed|Qwen2.5-32B", "Qwen2.5-32B", axes[1], (-2.0, 110.0)),
    ]

    colors = {
        "active_cegis": "#1f77b4",            # Blue
        "sampled_cegis": "#2ca02c",           # Green
        "active_cegis_no_balance": "#d62728", # Red
    }
    markers = {
        "active_cegis": "o",
        "sampled_cegis": "s",
        "active_cegis_no_balance": "^",
    }
    short_labels = {
        "active_cegis": "Active",
        "sampled_cegis": "Sampled",
        "active_cegis_no_balance": "No-Balance",
    }

    rounds = np.array([0, 1, 2, 3])

    for key, title, ax, ylim in panels:
        pdata = trajectories.get(key, {})
        for method in ["active_cegis", "sampled_cegis", "active_cegis_no_balance"]:
            mdata = pdata.get(method)
            if not mdata:
                continue
            rstats = mdata["round_stats"]
            meds = [s["median_D"] for s in rstats]
            q25 = [s["iqr_D"][0] for s in rstats]
            q75 = [s["iqr_D"][1] for s in rstats]
            tot = rstats[0]["n_eval"]

            label = f"{short_labels[method]} (N={tot})"
            ax.plot(rounds, meds, label=label, color=colors[method],
                    marker=markers[method], linewidth=1.8, markersize=5)
            ax.fill_between(rounds, q25, q75, color=colors[method], alpha=0.15)

        # Domain-specific clean annotations
        if title == "Gemma-4-31B":
            act = pdata["active_cegis"]["round_stats"]
            smp = pdata["sampled_cegis"]["round_stats"]
            nob = pdata["active_cegis_no_balance"]["round_stats"]

            ax.annotate("20/20 (all)", (0, 23.5), fontsize=6.5, color="black", ha="center", va="bottom", fontweight="bold")

            ax.annotate(f"{act[1]['n_alive']}/20", (1, 6.2), fontsize=6.5, color="#1f77b4", ha="center", va="top")
            ax.annotate(f"{smp[1]['n_alive']}/20", (1, 9.8), fontsize=6.5, color="#2ca02c", ha="center", va="bottom")
            ax.annotate(f"{nob[1]['n_alive']}/20", (1, 20.5), fontsize=6.5, color="#d62728", ha="center", va="top")

            ax.annotate(f"{act[2]['n_alive']}/20", (2, 2.2), fontsize=6.5, color="#1f77b4", ha="center", va="top")
            ax.annotate(f"{smp[2]['n_alive']}/20", (2, 9.8), fontsize=6.5, color="#2ca02c", ha="center", va="bottom")
            ax.annotate(f"{nob[2]['n_alive']}/20", (2, 20.5), fontsize=6.5, color="#d62728", ha="center", va="top")

            ax.annotate(f"{act[3]['n_alive']}/20", (3, 1.5), fontsize=6.5, color="#1f77b4", ha="center", va="bottom")
            ax.annotate(f"{smp[3]['n_alive']}/20", (3, 9.8), fontsize=6.5, color="#2ca02c", ha="center", va="bottom")
            ax.annotate(f"{nob[3]['n_alive']}/20", (3, 20.5), fontsize=6.5, color="#d62728", ha="center", va="top")

        elif title == "Qwen2.5-32B":
            act = pdata["active_cegis"]["round_stats"]
            smp = pdata["sampled_cegis"]["round_stats"]
            nob = pdata["active_cegis_no_balance"]["round_stats"]

            ax.annotate(f"{act[0]['n_alive']}/17", (0, 42.0), fontsize=6.5, color="#1f77b4", ha="center", va="top")
            ax.annotate(f"{nob[0]['n_alive']}/14", (0, 74.0), fontsize=6.5, color="#d62728", ha="center", va="bottom")
            ax.annotate(f"{smp[0]['n_alive']}/15", (0, 93.0), fontsize=6.5, color="#2ca02c", ha="center", va="bottom")

            ax.annotate(f"{act[1]['n_alive']}/17", (0.87, 42.0), fontsize=6.5, color="#1f77b4", ha="center", va="top")
            ax.annotate(f"{smp[1]['n_alive']}/15", (1.13, 42.0), fontsize=6.5, color="#2ca02c", ha="center", va="top")
            ax.annotate(f"{nob[1]['n_alive']}/14", (1.00, 56.0), fontsize=6.5, color="#d62728", ha="center", va="bottom")

            ax.annotate(f"{act[2]['n_alive']}/17", (2.00, 27.0), fontsize=6.5, color="#1f77b4", ha="center", va="top")
            ax.annotate(f"{smp[2]['n_alive']}/15", (1.92, 35.0), fontsize=6.5, color="#2ca02c", ha="center", va="top")
            ax.annotate(f"{nob[2]['n_alive']}/14", (2.08, 49.0), fontsize=6.5, color="#d62728", ha="center", va="bottom")

            ax.annotate(f"{act[3]['n_alive']}/17", (3.00, 27.0), fontsize=6.5, color="#1f77b4", ha="center", va="top")
            ax.annotate(f"{smp[3]['n_alive']}/15", (2.88, 42.0), fontsize=6.5, color="#2ca02c", ha="center", va="bottom")
            ax.annotate(f"{nob[3]['n_alive']}/14", (3.06, 50.0), fontsize=6.5, color="#d62728", ha="center", va="bottom")

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Round (Refinement Step)", fontsize=8.5)
        ax.set_xticks(rounds)
        ax.set_xticklabels(["0 (Init)", "1", "2", "3"])
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)
        ax.set_ylim(-1.0, 32.0 if title == "Gemma-4-31B" else 112.0)
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

    axes[0].set_ylabel("residual closure defects $D$ (full closure)", fontsize=8.5)
    axes[1].set_ylabel("residual closure defects $D$ (full closure)", fontsize=8.5)

    axes[0].legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    axes[1].legend(loc="upper right", fontsize=7.5, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_pdf, format="pdf", dpi=300)
    plt.close(fig)
    print(f"Generated full-closure convergence figure: {out_pdf}")


def generate_latex_fragment(trajectories: dict[str, Any], out_tex: Path) -> None:
    """Generate LaTeX fragment for s20_trajectory.tex replacing Table 4."""
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"% Auto-generated by scripts/closure_trajectory.py on " + time.strftime("%Y-%m-%d"),
        r"% Replaces Table 4 of s19_appendix_graded.tex with full-closure trajectory evaluation",
        r"\subsection{Round-by-Round Convergence Trajectories (Full Closure)}",
        r"Table~\ref{tab:round-trajectories} details the trajectory of residual closure defects $D_0 \to D_1 \to D_2 \to D_3$ evaluated over the \emph{full reachable closure} ($|\mathcal{R}| = 288$ for Shopping, $1{,}294$ for $\tau$-bench, $11{,}712$ for Deployment). Each cell reports the median defect [IQR] along with the run distribution as $(n_{\text{alive}} / n_{\text{carried}})$ out of $N$. Early-terminated runs carry forward their final evaluation to avoid survivorship bias; parse failures at Round~0 are excluded from evaluation denominators.",
        r"",
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Round-by-round convergence of residual closure defects $D$ across refinement rounds over the full closure. Median [IQR] and run distribution ($n_{\text{alive}} / n_{\text{carried}}$ out of $N$) per round.}",
        r"\label{tab:round-trajectories}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Setting & Method & $N$ & Round 0 ($D_0$) & Round 1 ($D_1$) & Round 2 ($D_2$) & Round 3 ($D_3$) \\",
        r"\midrule",
    ]

    for dom, var, model in TARGET_COMBINATIONS:
        group_key = f"{dom}|{var}|{model}"
        pdata = trajectories.get(group_key, {})
        dom_clean = dom.replace("&", r"\&")
        var_clean = var
        setting_label = f"{dom_clean} ({var_clean}) -- {model}"

        for method in METHODS:
            mdata = pdata.get(method)
            if not mdata:
                continue
            rstats = mdata["round_stats"]
            n_eval_r0 = rstats[0]["n_eval"]
            n_tot = mdata["n_runs"]
            n_str = f"{n_eval_r0}" if n_eval_r0 == n_tot else f"{n_eval_r0} ({n_tot - n_eval_r0} pf)"

            round_cols = []
            for s in rstats:
                if s["median_D"] is not None:
                    med = s["median_D"]
                    q25, q75 = s["iqr_D"]
                    a = s["n_alive"]
                    c = s["n_carried"]
                    round_cols.append(f"{med:.1f} [{q25:.1f},{q75:.1f}] ({a}/{c})")
                else:
                    round_cols.append("--")

            m_display = METHOD_NAMES.get(method, method)
            lines.append(f"{setting_label} & {m_display:<12} & {n_str} & " + " & ".join(round_cols) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        r"",
    ])

    out_tex.write_text("\n".join(lines))
    print(f"Generated LaTeX fragment: {out_tex}")


def generate_markdown(trajectories: dict[str, Any], out_md: Path) -> None:
    """Generate comprehensive markdown summary."""
    lines = [
        "# Round-by-Round Convergence Trajectory (Full Closure Evaluation)",
        "",
        "- Date: 2026-09-04",
        "- Generated by: `scripts/closure_trajectory.py`",
        "- Metric: $D = \\mathrm{FA} + \\mathrm{FR} + \\mathrm{PV}$ evaluated over the full reachable closure",
        "  - Shopping (price, relaxed): 288 states",
        "  - tau-bench Retail (full): 1,294 states",
        "  - Deployment (full): 11,712 states",
        "",
        "## Summary Table: Median Defect [IQR] ($n_{\\text{alive}} / n_{\\text{carried}}$)",
        "",
        "| Setting | Method | $N_{\\text{valid}}$ | Round 0 ($D_0$) | Round 1 ($D_1$) | Round 2 ($D_2$) | Round 3 ($D_3$) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for dom, var, model in TARGET_COMBINATIONS:
        group_key = f"{dom}|{var}|{model}"
        pdata = trajectories.get(group_key, {})
        setting_label = f"**{dom} ({var}) — {model}**"

        for method in METHODS:
            mdata = pdata.get(method)
            if not mdata:
                continue
            rstats = mdata["round_stats"]
            n_eval_r0 = rstats[0]["n_eval"]
            n_tot = mdata["n_runs"]
            n_str = f"{n_eval_r0}/{n_tot}" if n_eval_r0 == n_tot else f"{n_eval_r0}/{n_tot} ({n_tot - n_eval_r0} pf)"

            round_cols = []
            for s in rstats:
                if s["median_D"] is not None:
                    med = s["median_D"]
                    q25, q75 = s["iqr_D"]
                    a = s["n_alive"]
                    c = s["n_carried"]
                    round_cols.append(f"{med:.1f} [{q25:.1f}, {q75:.1f}] ({a}a, {c}c)")
                else:
                    round_cols.append("--")

            m_display = METHOD_NAMES.get(method, method)
            lines.append(f"| {setting_label} | {m_display} | {n_str} | " + " | ".join(round_cols) + " |")

    lines.append("")
    out_md.write_text("\n".join(lines))
    print(f"Generated Markdown report: {out_md}")


def main() -> None:
    t0 = time.time()
    trajectories = compute_all_trajectories()

    # Save JSON
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(trajectories, indent=2))
    print(f"Saved full closure trajectories JSON: {JSON_OUT}")

    # Generate Markdown
    generate_markdown(trajectories, MD_OUT)

    # Generate Plot
    generate_plot(trajectories, PDF_OUT)

    # Generate LaTeX fragment
    generate_latex_fragment(trajectories, TEX_OUT)

    print(f"All outputs generated successfully in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
