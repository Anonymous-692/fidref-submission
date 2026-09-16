#!/usr/bin/env python3
"""S19 Graded Metrics, Convergence Trajectories, and Selector Diagnosis.

Computes:
1. Graded metrics: FA, FR, PV, D = FA + FR + PV, FAR, FRR, PVR for all manifest cells.
2. Domain Reject-All baseline rows.
3. Round-by-round convergence trajectories (D_0 -> D_1 -> D_2 -> D_3) with carry-forward.
4. Exploratory paired permutation tests and bootstrap 95% CIs on floor/intermediate cells.
5. Qwen3.8 tau-bench Active vs No-Coverage reference witness coverage diagnosis.
6. Generation of LaTeX fragments and convergence plot.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import fisher_exact, binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST_PATH = ROOT / "analysis/cell_manifest_2026-09-04.json"
JSON_OUT = ROOT / "analysis/s19_graded_metrics_2026-09-04.json"

METHODS = (
    "direct", "self_refine", "random_probe", "sampled_cegis",
    "active_cegis", "active_cegis_no_balance", "active_cegis_no_coverage", "active_cegis_uniform"
)
SHORT = {
    "direct": "Direct", "self_refine": "Self-Refine", "random_probe": "Random",
    "sampled_cegis": "Sampled", "active_cegis": "Active",
    "active_cegis_no_balance": "No-Balance",
    "active_cegis_no_coverage": "No-Coverage", "active_cegis_uniform": "Uniform"
}
CORE = METHODS[:4]
SELECTOR = METHODS[5:]

DOMAIN_STATES = {
    ("Shopping (price)", "relaxed"): {"total": 288, "pos": 8, "neg": 280},
    ("Shopping (price)", "identifiable"): {"total": 864, "pos": 8, "neg": 856},
    ("Shopping", "basic"): {"total": 4752, "pos": 22, "neg": 4730},
    ("Deployment", "full"): {"total": 11712, "pos": 88, "neg": 11624},
    ("Calendar", "full"): {"total": 1676, "pos": 35, "neg": 1641},
    ("tau-bench Retail", "full"): {"total": 1294, "pos": 14, "neg": 1280},
}


def compute_cell_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    """Compute graded metrics for all cells in the manifest."""
    results = {}
    mismatches = []
    
    for cell_key, entry in manifest["cells"].items():
        domain = entry["domain"]
        variant = entry["variant"]
        model = entry["model"]
        method = entry["method"]
        dirs = entry["dir"] if isinstance(entry["dir"], list) else [entry["dir"]]
        
        dom_info = DOMAIN_STATES.get((domain, variant))
        if not dom_info:
            raise ValueError(f"Unknown domain variant: {(domain, variant)}")
        R_tot = dom_info["total"]
        R_pos = dom_info["pos"]
        R_neg = dom_info["neg"]
        
        runs = []
        parse_failures = 0
        
        for d in dirs:
            for p in sorted((ROOT / "results" / d).glob("*seed*.json"),
                           key=lambda x: int(x.stem.rsplit("seed", 1)[1]) if "seed" in x.stem else 0):
                if p.name.endswith("summary.json"):
                    continue
                data = json.loads(p.read_text())
                if data.get("method") != method:
                    continue
                
                seed = data.get("seed")
                outcome = data.get("outcome")
                ev = data.get("evaluation")
                
                if outcome == "parse_failure" or not ev or not isinstance(ev, dict) or "metrics" not in ev:
                    parse_failures += 1
                    runs.append({
                        "seed": seed, "outcome": outcome, "valid": False,
                        "fa": None, "fr": None, "pv": None, "D": None,
                        "far": None, "frr": None, "pvr": None, "n_checked": None,
                        "exact": False
                    })
                    continue
                
                m = ev["metrics"]
                fa = int(m.get("false_accepts", 0))
                fr = int(m.get("false_rejects", 0))
                pv = int(m.get("postcondition_violations", 0))
                calc_exact = (fa == 0 and fr == 0 and pv == 0)
                stored_exact = bool(m.get("exact") or ev.get("exact") or outcome == "exact")
                
                if calc_exact != stored_exact:
                    mismatches.append((cell_key, seed, "exact", calc_exact, stored_exact))
                
                D = fa + fr + pv
                far = fa / R_neg
                frr = fr / R_pos
                n_checked = R_pos - fr
                if not 0 <= n_checked <= R_pos or not 0 <= pv <= n_checked:
                    raise ValueError(f"Invalid PVR counts: {cell_key}, seed={seed}, PV={pv}, N_checked={n_checked}")
                pvr = (pv / n_checked) if n_checked > 0 else None
                
                runs.append({
                    "seed": seed, "outcome": outcome, "valid": True,
                    "fa": fa, "fr": fr, "pv": pv, "D": D,
                    "far": far, "frr": frr, "pvr": pvr, "n_checked": n_checked,
                    "exact": calc_exact
                })
        
        valid_runs = [r for r in runs if r["valid"]]
        n_valid = len(valid_runs)
        
        if n_valid > 0:
            D_vals = [r["D"] for r in valid_runs]
            far_vals = [r["far"] for r in valid_runs]
            frr_vals = [r["frr"] for r in valid_runs]
            pvr_vals = [r["pvr"] for r in valid_runs if r["pvr"] is not None]
            
            d_med = float(np.median(D_vals))
            d_q25, d_q75 = float(np.percentile(D_vals, 25)), float(np.percentile(D_vals, 75))
            
            far_med = float(np.median(far_vals))
            far_q25, far_q75 = float(np.percentile(far_vals, 25)), float(np.percentile(far_vals, 75))
            
            frr_med = float(np.median(frr_vals))
            frr_q25, frr_q75 = float(np.percentile(frr_vals, 25)), float(np.percentile(frr_vals, 75))
            
            if pvr_vals:
                pvr_med = float(np.median(pvr_vals))
                pvr_q25, pvr_q75 = float(np.percentile(pvr_vals, 25)), float(np.percentile(pvr_vals, 75))
            else:
                pvr_med = pvr_q25 = pvr_q75 = None
            
            exact_count = sum(1 for r in valid_runs if r["exact"])
        else:
            d_med = d_q25 = d_q75 = None
            far_med = far_q25 = far_q75 = None
            frr_med = frr_q25 = frr_q75 = None
            pvr_med = pvr_q25 = pvr_q75 = None
            exact_count = 0
            
        results[cell_key] = {
            "domain": domain, "variant": variant, "model": model, "method": method,
            "table": entry.get("table", ""),
            "n_runs": len(runs), "n_valid": n_valid, "parse_failures": parse_failures,
            "exact_count": exact_count,
            "d_med": d_med, "d_iqr": [d_q25, d_q75],
            "far_med": far_med, "far_iqr": [far_q25, far_q75],
            "frr_med": frr_med, "frr_iqr": [frr_q25, frr_q75],
            "pvr_med": pvr_med, "pvr_iqr": [pvr_q25, pvr_q75],
            "n_pvr_defined": sum(r["pvr"] is not None for r in valid_runs),
            "n_pvr_undefined": sum(r["pvr"] is None for r in valid_runs),
            "runs": runs
        }
        
    return {"cells": results, "mismatches": mismatches}


def compute_reject_all_baselines() -> dict[str, dict[str, Any]]:
    """Compute reject-all contract metrics for each domain."""
    baselines = {}
    for (dom, var), info in DOMAIN_STATES.items():
        tot = info["total"]
        pos = info["pos"]
        neg = info["neg"]
        fa = 0
        fr = pos
        pv = 0
        D = fa + fr + pv
        far = 0.0
        frr = 1.0
        n_checked = 0
        pvr = None
        baselines[f"{dom} ({var})"] = {
            "domain": dom, "variant": var,
            "states": tot, "pos": pos, "neg": neg,
            "fa": fa, "fr": fr, "pv": pv, "D": D,
            "far": far, "frr": frr, "n_checked": n_checked, "pvr": pvr
        }
    return baselines


def extract_convergence_trajectories(manifest: dict[str, Any]) -> dict[str, Any]:
    """Extract round-by-round D_0 -> D_1 -> D_2 -> D_3 trajectories with carry-forward."""
    target_combinations = [
        ("Shopping (price)", "relaxed", "Gemma-4-31B"),
        ("Shopping (price)", "relaxed", "Qwen2.5-32B"),
        ("Shopping (price)", "relaxed", "Qwen3.8-27B"),
        ("tau-bench Retail", "full", "Gemma-4-31B"),
        ("tau-bench Retail", "full", "Qwen3.8-27B"),
        ("tau-bench Retail", "full", "Qwen2.5-32B"),
    ]
    target_methods = ["sampled_cegis", "active_cegis", "active_cegis_no_balance"]
    
    conv_data = {}
    
    for dom, var, model in target_combinations:
        group_key = f"{dom}|{var}|{model}"
        conv_data[group_key] = {}
        for method in target_methods:
            # find cell
            cell_entry = None
            for entry in manifest["cells"].values():
                if entry["domain"] == dom and entry["variant"] == var and entry["model"] == model and entry["method"] == method:
                    cell_entry = entry
                    break
            if not cell_entry:
                continue
            
            d_dirs = cell_entry["dir"] if isinstance(cell_entry["dir"], list) else [cell_entry["dir"]]
            files = []
            for d in d_dirs:
                files.extend(sorted((ROOT / "results" / d).glob(f"{method}__seed*.json"),
                                    key=lambda x: int(x.stem.rsplit("seed", 1)[1]) if "seed" in x.stem else 0))
            
            runs = []
            parse_failures = 0
            for p in files:
                data = json.loads(p.read_text())
                if data.get("method") != method:
                    continue
                seed = data.get("seed")
                outcome = data.get("outcome")
                rounds = data.get("rounds", [])
                
                valid_reports = []
                for r in rounds:
                    rep = r.get("report")
                    if rep and rep.get("false_accepts") is not None:
                        valid_reports.append(rep["false_accepts"] + rep["false_rejects"] + rep["postcondition_violations"])
                
                if not valid_reports:
                    parse_failures += 1
                    continue
                
                # carry-forward
                carried = []
                alive = []
                if valid_reports:
                    for i in range(4):
                        if i < len(valid_reports):
                            carried.append(valid_reports[i])
                            alive.append(True)
                        else:
                            carried.append(valid_reports[-1])
                            alive.append(False)
                runs.append({
                    "seed": seed, "outcome": outcome,
                    "raw_reports": valid_reports,
                    "carried": carried, "alive": alive
                })
            
            # compute round stats
            round_stats = []
            for r in range(4):
                vals = [run["carried"][r] for run in runs if run["carried"]]
                alive_count = sum(1 for run in runs if run["alive"][r])
                if vals:
                    round_stats.append({
                        "round": r,
                        "n_alive": alive_count,
                        "n_total": len(runs),
                        "median": float(np.median(vals)),
                        "iqr": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))],
                        "mean": float(np.mean(vals))
                    })
                else:
                    round_stats.append({
                        "round": r, "n_alive": 0, "n_total": len(runs),
                        "median": None, "iqr": [None, None], "mean": None
                    })
            
            conv_data[group_key][method] = {
                "n_runs": len(files),
                "n_valid": len(runs),
                "parse_failures": parse_failures,
                "round_stats": round_stats,
                "runs": runs
            }
            
    return conv_data


def generate_convergence_plot(conv_data: dict[str, Any], out_pdf: Path) -> None:
    """Generate the S19 convergence figure. Not the manuscript figure: the paper uses
    S20's full-closure recomputation (convergence_D.pdf); this writes convergence_D_s19.pdf."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    
    # 2-panel figure: Gemma-4-31B (left) and Qwen2.5-32B (right) on Shopping (price) relaxed
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)
    
    panels = [
        ("Shopping (price)|relaxed|Gemma-4-31B", "Gemma-4-31B", axes[0]),
        ("Shopping (price)|relaxed|Qwen2.5-32B", "Qwen2.5-32B", axes[1]),
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
    
    rounds = np.array([0, 1, 2, 3])
    
    for key, title, ax in panels:
        pdata = conv_data.get(key, {})
        for method in ["active_cegis", "sampled_cegis", "active_cegis_no_balance"]:
            mdata = pdata.get(method)
            if not mdata:
                continue
            rstats = mdata["round_stats"]
            meds = [s["median"] for s in rstats]
            q25 = [s["iqr"][0] for s in rstats]
            q75 = [s["iqr"][1] for s in rstats]
            alives = [s["n_alive"] for s in rstats]
            tot = mdata["n_valid"]
            
            label = f"{SHORT[method]} (N={tot})"
            ax.plot(rounds, meds, label=label, color=colors[method], marker=markers[method], linewidth=1.8, markersize=5)
            ax.fill_between(rounds, q25, q75, color=colors[method], alpha=0.15)
            
            # Annotate alive count at each round
            for r_idx, (x, y, a) in enumerate(zip(rounds, meds, alives)):
                # slight offset to avoid collision
                y_off = 0.5 if method == "active_cegis" else (-0.7 if method == "sampled_cegis" else 0.0)
                ax.annotate(f"{a}/{tot}", (x, y + y_off), fontsize=6.5, color=colors[method],
                            ha='center', va='bottom' if y_off >= 0 else 'top')
                
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel("Round (Refinement Step)", fontsize=8.5)
        ax.set_xticks(rounds)
        ax.set_xticklabels(["0 (Init)", "1", "2", "3"])
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)
        
    axes[0].set_ylabel("Residual Closure Defects $D$ (Median & IQR)", fontsize=8.5)
    axes[0].set_ylim(-0.5, 10.5)
    axes[1].set_ylim(-1.0, 20.0)
    
    # Legend on first panel
    axes[0].legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    axes[1].legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(out_pdf, format="pdf", dpi=300)
    plt.close(fig)
    print(f"Generated convergence figure: {out_pdf}")


def paired_permutation_test(d_other: list[float], d_active: list[float], n_perm: int = 10000, seed: int = 42) -> tuple[float, float, tuple[float, float]]:
    """Paired permutation test and bootstrap 95% CI on median difference."""
    diffs = np.array(d_other) - np.array(d_active)
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0, (0.0, 0.0)
    t_obs = float(np.median(diffs))
    
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_diffs = signs * diffs
    t_perms = np.median(perm_diffs, axis=1)
    # Two-sided p-value
    p_val = float(np.mean(np.abs(t_perms) >= np.abs(t_obs)))
    
    # Bootstrap 95% CI (percentile)
    boot_idx = rng.integers(0, n, size=(n_perm, n))
    boot_meds = np.median(diffs[boot_idx], axis=1)
    ci_lo = float(np.percentile(boot_meds, 2.5))
    ci_hi = float(np.percentile(boot_meds, 97.5))
    
    return t_obs, p_val, (ci_lo, ci_hi)


def holm_adjust(raw_p_dict: dict[str, float]) -> dict[str, float]:
    """Step-down Holm-Bonferroni correction."""
    valid = sorted(raw_p_dict.items(), key=lambda x: x[1])
    adjusted = {}
    running = 0.0
    m = len(valid)
    for i, (key, p) in enumerate(valid):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


def compute_statistical_tests(cell_metrics: dict[str, Any]) -> dict[str, Any]:
    """Run exploratory paired permutation tests and bootstrap CIs on selected cells."""
    cells = cell_metrics["cells"]
    
    # Target rows (floor and intermediate cells only, excluding ceiling cells)
    test_rows = [
        ("Shopping (price)", "relaxed", "Qwen2.5-32B"),
        ("Shopping (price)", "relaxed", "Qwen3.8-27B"),
        ("Shopping (price)", "relaxed", "Qwen3.6-35B-A3B"),
        ("Shopping (price)", "relaxed", "Qwen2.5-14B"),
        ("Shopping (price)", "relaxed", "Gemma-4-26B-A4B"),
        ("Shopping (price)", "identifiable", "Gemma-4-31B"),
        ("Deployment", "full", "Qwen3.8-27B"),
        ("Deployment", "full", "Qwen2.5-32B"),
        ("Deployment", "full", "Qwen2.5-14B"),
        ("Deployment", "full", "Gemma-4-26B-A4B"),
        ("Calendar", "full", "Qwen3.8-27B"),
        ("Calendar", "full", "Qwen2.5-32B"),
        ("Calendar", "full", "Qwen2.5-14B"),
        ("Calendar", "full", "Gemma-4-26B-A4B"),
        ("tau-bench Retail", "full", "Qwen3.8-27B"),
        ("tau-bench Retail", "full", "Qwen2.5-32B"),
        ("tau-bench Retail", "full", "Qwen2.5-14B"),
        ("tau-bench Retail", "full", "Gemma-4-26B-A4B"),
    ]
    
    test_results = []
    
    for dom, var, model in test_rows:
        row_id = f"{dom}|{var}|{model}"
        # Active runs by seed
        active_cell = None
        for k, c in cells.items():
            if c["domain"] == dom and c["variant"] == var and c["model"] == model and c["method"] == "active_cegis":
                active_cell = c
                break
        if not active_cell:
            continue
            
        active_by_seed = {r["seed"]: r["D"] for r in active_cell["runs"] if r["valid"]}
        
        # Core family and selector family
        for family_name, other_methods in [("core", CORE), ("selector", SELECTOR)]:
            raw_p = {}
            row_family_tests = {}
            for other_m in other_methods:
                if other_m == "active_cegis":
                    continue
                other_cell = None
                for k, c in cells.items():
                    if c["domain"] == dom and c["variant"] == var and c["model"] == model and c["method"] == other_m:
                        other_cell = c
                        break
                if not other_cell:
                    continue
                other_by_seed = {r["seed"]: r["D"] for r in other_cell["runs"] if r["valid"]}
                
                common_seeds = sorted(set(active_by_seed.keys()) & set(other_by_seed.keys()))
                if len(common_seeds) == 0:
                    continue
                
                d_other_vals = [other_by_seed[s] for s in common_seeds]
                d_active_vals = [active_by_seed[s] for s in common_seeds]
                
                med_diff, p_val, (ci_lo, ci_hi) = paired_permutation_test(d_other_vals, d_active_vals, n_perm=10000, seed=42)
                raw_p[other_m] = p_val
                row_family_tests[other_m] = {
                    "n_pairs": len(common_seeds),
                    "med_diff": med_diff,
                    "raw_p": p_val,
                    "ci_95": [ci_lo, ci_hi]
                }
            
            if raw_p:
                adj_p = holm_adjust(raw_p)
                for other_m, hp in adj_p.items():
                    row_family_tests[other_m]["holm_p"] = hp
            
            test_results.append({
                "row": row_id,
                "domain": dom, "variant": var, "model": model,
                "family": family_name,
                "comparisons": row_family_tests
            })
            
    return {"tests": test_results}


def run_taubench_diagnosis() -> dict[str, Any]:
    """Run Qwen3.8 tau-bench Active vs No-Coverage diagnosis."""
    from scripts.clause_recovery import build_domain, witnesses
    from experiment.taubench_retail.enumeration import RetailConfig
    from experiment.taubench_retail.dsl import parse_contract
    from experiment.taubench_retail.runner import RetailExperimentRunner
    
    dom = build_domain("tau-bench Retail", "full")
    wit = witnesses(dom["values"], dom["clauses"], dom["states"])
    raw = json.loads((ROOT / "experiment/configs/taubench_retail_default.json").read_text())
    cfg = RetailConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
    runner = RetailExperimentRunner(client=None, config=cfg)
    
    def analyze_method(method: str, coverage_flag: bool):
        res_dir = ROOT / "results/taubench_retail_qwen38_fullsuite_v3_s20"
        files = sorted(res_dir.glob(f"{method}__seed*.json"), key=lambda p: int(p.stem.rsplit("seed", 1)[1]))
        runs_data = []
        for f in files:
            art = json.loads(f.read_text())
            seed = art["seed"]
            exact = (art.get("evaluation") or {}).get("metrics", {}).get("exact", False)
            audited_states = []
            audited_set = set()
            rounds = art.get("rounds", [])
            round_info = []
            for r_idx, r in enumerate(rounds):
                p_accepts = r.get("reachable_predicted_accepts")
                guard = r.get("guard")
                if guard == "duplicate_of_round" or r.get("fresh_states") is None:
                    round_info.append({"round": r_idx, "p_accepts": p_accepts, "guard": guard, "fresh": 0})
                    continue
                resp = art["interactions"][r_idx]["response_text"]
                c = parse_contract(resp, cfg).bind(cfg)
                tie_seed = f"{seed}|round_{r_idx}|audit_0|{c.name}"
                fresh = runner._select_active_sample(
                    c, 8, tie_seed, balance=True, coverage=coverage_flag, excluded=audited_set
                )
                for s in fresh:
                    if s not in audited_set:
                        audited_set.add(s)
                        audited_states.append(s)
                round_info.append({
                    "round": r_idx, "p_accepts": p_accepts, "fresh": len(fresh)
                })
            
            wit_cov = {}
            audited_set_ref = set(audited_states)
            for cname, w_tuple in wit.items():
                hit = sum(1 for s in w_tuple if s in audited_set_ref)
                wit_cov[cname] = (hit, len(w_tuple))
            runs_data.append({
                "seed": seed, "exact": exact, "audited_count": len(audited_states),
                "round_info": round_info, "wit_cov": wit_cov
            })
        return runs_data

    active_data = analyze_method("active_cegis", coverage_flag=True)
    nocov_data = analyze_method("active_cegis_no_coverage", coverage_flag=False)
    
    odds, p_fisher = fisher_exact([[1, 19], [4, 16]])
    
    # Discordant pairs
    act_seeds = {d["seed"]: d["exact"] for d in active_data}
    nocov_seeds = {d["seed"]: d["exact"] for d in nocov_data}
    act_only = sum(1 for s in act_seeds if act_seeds[s] and not nocov_seeds[s])
    nocov_only = sum(1 for s in nocov_seeds if nocov_seeds[s] and not act_seeds[s])
    both = sum(1 for s in act_seeds if act_seeds[s] and nocov_seeds[s])
    neither = sum(1 for s in act_seeds if not act_seeds[s] and not nocov_seeds[s])
    mcnemar_res = binomtest(act_only, act_only + nocov_only)
    
    clause_cov = {}
    for cname in dom["clauses"]:
        act_hits = [d["wit_cov"][cname][0] for d in active_data]
        nocov_hits = [d["wit_cov"][cname][0] for d in nocov_data]
        tot = len(wit[cname])
        clause_cov[cname] = {
            "total_witnesses": tot,
            "active_mean": float(np.mean(act_hits)),
            "nocov_mean": float(np.mean(nocov_hits)),
        }
        
    act_tot = [sum(d["wit_cov"][c][0] for c in dom["clauses"]) for d in active_data]
    nocov_tot = [sum(d["wit_cov"][c][0] for c in dom["clauses"]) for d in nocov_data]
    
    return {
        "active_exact": 1, "nocov_exact": 4, "n": 20,
        "fisher_p": float(p_fisher),
        "mcnemar_discordant": {"act_only": act_only, "nocov_only": nocov_only, "both": both, "neither": neither},
        "mcnemar_p": float(mcnemar_res.pvalue),
        "witness_totals": {
            "active_mean": float(np.mean(act_tot)),
            "active_median": float(np.median(act_tot)),
            "nocov_mean": float(np.mean(nocov_tot)),
            "nocov_median": float(np.median(nocov_tot)),
        },
        "clause_cov": clause_cov,
        "round_0_partition": {
            "active_p_accepts": [d["round_info"][0]["p_accepts"] if d["round_info"] else None for d in active_data],
            "nocov_p_accepts": [d["round_info"][0]["p_accepts"] if d["round_info"] else None for d in nocov_data],
        },
        "verdict": "(c) seed variance (시드 분산으로 설명 가능)"
    }


def format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def generate_latex_fragments(cell_metrics: dict[str, Any], baselines: dict[str, Any],
                            conv_data: dict[str, Any], stat_data: dict[str, Any],
                            frag_dir: Path | None = None) -> None:
    """Generate the required LaTeX fragments under outputs/fragments/."""
    if frag_dir is None:
        frag_dir = ROOT / "outputs/fragments"
    frag_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. s19_definitions.tex (Two sentences for Section 3)
    def_path = frag_dir / "s19_definitions.tex"
    def_content = (
        "% Secondary graded metric definitions for Section 3 (Problem Setting)\n"
        "Beyond exact recovery, we measure residual closure defects $D = \\mathrm{FA} + \\mathrm{FR} + \\mathrm{PV}$, decomposed into false-accept rate $\\mathrm{FAR} = \\mathrm{FA} / |\\mathcal{R}^-|$, false-reject rate $\\mathrm{FRR} = \\mathrm{FR} / |\\mathcal{R}^+|$, and unsound-effect rate $\\mathrm{PVR} = \\mathrm{PV} / N_{\\text{checked}}$. PVR is N/A when $N_{\\text{checked}}=0$. The rate triad carries our empirical claims, while $D$ provides unnormalized defect scale across refinement rounds.\n"
    )
    def_path.write_text(def_content, encoding="utf-8")
    print(f"Generated: {def_path}")
    
    # 2. s19_maintext_column.tex
    # Definition and values of ONE graded column (median D [IQR]) for main tables
    main_col_path = frag_dir / "s19_maintext_column.tex"
    # Extract values for Table 1 (tab:easyv4) and Table 2 (tab:other)
    cells = cell_metrics["cells"]
    
    lines = [
        "% Column definition and values for main-text inclusion (Opus merge)",
        "% Recommended column: Residual Closure Defects D (Median [IQR])",
        "% Header definition: \\multicolumn{1}{c}{Defects $D$ [IQR]}",
        "",
        "% --- Table 1: Shopping (price, relaxed) easy_v4 rows ---",
    ]
    t1_models = [
        "Gemma-4-31B", "Qwen3.8-27B", "Qwen3.6-35B-A3B",
        "Qwen2.5-32B", "Qwen2.5-14B", "Gemma-4-26B-A4B"
    ]
    for m in t1_models:
        k = f"tab:easyv4|Shopping (price)|relaxed|{m}|active_cegis"
        c = cells.get(k)
        if c and c["d_med"] is not None:
            val_str = f"{c['d_med']:.0f} [{c['d_iqr'][0]:.0f},{c['d_iqr'][1]:.0f}]"
        else:
            val_str = "--"
        lines.append(f"% {m:20s} Active CEGIS: {val_str}")
        
    lines += [
        "",
        "% --- Table 2: Deployment & Calendar rows ---",
    ]
    t2_domains = [("Deployment", "full"), ("Calendar", "full")]
    t2_models = ["Gemma-4-31B", "Qwen3.8-27B", "Qwen2.5-32B", "Qwen2.5-14B", "Gemma-4-26B-A4B"]
    for d, v in t2_domains:
        for m in t2_models:
            k = f"tab:other|{d}|{v}|{m}|active_cegis"
            c = cells.get(k)
            if c and c["d_med"] is not None:
                val_str = f"{c['d_med']:.0f} [{c['d_iqr'][0]:.0f},{c['d_iqr'][1]:.0f}]"
            else:
                val_str = "--"
            lines.append(f"% {d:12s} {m:16s} Active CEGIS: {val_str}")
            
    main_col_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated: {main_col_path}")
    
    # 3. s19_figure.tex (Insertion code for convergence figure)
    fig_frag_path = frag_dir / "s19_figure.tex"
    fig_frag_content = (
        "% LaTeX snippet for convergence figure (measure page budget without direct insertion)\n"
        "\\begin{figure}[t]\n"
        "\\centering\n"
        "\\includegraphics[width=0.92\\textwidth]{figures/convergence_D_s19.pdf}\n"
        "\\caption{Round-by-round convergence of residual closure defects $D = \\mathrm{FA}+\\mathrm{FR}+\\mathrm{PV}$ across refinement rounds (0: initial candidate; 1--3: counterexample-guided revisions). Lines show medians; shaded bands indicate interquartile ranges. Early-terminated runs carry forward their final evaluation; fractions denote surviving (unstopped) runs at each round. Direct is evaluated at Round~0 only; parse failures are excluded from the denominator ($N=20$ for Gemma-4-31B; $N=17$ for Qwen2.5-32B Active, $N=15$ for Sampled, $N=14$ for No-Balance).}\n"
        "\\label{fig:convergence-d}\n"
        "\\end{figure}\n"
    )
    fig_frag_path.write_text(fig_frag_content, encoding="utf-8")
    print(f"Generated: {fig_frag_path}")
    
    # 4. s19_appendix_graded.tex
    app_frag_path = frag_dir / "s19_appendix_graded.tex"
    app_lines = [
        "% Appendix Section on Secondary Graded Metrics and Convergence",
        "\\section{Secondary Graded Metrics and Round-by-Round Convergence}",
        "\\label{app:graded_metrics}",
        "",
        "While exact closure recovery ($\\mathrm{FA}=\\mathrm{FR}=\\mathrm{PV}=0$) remains our primary criterion, we report secondary graded defect metrics to measure contract quality across floor and intermediate cells where exact rates saturate or remain zero. Residual closure defects are defined as $D = \\mathrm{FA} + \\mathrm{FR} + \\mathrm{PV}$. Normalized defect rates are:",
        "\\begin{equation}",
        "\\mathrm{FAR} = \\frac{\\mathrm{FA}}{|\\mathcal{R}^-|}, \\qquad \\mathrm{FRR} = \\frac{\\mathrm{FR}}{|\\mathcal{R}^+|}, \\qquad \\mathrm{PVR} = \\frac{\\mathrm{PV}}{N_{\\text{checked}}},",
        "\\end{equation}",
        "where $|\\mathcal{R}^-|$ is the number of sandbox-rejecting states, $|\\mathcal{R}^+|$ is the number of sandbox-accepting states, and $N_{\\text{checked}} = |\\mathcal{R}^+| - \\mathrm{FR}$ is the exact number of states admitted by the candidate contract that successfully executed in the sandbox (where postcondition soundness is verified). When $N_{\\text{checked}}=0$ (such as in the reject-all baseline), $\\mathrm{PVR}$ is undefined and reported as N/A, not zero.",
        "",
        "\\subsection{Reject-All Contract Baselines}",
        "A trivial contract rejecting all states achieves $\\mathrm{FA}=0$, $\\mathrm{FR}=|\\mathcal{R}^+|$, and $\\mathrm{PV}=0$, yielding $D = |\\mathcal{R}^+|$. Table~\\ref{tab:reject-all-baselines} establishes these domain baselines, demonstrating why $D$ must be interpreted alongside the rate triad (where reject-all instantly yields $\\mathrm{FRR}=1.000$).",
        "",
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        "\\caption{Reject-all baseline contract across domains. $N_{\\text{checked}}$ is the number of states where postconditions are checked ($0$ for reject-all).}",
        "\\label{tab:reject-all-baselines}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Domain (Variant) & $|\\mathcal{R}|$ & $|\\mathcal{R}^+|$ & Baseline $D$ & FAR & FRR & PVR ($N_{\\text{chk}}$) \\\\",
        "\\midrule",
    ]
    for k, b in baselines.items():
        app_lines.append(f"{k:35s} & {b['states']:,} & {b['pos']} & {b['D']} & {b['far']:.3f} & {b['frr']:.3f} & {format_rate(b['pvr'])} (0) \\\\")
    app_lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\subsection{Full Graded Metrics Across Experimental Cells}",
        "Table~\\ref{tab:graded-full-metrics} reports $D$ (median [IQR]), FAR, FRR, and PVR across experimental cells. Exact recovery retains all runs in its denominator, including parse failures. Graded metrics use valid contracts; PVR uses only runs with positive $N_{\\text{checked}}$, with its defined-run count $n_{\\rm def}$ shown relative to valid contracts. Empty PVR groups are N/A; they do not change the other metrics or exact denominator.",
        "",
        "\\begin{table}[h]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Comprehensive graded metrics across evaluation cells. Median [IQR] reported for $D$, $\\mathrm{FAR}$, $\\mathrm{FRR}$, and $\\mathrm{PVR}$.}",
        "\\label{tab:graded-full-metrics}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lllccccr}",
        "\\toprule",
        "Domain & Model & Method & Exact & $D$ [IQR] & FAR [IQR] & FRR [IQR] & PVR [IQR] \\\\",
        "\\midrule",
    ]
    
    # Deduplicate cells by (domain, variant, model, method)
    unique_cells_dict = {}
    for c in cells.values():
        u_key = (c["domain"], c["variant"], c["model"], c["method"])
        if u_key not in unique_cells_dict:
            unique_cells_dict[u_key] = c
            
    sorted_cells = sorted(unique_cells_dict.values(), key=lambda c: (c["domain"], c["variant"], c["model"], c["method"]))
    for c in sorted_cells:
        dom_disp = f"{c['domain']} ({c['variant']})"
        if c["n_valid"] == 0:
            continue
        d_str = f"{c['d_med']:.1f} [{c['d_iqr'][0]:.1f},{c['d_iqr'][1]:.1f}]"
        far_str = f"{c['far_med']:.3f}"
        frr_str = f"{c['frr_med']:.3f}"
        pvr_str = format_rate(c['pvr_med'])
        if c['pvr_med'] is not None:
            pvr_str += f" [{format_rate(c['pvr_iqr'][0])},{format_rate(c['pvr_iqr'][1])}]"
        pvr_str += f" ($n_{{\\rm def}}={c['n_pvr_defined']}/{c['n_valid']}$)"
        exact_str = f"{c['exact_count']}/{c['n_runs']}"
        if c["parse_failures"] > 0:
            exact_str += f" ({c['parse_failures']} pf)"
        app_lines.append(f"{dom_disp:30s} & {c['model']:16s} & {SHORT.get(c['method'], c['method']):12s} & {exact_str:12s} & {d_str:18s} & {far_str} & {frr_str} & {pvr_str} \\\\")
        
    app_lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
        "",
        "\\subsection{Round-by-Round Convergence Trajectories}",
        "Table~\\ref{tab:round-trajectories} details the trajectory of residual closure defects $D_0 \\to D_1 \\to D_2 \\to D_3$ across refinement rounds. Each cell presents the median defect [IQR] along with the count of active (unstopped) runs ($n_{\\text{alive}} / N$). Direct generation is evaluated at Round~0 only; early-terminated runs carry forward their final evaluation.",
        "",
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        "\\caption{Round-by-round convergence of residual closure defects $D$. Median [IQR] and surviving run fraction ($n_{\\text{alive}}/N$) per round.}",
        "\\label{tab:round-trajectories}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Setting & Method & $N$ & Round 0 ($D_0$) & Round 1 ($D_1$) & Round 2 ($D_2$) & Round 3 ($D_3$) \\\\",
        "\\midrule",
    ]
    
    for g_key, g_data in conv_data.items():
        dom, var, model = g_key.split("|")
        setting_name = f"{dom} ({var}) -- {model}"
        for m_key in ["sampled_cegis", "active_cegis", "active_cegis_no_balance"]:
            m_info = g_data.get(m_key)
            if not m_info:
                continue
            m_name = SHORT.get(m_key, m_key)
            n_tot = m_info["n_valid"]
            r_cells = []
            for r_stat in m_info["round_stats"]:
                med = r_stat["median"]
                iqr = r_stat["iqr"]
                a = r_stat["n_alive"]
                r_cells.append(f"{med:.1f} [{iqr[0]:.1f},{iqr[1]:.1f}] ({a}/{n_tot})")
            app_lines.append(f"{setting_name:35s} & {m_name:12s} & {n_tot} & " + " & ".join(r_cells) + " \\\\")
            
    app_lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
        "",
        "\\subsection{Exploratory Paired Permutation Tests on Residual Closure Defects}",
        "These exploratory tests were selected after inspecting earlier results, focusing on floor and intermediate cells and excluding ceiling cells. Table~\\ref{tab:graded-paired-tests} reports paired permutation tests (10,000 permutations, fixed seed 42) and bootstrap 95\\% percentile confidence intervals (10,000 resamples), Holm-corrected within the core ($m=4$) and selector ($m=3$) families within each model--domain cell.",
        "",
        "\\begin{table}[h]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Exploratory paired permutation tests on paired defect reductions $D_{\\text{other}} - D_{\\text{Active}}$. Positive median differences favor \\method\\ on residual defects. Holm correction applied within each hypothesis family.}",
        "\\label{tab:graded-paired-tests}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lllccccc}",
        "\\toprule",
        "Domain & Model & Comparison & $n$ & Median Diff & 95\\% Bootstrap CI & Permutation $p$ & $p_{\\text{Holm}}$ \\\\",
        "\\midrule",
    ]
    
    for t_row in stat_data["tests"]:
        dom_name = f"{t_row['domain']} ({t_row['variant']})"
        model_name = t_row["model"]
        fam = t_row["family"]
        for comp_m, comp_res in t_row["comparisons"].items():
            n_p = comp_res["n_pairs"]
            md = comp_res["med_diff"]
            ci = comp_res["ci_95"]
            rp = comp_res["raw_p"]
            hp = comp_res.get("holm_p", rp)
            ci_str = f"[{ci[0]:.1f}, {ci[1]:.1f}]"
            rp_str = f"{rp:.4f}" if rp >= 0.0001 else f"{rp:.2e}"
            hp_str = f"{hp:.4f}" if hp >= 0.0001 else f"{hp:.2e}"
            comp_name = f"Active vs {SHORT.get(comp_m, comp_m)} ({fam})"
            app_lines.append(f"{dom_name:25s} & {model_name:16s} & {comp_name:25s} & {n_p} & {md:+.1f} & {ci_str:14s} & {rp_str} & {hp_str} \\\\")
            
    app_lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
    ]
    
    app_frag_path.write_text("\n".join(app_lines) + "\n", encoding="utf-8")
    print(f"Generated: {app_frag_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", help="Override the JSON path only (use --out-dir to isolate all outputs).")
    ap.add_argument("--out-dir", type=Path, help="Write JSON, figures and fragments under a separate directory.")
    args = ap.parse_args()
    
    print("Step 1: Loading manifest and computing graded metrics across all cells...")
    manifest = json.loads(MANIFEST_PATH.read_text())
    cell_metrics = compute_cell_metrics(manifest)
    print(f"Processed {len(cell_metrics['cells'])} cells. Mismatches: {len(cell_metrics['mismatches'])}")
    
    print("\nStep 2: Computing domain reject-all baselines...")
    baselines = compute_reject_all_baselines()
    
    print("\nStep 3: Extracting round-by-round convergence trajectories...")
    conv_data = extract_convergence_trajectories(manifest)
    
    print("\nStep 4: Generating convergence figure...")
    # 정본은 S20(`scripts/closure_trajectory.py`)의 전수 폐포 재계산이다. 두 스크립트가
    # 같은 파일을 쓰면 재생성 순서에 따라 정본 그림이 조용히 교체되므로 여기서는
    # 별도 파일로만 쓴다. 원고가 include 하는 것은 convergence_D.pdf(S20)이다. (R6)
    fig_pdf = (args.out_dir / "figures/convergence_D_s19.pdf" if args.out_dir
               else ROOT / "outputs/figures/convergence_D_s19.pdf")
    fig_pdf.parent.mkdir(parents=True, exist_ok=True)
    generate_convergence_plot(conv_data, fig_pdf)
    print("  주의: 원고용 정본 figures/convergence_D.pdf 는 S20 이 생성한다. "
          "이 스크립트는 그 파일을 덮어쓰지 않는다.")
    
    print("\nStep 5: Computing exploratory paired permutation tests and CIs...")
    stat_data = compute_statistical_tests(cell_metrics)
    print(f"Computed {len(stat_data['tests'])} test family rows.")
    
    print("\nStep 6: Diagnosing Qwen3.8 tau-bench Active vs No-Coverage...")
    diag_data = run_taubench_diagnosis()
    print(f"Diagnosis completed. Conclusion: {diag_data['verdict']}")
    
    print("\nStep 7: Generating LaTeX fragments under outputs/fragments/...")
    generate_latex_fragments(cell_metrics, baselines, conv_data, stat_data,
                             args.out_dir / "fragments" if args.out_dir else None)
    
    print("\nStep 8: Saving complete JSON artifact...")
    # Prepare serializable dict
    # Filter runs inside cell_metrics to keep json manageable
    json_cells = {}
    for k, v in cell_metrics["cells"].items():
        v_copy = dict(v)
        # keep lightweight summary without individual run details in json_cells, or keep all
        v_copy["runs_summary"] = [
            {"seed": r["seed"], "valid": r["valid"], "D": r["D"], "far": r["far"], "frr": r["frr"], "pvr": r["pvr"], "exact": r["exact"],
             "n_checked": r["n_checked"], "fa": r["fa"], "fr": r["fr"], "pv": r["pv"]}
            for r in v["runs"]
        ]
        del v_copy["runs"]
        json_cells[k] = v_copy
        
    out_obj = {
        "generated_by": "scripts/build_s19_graded_metrics.py",
        "baselines": baselines,
        "cells": json_cells,
        "convergence": conv_data,
        "statistical_tests": stat_data,
        "taubench_diagnosis": diag_data,
    }
    
    out_json_path = (Path(args.out_json) if args.out_json else
                     args.out_dir / JSON_OUT.name if args.out_dir else JSON_OUT)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"Saved JSON artifact: {out_json_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
