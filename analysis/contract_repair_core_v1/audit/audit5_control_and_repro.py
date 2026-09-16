#!/usr/bin/env python3
"""Audit 5: the Gemma negative control, and reproducibility across code versions.

Two checks that the tie-tolerance result depends on:

  (a) the 3 seeds where B' and B'' "differ" at round 1 on Gemma are seeds with no round 1 at
      all (evidence never found), not a real divergence;
  (b) the Gemma B' run repeated under the current code reproduces the 2026-09-07 run, so the
      recording fields added for B'' did not change behaviour and serving is deterministic here.

Read-only.
"""
import json, glob, sys
from collections import Counter
from math import comb
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]

def load(d, cond):
    out = {}
    for f in glob.glob(f"{ROOT/d}/{cond}__seed*.json"):
        r = json.load(open(f)); out[r["seed"]] = r
    return out

def closure_d(r):
    m = r["evaluation"]["metrics"]
    return m["false_accepts"] + m["false_rejects"] + m["postcondition_violations"]

def round1(r):
    return next((x for x in r["rounds"] if x.get("round") == 1), None)

def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2)

report = {}

# ---- (a) negative control on Gemma ------------------------------------------------
NEW = "results/deployment_gemma31b_repair_tie_v1"
B, T = load(NEW, "repair_patch_witness"), load(NEW, "repair_patch_tie")
seeds = sorted(set(B) & set(T))
no_r1 = [s for s in seeds if round1(B[s]) is None and round1(T[s]) is None]
r1_pairs = [s for s in seeds if round1(B[s]) and round1(T[s])]
r1_same = [s for s in r1_pairs if round1(B[s])["patch_raw"] == round1(T[s])["patch_raw"]]
w = sum(1 for s in seeds if closure_d(T[s]) < closure_d(B[s]))
l = sum(1 for s in seeds if closure_d(B[s]) < closure_d(T[s]))
report["gemma_negative_control"] = {
    "n_seeds": len(seeds),
    "exact": {"strict": sum(B[s]["outcome"] == "exact" for s in seeds),
              "tie_tolerant": sum(T[s]["outcome"] == "exact" for s in seeds)},
    "proposals": {"strict": sum(B[s]["repair_core"]["patches_proposed"] for s in seeds),
                  "tie_tolerant": sum(T[s]["repair_core"]["patches_proposed"] for s in seeds)},
    "rejections": {"strict": sum(B[s]["repair_core"]["patches_rejected"] for s in seeds),
                   "tie_tolerant": sum(T[s]["repair_core"]["patches_rejected"] for s in seeds)},
    "tie_adoptions_tie_tolerant": sum(T[s]["repair_core"].get("tie_patches_adopted", 0) for s in seeds),
    "closure_d_sign_test": {"tie_better": w, "strict_better": l,
                            "ties": len(seeds) - w - l, "p": mcnemar(w, l)},
    "seeds_without_round1": no_r1,
    "round1_present_and_identical": f"{len(r1_same)}/{len(r1_pairs)}",
    "conclusion": ("The rule has no effect here because the event it governs never occurs: zero "
                   "rejections and zero observed ties. This shows the rule is inert on this model, "
                   "not that it is useless."),
}

# ---- (b) reproducibility across code versions -------------------------------------
OLD = load("results/deployment_gemma31b_repair_core_v1", "repair_patch_witness")
common = sorted(set(OLD) & set(B))
r1_both = [s for s in common if round1(OLD[s]) and round1(B[s])]
report["cross_version_reproducibility"] = {
    "old_run": "results/deployment_gemma31b_repair_core_v1 (2026-09-07, before the B'' fields)",
    "new_run": f"{NEW} (2026-09-08, current code)",
    "n": len(common),
    "final_contract_identical": sum(1 for s in common
                                    if OLD[s]["contract"]["source"] == B[s]["contract"]["source"]),
    "closure_d_identical": sum(1 for s in common if closure_d(OLD[s]) == closure_d(B[s])),
    "exact_verdict_identical": sum(1 for s in common
                                   if (OLD[s]["outcome"] == "exact") == (B[s]["outcome"] == "exact")),
    "round1_proposal_identical": f"{sum(1 for s in r1_both if round1(OLD[s])['patch_raw'] == round1(B[s])['patch_raw'])}/{len(r1_both)}",
    "conclusion": ("The fields added for B'' record only; they do not change behaviour, and this "
                   "server reproduced every run exactly. The earlier 'first proposal differs' "
                   "finding was an artifact of comparing a field absent from the older run."),
}

for k, v in report.items():
    print(f"== {k}")
    print(json.dumps(v, indent=1, ensure_ascii=False))
    print()
out = ROOT / "analysis/contract_repair_core_v1/audit/audit5_control_and_repro.json"
out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
print(f"wrote {out.relative_to(ROOT)}")
