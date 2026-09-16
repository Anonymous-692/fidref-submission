#!/usr/bin/env python3
"""Aggregate the contract-repair comparison.

Reports performance *and* the repair-opportunity accounting the pilot showed is essential:
a seed whose observed pool is already clean at round 0 never enters repair. Those seeds are
"no patch opportunity", not failed patches, and lumping them together would misread a
counterexample-finding bottleneck as a repair-mechanism result.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

CONDITIONS = ["repair_free_pool", "repair_patch_pool", "repair_free_active"]
LABEL = {"repair_free_pool": "A free-rewrite/pool",
         "repair_patch_pool": "B patch/pool",
         "repair_free_active": "C free-rewrite/active"}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / den
    return (c - h, c + h)


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def load(directory: str):
    runs: dict[tuple[str, int], dict] = {}
    for f in sorted(glob.glob(f"{directory}/*seed*.json")):
        a = json.load(open(f))
        runs[(a["method"], a["seed"])] = a
    return runs


def main() -> int:
    directory = sys.argv[1] if len(sys.argv) > 1 else "results/deployment_qwen14b_repair_core_v1_pilot2"
    runs = load(directory)
    seeds = sorted({s for _, s in runs})
    print(f"# contract-repair comparison — {directory}")
    print(f"seeds: {seeds}\n")

    print("## outcome and repair opportunity")
    hdr = (f"{'condition':22} {'exact':>12} {'pre-complete':>12} {'entered':>8} "
           f"{'proposed':>9} {'adopted':>8} {'rejected':>9} {'invalid':>8} {'calls':>6}")
    print(hdr)
    table = {}
    for cond in CONDITIONS:
        rs = [runs[(cond, s)] for s in seeds if (cond, s) in runs]
        if not rs:
            continue
        n = len(rs)
        ex = sum(r.get("outcome") == "exact" for r in rs)
        met = [(r.get("evaluation") or {}).get("metrics") or {} for r in rs]
        pc = sum(1 for m in met if m.get("false_accepts") == 0 and m.get("false_rejects") == 0)
        rc = [r.get("repair_core") or {} for r in rs]
        entered = sum(1 for x in rc if x.get("patches_proposed", 0) > 0)
        proposed = sum(x.get("patches_proposed", 0) for x in rc)
        adopted = sum(x.get("patches_adopted", 0) for x in rc)
        rejected = sum(x.get("patches_rejected", 0) for x in rc)
        invalid = sum(1 for r in rs for rd in r.get("rounds", [])
                      if rd.get("patch_valid") is False)
        calls = sum(x.get("loop_model_calls", 0) for x in rc) / n
        lo, hi = wilson(ex, n)
        table[cond] = dict(n=n, exact=ex, pc=pc)
        print(f"{LABEL[cond]:22} {ex:2d}/{n:<2d}[{lo:.2f},{hi:.2f}] {pc:3d}/{n:<8d} "
              f"{entered:2d}/{n:<5d} {proposed:9d} {adopted:8d} {rejected:9d} {invalid:8d} {calls:6.2f}")

    print("\n## why runs did not enter repair (round-0 observed pool already clean)")
    for cond in CONDITIONS:
        rs = [(s, runs[(cond, s)]) for s in seeds if (cond, s) in runs]
        clean = [s for s, r in rs
                 if (r.get("repair_core") or {}).get("patches_proposed", 0) == 0]
        print(f"  {LABEL[cond]:22} no-opportunity seeds: {clean}")

    print("\n## closure defects vs observed defects (the counterexample-finding bottleneck)")
    for cond in CONDITIONS:
        for s in seeds:
            r = runs.get((cond, s))
            if r is None:
                continue
            m = (r.get("evaluation") or {}).get("metrics") or {}
            d_cl = (m.get("false_accepts", 0) + m.get("false_rejects", 0)
                    + m.get("postcondition_violations", 0))
            r0 = next((x for x in r.get("rounds", []) if x.get("round") == 0), {})
            d_ob = (r0.get("observed") or {}).get("d_obs")
            print(f"  {LABEL[cond]:22} seed{s}: observed D0={d_ob}  closure D={d_cl}  "
                  f"outcome={r.get('outcome')}  stop={r.get('stopped_because')}")

    print("\n## paired tests on closure-exact (one Holm family, m=3; primary question is A-B)")
    raw = {}
    for a, b in (("repair_free_pool", "repair_patch_pool"),
                 ("repair_free_pool", "repair_free_active"),
                 ("repair_patch_pool", "repair_free_active")):
        common = [s for s in seeds if (a, s) in runs and (b, s) in runs]
        ba = sum(1 for s in common if runs[(a, s)]["outcome"] == "exact"
                 and runs[(b, s)]["outcome"] != "exact")
        ab = sum(1 for s in common if runs[(b, s)]["outcome"] == "exact"
                 and runs[(a, s)]["outcome"] != "exact")
        raw[f"{LABEL[a]} vs {LABEL[b]}"] = mcnemar(ba, ab)
        print(f"  {LABEL[a]:22} vs {LABEL[b]:22} n={len(common)} {ba}:{ab} p={mcnemar(ba, ab):.4f}")
    adj = holm(raw)
    for k, v in adj.items():
        print(f"  Holm  {k:48} {v:.4f}")
    print("\nA non-significant result is 'no difference detected', never equivalence. Seeds are "
          "repetitions of one task, not different tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
