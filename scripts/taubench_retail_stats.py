#!/usr/bin/env python3
"""taubench_retail 결과 통계: exact/20 + Wilson 95% CI, McNemar vs Direct/Active, Holm 보정.

사용: .venv/bin/python scripts/taubench_retail_stats.py [--out analysis/taubench_retail_stats_2026-09-04.txt]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

MODELS = {
    "Qwen2.5-14B": "results/taubench_retail_qwen14b_fullsuite_v3b_s20",
    "Gemma-4-31B": "results/taubench_retail_gemma4_fullsuite_v3_s20",
    "Qwen3.8-27B": "results/taubench_retail_qwen38_fullsuite_v3_s20",
    "Qwen2.5-32B": "results/taubench_retail_qwen32b_fullsuite_v3_s20",
}
METHODS = [
    "direct",
    "self_refine",
    "random_probe",
    "sampled_cegis",
    "active_cegis",
    "active_cegis_no_balance",
    "active_cegis_no_coverage",
    "active_cegis_uniform",
]
SHORT = {
    "direct": "Direct",
    "self_refine": "Self-Refine",
    "random_probe": "Random",
    "sampled_cegis": "Sampled",
    "active_cegis": "Active",
    "active_cegis_no_balance": "No-Balance",
    "active_cegis_no_coverage": "No-Coverage",
    "active_cegis_uniform": "Uniform",
}
CORE_METHODS = ["self_refine", "random_probe", "sampled_cegis", "active_cegis"]
SELECTOR_METHODS = ["active_cegis_no_balance", "active_cegis_no_coverage", "active_cegis_uniform"]
EXPECTED_SANDBOX = "1c3fb534"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """b = A성공·B실패, c = A실패·B성공. 양측 exact binomial."""
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    # Filter out NaNs for ranking
    valid = {k: v for k, v in pvals.items() if not math.isnan(v)}
    items = sorted(valid.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    for k in pvals:
        if k not in out:
            out[k] = float("nan")
    return out


def load(path_str: str) -> dict[tuple[str, int], dict]:
    runs: dict[tuple[str, int], dict] = {}
    p = Path(path_str)
    if not p.exists():
        return runs
    for fp in p.glob("*seed*.json"):
        if fp.name.endswith("summary.json"):
            continue
        try:
            r = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "method" not in r or "seed" not in r:
            continue
        if r.get("hashes", {}).get("sandbox_config_sha256", "")[:8] != EXPECTED_SANDBOX:
            continue
        key = (r["method"], int(r["seed"]))
        if key in runs:
            raise SystemExit(f"중복 (method, seed) {key}: {fp} vs 이전 항목")
        runs[key] = r
    return runs


def generate_report() -> str:
    lines: list[str] = []
    P = lines.append
    P("# τ-bench Retail 스위트 통계 분석 (1,294 states, seed 0–19)\n")
    P("작성 일시: 2026-09-04 | 평가 도메인: `taubench_retail` (`exchange_delivered_order_items`)\n")
    P("> **주의(B17 감사).** 이 표의 `Sampled` 열은 레거시 arm 이며 실제로는 후보 분할 균일 표집을, "
      "`Random` 은 반복 Uniform 정책을 수행한다. 다른 도메인의 고정 풀·1회성 대조군이 아니므로 "
      "**원고 tab:taubench 의 Fixed-pool 열과 같지 않다**. 고정 풀 수치는 `scripts/build_s17_stats.py` 가 산출한다.\n")
    P("exact/20 [Wilson 95% CI] · 호출 수 평균 · parse failure 수. 셀이 20런 미만이면 `n=`을 표기한다.\n")

    hdr = "| 모델 | " + " | ".join(SHORT[m] for m in METHODS) + " |"
    P(hdr)
    P("|" + " :-- |" + " ---: |" * len(METHODS))

    all_model_runs: dict[str, dict[tuple[str, int], dict]] = {}
    for model, path_str in MODELS.items():
        runs = load(path_str)
        all_model_runs[model] = runs
        if not runs:
            cells = ["·" for _ in METHODS]
            P(f"| {model} | " + " | ".join(cells) + " |")
            continue

        by_m: dict[str, dict[int, bool]] = defaultdict(dict)
        for (m, s), r in runs.items():
            exact = bool(
                r.get("outcome") == "exact"
                or (r.get("evaluation") or {}).get("metrics", {}).get("exact")
            )
            by_m[m][s] = exact

        cells = []
        for m in METHODS:
            d = by_m.get(m, {})
            n = len(d)
            k = sum(d.values())
            if n == 0:
                cells.append("·")
                continue
            lo, hi = wilson(k, n)
            calls = sum(len(runs[(m, s)].get("interactions", [])) for s in d) / n
            pf = sum(1 for s in d if runs[(m, s)].get("outcome") == "parse_failure")
            cell = f"**{k}/{n}**" if m == "active_cegis" else f"{k}/{n}"
            cell += f" [{lo:.2f},{hi:.2f}] c={calls:.2f} pf={pf}"
            if n != 20:
                cell += f" n={n}"
            cells.append(cell)
        P(f"| {model} | " + " | ".join(cells) + " |")

    # Detailed statistics per model
    for model, runs in all_model_runs.items():
        if not runs:
            continue
        P(f"\n---\n\n## {model} 상세 분석\n")

        by_m = defaultdict(dict)
        for (m, s), r in runs.items():
            exact = bool(
                r.get("outcome") == "exact"
                or (r.get("evaluation") or {}).get("metrics", {}).get("exact")
            )
            by_m[m][s] = exact

        # 1. Stopped because distribution & sample satisfaction vs exact
        P("### 종료 원인 (`stopped_because`) 및 표본 과적합 분포\n")
        P("| 방법 | 완료 수 | query_budget | vacuous_unrepaired | dup_output | method_complete | sample_sat (exact=T) | sample_sat (exact=F) |")
        P("| :-- | --: | --: | --: | --: | --: | --: | --: |")
        for m in METHODS:
            seeds_for_m = [s for (method, s) in runs.keys() if method == m]
            n_m = len(seeds_for_m)
            if n_m == 0:
                continue
            sb_counts = Counter(runs[(m, s)].get("stopped_because") for s in seeds_for_m)
            sample_sat_exact_t = sum(
                1
                for s in seeds_for_m
                if runs[(m, s)].get("stopped_because") == "sampled_oracle_satisfied"
                and by_m[m][s]
            )
            sample_sat_exact_f = sum(
                1
                for s in seeds_for_m
                if runs[(m, s)].get("stopped_because") == "sampled_oracle_satisfied"
                and not by_m[m][s]
            )
            P(
                f"| {SHORT[m]} | {n_m} | {sb_counts.get('query_budget_exhausted', 0)} | "
                f"{sb_counts.get('vacuous_candidate_unrepaired', 0)} | "
                f"{sb_counts.get('duplicate_model_output', 0)} | "
                f"{sb_counts.get('method_complete', 0)} | "
                f"{sample_sat_exact_t} | {sample_sat_exact_f} |"
            )

        # 2. Parse failure causes
        pf_errors = Counter()
        for (m, s), r in runs.items():
            if r.get("outcome") == "parse_failure":
                err = r.get("contract", {}).get("error") or "unknown parse failure"
                pf_errors[err] += 1
        P("\n### 파싱 실패 원인 상위 내역\n")
        if not pf_errors:
            P("파싱 실패 0건.\n")
        else:
            P("| 순위 | 발생 건수 | 에러 원인 메시지 |")
            P("| --: | --: | :-- |")
            for rank, (err_msg, cnt) in enumerate(pf_errors.most_common(5), 1):
                P(f"| {rank} | {cnt} | `{err_msg}` |")

        # 3. Core Family McNemar vs Direct (Holm m=4)
        P("\n### Core Family 유의성 검정 (대조군: Direct, 양측 exact McNemar, Holm m=4)\n")
        P("| 대비 (vs Direct) | n | 대비군만 성공 (b) | Direct만 성공 (c) | p (McNemar) | p (Holm) | 판정 |")
        P("| :-- | --: | --: | --: | :-- | :-- | :-- |")
        dir_dict = by_m.get("direct", {})
        raw_core: dict[str, float] = {}
        detail_core: dict[str, tuple[int, int, int]] = {}
        for m in CORE_METHODS:
            m_dict = by_m.get(m, {})
            common = sorted(set(dir_dict) & set(m_dict))
            b = sum(1 for s in common if m_dict[s] and not dir_dict[s])
            c = sum(1 for s in common if not m_dict[s] and dir_dict[s])
            detail_core[m] = (len(common), b, c)
            raw_core[m] = mcnemar_exact(b, c)

        adj_core = holm(raw_core)
        for m in CORE_METHODS:
            n_c, b, c = detail_core[m]
            p_raw = raw_core[m]
            p_adj = adj_core[m]
            if math.isnan(p_raw):
                p_str = "no discordant pairs"
                padj_str = "no discordant pairs"
                verdict = "비유의 (0:0)"
            else:
                p_str = f"{p_raw:.4f}"
                padj_str = f"{p_adj:.4f}"
                verdict = "유의" if p_adj < 0.05 else "비유의"
            P(f"| {SHORT[m]} vs Direct | {n_c} | {b} | {c} | {p_str} | {padj_str} | {verdict} |")

        # 4. Selector Family McNemar vs Active (Holm m=3)
        P("\n### Selector Family 유의성 검정 (소거군 vs Active, 양측 exact McNemar, Holm m=3)\n")
        P("| 대비 (vs Active) | n | Active만 성공 (b) | 소거군만 성공 (c) | p (McNemar) | p (Holm) | 판정 |")
        P("| :-- | --: | --: | --: | :-- | :-- | :-- |")
        act_dict = by_m.get("active_cegis", {})
        raw_sel: dict[str, float] = {}
        detail_sel: dict[str, tuple[int, int, int]] = {}
        for o in SELECTOR_METHODS:
            o_dict = by_m.get(o, {})
            common = sorted(set(act_dict) & set(o_dict))
            b = sum(1 for s in common if act_dict[s] and not o_dict[s])
            c = sum(1 for s in common if not act_dict[s] and o_dict[s])
            detail_sel[o] = (len(common), b, c)
            raw_sel[o] = mcnemar_exact(b, c)

        adj_sel = holm(raw_sel)
        for o in SELECTOR_METHODS:
            n_c, b, c = detail_sel[o]
            p_raw = raw_sel[o]
            p_adj = adj_sel[o]
            if math.isnan(p_raw):
                p_str = "no discordant pairs"
                padj_str = "no discordant pairs"
                verdict = "비유의 (0:0)"
            else:
                p_str = f"{p_raw:.4f}"
                padj_str = f"{p_adj:.4f}"
                verdict = "유의" if p_adj < 0.05 else "비유의"
            P(f"| Active vs {SHORT[o]} | {n_c} | {b} | {c} | {p_str} | {padj_str} | {verdict} |")

    P("\n주의: `p=1.0` 또는 `no discordant pairs`는 동등성의 증거가 아니다. 비유의 셀은 \"차이를 확인하지 못함\"으로 해석한다.\n")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Output markdown path")
    args = parser.parse_args()

    report_text = generate_report()
    print(report_text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_text, encoding="utf-8")
        print(f"Wrote report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
