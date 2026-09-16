#!/usr/bin/env python3
"""easy_v4 결과 통계: exact/20 + Wilson 95% CI, 동일 seed McNemar(exact binomial), 가설군별 Holm 보정.

읽기 전용. 사용: .venv/bin/python scripts/easy_v4_stats.py [--out analysis/easy_v4_stats.txt]
가설군(EXPERIMENT_PLAN §9): core = Active vs {Direct, Self-Refine, Random, Sampled};
selector = Active vs {No-Balance, No-Coverage, Uniform}. Holm은 (모델 × 가설군) 단위.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

# 디렉터리 글롭은 쓰지 않는다. 글롭은 모델(Gemma-4-31B와 Gemma-4-26B-A4B)과
# 프로토콜(structureddiff와 transitiondiff)을 함께 잡아 셀을 섞는다. 경로·시드는
# cell manifest 의 명시 항목에서만 해석한다(R6).
DEFAULT_MANIFEST = "analysis/cell_manifest_2026-09-04.json"
DOMAIN = "Shopping (price)"
VARIANT = "relaxed"
MODEL_ORDER = ["Gemma-4-31B", "Qwen3.8-27B", "Qwen2.5-32B", "Qwen2.5-14B",
               "Gemma-4-26B-A4B", "Qwen3.6-35B-A3B"]
METHODS = ["direct", "self_refine", "random_probe", "sampled_cegis", "active_cegis",
           "active_cegis_no_balance", "active_cegis_no_coverage", "active_cegis_uniform"]
SHORT = {"direct": "Direct", "self_refine": "Self-Refine", "random_probe": "Random", "sampled_cegis": "Sampled",
         "active_cegis": "Active", "active_cegis_no_balance": "No-Balance",
         "active_cegis_no_coverage": "No-Coverage", "active_cegis_uniform": "Uniform"}
FAMILIES = {"core": ["direct", "self_refine", "random_probe", "sampled_cegis"],
            "selector": ["active_cegis_no_balance", "active_cegis_no_coverage", "active_cegis_uniform"]}
EXPECTED_SANDBOX = "4d4a39e6"


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
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def resolve(manifest_path: str) -> dict[str, dict[str, dict]]:
    """manifest -> {model: {method: cell}}. relaxed price-aware Shopping 셀만 추린다.

    한 (model, method) 가 서로 다른 디렉터리로 두 번 나오면 그 자리에서 멈춘다. 여기서
    조용히 하나를 고르면 프로토콜이 섞인 표가 만들어진다.
    """
    data = json.loads(Path(manifest_path).read_text())
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for key, cell in data["cells"].items():
        if cell.get("domain") != DOMAIN or cell.get("variant") != VARIANT:
            continue
        model, method = cell["model"], cell["method"]
        prev = out[model].get(method)
        if prev is not None and prev["dir"] != cell["dir"]:
            raise SystemExit(
                f"manifest 충돌 ({model}, {method}): {prev['dir']} vs {cell['dir']}")
        out[model][method] = cell
    return out


def load(cells: dict[str, dict]) -> dict[tuple[str, int], dict]:
    """manifest 가 지목한 디렉터리·시드만 읽는다. 읽기 전용."""
    runs: dict[tuple[str, int], dict] = {}
    for method, cell in cells.items():
        # 한 셀이 여러 디렉터리에 샤드로 나뉜 경우가 있다(Qwen3.6 은 시드 0-6 / 7-13 / 14-19).
        dirs = cell["dir"] if isinstance(cell["dir"], list) else [cell["dir"]]
        for seed in cell["seeds"]:
            found = [Path("results") / d / f"{method}__seed{seed}.json" for d in dirs]
            found = [q for q in found if q.exists()]
            if not found:
                raise SystemExit(f"manifest 가 지목한 아티팩트 없음: {method}__seed{seed} in {dirs}")
            if len(found) > 1:
                raise SystemExit(f"샤드 중복 {method}__seed{seed}: {found}")
            p = found[0]
            r = json.loads(p.read_text())
            if r.get("method") != method or r.get("seed") != seed:
                raise SystemExit(f"아티팩트 불일치 {p}: {r.get('method')}/{r.get('seed')}")
            got = r.get("hashes", {}).get("sandbox_config_sha256", "")[:8]
            if got != EXPECTED_SANDBOX:
                raise SystemExit(f"sandbox 해시 불일치 {p}: {got} != {EXPECTED_SANDBOX}")
            key = (method, seed)
            if key in runs:
                raise SystemExit(f"중복 (method, seed) {key}: {p} — 혼합 금지 규칙 위반")
            runs[key] = r
    return runs


def check_against_manifest(model: str, cells: dict[str, dict],
                           runs: dict[tuple[str, int], dict]) -> list[str]:
    """재계산한 exact 수가 manifest 기록과 같은지 확인한다."""
    problems = []
    for method, cell in cells.items():
        recomputed = sum(1 for s in cell["seeds"]
                         if runs[(method, s)].get("outcome") == "exact")
        if "exact" in cell and recomputed != cell["exact"]:
            problems.append(f"{model}/{method}: 재계산 {recomputed} != manifest {cell['exact']}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="셀 경로·시드를 해석할 manifest (기본: %(default)s)")
    args = ap.parse_args()
    resolved = resolve(args.manifest)
    mismatches: list[str] = []
    lines: list[str] = []
    P = lines.append
    P("# easy_v4 통계 (288 states, seed 0–19)\n")
    P("exact/20 [Wilson 95% CI] · 호출 수 평균 · parse failure 수. 셀이 20런 미만이면 `n=`을 표기한다.\n")
    hdr = "| 모델 | " + " | ".join(SHORT[m] for m in METHODS) + " |"
    P(hdr); P("|" + " :-- |" + " ---: |" * len(METHODS))
    tests: list[str] = []
    models = [m for m in MODEL_ORDER if m in resolved]
    models += [m for m in sorted(resolved) if m not in MODEL_ORDER]
    for model in models:
        cells_spec = resolved[model]
        runs = load(cells_spec)
        if not runs:
            continue
        mismatches.extend(check_against_manifest(model, cells_spec, runs))
        cells = []
        by_m: dict[str, dict[int, bool]] = defaultdict(dict)
        for (m, s), r in runs.items():
            by_m[m][s] = (r.get("outcome") == "exact")
        for m in METHODS:
            d = by_m.get(m, {})
            n = len(d); k = sum(d.values())
            if n == 0:
                cells.append("·"); continue
            lo, hi = wilson(k, n)
            calls = sum((runs[(m, s)].get("spend") or {}).get("model_calls", 0) for s in d) / n
            pf = sum(1 for s in d if runs[(m, s)].get("outcome") == "parse_failure")
            cell = f"**{k}/{n}**" if m == "active_cegis" else f"{k}/{n}"
            cell += f" [{lo:.2f},{hi:.2f}] c={calls:.2f} pf={pf}"
            if n != 20:
                cell += f" n={n}"
            cells.append(cell)
        P(f"| {model} | " + " | ".join(cells) + " |")
        # paired tests
        A = by_m.get("active_cegis", {})
        for fam, others in FAMILIES.items():
            raw: dict[str, float] = {}
            detail: dict[str, tuple[int, int, int]] = {}
            for o in others:
                B = by_m.get(o, {})
                common = sorted(set(A) & set(B))
                if len(common) < 20:
                    continue
                b = sum(1 for s in common if A[s] and not B[s])
                c = sum(1 for s in common if B[s] and not A[s])
                raw[o] = mcnemar_exact(b, c); detail[o] = (len(common), b, c)
            if not raw:
                continue
            adj = holm(raw)
            tests.append(f"\n### {model} — {fam} 가설군 (Holm m={len(raw)})\n")
            tests.append("| 대비 | n | Active만 성공 | 비교군만 성공 | p (McNemar) | p (Holm) | 판정 |")
            tests.append("| :-- | --: | --: | --: | --: | --: | :-- |")
            for o in others:
                if o not in raw:
                    continue
                n, b, c = detail[o]
                verdict = "유의" if adj[o] < 0.05 else "비유의"
                tests.append(f"| Active vs {SHORT[o]} | {n} | {b} | {c} | {raw[o]:.2e} | {adj[o]:.2e} | {verdict} |")
    lines.extend(tests)
    P("\n주의: `p=1.0`은 동등성의 증거가 아니다(Plan §9). 비유의 셀은 \"차이를 확인하지 못함\"으로만 쓴다.")
    P(f"\n경로 출처: `{args.manifest}` 의 명시 항목 (domain={DOMAIN}, variant={VARIANT}). 디렉터리 글롭은 사용하지 않는다.")
    if mismatches:
        P("\n**manifest 대조 불일치**\n")
        for m in mismatches:
            P(f"- {m}")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    if mismatches:
        print("manifest 대조 불일치:", *mismatches, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
