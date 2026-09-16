"""Closure discrimination audit: does each reference clause do observable work?

For every domain we enumerate the full reachable closure used in the paper and ask,
for each reference clause c:
  witnesses(c) = #{states : every other clause holds, but c fails}
A clause with zero witnesses is not tested by the closure -- a candidate contract
that omits it is still exact, so the closure cannot distinguish the two hypotheses.
We also report state variables that are constant across the closure (dead dimensions).
Read-only: no artifact is written or modified.
"""
from __future__ import annotations
import argparse, dataclasses, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _shopping(cfg_name: str = "shopping_price_easy_v4.json", label: str = "Shopping easy_v4"):
    from experiment.shopping_price.config import PriceShoppingConfig
    from experiment.shopping_price.enumeration import enumerate_reachable
    from experiment.shopping_price.ground_truth import CLAUSE_NAMES, clause_values
    cfg = PriceShoppingConfig.from_json(ROOT / "experiment/configs" / cfg_name)
    states = enumerate_reachable(cfg).states
    return label, cfg, states, list(CLAUSE_NAMES), [clause_values(s, cfg) for s in states]


def _legacy_shopping():
    from experiment.environment import EnvConfig, enumerate_reachable
    from experiment.evaluation.ground_truth import GROUND_TRUTH_CLAUSES
    from experiment.environment.state import OrderStatus
    from experiment.environment.state import counts_covers
    cfg = EnvConfig()
    states = enumerate_reachable(cfg).states
    def vals(s):
        return (
            bool(s.logged_in),
            s.order_status is OrderStatus.NONE,
            not s.cart_is_empty,
            cfg.is_valid_payment(s.payment_method),
            cfg.is_valid_address(s.shipping_address),
            counts_covers(s.stock, s.cart),
        )
    return "Shopping legacy", cfg, states, list(GROUND_TRUTH_CLAUSES), [vals(s) for s in states]


def _calendar():
    from experiment.calendar.config import CalendarConfig
    from experiment.calendar.enumeration import enumerate_reachable
    from experiment.calendar.ground_truth import CLAUSE_NAMES, clause_values
    cfg = CalendarConfig.from_json(ROOT / "experiment/configs/calendar_default.json")
    states = enumerate_reachable(cfg).states
    return "Calendar", cfg, states, list(CLAUSE_NAMES), [clause_values(s, cfg) for s in states]


def _deployment():
    from experiment.deployment.config import DeploymentConfig
    from experiment.deployment.enumeration import enumerate_reachable
    from experiment.deployment.ground_truth import GROUND_TRUTH_CLAUSES
    from experiment.deployment.state import DeploymentStatus
    from experiment.deployment.env import counts_covers
    cfg = DeploymentConfig.from_json_file(ROOT / "experiment/configs/deployment_default.json")
    states = enumerate_reachable(cfg).states
    def vals(s):
        return (
            bool(s.authenticated),
            s.deployment_status is DeploymentStatus.IDLE,
            not s.is_empty_allocation,
            cfg.is_valid_region(s.target_region),
            cfg.is_valid_tier(s.cluster_tier),
            counts_covers(s.available_quota, s.allocated_resources),
        )
    return "Deployment", cfg, states, list(GROUND_TRUTH_CLAUSES), [vals(s) for s in states]


def _taubench(config_name: str, label: str):
    from experiment.taubench_retail.enumeration import enumerate_reachable, RetailConfig
    from experiment.taubench_retail.ground_truth import CLAUSE_NAMES, clause_values
    raw = json.loads((ROOT / "experiment/configs" / config_name).read_text())
    cfg = RetailConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()})
    states = enumerate_reachable(cfg).states
    return label, cfg, states, list(CLAUSE_NAMES), [clause_values(s, cfg) for s in states]


DOMAINS = [
    _shopping,
    lambda: _shopping("shopping_price_moderate_v2.json", "Shopping moderate_v2"),
    lambda: _shopping("shopping_price_moderate_v3.json", "Shopping moderate_v3"),
    lambda: _shopping("shopping_price_default.json", "Shopping price original"),
    _legacy_shopping,
    _calendar, _deployment,
    lambda: _taubench("taubench_retail_default.json", "tau-bench retail"),
    lambda: _taubench("taubench_retail_w2del.json", "tau-bench retail (W2-deliverable)"),
]


def constant_fields(states):
    if not states or not dataclasses.is_dataclass(states[0]):
        return {}
    out = {}
    for f in dataclasses.fields(states[0]):
        vals = {getattr(s, f.name) for s in states}
        if len(vals) == 1:
            out[f.name] = next(iter(vals))
    return out


def audit(build):
    name, cfg, states, clause_names, rows = build()
    n = len(states)
    accepting = sum(1 for r in rows if all(r))
    lines = [f"### {name}", "",
             f"- closure: {n:,} states, accepting {accepting} ({accepting / n * 100:.2f}%)",
             f"- reference clauses: {len(clause_names)}", "",
             "| # | clause | witnesses | verdict |", "| --: | :-- | --: | :-- |"]
    redundant = []
    for i, cname in enumerate(clause_names):
        w = sum(1 for r in rows if not r[i] and all(v for j, v in enumerate(r) if j != i))
        if w == 0:
            redundant.append(cname)
        lines.append(f"| {i} | {cname} | {w} | {'**UNTESTED**' if w == 0 else 'tested'} |")
    const = constant_fields(states)
    lines += ["", f"- untested clauses: {len(redundant)}" + (f" -> {redundant}" if redundant else ""),
              f"- constant state fields across closure: {const if const else 'none'}", ""]
    return "\n".join(lines), len(redundant), const


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/closure_discrimination_audit_2026-09-04.txt")
    args = ap.parse_args()
    parts = ["# Closure discrimination audit (2026-09-04)", "",
             "For each reference clause c, `witnesses(c)` counts closure states where every *other*",
             "clause holds but c fails. A clause with zero witnesses is not tested by the closure:",
             "a candidate that omits it is still scored exact, so `exact` cannot separate the two.", ""]
    total = 0
    for build in DOMAINS:
        text, nred, _ = audit(build)
        parts.append(text)
        total += nred
    parts.append(f"**Total untested reference clauses across domains: {total}.**")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
