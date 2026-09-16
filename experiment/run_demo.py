#!/usr/bin/env python3
"""Walk the shopping sandbox and score the ground truth against each defect."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.environment import (  # noqa: E402  (path set up above)
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_STATES,
    EnvConfig,
    ShoppingAction,
    ShoppingEnv,
    enumerate_reachable,
)
from experiment.evaluation import (  # noqa: E402
    all_fixtures,
    evaluate_contract,
    ground_truth_contract,
)


def scripted_episode(env: ShoppingEnv) -> list[dict[str, object]]:
    """One hand-written trajectory that exercises the whole order lifecycle."""
    script = [
        ShoppingAction.place_order(),  # refused: nobody is logged in yet
        ShoppingAction.login(),
        ShoppingAction.add_to_cart(env.config.items[0]),
        ShoppingAction.place_order(),  # refused: no payment method
        ShoppingAction.set_payment(env.config.valid_payment_methods[0]),
        ShoppingAction.place_order(),  # refused: no shipping address
        ShoppingAction.set_address(env.config.valid_addresses[0]),
        ShoppingAction.place_order(),  # accepted
        ShoppingAction.confirm_order(),
        ShoppingAction.clear_order(),
    ]
    trace: list[dict[str, object]] = []
    for action in script:
        result = env.step(action)
        trace.append(
            {
                "action": str(action),
                "ok": result.ok,
                "error": result.error,
                "state": result.state.describe(),
            }
        )
    return trace


def snapshot_probe(env: ShoppingEnv) -> dict[str, object]:
    """Show that a snapshot round-trips exactly after unrelated activity."""
    snapshot = env.snapshot()
    before = env.state
    env.step(ShoppingAction.login())
    env.step(ShoppingAction.add_to_cart(env.config.items[-1]))
    disturbed = env.state
    env.restore(snapshot)
    return {
        "state_before": before.describe(),
        "state_after_disturbance": disturbed.describe(),
        "restored_exactly": env.state == before,
        "step_count_restored": env.step_count == snapshot.step_count,
    }


def build_report(config: EnvConfig, max_depth: int, max_states: int) -> dict[str, object]:
    env = ShoppingEnv(config)
    trace = scripted_episode(env)
    env.reset()
    probe = snapshot_probe(env)

    first = enumerate_reachable(config, max_depth=max_depth, max_states=max_states)
    second = enumerate_reachable(config, max_depth=max_depth, max_states=max_states)
    states = first.states

    truth = evaluate_contract(ground_truth_contract(config), states, config)
    fixtures = []
    for fixture in all_fixtures(config):
        report = evaluate_contract(fixture.contract, states, config)
        fixtures.append(
            {
                "fixture": fixture.summary(),
                "report": report.summary(),
                "detected": report.symptoms == fixture.expected_symptoms,
                "example": report.counterexamples[0].to_dict() if report.counterexamples else None,
            }
        )

    return {
        "config": config.to_dict(),
        "episode": trace,
        "snapshot_probe": probe,
        "enumeration": first.summary() | {"reproducible": first.states == second.states},
        "ground_truth": truth.summary(),
        "fixtures": fixtures,
    }


def render(report: dict[str, object]) -> str:
    lines: list[str] = []
    config = report["config"]
    lines.append("== controlled shopping sandbox ==")
    lines.append(
        f"config: seed={config['seed']} items={config['items']} "
        f"max_stock={config['max_stock']} cart_capacity={config['cart_capacity']}"
    )

    lines.append("")
    lines.append("-- scripted episode --")
    for entry in report["episode"]:
        mark = "ok " if entry["ok"] else "REJ"
        note = "" if entry["ok"] else f"  ({entry['error']})"
        lines.append(f"  {mark} {entry['action']:<24}{entry['state']}{note}")

    probe = report["snapshot_probe"]
    lines.append("")
    lines.append("-- snapshot / restore --")
    lines.append(f"  restored exactly: {probe['restored_exactly']}")
    lines.append(f"  counters restored: {probe['step_count_restored']}")

    enumeration = report["enumeration"]
    lines.append("")
    lines.append("-- reachable states --")
    lines.append(
        f"  states={enumeration['states']} transitions={enumeration['transitions']} "
        f"deepest_level={enumeration['deepest_level']} truncated={enumeration['truncated']}"
    )
    lines.append(f"  reproducible across runs: {enumeration['reproducible']}")

    truth = report["ground_truth"]
    lines.append("")
    lines.append("-- ground truth vs sandbox --")
    lines.append(
        f"  successes={truth['successes']} failures={truth['failures']} "
        f"false_accepts={truth['false_accepts']} false_rejects={truth['false_rejects']} "
        f"postcondition_violations={truth['postcondition_violations']} exact={truth['exact']}"
    )

    lines.append("")
    lines.append("-- defect fixtures --")
    for entry in report["fixtures"]:
        fixture = entry["fixture"]
        observed = entry["report"]["symptoms"]
        status = "detected" if entry["detected"] else "MISSED"
        lines.append(f"  {fixture['category']:<26}{status:<10}symptoms={observed}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "default.json",
        help="sandbox configuration JSON file",
    )
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--max-states", type=int, default=DEFAULT_MAX_STATES)
    parser.add_argument("--json", action="store_true", help="print the raw report as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = EnvConfig.from_json_file(args.config)
    report = build_report(config, args.max_depth, args.max_states)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render(report))

    problems: list[str] = []
    if not report["ground_truth"]["exact"]:
        problems.append("ground truth disagrees with the sandbox")
    if not report["enumeration"]["reproducible"]:
        problems.append("enumeration was not reproducible")
    if not report["snapshot_probe"]["restored_exactly"]:
        problems.append("snapshot restore did not round-trip")
    missed = [entry["fixture"]["name"] for entry in report["fixtures"] if not entry["detected"]]
    if missed:
        problems.append(f"undetected defect fixtures: {missed}")

    if problems:
        print("\nFAILED: " + "; ".join(problems), file=sys.stderr)
        return 1
    print("\nall demo checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
