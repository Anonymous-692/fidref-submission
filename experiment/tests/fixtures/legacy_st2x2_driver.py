#!/usr/bin/env python3
"""B16 driver: selection x termination factorial on Deployment and tau-bench.

Two subcommands.

``initial``  Generate one initial contract per (model, domain, seed) and store the request,
             the response, the token usage, and the contract hash. These are the shared
             round-0 candidates; parse failures are kept exactly as they came out, never
             discarded and never redrawn.

``run``      Run one condition over the stored initial contracts. Every condition replays the
             same round-0 source and is charged one model call for it, so the four conditions
             differ only in {selection, termination}.

Artifacts are written through each domain's own ``_finish``, so their schema, hashes, and
serving metadata match every other run in the project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.modeling.client import ChatClient  # noqa: E402
from experiment.modeling.runner import (  # noqa: E402
    ST2X2_PARTITIONED_EXHAUST,
    ST2X2_PARTITIONED_STOP,
    ST2X2_UNIFORM_EXHAUST,
    ST2X2_UNIFORM_STOP,
    Budgets,
    Decoding,
    RunSpec,
)
from experiment.modeling.st2x2 import run_st2x2  # noqa: E402
from experiment.modeling.st2x2_domains import deployment_adapter, taubench_adapter  # noqa: E402

CONDITIONS = {
    ST2X2_PARTITIONED_STOP: (True, False),
    ST2X2_UNIFORM_STOP: (False, False),
    ST2X2_PARTITIONED_EXHAUST: (True, True),
    ST2X2_UNIFORM_EXHAUST: (False, True),
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_runner(domain: str, base_url: str, model: str, guided_json: bool):
    client = ChatClient(model=model, base_url=base_url, timeout=600.0, allow_remote=False)
    serving = {"model": model, "endpoint": f"{base_url}/chat/completions"}
    if domain == "taubench":
        from experiment.taubench_retail.enumeration import RetailConfig
        from experiment.taubench_retail.runner import RetailExperimentRunner

        raw = json.loads((ROOT / "experiment/configs/taubench_retail_default.json").read_text())
        cfg = RetailConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()})
        runner = RetailExperimentRunner(
            client,
            config=cfg,
            max_depth=20,
            max_states=5000,
            compact_context=True,
            context_token_limit=8192,
            counterexample_limit=3,
            refinement_protocol="self_contained_v3",
            serving_metadata=serving,
        )
        return runner, taubench_adapter(runner), "taubench_retail"
    from experiment.deployment.config import DeploymentConfig
    from experiment.deployment.runner import DeploymentExperimentRunner

    cfg = DeploymentConfig.from_json_file(ROOT / "experiment/configs/deployment_default.json")
    runner = DeploymentExperimentRunner(
        cfg, client, max_depth=20, max_states=15000, serving_metadata=serving
    )
    return runner, deployment_adapter(runner), "deployment"


def make_session(runner, spec, domain: str):
    """The two sessions take their arguments in opposite orders; both are positional."""
    session_cls = sys.modules[type(runner).__module__]._Session
    if domain == "taubench":
        return session_cls(spec, runner)
    return session_cls(runner, spec)


def make_ask(session, domain: str):
    """Normalise the two session APIs to ``ask(role, prompt) -> str | None``."""
    if domain == "taubench":
        return session.ask
    def ask(role: str, prompt: str):
        interaction = session.ask(role, prompt)
        if interaction is None:
            return None
        return interaction.response_text
    return ask


def cmd_initial(args) -> None:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    store: dict[str, Any] = {"generated": [], "note": (
        "Shared round-0 candidates for the B16 factorial. One generation call per "
        "(model, domain, seed); every condition replays this source and is charged one call. "
        "Parse failures are retained as produced."
    )}
    runner, adapter, dom = build_runner(args.domain, args.base_url, args.model, args.guided_json)
    for seed in args.seeds:
        spec = RunSpec(
            method=ST2X2_PARTITIONED_STOP,
            seed=seed,
            decoding=Decoding(temperature=0.2, top_p=0.95, max_tokens=2048),
            budgets=Budgets(state_budget=args.state_budget, query_budget=args.query_budget, token_budget=16000),
            use_guided_json=args.guided_json,
        )
        session = make_session(runner, spec, args.domain)
        ask = make_ask(session, args.domain)
        text = ask("synthesis", adapter.direct_prompt())
        parsed_ok, err = True, None
        try:
            adapter.parse(text or "")
        except Exception as exc:  # noqa: BLE001
            parsed_ok, err = False, str(exc)
        inter = session.interactions[-1] if session.interactions else None
        store["generated"].append({
            "domain": dom,
            "model": args.model,
            "seed": seed,
            "source": text,
            "source_sha256": sha(text) if text else None,
            "parsed": parsed_ok,
            "parse_error": err,
            "usage": dict(getattr(inter, "usage", {}) or {}),
            "prompt_sha256": getattr(inter, "prompt_sha256", ""),
            "response_sha256": getattr(inter, "response_sha256", ""),
            "finish_reason": getattr(inter, "finish_reason", None),
            "latency_s": getattr(inter, "latency_s", None),
        })
        print(f"seed {seed}: parsed={parsed_ok} chars={len(text or '')}", flush=True)
    out.write_text(json.dumps(store, indent=1))
    print(f"wrote {out} ({len(store['generated'])} initial contracts)")


def cmd_run(args) -> None:
    initial = {
        (g["domain"], g["model"], g["seed"]): g
        for f in args.initial
        for g in json.loads(Path(f).read_text())["generated"]
    }
    runner, adapter, dom = build_runner(args.domain, args.base_url, args.model, args.guided_json)
    balance, exhaust = CONDITIONS[args.condition]
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        target = outdir / f"{args.condition}__seed{seed}.json"
        if target.exists():
            print(f"seed {seed}: exists, skipping", flush=True)
            continue
        entry = initial[(dom, args.model, seed)]
        spec = RunSpec(
            method=args.condition,
            seed=seed,
            decoding=Decoding(temperature=0.2, top_p=0.95, max_tokens=2048),
            budgets=Budgets(state_budget=args.state_budget, query_budget=args.query_budget, token_budget=16000),
            use_guided_json=args.guided_json,
        )
        session = make_session(runner, spec, args.domain)
        ask = make_ask(session, args.domain)
        session.ledger.calls += 1  # the replayed round-0 call, charged identically
        result = run_st2x2(
            adapter,
            ask=ask,
            seed=seed,
            state_budget=args.state_budget,
            query_budget=args.query_budget,
            balance=balance,
            exhaust=exhaust,
            initial_source=entry["source"],
        )
        session.rounds.extend(result["rounds"])
        session.stopped_because = result["stopped_because"] or "method_complete"
        _adopt_final(session, adapter, result, entry)
        artifact = runner._finish(session)
        payload = asdict(artifact) if is_dataclass(artifact) else artifact
        payload["st2x2"] = {
            "selection": result["selection"],
            "termination": result["termination"],
            "batch_size": result["batch_size"],
            "loop_model_calls": result["model_calls"],
            "states_observed_loop": result["states_observed"],
            "initial_contract_sha256": entry["source_sha256"],
            "initial_contract_replayed": True,
            "initial_generation_usage": entry["usage"],
        }
        target.write_text(json.dumps(payload, indent=1, default=str))
        ev = ((payload.get("evaluation") or {}).get("metrics") or {})
        print(
            f"seed {seed}: stop={result['stopped_because']} calls={result['model_calls']} "
            f"states={result['states_observed']} exact={ev.get('exact')}",
            flush=True,
        )


def _adopt_final(session, adapter, result, entry) -> None:
    """Put the loop's final candidate where the domain's ``_finish`` expects it."""
    source = result["source"]
    module = sys.modules[type(session.runner).__module__]
    record_cls = module.ContractRecord
    if result["parsed"] is None:
        session.contract = record_cls(module.STATUS_PARSE_FAILURE, source=source, error="unrepaired parse failure")
        session.parsed = None
        return
    parsed = result["parsed"]
    session.parsed = parsed
    node_count = getattr(parsed, "node_count", None)
    try:
        session.contract = record_cls(module.STATUS_PARSED, source=source, node_count=node_count)
    except TypeError:
        session.contract = record_cls(status=module.STATUS_PARSED, source=source)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("initial", "run"):
        p = sub.add_parser(name)
        p.add_argument("--domain", choices=["deployment", "taubench"], required=True)
        p.add_argument("--base-url", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
        p.add_argument("--state-budget", type=int, default=48)
        p.add_argument("--query-budget", type=int, default=4)
        p.add_argument("--guided-json", action="store_true")
        p.add_argument("--output", required=True)
        if name == "run":
            p.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
            p.add_argument("--initial", nargs="+", required=True)
    args = ap.parse_args()
    (cmd_initial if args.cmd == "initial" else cmd_run)(args)


if __name__ == "__main__":
    main()
