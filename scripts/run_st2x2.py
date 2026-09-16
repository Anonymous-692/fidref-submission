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
from experiment.modeling import g2_runtime  # noqa: E402
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


def build_runner(domain: str, base_url: str, model: str, guided_json: bool, *,
                 sandbox_config: str | None = None, fixture_id: str | None = None,
                 max_depth: int | None = None, max_states: int | None = None,
                 replay_protocol: str | None = None, context_window: int | None = None,
                 context_margin: int = 256):
    if domain not in ("deployment", "taubench"):
        raise ValueError(f"unknown domain: {domain}")
    custom = any(v is not None for v in (sandbox_config, fixture_id, max_depth, max_states))
    if custom and (not fixture_id or not fixture_id.strip()):
        raise ValueError("custom fixtures/bounds require --fixture-id")
    if replay_protocol is not None:
        if replay_protocol != g2_runtime.VERSION or domain != 'deployment' or not custom:
            raise ValueError('G2 replay protocol requires a custom Deployment fixture')
        if context_window is None or context_window < 1 or not 0 <= context_margin < context_window:
            raise ValueError('G2 requires explicit context window and valid margin')
    elif context_window is not None:
        raise ValueError('context-window requires the opt-in G2 replay protocol')
    depth = max_depth if max_depth is not None else 20
    cap = max_states if max_states is not None else (5000 if domain == "taubench" else 15000)
    if depth < 0 or cap < 1:
        raise ValueError("invalid enumeration bounds")
    default_path = ROOT / ("experiment/configs/taubench_retail_default.json" if domain == "taubench"
                           else "experiment/configs/deployment_default.json")
    config_path = Path(sandbox_config).resolve() if sandbox_config else default_path
    config_bytes = config_path.read_bytes()
    raw = json.loads(config_bytes)
    embedded_id = raw.pop("fixture_id", None)
    if embedded_id is not None and embedded_id != fixture_id:
        raise ValueError("--fixture-id disagrees with configuration fixture_id")
    client = ChatClient(model=model, base_url=base_url, timeout=600.0, allow_remote=False)
    serving = {"model": model, "endpoint": f"{base_url}/chat/completions"}
    if domain == "taubench":
        from experiment.taubench_retail.enumeration import RetailConfig
        from experiment.taubench_retail.runner import RetailExperimentRunner

        cfg = RetailConfig.from_mapping(raw)
        runner = RetailExperimentRunner(
            client,
            config=cfg,
            max_depth=depth,
            max_states=cap,
            compact_context=True,
            context_token_limit=8192,
            counterexample_limit=3,
            refinement_protocol="self_contained_v3",
            serving_metadata=serving,
            **({"config_path": str(config_path)} if custom else {}),
        )
        _attach_fixture(runner, custom, fixture_id, config_path, config_bytes)
        return runner, taubench_adapter(runner), "taubench_retail"
    from experiment.deployment.config import DeploymentConfig
    from experiment.deployment.runner import DeploymentExperimentRunner

    cfg = DeploymentConfig.from_dict(raw)
    runner = DeploymentExperimentRunner(
        cfg, client, max_depth=depth, max_states=cap, serving_metadata=serving,
        **({"config_path": str(config_path)} if custom else {}),
    )
    _attach_fixture(runner, custom, fixture_id, config_path, config_bytes)
    if replay_protocol is not None:
        g2_runtime.configure(runner, context_window, context_margin)
    return runner, deployment_adapter(runner), "deployment"


def _attach_fixture(runner, custom, fixture_id, config_path, config_bytes):
    if not custom:
        return
    summary = runner.enumeration_summary
    summary = summary() if callable(summary) else summary
    if summary["truncated"]:
        raise ValueError("custom fixture enumeration is truncated; no model call allowed")
    runner.fixture_context = {
        "version": "g2_fixture_context_v1", "fixture_id": fixture_id,
        "config_path": str(config_path),
        "config_file_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "sandbox_config_sha256": sha(json.dumps(runner.config.to_dict(), sort_keys=True)),
        "max_depth": runner.max_depth, "max_states": runner.max_states,
    }


def _runner_from_args(args):
    return build_runner(args.domain, args.base_url, args.model, args.guided_json,
        sandbox_config=getattr(args, "sandbox_config", None), fixture_id=getattr(args, "fixture_id", None),
        max_depth=getattr(args, "max_depth", None), max_states=getattr(args, "max_states", None),
        replay_protocol=getattr(args, 'replay_protocol', None),
        context_window=getattr(args, 'context_window', None),
        context_margin=getattr(args, 'context_margin', 256))


def _initial_context(runner, args):
    if not hasattr(runner, "fixture_context"):
        return None
    return {"fixture": runner.fixture_context, "guided_json": args.guided_json,
            "state_budget": args.state_budget, "query_budget": args.query_budget,
            "decoding": {"temperature": 0.2, "top_p": 0.95, "max_tokens": 2048},
            "token_budget": 16000,
            **({'runtime': runner.g2_runtime} if hasattr(runner, 'g2_runtime') else {})}


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
    runner, adapter, dom = _runner_from_args(args)
    if hasattr(runner, "fixture_context") and out.exists():
        raise ValueError("custom fixture initial output already exists; use a new path")
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
            **({"initial_context": _initial_context(runner, args)} if hasattr(runner, "fixture_context") else {}),
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
        if hasattr(runner, 'g2_runtime'):
            row = store['generated'][-1]
            row.update(g2_runtime.initial_metadata(session))
            row['record_sha256'] = g2_runtime.record_digest(row)
        print(f"seed {seed}: parsed={parsed_ok} chars={len(text or '')}", flush=True)
    out.write_text(json.dumps(store, indent=1))
    print(f"wrote {out} ({len(store['generated'])} initial contracts)")


def cmd_run(args) -> None:
    records = [g for f in args.initial for g in json.loads(Path(f).read_text())["generated"]]
    initial = {(g["domain"], g["model"], g["seed"]): g for g in records}
    runner, adapter, dom = _runner_from_args(args)
    expected_context = _initial_context(runner, args)
    if expected_context is not None:
        if len(initial) != len(records):
            raise ValueError("duplicate initial contract keys; use one fixture per invocation")
    for seed in args.seeds:
        entry = initial[(dom, args.model, seed)]
        if entry.get("initial_context") != expected_context:
            raise ValueError(f"seed {seed}: initial fixture/protocol mismatch")
        if expected_context is not None and entry["source_sha256"] != (sha(entry['source']) if entry['source'] else None):
            raise ValueError(f"seed {seed}: initial source hash mismatch")
        if hasattr(runner, 'g2_runtime'):
            g2_runtime.validate_initial(entry)
    balance, exhaust = CONDITIONS[args.condition]
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        target = outdir / f"{args.condition}__seed{seed}.json"
        if target.exists():
            if expected_context is not None:
                existing = json.loads(target.read_text())
                if (existing.get("initial_context") != expected_context
                        or existing.get("model") != args.model
                        or existing.get("method") != args.condition
                        or existing.get("seed") != seed
                        or existing.get("st2x2", {}).get("initial_contract_sha256") != initial[(dom, args.model, seed)]["source_sha256"]):
                    raise ValueError(f"existing artifact does not match requested fixture: {target}")
                if (hasattr(runner, 'g2_runtime') and existing.get('g2_accounting', {}).get('initial_record_sha256')
                        != initial[(dom, args.model, seed)]['record_sha256']):
                    raise ValueError(f'existing artifact initial record mismatch: {target}')
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
        g2 = hasattr(runner, 'g2_runtime')
        if g2:
            g2_runtime.replay_initial(session, entry)
        else:
            session.ledger.calls += 1  # legacy B16 accounting, preserved
        if g2 and entry['source'] is None:
            result = g2_runtime.missing_result(entry, state_budget=args.state_budget,
                query_budget=args.query_budget, balance=balance, exhaust=exhaust)
        else:
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
        session.stopped_because = ((session.stopped_because if g2 else None)
                                  or result["stopped_because"] or "method_complete")
        if g2:
            g2_runtime.sync_ledger(session, result)
        _adopt_final(session, adapter, result, entry)
        artifact = runner._finish(session)
        payload = asdict(artifact) if is_dataclass(artifact) else artifact
        if expected_context is not None:
            payload["initial_context"] = expected_context
            payload["fixture_context"] = runner.fixture_context
            base_hash = payload["hashes"]["protocol_sha256"]
            payload["hashes"]["base_protocol_sha256"] = base_hash
            payload["hashes"]["protocol_sha256"] = sha(json.dumps(
                {"base_protocol_sha256": base_hash, "initial_context": expected_context}, sort_keys=True))
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
        if g2:
            payload['g2_accounting'] = g2_runtime.accounting(session, entry, result)
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
        if hasattr(session.runner, 'g2_runtime') and entry['source'] is None:
            session.contract = record_cls(module.STATUS_MISSING, source=None, error=entry.get('initial_error'))
            session.parsed = None
            return
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
        p.add_argument("--sandbox-config")
        p.add_argument("--fixture-id")
        p.add_argument("--max-depth", type=int)
        p.add_argument("--max-states", type=int)
        p.add_argument('--replay-protocol', choices=[g2_runtime.VERSION])
        p.add_argument('--context-window', type=int)
        p.add_argument('--context-margin', type=int, default=256)
        p.add_argument("--output", required=True)
        if name == "run":
            p.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
            p.add_argument("--initial", nargs="+", required=True)
    args = ap.parse_args()
    (cmd_initial if args.cmd == "initial" else cmd_run)(args)


if __name__ == "__main__":
    main()
