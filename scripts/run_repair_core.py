#!/usr/bin/env python3
"""Driver for the contract-repair comparison on Deployment (conditions A, B, C).

``initial``  one round-0 contract per seed, shared by all three conditions; parse failures are
             kept exactly as produced and charged identically in each condition.
``run``      one condition over the stored initial contracts.

Artifacts go through the domain's own ``_finish`` so schema, hashes and serving metadata match
every other run in the project, plus a ``repair_core`` block with the adoption trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.deployment import dsl, repair_prompts  # noqa: E402
from experiment.deployment.config import DeploymentConfig  # noqa: E402
from experiment.deployment.contracts import evaluate_contract  # noqa: E402
from experiment.deployment.patch import (  # noqa: E402
    DEFAULT_INSERT_NODE_CAP,
    DEFAULT_PATCH_CAP,
    patch_json_schema,
)
from experiment.deployment.repair_core import (  # noqa: E402
    CONDITIONS,
    PATCH_CONDITIONS,
    TIE_TOLERANT_CONDITIONS,
    WITNESS_CONDITIONS,
    Observation,
    RepairAdapter,
    run_repair,
)
from experiment.deployment.runner import DeploymentExperimentRunner  # noqa: E402
from experiment.deployment.witness import find_witness_pool  # noqa: E402
from experiment.modeling.client import ChatClient  # noqa: E402
from experiment.modeling.runner import Budgets, Decoding, RunSpec  # noqa: E402

COUNTEREXAMPLE_LIMIT = 3


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(base_url: str, model: str):
    client = ChatClient(model=model, base_url=base_url, timeout=600.0, allow_remote=False)
    cfg = DeploymentConfig.from_json_file(ROOT / "experiment/configs/deployment_default.json")
    # Context policy matches every other Deployment cell in the project
    # (deployment_*_final_v3_s20.json): accumulate the conversation but clamp the completion
    # allowance against the window. Omitting context_token_limit disables the clamp entirely,
    # which produced HTTP 400 context overflows in the first pilot.
    runner = DeploymentExperimentRunner(
        cfg, client, max_depth=20, max_states=15000,
        compact_context=False, context_token_limit=8192, context_margin=256,
        serving_metadata={"model": model, "endpoint": f"{base_url}/chat/completions"},
    )
    return runner, cfg


def make_adapter(runner, cfg, patch_cap: int, insert_cap: int) -> RepairAdapter:
    def parse(text: str):
        return dsl.parse_contract_text(text)

    def score_subset(parsed, subset):
        rep = evaluate_contract(parsed.bind(cfg), tuple(subset), cfg,
                                max_counterexamples=COUNTEREXAMPLE_LIMIT)
        ces = [c.to_dict() if hasattr(c, "to_dict") else c for c in (rep.counterexamples or [])]
        return Observation(states_checked=len(subset), false_accepts=int(rep.false_accepts),
                           false_rejects=int(rep.false_rejects),
                           postcondition_violations=int(rep.postcondition_violations),
                           counterexamples=ces[:COUNTEREXAMPLE_LIMIT])

    def fixed_pool(seed: int, count: int):
        # candidate-independent by construction: probe_states never sees a contract
        return runner.probe_states(seed, count)

    def select_active(parsed, count, *, seed, audit_index, excluded=()):
        return runner.candidate_aware_states(parsed, seed=seed, count=count,
                                             audit_index=audit_index, excluded=tuple(excluded),
                                             balance=True, coverage=False)

    return RepairAdapter(
        parse=parse, score_subset=score_subset, fixed_pool=fixed_pool,
        select_active=select_active,
        free_rewrite_prompt=repair_prompts.free_rewrite_prompt,
        patch_prompt=lambda s, m, c: repair_prompts.patch_prompt(
            s, m, c, patch_cap=patch_cap, insert_node_cap=insert_cap),
        patch_repair_prompt=lambda e: repair_prompts.patch_repair_prompt(
            e, patch_cap=patch_cap, insert_node_cap=insert_cap),
        parse_repair_prompt=lambda s, e: __import__(
            "experiment.deployment.prompts", fromlist=["prompts"]
        ).parse_repair_prompt(cfg, error=e, previous_output=s),
    )


def make_spec(method: str, seed: int, args) -> RunSpec:
    return RunSpec(
        method=method, seed=seed,
        decoding=Decoding(temperature=args.temperature, top_p=args.top_p,
                          max_tokens=args.max_tokens),
        budgets=Budgets(state_budget=args.state_budget, query_budget=args.query_budget,
                        token_budget=args.token_budget),
        use_guided_json=True,
    )


_SCHEMA_LOCK = threading.Lock()
_SCHEMA_OWNER: int | None = None


@contextmanager
def _guided_schema(schema_fn):
    """Swap the guided-decoding schema for the duration of one call.

    ``_Session.ask`` selects ``dsl.contract_json_schema()`` inline, so a patch response would
    otherwise be constrained to the contract grammar and always fail. Rather than change the
    shipped session (every other method depends on it), the schema factory is swapped around
    the single call and restored in ``finally``. Which schema each call used is recorded in the
    artifact, so this is auditable rather than invisible.

    The swap mutates a module-level attribute, which is process-global: two concurrent requests
    would receive each other's schema. This driver is therefore single-threaded by contract, and
    the guard below turns a violation into an immediate error instead of silently mixing
    schemas across runs. Do not add workers without replacing this mechanism.
    """
    global _SCHEMA_OWNER
    me = threading.get_ident()
    if not _SCHEMA_LOCK.acquire(blocking=False):
        raise RuntimeError(
            "guided-schema swap is already active on another thread; this driver must run "
            f"single-threaded (holder={_SCHEMA_OWNER}, caller={me})"
        )
    _SCHEMA_OWNER = me
    original = dsl.contract_json_schema
    dsl.contract_json_schema = schema_fn  # type: ignore[assignment]
    try:
        yield
    finally:
        dsl.contract_json_schema = original  # type: ignore[assignment]
        _SCHEMA_OWNER = None
        _SCHEMA_LOCK.release()


def make_ask(session, runner, patch_cap: int, schema_log: list[dict[str, Any]]):
    """Normalise to ask(role, prompt, schema_kind); pick the guided schema per response type."""
    contract_schema = dsl.contract_json_schema

    def ask(role: str, prompt: str, schema_kind: str | None = None):
        kind = schema_kind or "contract"
        schema_log.append({"role": role, "guided_schema": kind})
        if kind == "patch":
            with _guided_schema(lambda: patch_json_schema(patch_cap)):
                interaction = session.ask(role, prompt)
        else:
            interaction = session.ask(role, prompt)
        if interaction is None:
            return None
        return interaction.response_text
    return ask


def cmd_initial(args) -> None:
    runner, cfg = build(args.base_url, args.model)
    from experiment.deployment import prompts as dprompts
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    store: dict[str, Any] = {"generated": [], "note": (
        "Shared round-0 candidates for the contract-repair comparison. One generation call per "
        "seed; all three conditions replay this source and are charged one call for it. "
        "Parse failures are retained as produced."
    )}
    session_cls = sys.modules[type(runner).__module__]._Session
    for seed in args.seeds:
        spec = make_spec(CONDITIONS[0], seed, args)
        session = session_cls(runner, spec)
        schema_log: list[dict[str, Any]] = []
        ask = make_ask(session, runner, args.patch_cap, schema_log)
        text = ask("synthesis", dprompts.direct_prompt(cfg), "contract")
        ok, err = True, None
        try:
            dsl.parse_contract_text(text or "")
        except Exception as exc:  # noqa: BLE001
            ok, err = False, str(exc)
        inter = session.interactions[-1] if session.interactions else None
        store["generated"].append({
            "domain": "deployment", "model": args.model, "seed": seed,
            "source": text, "source_sha256": sha(text) if text else None,
            "parsed": ok, "parse_error": err,
            "usage": dict(getattr(inter, "usage", {}) or {}),
            "prompt_sha256": getattr(inter, "prompt_sha256", ""),
            "response_sha256": getattr(inter, "response_sha256", ""),
            "finish_reason": getattr(inter, "finish_reason", None),
            "latency_s": getattr(inter, "latency_s", None),
        })
        print(f"seed {seed}: parsed={ok} chars={len(text or '')}", flush=True)
    out.write_text(json.dumps(store, indent=1))
    print(f"wrote {out} ({len(store['generated'])} initial contracts)")


def cmd_run(args) -> None:
    initial = {g["seed"]: g for f in args.initial
               for g in json.loads(Path(f).read_text())["generated"]}
    runner, cfg = build(args.base_url, args.model)
    adapter = make_adapter(runner, cfg, args.patch_cap, args.insert_node_cap)
    session_cls = sys.modules[type(runner).__module__]._Session
    module = sys.modules[type(runner).__module__]
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        target = outdir / f"{args.condition}__seed{seed}.json"
        if target.exists():
            print(f"seed {seed}: exists, skipping", flush=True)
            continue
        entry = initial[seed]
        spec = make_spec(args.condition, seed, args)
        session = session_cls(runner, spec)
        schema_log: list[dict[str, Any]] = []
        ask = make_ask(session, runner, args.patch_cap, schema_log)
        session.ledger.calls += 1  # the replayed round-0 call, charged identically

        # A'/B' run on fixed evidence that is known to expose an observed error for the shared
        # initial candidate. The search executes states and that cost is recorded, never hidden.
        witness = None
        evidence = None
        if args.condition in WITNESS_CONDITIONS:
            try:
                initial_parsed = adapter.parse(entry["source"])
            except Exception:  # noqa: BLE001 - an unparsable initial candidate has no evidence
                initial_parsed = None
            if initial_parsed is not None:
                witness = find_witness_pool(
                    draw_pool=lambda sub, n: runner.probe_states(sub, n),
                    score=lambda states: adapter.score_subset(initial_parsed, tuple(states)).d_obs,
                    seed=seed, size=args.state_budget, max_pools=args.max_witness_pools)
                evidence = witness.states
                print(f"  witness: found={witness.found} pools={witness.pools_drawn} "
                      f"executed={witness.states_executed} errors={witness.observed_errors}",
                      flush=True)

        result = run_repair(adapter, ask=ask, condition=args.condition, seed=seed,
                            state_budget=args.state_budget, query_budget=args.query_budget,
                            initial_source=entry["source"], patch_cap=args.patch_cap,
                            insert_node_cap=args.insert_node_cap, evidence=evidence)
        session.rounds.extend(result["rounds"])
        session.stopped_because = result["stopped_because"] or "method_complete"
        _adopt_final(session, module, result)
        artifact = runner._finish(session)
        payload = asdict(artifact) if is_dataclass(artifact) else artifact
        payload["repair_core"] = {
            "condition": result["condition"],
            # Derive from the loop's own condition set rather than naming one condition, which
            # is how repair_patch_witness came to be labelled "always_replace" in the
            # 2026-09-07 Gemma run even though the loop applied the strict rule correctly.
            "adoption_rule": (
                "non_increasing_d_obs" if result["condition"] in TIE_TOLERANT_CONDITIONS
                else "strict_d_obs_decrease" if result["condition"] in PATCH_CONDITIONS
                else "always_replace"),
            "tie_patches_adopted": sum(1 for r in result["rounds"] if r.get("tie_adopted")),
            "duplicate_stop_fired": any(r.get("guard") == "duplicate_output"
                                        for r in result["rounds"]),
            "repeated_patch_texts": sum(1 for r in result["rounds"]
                                        if r.get("patch_text_seen_before")),
            "patch_cap": args.patch_cap, "insert_node_cap": args.insert_node_cap,
            "loop_model_calls": result["model_calls"],
            "states_observed_loop": result["states_observed"],
            "patches_proposed": result["patches_proposed"],
            "patches_adopted": result["patches_adopted"],
            "patches_rejected": result["patches_rejected"],
            "total_edits": result["total_edits"],
            "initial_contract_sha256": entry["source_sha256"],
            "initial_contract_replayed": True,
            "initial_generation_usage": entry["usage"],
            "last_candidate_sha256": (sha(result["last_candidate_source"])
                                      if result["last_candidate_source"] else None),
            "guided_schema_per_call": schema_log,
            "witness_search": witness.to_dict() if witness is not None else None,
        }
        target.write_text(json.dumps(payload, indent=1, default=str))
        ev = ((payload.get("evaluation") or {}).get("metrics") or {})
        print(f"seed {seed}: stop={result['stopped_because']} calls={result['model_calls']} "
              f"states={result['states_observed']} adopted={result['patches_adopted']}"
              f"/{result['patches_proposed']} exact={ev.get('exact')}", flush=True)


def _adopt_final(session, module, result) -> None:
    """The incumbent is the final contract. A broken proposal never replaces it."""
    source = result["source"]
    if result["parsed"] is None:
        session.contract = module.ContractRecord(module.STATUS_PARSE_FAILURE, source=source,
                                                 error="unrepaired parse failure")
        session.parsed = None
        return
    session.parsed = result["parsed"]
    try:
        session.contract = module.ContractRecord(module.STATUS_PARSED, source=source)
    except TypeError:
        session.contract = module.ContractRecord(status=module.STATUS_PARSED, source=source)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("initial", "run"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
        p.add_argument("--state-budget", type=int, default=48)
        p.add_argument("--query-budget", type=int, default=4)
        p.add_argument("--token-budget", type=int, default=16000)
        p.add_argument("--max-tokens", type=int, default=2048)
        p.add_argument("--temperature", type=float, default=0.2)
        p.add_argument("--top-p", type=float, default=0.95)
        p.add_argument("--patch-cap", type=int, default=DEFAULT_PATCH_CAP)
        p.add_argument("--insert-node-cap", type=int, default=DEFAULT_INSERT_NODE_CAP)
        p.add_argument("--max-witness-pools", type=int, default=12)
        p.add_argument("--output", required=True)
        if name == "run":
            p.add_argument("--condition", choices=list(CONDITIONS), required=True)
            p.add_argument("--initial", nargs="+", required=True)
    args = ap.parse_args()
    (cmd_initial if args.cmd == "initial" else cmd_run)(args)


if __name__ == "__main__":
    main()
