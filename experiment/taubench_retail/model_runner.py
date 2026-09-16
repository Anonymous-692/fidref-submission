#!/usr/bin/env python3
"""Run contract synthesis experiments on the tau-bench retail domain."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.modeling import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    Budgets,
    ChatClient,
    Decoding,
    RunSpec,
    TransportError,
    write_artifact,
    write_summary,
)
from experiment.modeling.artifacts import COUNTEREXAMPLE_FORMAT_VERSION
from experiment.taubench_retail import (
    DEFAULT_RETAIL_CONFIG,
    RetailConfig,
    RetailExperimentRunner,
)
from experiment.taubench_retail.runner import (
    METHODS,
    REFINEMENT_PROTOCOL_LEGACY_V2,
    REFINEMENT_PROTOCOLS,
)

from experiment.taubench_retail.dsl import VOCABULARY_LEGACY, VOCABULARY_PROTOCOLS

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "taubench_retail_default.json"
)

CONFIG_KEYS = (
    "description",
    "domain",
    "sandbox_config",
    "base_url",
    "model",
    "methods",
    "seeds",
    "temperature",
    "top_p",
    "max_tokens",
    "state_budget",
    "query_budget",
    "token_budget",
    "max_depth",
    "max_states",
    "timeout",
    "workers",
    "compact_context",
    "context_token_limit",
    "context_margin",
    "counterexample_limit",
    "refinement_protocol",
    "guided_json",
    "vocabulary_protocol",
    "reasoning_effort",
    "thinking_token_budget",
    "chat_template_kwargs",
    "return_token_ids",
)


def load_experiment_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    unknown = sorted(set(payload) - set(CONFIG_KEYS))
    if unknown:
        raise ValueError(f"{path}: unknown configuration keys: {unknown}")
    return dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory to write run artifacts into (created if absent)",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        help="JSON file holding default settings",
    )
    parser.add_argument("--sandbox-config", type=Path, help="RetailConfig JSON file")
    parser.add_argument("--base-url", help=f"OpenAI-compatible base URL (default {DEFAULT_BASE_URL})")
    parser.add_argument("--api-key", help="bearer token, if required")
    parser.add_argument("--model", help="served model name")
    parser.add_argument("--methods", nargs="+", choices=METHODS, help="methods to run")
    parser.add_argument("--seeds", nargs="+", type=int, help="seeds to run each method under")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--state-budget", type=int)
    parser.add_argument("--query-budget", type=int)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--max-states", type=int, default=5000)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--allow-remote-host", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--guided-json", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compact-context", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--context-token-limit", type=int)
    parser.add_argument("--context-margin", type=int)
    parser.add_argument("--counterexample-limit", type=int)
    parser.add_argument("--refinement-protocol", choices=REFINEMENT_PROTOCOLS)
    parser.add_argument("--vocabulary-protocol", choices=VOCABULARY_PROTOCOLS)
    parser.add_argument("--wait-server", type=float, default=600.0)
    return parser


def _pick(value: Any, config: Mapping[str, Any], key: str, fallback: Any) -> Any:
    if value is not None:
        return value
    if key in config and config[key] is not None:
        return config[key]
    return fallback


def execute_from_args(
    args: argparse.Namespace,
    experiment_config: Mapping[str, Any] | None = None,
    client: ChatClient | None = None,
    transport: Any | None = None,
) -> dict[str, Any]:
    cfg = dict(experiment_config or {})
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    sandbox_path = Path(
        _pick(args.sandbox_config, cfg, "sandbox_config", "experiment/configs/taubench_retail_default.json")
    )
    sandbox_config = RetailConfig.from_json(sandbox_path) if sandbox_path.exists() else DEFAULT_RETAIL_CONFIG

    base_url = str(_pick(args.base_url, cfg, "base_url", DEFAULT_BASE_URL))
    model = str(_pick(args.model, cfg, "model", "qwen2.5-14b-instruct"))
    methods = tuple(_pick(args.methods, cfg, "methods", ["direct", "active_cegis"]))
    seeds = tuple(int(seed) for seed in _pick(args.seeds, cfg, "seeds", [0]))

    decoding = Decoding(
        temperature=float(_pick(args.temperature, cfg, "temperature", 0.2)),
        top_p=float(_pick(args.top_p, cfg, "top_p", 0.95)),
        max_tokens=int(_pick(args.max_tokens, cfg, "max_tokens", 2048)),
    )
    budgets = Budgets(
        state_budget=int(_pick(args.state_budget, cfg, "state_budget", 48)),
        query_budget=int(_pick(args.query_budget, cfg, "query_budget", 4)),
        token_budget=int(_pick(args.token_budget, cfg, "token_budget", 16000)),
    )
    max_depth = int(_pick(args.max_depth, cfg, "max_depth", 20))
    max_states = int(_pick(args.max_states, cfg, "max_states", 5000))
    workers = int(_pick(args.workers, cfg, "workers", 1))
    compact_context = bool(_pick(args.compact_context, cfg, "compact_context", True))
    context_token_limit_raw = _pick(args.context_token_limit, cfg, "context_token_limit", 8192)
    context_token_limit = int(context_token_limit_raw) if context_token_limit_raw is not None else None
    context_margin = int(_pick(args.context_margin, cfg, "context_margin", 256))
    counterexample_limit = int(_pick(args.counterexample_limit, cfg, "counterexample_limit", 3))
    refinement_protocol = str(
        _pick(
            getattr(args, "refinement_protocol", None),
            cfg,
            "refinement_protocol",
            REFINEMENT_PROTOCOL_LEGACY_V2,
        )
    )
    use_guided_json = bool(_pick(args.guided_json, cfg, "guided_json", False))
    reasoning_effort = _pick(None, cfg, "reasoning_effort", None)
    thinking_token_budget_raw = _pick(None, cfg, "thinking_token_budget", None)
    thinking_token_budget = int(thinking_token_budget_raw) if thinking_token_budget_raw is not None else None
    chat_template_kwargs = dict(_pick(None, cfg, "chat_template_kwargs", {}))
    return_token_ids = bool(_pick(None, cfg, "return_token_ids", False))

    if client is None:
        client = ChatClient(
            model=model,
            base_url=base_url,
            api_key=args.api_key,
            timeout=float(_pick(args.timeout, cfg, "timeout", DEFAULT_TIMEOUT)),
            transport=transport,
            allow_remote=args.allow_remote_host,
            reasoning_effort=reasoning_effort,
            thinking_token_budget=thinking_token_budget,
            chat_template_kwargs=chat_template_kwargs,
            return_token_ids=return_token_ids,
        )

    if transport is None and getattr(client, "transport", None) is None:
        from experiment.serving.smoke_openai import wait_for_server
        server_info = wait_for_server(client.base_url, api_key=args.api_key, timeout=args.wait_server, model=model)
        _structured_style = client.set_structured_output_style_from_version(
            (server_info or {}).get("vllm_version")
        )
        serving_metadata = {
            "model": model,
            "endpoint": client.endpoint,
            "vllm_version": server_info.get("vllm_version") if server_info else None,
            "structured_output_style": _structured_style,
            "max_model_len": server_info.get("max_model_len") if server_info else None,
        }
    else:
        serving_metadata = {
            "model": model,
            "endpoint": client.endpoint,
            "mock": True,
        }
    if client.reasoning_config:
        serving_metadata["reasoning_request"] = client.reasoning_config

    runner = RetailExperimentRunner(
        client=client,
        config=sandbox_config,
        config_path=str(sandbox_path),
        max_depth=max_depth,
        max_states=max_states,
        compact_context=compact_context,
        context_token_limit=context_token_limit,
        context_margin=context_margin,
        counterexample_limit=counterexample_limit,
        refinement_protocol=refinement_protocol,
        vocabulary_protocol=str(_pick(getattr(args, "vocabulary_protocol", None), cfg,
                                      "vocabulary_protocol", VOCABULARY_LEGACY)),
        serving_metadata=serving_metadata,
    )

    run_specs: list[RunSpec] = []
    for method in methods:
        for seed in seeds:
            run_specs.append(
                RunSpec(
                    method=method,
                    seed=seed,
                    decoding=decoding,
                    budgets=budgets,
                    use_guided_json=use_guided_json,
                )
            )

    artifacts: list[Any] = []

    def run_one(spec: RunSpec) -> Any:
        artifact_path = output_dir / f"{spec.method}__seed{spec.seed}.json"
        if artifact_path.exists():
            print(f"[SKIP] Existing artifact found: {artifact_path.name}")
            try:
                from experiment.modeling.artifacts import read_artifact
                return read_artifact(artifact_path)
            except Exception:
                pass
        try:
            artifact = runner.run_method(spec)
        except Exception as exc:
            import datetime
            import traceback
            from experiment.modeling.artifacts import (
                ContractRecord,
                RunArtifact,
                STATUS_MISSING,
                sha256_json,
            )
            logger.exception("Runner error on %s seed %d: %s", spec.method, spec.seed, exc)
            print(f"[ERROR] Runner error on {spec.method} seed {spec.seed}: {exc}")
            tb_str = traceback.format_exc()
            try:
                protocol_hash = runner.compute_protocol_hash(spec)
            except Exception:
                protocol_hash = ""
            # Known legacy bug: this fallback discards partial request/response
            # traces and reasoning_sha256, and writes placeholder zero usage.
            # Observed in 9 Gemma reasoning retail runs (unhashable type: 'list').
            # Keep these runs as failures in the denominator; zero counters are
            # missing measurements, not evidence of no calls or zero cost.
            # TODO: preserve partial session telemetry when serializing errors.
            artifact = RunArtifact(
                method=spec.method,
                model=client.model,
                endpoint=client.endpoint,
                seed=spec.seed,
                decoding=spec.decoding.to_dict(spec.seed),
                budgets=spec.budgets.to_dict(),
                sandbox=runner.sandbox_context(),
                interactions=(),
                contract=ContractRecord(status=STATUS_MISSING, error=str(exc)),
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0},
                latency={"total_s": 0.0, "per_call_s": []},
                spend={
                    "calls": 0,
                    "parse_failures": 0,
                    "states_observed": 0,
                    "oracle_feedback_queries": 0,
                    "sampled_feedback_queries": 0,
                    "sampled_states_checked": 0,
                },
                outcome="runner_error",
                evaluation=None,
                rounds=(),
                stopped_because=f"runner_error: {type(exc).__name__}: {exc}",
                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                serving=runner.serving_metadata,
                hashes={
                    "sandbox_config_sha256": sha256_json(json.loads(Path(runner.config_path).read_text())),
                    "decoding_sha256": sha256_json(spec.decoding.to_dict(spec.seed)),
                    "budgets_sha256": sha256_json(spec.budgets.to_dict()),
                    "protocol_sha256": protocol_hash,
                    "error": str(exc),
                    "traceback": tb_str[:2000],
                },
            )
        write_artifact(output_dir, artifact)
        return artifact

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(run_one, run_specs):
                artifacts.append(res)
    else:
        for spec in run_specs:
            artifacts.append(run_one(spec))

    summary_context = {
        "domain": "taubench_retail",
        "model": model,
        "endpoint": client.endpoint,
        "sandbox_config": str(sandbox_path),
        "sandbox": runner.sandbox_context(),
        "evaluation_states": len(runner.states),
        "methods": list(methods),
        "seeds": list(seeds),
        "workers": workers,
        "decoding": {"temperature": decoding.temperature, "top_p": decoding.top_p, "max_tokens": decoding.max_tokens},
        "budgets": budgets.to_dict(),
        "prompting": {
            "compact_context": compact_context,
            "context_token_limit": context_token_limit,
            "context_margin": context_margin,
            "counterexample_limit": counterexample_limit,
            "refinement_protocol": refinement_protocol,
            "guided_json": use_guided_json,
            "counterexample_format": COUNTEREXAMPLE_FORMAT_VERSION,
        },
    }
    summary_file = write_summary(output_dir, artifacts, summary_context)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(summary, indent=2))
    return summary


def main(argv: Sequence[str] | None = None, transport: Any | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    experiment_config = {}
    if args.experiment_config and args.experiment_config.exists():
        experiment_config = load_experiment_config(args.experiment_config)

    execute_from_args(args, experiment_config, transport=transport)
    return 0


if __name__ == "__main__":
    sys.exit(main())
