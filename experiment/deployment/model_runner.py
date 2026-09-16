#!/usr/bin/env python3
"""Run contract synthesis experiments on the cloud deployment domain."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.deployment import (
    DEFAULT_DEPLOYMENT_CONFIG,
    DeploymentConfig,
    DeploymentExperimentRunner,
)
from experiment.modeling import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    METHODS,
    Budgets,
    ChatClient,
    Decoding,
    RunSpec,
    TransportError,
    write_artifact,
    write_summary,
)
from experiment.modeling.artifacts import COUNTEREXAMPLE_FORMAT_VERSION

DEFAULT_EXPERIMENT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "deployment_model_experiment.json"
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
    "guided_json",
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
        default=DEFAULT_EXPERIMENT_CONFIG,
        help="JSON file holding default settings",
    )
    parser.add_argument("--sandbox-config", type=Path, help="DeploymentConfig JSON file")
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
    parser.add_argument("--max-states", type=int, default=15000)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--allow-remote-host", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--guided-json", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compact-context", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--context-token-limit", type=int)
    parser.add_argument("--context-margin", type=int)
    parser.add_argument("--counterexample-limit", type=int)
    parser.add_argument("--wait-server", type=float, default=600.0)
    return parser


def _pick(value: Any, config: Mapping[str, Any], key: str, fallback: Any) -> Any:
    if value is not None:
        return value
    if key in config and config[key] is not None:
        return config[key]
    return fallback


def _resolve_sandbox_path(raw: Any) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (ROOT / path)


def render(rows: Sequence[Mapping[str, Any]], context: Mapping[str, Any]) -> str:
    lines = [
        "== cloud deployment contract experiment ==",
        f"model={context['model']} endpoint={context['endpoint']}",
        f"sandbox={context['sandbox_config']} evaluation_states={context['evaluation_states']}",
        f"budgets={context['budgets']}",
        "",
        f"{'method':<24}{'seed':>5}{'outcome':>16}{'fa':>5}{'fr':>5}{'pv':>5}"
        f"{'calls':>7}{'tokens':>8}{'states':>8}{'oracle':>8}",
    ]
    for row in rows:
        lines.append(
            f"{row['method']:<24}{row['seed']:>5}{row['outcome']:>16}"
            f"{_cell(row['false_accepts']):>5}{_cell(row['false_rejects']):>5}"
            f"{_cell(row['postcondition_violations']):>5}"
            f"{row['model_calls']:>7}{row['total_tokens']:>8}"
            f"{row['states_observed']:>8}{row['oracle_feedback_queries']:>8}"
        )
    from experiment.modeling.artifacts import wilson_interval
    exact = sum(1 for row in rows if row["exact"])
    n = len(rows)
    center, lower, upper = wilson_interval(exact, n)
    lines.append("")
    lines.append(
        f"exact contracts: {exact}/{n} "
        f"(95% CI: {lower:.1%} - {upper:.1%} | center {center:.1%})"
    )
    return "\n".join(lines)


def _cell(value: Any) -> str:
    return "-" if value is None else str(value)


def main(argv: Sequence[str] | None = None, *, transport: Any = None) -> int:
    args = build_parser().parse_args(argv)

    config: dict[str, Any] = {}
    if args.experiment_config and args.experiment_config.is_file():
        try:
            config = load_experiment_config(args.experiment_config)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"could not read experiment configuration: {exc}", file=sys.stderr)
            return 2

    model = _pick(args.model, config, "model", None)
    if not model:
        print("no model given; pass --model or set it in config", file=sys.stderr)
        return 2

    sandbox_path = _resolve_sandbox_path(
        _pick(args.sandbox_config, config, "sandbox_config", "experiment/configs/deployment_default.json")
    )
    try:
        sandbox_config = DeploymentConfig.from_json_file(sandbox_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"could not read deployment sandbox config: {exc}", file=sys.stderr)
        return 2

    decoding = Decoding(
        temperature=float(_pick(args.temperature, config, "temperature", 0.2)),
        top_p=float(_pick(args.top_p, config, "top_p", 0.95)),
        max_tokens=int(_pick(args.max_tokens, config, "max_tokens", 2048)),
    )
    budgets = Budgets(
        state_budget=int(_pick(args.state_budget, config, "state_budget", 48)),
        query_budget=int(_pick(args.query_budget, config, "query_budget", 4)),
        token_budget=int(_pick(args.token_budget, config, "token_budget", 16000)),
    )
    methods = tuple(_pick(args.methods, config, "methods", ["active_cegis", "active_cegis_no_coverage"]))
    seeds = tuple(int(seed) for seed in _pick(args.seeds, config, "seeds", [0]))
    max_depth = int(_pick(args.max_depth, config, "max_depth", 20))
    max_states = int(_pick(args.max_states, config, "max_states", 6000))
    workers = int(_pick(args.workers, config, "workers", 1))
    compact_context = bool(_pick(args.compact_context, config, "compact_context", False))
    context_token_limit_raw = _pick(
        args.context_token_limit, config, "context_token_limit", None
    )
    context_token_limit = (
        int(context_token_limit_raw) if context_token_limit_raw is not None else None
    )
    context_margin = int(_pick(args.context_margin, config, "context_margin", 256))
    counterexample_limit = int(
        _pick(args.counterexample_limit, config, "counterexample_limit", 5)
    )
    use_guided_json = bool(_pick(args.guided_json, config, "guided_json", False))
    reasoning_effort = _pick(None, config, "reasoning_effort", None)
    thinking_token_budget_raw = _pick(None, config, "thinking_token_budget", None)
    thinking_token_budget = int(thinking_token_budget_raw) if thinking_token_budget_raw is not None else None
    chat_template_kwargs = dict(_pick(None, config, "chat_template_kwargs", {}))
    return_token_ids = bool(_pick(None, config, "return_token_ids", False))

    if workers < 1:
        print("workers must be at least 1", file=sys.stderr)
        return 2

    try:
        client = ChatClient(
            model=model,
            base_url=_pick(args.base_url, config, "base_url", DEFAULT_BASE_URL),
            api_key=args.api_key,
            timeout=float(_pick(args.timeout, config, "timeout", DEFAULT_TIMEOUT)),
            transport=transport,
            allow_remote=args.allow_remote_host,
            reasoning_effort=reasoning_effort,
            thinking_token_budget=thinking_token_budget,
            chat_template_kwargs=chat_template_kwargs,
            return_token_ids=return_token_ids,
        )
    except ValueError as exc:
        print(f"could not build client: {exc}", file=sys.stderr)
        return 2

    if transport is None:
        from experiment.serving.smoke_openai import wait_for_server
        server_info = wait_for_server(client.base_url, api_key=args.api_key, timeout=args.wait_server, model=model)
        if server_info is None:
            print(f"server at {client.base_url} failed to become ready within {args.wait_server}s", file=sys.stderr)
            return 1
        # 서버가 보고한 vLLM 버전에 맞춰 구조화 디코딩 필드를 확정한다.
        # 0.19 이상은 최상위 guided_json 을 무시하므로 response_format 을 써야 한다.
        _structured_style = client.set_structured_output_style_from_version(
            (server_info or {}).get("vllm_version")
        )
        serving_metadata = {
            "model": model,
            "endpoint": client.endpoint,
            "vllm_version": server_info.get("vllm_version"),
            "structured_output_style": _structured_style,
            "max_model_len": server_info.get("max_model_len"),
        }
    else:
        serving_metadata = {
            "model": model,
            "endpoint": client.endpoint,
            "mock": True,
        }
    if client.reasoning_config:
        serving_metadata["reasoning_request"] = client.reasoning_config
    serving_metadata["request_concurrency"] = workers

    try:
        runner = DeploymentExperimentRunner(
            sandbox_config,
            client,
            max_depth=max_depth,
            max_states=max_states,
            config_path=str(sandbox_path),
            serving_metadata=serving_metadata,
            compact_context=compact_context,
            context_token_limit=context_token_limit,
            context_margin=context_margin,
            counterexample_limit=counterexample_limit,
        )
    except ValueError as exc:
        print(f"could not initialize deployment runner: {exc}", file=sys.stderr)
        return 2
    specs = tuple(
        RunSpec(
            method=method,
            seed=seed,
            decoding=decoding,
            budgets=budgets,
            use_guided_json=use_guided_json,
        )
        for method in methods
        for seed in seeds
    )

    artifacts = []
    if workers == 1:
        executor = None
        futures = (None for _ in specs)
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="deployment_exp")
        futures = tuple(executor.submit(runner.run, spec) for spec in specs)

    try:
        for spec, future in zip(specs, futures):
            try:
                artifact = runner.run(spec) if future is None else future.result()
            except TransportError as exc:
                print(f"{spec.method} seed {spec.seed}: {exc}", file=sys.stderr)
                return 1
            write_artifact(args.output, artifact)
            artifacts.append(artifact)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    context = {
        "domain": "deployment",
        "model": model,
        "endpoint": client.endpoint,
        "sandbox_config": str(sandbox_path),
        "sandbox": runner.sandbox_context(),
        "evaluation_states": len(runner.states),
        "methods": list(methods),
        "seeds": list(seeds),
        "workers": workers,
        "decoding": {
            "temperature": decoding.temperature,
            "top_p": decoding.top_p,
            "max_tokens": decoding.max_tokens,
        },
        "budgets": budgets.to_dict(),
        "prompting": {
            "compact_context": compact_context,
            "context_token_limit": context_token_limit,
            "context_margin": context_margin,
            "counterexample_limit": counterexample_limit,
            "guided_json": use_guided_json,
            "counterexample_format": COUNTEREXAMPLE_FORMAT_VERSION,
        },
    }
    summary_path = write_summary(args.output, artifacts, context)
    rows = [artifact.summary_row() for artifact in artifacts]

    if args.json:
        print(json.dumps({"context": context, "runs": rows}, ensure_ascii=False, indent=2))
    else:
        print(render(rows, context))
        print(f"\nartifacts written to {args.output} (summary: {summary_path.name})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
