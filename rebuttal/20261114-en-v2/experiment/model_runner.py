#!/usr/bin/env python3
"""Run the model-driven contract experiment against a local OpenAI-compatible server.

The endpoint defaults to loopback (``http://127.0.0.1:8000/v1``) because the model
is served locally by ``experiment/serving``; a non-loopback base URL is refused
unless ``--allow-remote-host`` is passed. Artifacts are written only to the
directory named by ``--output``, which has no default.

    python experiment/model_runner.py --output results/model_run --model qwen2.5-14b-instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.deployment import (  # noqa: E402
    DeploymentConfig,
    DeploymentExperimentRunner,
)
from experiment.environment import (  # noqa: E402  (path set up above)
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_STATES,
    EnvConfig,
)
from experiment.modeling import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    METHODS,
    Budgets,
    ChatClient,
    Decoding,
    ExperimentRunner,
    RunSpec,
    TransportError,
    write_artifact,
    write_summary,
)
from experiment.modeling.artifacts import COUNTEREXAMPLE_FORMAT_VERSION  # noqa: E402

DEFAULT_EXPERIMENT_CONFIG = Path(__file__).resolve().parent / "configs" / "model_experiment.json"

# Keys accepted in the experiment configuration file. Each has a matching CLI
# flag, and the flag wins when both are given.
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
)


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Read the defaults file, rejecting keys the runner does not understand."""
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
        help="JSON file holding the defaults below",
    )
    parser.add_argument(
        "--domain",
        choices=["shopping", "deployment"],
        help="environment domain (default: shopping, or inferred from sandbox-config)",
    )
    parser.add_argument("--sandbox-config", type=Path, help="sandbox EnvConfig JSON file")
    parser.add_argument("--base-url", help=f"OpenAI-compatible base URL (default {DEFAULT_BASE_URL})")
    parser.add_argument("--api-key", help="bearer token, if the server requires one")
    parser.add_argument("--model", help="served model name")
    parser.add_argument("--methods", nargs="+", choices=METHODS, help="methods to run")
    parser.add_argument("--seeds", nargs="+", type=int, help="seeds to run each method under")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int, help="completion tokens per call")
    parser.add_argument("--state-budget", type=int, help="sandbox states a method may observe")
    parser.add_argument("--query-budget", type=int, help="model calls per run")
    parser.add_argument("--token-budget", type=int, help="prompt+completion tokens per run")
    parser.add_argument("--max-depth", type=int, help="enumeration depth bound for scoring")
    parser.add_argument("--max-states", type=int, help="enumeration size bound for scoring")
    parser.add_argument("--timeout", type=float, help="per-request timeout in seconds")
    parser.add_argument(
        "--workers",
        type=int,
        help="runs to execute concurrently (default: 1)",
    )
    parser.add_argument(
        "--allow-remote-host",
        action="store_true",
        help="permit a non-loopback base URL (off by default)",
    )
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument("--guided-json", action="store_true", help="constrain output to valid contract JSON schema")
    parser.add_argument("--wait-server", type=float, default=600.0, help="seconds to wait for server readiness (default: 600)")
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
    """A compact table; the oracle scores are the right-hand columns."""
    lines = [
        "== model contract experiment ==",
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
    lines.append(
        "fa=false accepts, fr=false rejects, pv=postcondition violations; "
        "oracle=evaluator queries consumed as method feedback"
    )
    return "\n".join(lines)


def _cell(value: Any) -> str:
    return "-" if value is None else str(value)


def main(argv: Sequence[str] | None = None, *, transport: Any = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_experiment_config(args.experiment_config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"could not read the experiment configuration: {exc}", file=sys.stderr)
        return 2

    model = _pick(args.model, config, "model", None)
    if not model:
        print("no model was given; pass --model or set it in the config file", file=sys.stderr)
        return 2

    sandbox_path = _resolve_sandbox_path(
        _pick(args.sandbox_config, config, "sandbox_config", "experiment/configs/default.json")
    )

    domain = _pick(args.domain, config, "domain", None)
    if domain is None:
        domain = "deployment" if "deployment" in str(sandbox_path) else "shopping"

    if domain == "deployment":
        try:
            sandbox_config = DeploymentConfig.from_json_file(sandbox_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"could not read the deployment sandbox configuration: {exc}", file=sys.stderr)
            return 2
        runner_cls = DeploymentExperimentRunner
    else:
        try:
            sandbox_config = EnvConfig.from_json_file(sandbox_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"could not read the sandbox configuration: {exc}", file=sys.stderr)
            return 2
        runner_cls = ExperimentRunner

    decoding = Decoding(
        temperature=float(_pick(args.temperature, config, "temperature", 0.2)),
        top_p=float(_pick(args.top_p, config, "top_p", 0.95)),
        max_tokens=int(_pick(args.max_tokens, config, "max_tokens", 1024)),
    )
    budgets = Budgets(
        state_budget=int(_pick(args.state_budget, config, "state_budget", 24)),
        query_budget=int(_pick(args.query_budget, config, "query_budget", 4)),
        token_budget=int(_pick(args.token_budget, config, "token_budget", 16000)),
    )
    methods = tuple(_pick(args.methods, config, "methods", list(METHODS)))
    seeds = tuple(int(seed) for seed in _pick(args.seeds, config, "seeds", [0]))
    max_depth = int(_pick(args.max_depth, config, "max_depth", DEFAULT_MAX_DEPTH))
    max_states = int(_pick(args.max_states, config, "max_states", DEFAULT_MAX_STATES))
    workers = int(_pick(args.workers, config, "workers", 1))

    if workers < 1:
        print("workers must be at least 1", file=sys.stderr)
        return 2

    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        print(f"unknown methods: {unknown}; known methods: {list(METHODS)}", file=sys.stderr)
        return 2

    try:
        client = ChatClient(
            model=model,
            base_url=_pick(args.base_url, config, "base_url", DEFAULT_BASE_URL),
            api_key=args.api_key,
            timeout=float(_pick(args.timeout, config, "timeout", DEFAULT_TIMEOUT)),
            transport=transport,
            allow_remote=args.allow_remote_host,
        )
    except ValueError as exc:
        print(f"could not build the client: {exc}", file=sys.stderr)
        return 2

    if transport is None:
        from experiment.serving.smoke_openai import wait_for_server
        server_info = wait_for_server(client.base_url, api_key=args.api_key, timeout=args.wait_server, model=model)
        if server_info is None:
            print(f"server at {client.base_url} failed to become ready within {args.wait_server}s", file=sys.stderr)
            return 1

        # Serving sidecar. serve_vllm.sh 로 띄운 로컬 서버만 사이드카를 남기므로,
        # 우리가 기동하지 않은 원격/터널 엔드포인트에는 사이드카가 없거나 다른 서버를 가리킨다.
        # 원칙: 값을 지어내지 않는다. 사이드카가 이 실행에 해당할 때만 TP/dtype 을 채우고,
        # 아닐 때는 라이브 서버가 보고한 값만 쓰고 나머지는 null 로 남긴다.
        sidecar_path = ROOT / "experiment" / "serving" / "logs" / "serving.json"
        sidecar: dict = {}
        sidecar_status = "absent"
        if sidecar_path.is_file():
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"ERROR: could not read serving sidecar {sidecar_path}: {exc}", file=sys.stderr)
                return 1
            sidecar_endpoint = str(sidecar.get("endpoint") or "")
            same_model = sidecar.get("served_model_name") == model
            same_endpoint = bool(sidecar_endpoint) and client.base_url.rstrip("/").startswith(
                sidecar_endpoint.rstrip("/")
            )
            if same_model and same_endpoint:
                sidecar_status = "applies"
            elif same_endpoint and not same_model:
                # 같은 엔드포인트인데 모델이 다르면 진짜 불일치다. 중단한다.
                print(
                    f"ERROR: sidecar model mismatch on the same endpoint {sidecar_endpoint}: "
                    f"sidecar has '{sidecar.get('served_model_name')}' but runner requested '{model}'",
                    file=sys.stderr,
                )
                return 1
            else:
                sidecar_status = "not_applicable"
                sidecar = {}

        if sidecar_status != "applies":
            print(
                f"NOTE: serving sidecar {sidecar_status} for {client.base_url}; "
                "tensor_parallel_size/dtype will be recorded as null rather than guessed.",
                file=sys.stderr,
            )

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
            "tensor_parallel_size": sidecar.get("tensor_parallel_size"),
            "dtype": sidecar.get("dtype"),
            "gpu_ids": sidecar.get("gpu_ids"),
            "profile": sidecar.get("profile"),
            "sidecar_status": sidecar_status,
            "max_num_seqs": sidecar.get("max_num_seqs"),
            "max_num_batched_tokens": sidecar.get("max_num_batched_tokens"),
            "swap_space_gib": sidecar.get("swap_space_gib"),
            "enforce_eager": sidecar.get("enforce_eager"),
            "disable_custom_all_reduce": sidecar.get("disable_custom_all_reduce"),
            "enable_prefix_caching": sidecar.get("enable_prefix_caching"),
            "enable_chunked_prefill": sidecar.get("enable_chunked_prefill"),
        }
    else:
        serving_metadata = {
            "model": model,
            "endpoint": client.endpoint,
            "mock": True,
        }
    serving_metadata["request_concurrency"] = workers

    runner = runner_cls(
        sandbox_config,
        client,
        max_depth=max_depth,
        max_states=max_states,
        config_path=str(sandbox_path),
        serving_metadata=serving_metadata,
    )
    specs = tuple(
        RunSpec(
            method=method,
            seed=seed,
            decoding=decoding,
            budgets=budgets,
            use_guided_json=args.guided_json,
        )
        for method in methods
        for seed in seeds
    )

    artifacts = []
    if workers == 1:
        executor = None
        futures = (None for _ in specs)
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="experiment")
        futures = tuple(executor.submit(runner.run, spec) for spec in specs)

    try:
        for spec, future in zip(specs, futures):
            try:
                artifact = runner.run(spec) if future is None else future.result()
            except TransportError as exc:  # pragma: no cover - client maps these already
                print(f"{spec.method} seed {spec.seed}: {exc}", file=sys.stderr)
                return 1
            write_artifact(args.output, artifact)
            artifacts.append(artifact)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    context = {
        "domain": domain,
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
        "counterexample_format": COUNTEREXAMPLE_FORMAT_VERSION,
    }
    summary_path = write_summary(args.output, artifacts, context)
    rows = [artifact.summary_row() for artifact in artifacts]

    if args.json:
        print(json.dumps({"context": context, "runs": rows}, ensure_ascii=False, indent=2))
    else:
        print(render(rows, context))
        print(f"\nartifacts written to {args.output} (summary: {summary_path.name})")

    unreachable = [
        artifact.name for artifact in artifacts if artifact.stopped_because == "transport_error"
    ]
    if unreachable:
        print(f"\nFAILED: the endpoint was unreachable for: {unreachable}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
