"""Retail V-only CLI: offline preflight by default; explicit bounded smoke only."""
import argparse
import json
from pathlib import Path
import signal

from ..modeling.client import ChatClient
from ..modeling.runner import RunSpec, Budgets, Decoding
from .checkpoint import RetailSuite, METHODS, atomic_json
from .config import RetailConfig
from .information_policy import CONDITIONS, source_hashes
from .runner import RetailExperimentRunner
from .state import SEMANTIC_FIELDS_PROTOCOL


def build_runner(client, condition, serving):
    fixture = Path(__file__).resolve().parents[1] / "configs/taubench_retail_default.json"
    return RetailExperimentRunner(client, config=RetailConfig.from_json(fixture), config_path=str(fixture),
        information_condition=condition, vocabulary_protocol=SEMANTIC_FIELDS_PROTOCOL,
        context_token_limit=8192, context_margin=256, compact_context=True,
        refinement_protocol="self_contained_v3", serving_metadata=serving)


def specs(seed):
    return [RunSpec(method=m, seed=seed, decoding=Decoding(max_tokens=2048),
                    budgets=Budgets(48, 4, 16000), use_guided_json=True) for m in METHODS]


def deployment_busy():
    found = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = path.read_bytes().split(b"\0")
        except (OSError, ProcessLookupError):
            continue
        if any(x.endswith(b"info_launch_20261113_v2/suite.py") for x in args) and b"queue" in args:
            found.append(int(path.parent.name))
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="execute three seed-1000 arms, never the main matrix")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--serving-record", type=Path)
    args = parser.parse_args(argv)
    if args.resume and not args.smoke:
        parser.error("resume applies to smoke execution only")
    client = ChatClient(model="gemma-4-31b-it", base_url="http://127.0.0.1:18002/v1", timeout=180,
                        structured_output_style="response_format", chat_template_kwargs={"enable_thinking": False})
    serving = json.loads(args.serving_record.read_text()) if args.serving_record else {}
    runner = build_runner(client, args.condition, serving)
    trials = specs(1000)
    if runner.enumeration.truncated or len(runner.states) != 1294:
        raise ValueError("unexpected/truncated Retail fixture")
    report = {"phase": "offline_preflight", "condition": args.condition, "states": 1294,
              "model_calls": 0, "deployment_queue_pids": deployment_busy(),
              "sources": source_hashes(), "protocols": {s.method: runner.compute_protocol_hash(s) for s in trials},
              "main_launch_enabled": False}
    if not args.smoke:
        atomic_json(args.output, report)
        print(json.dumps(report, indent=2))
        return
    if report["deployment_queue_pids"]:
        raise RuntimeError("Deployment queue is active; no Retail requests sent")
    if serving.get("dtype") not in ("bfloat16", "torch.bfloat16", "bf16") or serving.get("vllm_version") != "0.19.1":
        raise ValueError("verified BF16/vLLM serving record required")
    # Identity GET is read-only. This is not a substitute for operator verification
    # of GPU ownership, dtype, weights and template before approving this command.
    from urllib.request import urlopen
    with urlopen("http://127.0.0.1:18002/v1/models", timeout=15) as response:
        models = json.load(response)
    if client.model not in {r["id"] for r in models["data"]}:
        raise ValueError("served model identity mismatch")
    with RetailSuite(runner, trials, args.output, resume=args.resume) as suite:
        previous = {s: signal.signal(s, lambda *_: suite.stop_requested.set()) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            rows = suite.execute()
        finally:
            for sig, handler in previous.items(): signal.signal(sig, handler)
    for row in rows:
        if row["stopped_because"] in ("transport_error", "usage_unavailable", "token_budget_overrun"):
            raise RuntimeError("smoke infrastructure gate failed; preserve results")
        if row["usage"]["total_tokens"] > 16000 or row["spend"]["oracle_feedback_queries"] != 0:
            raise RuntimeError("smoke budget/isolation gate failed")
    print("Smoke saved; full matrix remains disabled.")


if __name__ == "__main__":
    main()
