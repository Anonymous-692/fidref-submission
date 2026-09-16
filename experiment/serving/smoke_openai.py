#!/usr/bin/env python3
"""Call a running vLLM server through its OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def request_json(
    url: str,
    *,
    api_key: str | None = None,
    payload: dict[str, object] | None = None,
    timeout: float = 120.0,
) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--model")
    parser.add_argument(
        "--prompt",
        default="Return exactly this JSON object: {\"status\": \"ok\"}",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--wait", type=float, default=0.0, help="seconds to poll until server is ready (0 = no polling)")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    return parser


def get_server_info(
    base_url: str,
    *,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Fetch measured runtime information directly from vLLM endpoints."""
    base_url = base_url.rstrip("/")
    root_url = base_url.removesuffix("/v1")
    
    vllm_version = None
    try:
        ver_payload = request_json(f"{root_url}/version", api_key=api_key, timeout=timeout)
        if isinstance(ver_payload, dict):
            vllm_version = ver_payload.get("version")
    except Exception:
        pass

    models_payload = request_json(f"{base_url}/models", api_key=api_key, timeout=timeout)
    entries = models_payload.get("data", [])
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"{base_url}/models returned no model entries")
    first_model = entries[0] if isinstance(entries[0], dict) else {}
    
    return {
        "vllm_version": vllm_version,
        "max_model_len": first_model.get("max_model_len"),
        "model_id": first_model.get("id"),
        "root_model": first_model.get("root"),
        "owned_by": first_model.get("owned_by"),
        "all_models": [m.get("id") for m in entries if isinstance(m, dict) and m.get("id")],
    }


def wait_for_server(
    base_url: str,
    *,
    api_key: str | None = None,
    timeout: float = 600.0,
    poll_interval: float = 2.0,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Poll /v1/models and execute a smoke test until the engine is fully ready, returning measured server info."""
    import time
    deadline = time.time() + timeout
    base_url = base_url.rstrip("/")
    while time.time() < deadline:
        try:
            models = request_json(f"{base_url}/models", api_key=api_key, timeout=10.0)
            entries = models.get("data", [])
            if isinstance(entries, list) and entries:
                target_model = model or entries[0].get("id")
                if target_model:
                    completion = request_json(
                        f"{base_url}/chat/completions",
                        api_key=api_key,
                        timeout=15.0,
                        payload={
                            "model": target_model,
                            "messages": [{"role": "user", "content": "ping"}],
                            "max_tokens": 1,
                        },
                    )
                    if completion.get("choices"):
                        return get_server_info(base_url, api_key=api_key, timeout=10.0)
        except Exception:
            pass
        time.sleep(poll_interval)
    return None



def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_url = args.base_url.rstrip("/")
    if args.wait > 0:
        print(f"waiting up to {args.wait}s for {base_url} to be ready...", file=sys.stderr)
        if not wait_for_server(base_url, api_key=args.api_key, timeout=args.wait, poll_interval=args.poll_interval, model=args.model):
            print(f"server at {base_url} failed to become ready within {args.wait}s", file=sys.stderr)
            return 1

    try:
        models = request_json(
            f"{base_url}/models", api_key=args.api_key, timeout=args.timeout
        )
        entries = models.get("data", [])
        if not isinstance(entries, list) or not entries:
            raise RuntimeError("/v1/models returned no models")
        model = args.model or entries[0].get("id")
        if not isinstance(model, str) or not model:
            raise RuntimeError("could not determine the served model name")
        completion = request_json(
            f"{base_url}/chat/completions",
            api_key=args.api_key,
            timeout=args.timeout,
            payload={
                "model": model,
                "messages": [{"role": "user", "content": args.prompt}],
                "temperature": 0,
                "max_tokens": 64,
            },
        )
        choices = completion.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("chat completion returned no choices")
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"smoke test failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(completion, ensure_ascii=False, indent=2))
    print("OpenAI-compatible smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


