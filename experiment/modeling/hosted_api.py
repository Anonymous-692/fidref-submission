"""Opt-in hosted JSON-mode transport; existing local clients remain unchanged.

One coordinator owns the budget, admission limits and durable request journal.
Successful responses are replayed on resume only when the entire request matches.
Uncertain/failed requests stop the suite rather than silently redrawing samples.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from .client import ChatClient, ChatResponse, Usage

ENDPOINT = "https://api.openai.com/v1/chat/completions"
# Conservative input price: no cache discounts; Luna includes cache-write uplift.
RATES = {"gpt-5.4-mini": (0.75, 4.50), "gpt-5.6-luna": (0.25, 1.20)}


class SuitePaused(RuntimeError):
    """Infrastructure/budget failure, not a scientific failure outcome."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_key(root: Path):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        for line in (root / ".env").read_text().splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip().removeprefix("export ").strip() == "OPENAI_API_KEY":
                key = value.strip().strip("\"'")
    if not key:
        raise SuitePaused("OPENAI_API_KEY is not configured")
    return key


class Coordinator:
    def __init__(self, root: Path, key: str, *, limit=12.0, prior_reserve=0.011226,
                 rpm=300, tpm=1_400_000, per_model=16):
        if not 0 < limit <= 12 or prior_reserve < 0:
            raise ValueError("invalid cost limit")
        self.root, self.key, self.limit = root, key, limit
        self.lock = threading.Lock()
        self.halted = threading.Event()
        self.windows = deque()
        self.rpm, self.tpm = rpm, tpm
        self.slots = {m: threading.Semaphore(per_model) for m in RATES}
        self.charged = prior_reserve
        self.reservations = {}
        for path in root.glob("requests/**/*.json"):
            item = json.loads(path.read_text())
            if item["status"] != "complete":
                raise SuitePaused(f"unresolved request requires audit: {path}")
            self.charged += item["cost_upper_usd"]

    def stop(self, reason):
        self.halted.set()
        raise SuitePaused(reason)

    def reserve(self, label, model, payload):
        # UTF-8 bytes plus framing allowance conservatively bound input tokens.
        tokens = len(json.dumps(payload, ensure_ascii=False).encode()) + 8192
        output = payload["max_completion_tokens"]
        bound = (tokens * RATES[model][0] + output * RATES[model][1]) / 1e6
        demand = tokens + output
        if demand > self.tpm:
            self.stop("one request exceeds token admission limit")
        while True:
            with self.lock:
                if self.halted.is_set():
                    raise SuitePaused("suite admission paused")
                now = time.monotonic()
                while self.windows and now - self.windows[0][0] >= 60:
                    self.windows.popleft()
                if len(self.windows) < self.rpm and sum(v for _, v in self.windows) + demand <= self.tpm:
                    if self.charged + sum(self.reservations.values()) + bound > self.limit:
                        self.stop("$12 admission ceiling reached, including in-flight reservations")
                    self.reservations[label] = bound
                    self.windows.append((now, demand))
                    return bound
            self.halted.wait(0.25)

    def call(self, relative: str, payload):
        path = self.root / "requests" / (relative + ".json")
        request_hash = digest(payload)
        if path.exists():
            item = json.loads(path.read_text())
            if (item["request_sha256"] != request_hash or item["status"] != "complete"
                    or digest(item["request"]) != request_hash
                    or digest(item["response"]) != item["response_sha256"]):
                self.stop(f"replay mismatch or unresolved request: {relative}")
            return self.response(item)
        model = payload["model"]
        with self.slots[model]:
            bound = self.reserve(relative, model, payload)
            item = {"status": "reserved", "request": payload,
                    "request_sha256": request_hash, "reserved_upper_usd": bound,
                    "created_at": time.time()}
            save(path, item)  # reservation reaches disk before the request is sent
            request = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(),
                headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    raw, status, headers = response.read(), response.status, response.headers
            except urllib.error.HTTPError as exc:
                raw, status, headers = exc.read(), exc.code, exc.headers
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                item.update(status="uncertain", error=str(exc).replace(self.key, "[REDACTED]"))
                save(path, item)
                self.stop(f"uncertain request; automatic retry disabled: {relative}")
            item.update(http_status=status, latency_s=time.monotonic() - started,
                        headers={k: v for k, v in headers.items()
                                 if k.lower().startswith("x-ratelimit") or k.lower() == "x-request-id"})
            try:
                data = json.loads(raw.decode().replace(self.key, "[REDACTED]"))
            except (ValueError, UnicodeError):
                item.update(status="uncertain", error="non-JSON API response")
                save(path, item)
                self.stop(f"non-JSON transport response: {relative}")
            item["response"] = data
            usage = data.get("usage") or {}
            if status != 200 or not all(isinstance(usage.get(k), int) and usage[k] >= 0
                                        for k in ("prompt_tokens", "completion_tokens")):
                item["status"] = "failed"
                save(path, item)
                self.stop(f"API/usage failure ({status}): {relative}; inspect journal")
            cost = (usage["prompt_tokens"] * RATES[model][0]
                    + usage["completion_tokens"] * RATES[model][1]) / 1e6
            item.update(status="complete", cost_upper_usd=cost, response_sha256=digest(data))
            save(path, item)
            with self.lock:
                self.charged += cost
                del self.reservations[relative]
            if cost > bound:
                self.stop("observed usage exceeded reservation bound")
            return self.response(item)

    @staticmethod
    def response(item):
        data = item["response"]
        choices = data.get("choices") or []
        if not choices:
            raise SuitePaused("successful response missing choices")
        content = choices[0]["message"].get("content") or ""
        if not isinstance(content, str):
            raise SuitePaused("unexpected response content shape")
        return ChatResponse(text=content, usage=Usage.from_payload(data["usage"]),
            latency_s=item["latency_s"], status=200, request=item["request"], raw=data,
            finish_reason=choices[0].get("finish_reason"))


class HostedClient(ChatClient):
    def __init__(self, model, coordinator, label, initial_label=None):
        super().__init__(model=model, base_url="https://api.openai.com/v1",
                         allow_remote=True, reasoning_effort="none")
        self.coordinator, self.label, self.initial_label = coordinator, label, initial_label
        self.calls = 0

    def build_request(self, messages, *, temperature, top_p, max_tokens, seed=None,
                      stop=None, guided_json=None, repetition_penalty=None):
        if guided_json is not None or stop or repetition_penalty is not None:
            raise ValueError("hosted protocol only supports JSON mode without local extensions")
        return {"model": self.model, "messages": [dict(m) for m in messages],
                "temperature": temperature, "top_p": top_p,
                "max_completion_tokens": max_tokens, "reasoning_effort": "none",
                "response_format": {"type": "json_object"},
                "store": False, "service_tier": "default", "stream": False}

    def complete(self, messages, *, temperature=0.2, top_p=0.95, max_tokens=2048,
                 seed=None, stop=None, guided_json=None, repetition_penalty=None):
        payload = self.build_request(messages, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, seed=seed, stop=stop, guided_json=guided_json,
            repetition_penalty=repetition_penalty)
        label = self.initial_label if self.calls == 0 and self.initial_label else f"{self.label}/{self.calls}"
        if self.calls == 0 and self.initial_label:
            if not (self.coordinator.root / "requests" / (label + ".json")).exists():
                raise SuitePaused("shared initial response missing")
        response = self.coordinator.call(label, payload)
        self.calls += 1
        return response
