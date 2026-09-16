#!/usr/bin/env python3
"""A dependency-free OpenAI-compatible chat client.

Only the Python standard library is used, and the transport is a plain callable
so that tests can drive every method without a server. The default endpoint is
loopback, and non-loopback hosts are refused unless the caller explicitly opts
in: this experiment talks to a locally served model, never to a hosted API.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_TIMEOUT = 120.0

# 구조화 디코딩 요청 필드 스타일
STRUCTURED_GUIDED_JSON = "guided_json"          # vLLM < 0.19
STRUCTURED_RESPONSE_FORMAT = "response_format"  # vLLM >= 0.19 (OpenAI 호환)


def structured_output_style_for(vllm_version: str | None) -> str:
    """서버 버전에 맞는 구조화 디코딩 필드 이름을 고른다.

    버전을 모르면 신형(response_format)을 택한다. 구형 서버는 알 수 없는 필드를
    무시하므로 잘못 택했을 때의 실패 양상이 조용한 미적용으로 동일하지만,
    현행 서버 다수가 0.19 이상이므로 기본값을 신형으로 둔다.
    """
    if not vllm_version:
        return STRUCTURED_RESPONSE_FORMAT
    head = str(vllm_version).split("+")[0].split("post")[0]
    parts = []
    for chunk in head.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    if len(parts) < 2:
        return STRUCTURED_RESPONSE_FORMAT
    major, minor = parts[0], parts[1]
    if (major, minor) >= (0, 19):
        return STRUCTURED_RESPONSE_FORMAT
    return STRUCTURED_GUIDED_JSON

# Hosts considered local. ``0.0.0.0`` is deliberately absent: it is a bind
# address, not a destination.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

Transport = Callable[[str, Mapping[str, str], bytes, float], "RawResponse"]


class TransportError(RuntimeError):
    """The endpoint could not be reached, or answered with something unusable."""


@dataclass(frozen=True)
class RawResponse:
    """What a transport hands back: an HTTP status and a body."""

    status: int
    body: bytes


@dataclass(frozen=True)
class Usage:
    """Token accounting for one completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=(
                (self.reasoning_tokens or 0) + (other.reasoning_tokens or 0)
                if self.reasoning_tokens is not None or other.reasoning_tokens is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, int]:
        payload = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.reasoning_tokens is not None:
            payload["reasoning_tokens"] = self.reasoning_tokens
            payload["non_reasoning_completion_tokens"] = max(
                self.completion_tokens - self.reasoning_tokens, 0
            )
        return payload

    @classmethod
    def from_payload(cls, payload: Any) -> "Usage":
        """Read the ``usage`` block defensively; absent fields count as zero."""
        if not isinstance(payload, Mapping):
            return cls()

        def value(key: str) -> int:
            raw = payload.get(key)
            if isinstance(raw, bool) or not isinstance(raw, int):
                return 0
            return max(raw, 0)

        prompt = value("prompt_tokens")
        completion = value("completion_tokens")
        total = value("total_tokens") or prompt + completion
        details = payload.get("completion_tokens_details")
        reasoning = None
        if isinstance(details, Mapping):
            raw_reasoning = details.get("reasoning_tokens")
            if isinstance(raw_reasoning, int) and not isinstance(raw_reasoning, bool):
                reasoning = max(raw_reasoning, 0)
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            reasoning_tokens=reasoning,
        )


@dataclass(frozen=True)
class ChatResponse:
    """One completion together with everything the artifact needs to record."""

    text: str
    usage: Usage
    latency_s: float
    status: int
    request: Mapping[str, Any]
    raw: Mapping[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None


def urllib_transport(
    url: str, headers: Mapping[str, str], body: bytes, timeout: float
) -> RawResponse:
    """Send one POST with :mod:`urllib`, mapping every failure to a clear error."""
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return RawResponse(status=response.status, body=response.read())
    except urllib.error.HTTPError as exc:  # a served error is still a response
        return RawResponse(status=exc.code, body=exc.read())
    except (urllib.error.URLError, OSError) as exc:
        raise TransportError(f"could not reach {url}: {exc}") from exc


def is_loopback(base_url: str) -> bool:
    """True when ``base_url`` points at this machine."""
    host = urllib.parse.urlsplit(base_url).hostname
    if host is None:
        return False
    return host.lower() in LOOPBACK_HOSTS


class ChatClient:
    """Minimal ``/v1/chat/completions`` client with a pluggable transport."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport: Transport | None = None,
        allow_remote: bool = False,
        clock: Callable[[], float] = time.perf_counter,
        structured_output_style: str | None = None,
        guided_decoding_backend: str | None = None,
        reasoning_effort: str | None = None,
        thinking_token_budget: int | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        return_token_ids: bool = False,
    ) -> None:
        if not model:
            raise ValueError("a model name is required")
        base_url = base_url.rstrip("/")
        if not base_url:
            raise ValueError("a base URL is required")
        if not allow_remote and not is_loopback(base_url):
            raise ValueError(
                f"{base_url} is not a loopback address; this runner talks to a locally "
                "served model. Pass allow_remote=True only if you really mean it."
            )
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.allow_remote = allow_remote
        self._transport = transport if transport is not None else urllib_transport
        self._clock = clock
        # 서버 버전을 아직 모를 때는 신형(response_format)을 기본으로 둔다.
        # 실제 버전 확인 후 set_structured_output_style() 로 갱신한다.
        self.structured_output_style = (
            structured_output_style or STRUCTURED_RESPONSE_FORMAT
        )
        self.guided_decoding_backend = guided_decoding_backend
        if reasoning_effort not in (None, "none", "low", "medium", "high"):
            raise ValueError("reasoning_effort must be none, low, medium, or high")
        if thinking_token_budget is not None and thinking_token_budget < 0:
            raise ValueError("thinking_token_budget must be non-negative")
        self.reasoning_effort = reasoning_effort
        self.thinking_token_budget = thinking_token_budget
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.return_token_ids = return_token_ids

    @property
    def reasoning_config(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.thinking_token_budget is not None:
            payload["thinking_token_budget"] = self.thinking_token_budget
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        if self.return_token_ids:
            payload["return_token_ids"] = True
        return payload

    def set_structured_output_style_from_version(self, vllm_version: str | None) -> str:
        """서버가 보고한 vLLM 버전에 맞춰 구조화 디코딩 필드를 확정한다."""
        self.structured_output_style = structured_output_style_for(vllm_version)
        return self.structured_output_style

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def tokenize_endpoint(self) -> str:
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        return f"{root}/tokenize"

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def build_request(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None = None,
        stop: Sequence[str] | None = None,
        guided_json: dict | None = None,
        repetition_penalty: float | None = None,
    ) -> dict[str, Any]:
        """Assemble the JSON body; optional fields are omitted when unset."""
        if not messages:
            raise ValueError("at least one message is required")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        payload.update(self.reasoning_config)
        if seed is not None:
            payload["seed"] = seed
        if repetition_penalty is not None:
            # vLLM 확장 필드. OpenAI 스펙에는 없으므로 설정한 실행에서만 넣는다.
            payload["repetition_penalty"] = repetition_penalty
        if stop:
            payload["stop"] = list(stop)
        if guided_json is not None:
            # 구조화 디코딩 필드는 vLLM 버전에 따라 이름이 다르다.
            # vLLM 0.19.x 는 최상위 "guided_json" 을 **조용히 무시**하므로
            # (검증: {"color": str, additionalProperties:false} 스키마에 산문 응답)
            # OpenAI 호환 "response_format": {"type": "json_schema"} 를 사용해야 한다.
            # 구버전(0.8.x)은 "guided_json" 만 인식한다.
            # 어느 경로를 썼는지는 payload 에 그대로 남아 provenance 로 추적된다.
            if self.structured_output_style == STRUCTURED_RESPONSE_FORMAT:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "contract",
                        "schema": guided_json,
                        "strict": True,
                    },
                }
            else:
                payload["guided_json"] = guided_json
                if self.guided_decoding_backend is not None:
                    # vLLM 0.8.x는 백엔드 옵션을 서버 CLI가 아니라 요청에서 받는다.
                    payload["guided_decoding_backend"] = self.guided_decoding_backend
        return payload

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        seed: int | None = None,
        stop: Sequence[str] | None = None,
        guided_json: dict | None = None,
        repetition_penalty: float | None = None,
    ) -> ChatResponse:
        """Request one completion and return it with usage and latency."""
        payload = self.build_request(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            stop=stop,
            guided_json=guided_json,
            repetition_penalty=repetition_penalty,
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started = self._clock()
        raw = self._transport(self.endpoint, self.headers(), body, self.timeout)
        latency = max(self._clock() - started, 0.0)

        if raw.status != 200:
            raise TransportError(
                f"{self.endpoint} answered HTTP {raw.status}: {_preview(raw.body)}"
            )
        try:
            decoded = json.loads(raw.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransportError(f"response body was not JSON: {exc}") from exc
        if not isinstance(decoded, Mapping):
            raise TransportError("response body was not a JSON object")

        text, finish_reason = _first_choice(decoded)
        return ChatResponse(
            text=text,
            usage=Usage.from_payload(decoded.get("usage")),
            latency_s=latency,
            status=raw.status,
            request=payload,
            raw=decoded,
            finish_reason=finish_reason,
        )

    def count_chat_tokens(self, messages: Sequence[Mapping[str, str]]) -> int:
        """Return the server-tokenized chat length before reserving output tokens."""
        payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "add_generation_prompt": True,
        }
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        raw = self._transport(self.tokenize_endpoint, self.headers(), body, self.timeout)
        if raw.status != 200:
            raise TransportError(
                f"{self.tokenize_endpoint} answered HTTP {raw.status}: {_preview(raw.body)}"
            )
        try:
            decoded = json.loads(raw.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransportError(f"tokenize response body was not JSON: {exc}") from exc
        count = decoded.get("count") if isinstance(decoded, Mapping) else None
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise TransportError("tokenize response carried no non-negative integer count")
        return count


def _first_choice(decoded: Mapping[str, Any]) -> tuple[str, str | None]:
    choices = decoded.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TransportError("response carried no choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise TransportError("first choice was not an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise TransportError("first choice carried no message")
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise TransportError("message content was not text")
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = None
    return content, finish_reason


def _preview(body: bytes, limit: int = 200) -> str:
    text = body.decode("utf-8", errors="replace")
    return text if len(text) <= limit else text[:limit] + "..."
