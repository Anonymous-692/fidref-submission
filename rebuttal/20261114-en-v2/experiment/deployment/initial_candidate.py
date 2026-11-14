"""Condition-scoped initial responses; no scoring or hidden reference candidates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..modeling.artifacts import Interaction, sha256_json
from ..modeling.client import Usage


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "deployment/runner.py", "deployment/model_runner.py", "deployment/initial_candidate.py",
        "deployment/checkpoint.py",
        "deployment/information_policy.py", "deployment/prompts.py", "deployment/dsl.py",
        "deployment/config.py", "deployment/state.py", "deployment/enumeration.py",
        "deployment/env.py", "deployment/contracts.py", "deployment/ground_truth.py",
        "modeling/client.py", "modeling/artifacts.py", "modeling/runner.py", "counterexamples.py",
    )
    return {path: hashlib.sha256((root / "experiment" / path).read_bytes()).hexdigest() for path in paths}


@dataclass(frozen=True)
class InitialCandidate:
    # Immutable serialization prevents one parallel arm from changing another's response.
    payload_json: str

    @classmethod
    def create(cls, context: Mapping[str, Any], interaction: Interaction | None,
               *, calls: int, usage: Usage, stopped_because: str | None) -> "InitialCandidate":
        payload = {
            "version": "deployment_initial_response_v1", "context": dict(context),
            "context_sha256": sha256_json(context),
            "interaction": interaction.to_dict() if interaction is not None else None,
            "calls": calls, "usage": usage.to_dict(), "stopped_because": stopped_because,
        }
        payload["sha256"] = sha256_json(payload)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InitialCandidate":
        candidate = cls(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False))
        candidate.validate(payload["context"])
        return candidate

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.payload_json)

    def validate(self, context: Mapping[str, Any]) -> None:
        payload = self.to_dict()
        digest = payload.pop("sha256")
        if digest != sha256_json(payload):
            raise ValueError("initial candidate content hash mismatch")
        if payload["version"] != "deployment_initial_response_v1":
            raise ValueError("unsupported initial candidate version")
        if payload["context_sha256"] != sha256_json(payload["context"]):
            raise ValueError("initial candidate context hash mismatch")
        if payload["context_sha256"] != sha256_json(context):
            raise ValueError("initial candidate belongs to a different information condition or run configuration")
        if type(payload["calls"]) is not int or payload["calls"] not in (0, 1):
            raise ValueError("initial candidate must contain at most one generation call")
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if type(payload["usage"].get(key)) is not int or payload["usage"][key] < 0:
                raise ValueError("initial candidate usage must be nonnegative integer counts")
        if payload["interaction"] is not None:
            interaction = Interaction.from_dict(payload["interaction"])
            if interaction.index != 0 or interaction.role != "propose":
                raise ValueError("initial response must be the first proposal")
            if any(payload["usage"][key] != interaction.usage.get(key, 0)
                   for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
                raise ValueError("initial candidate usage does not match its interaction")
            if payload["usage"].get("reasoning_tokens") != interaction.usage.get("reasoning_tokens"):
                raise ValueError("initial candidate reasoning usage does not match its interaction")
            if interaction.error is None and (payload["calls"] != 1 or interaction.response_text is None):
                raise ValueError("successful initial response must carry one generation call")
            if interaction.error is not None and (self.usage.total_tokens != 0 or payload["stopped_because"] is None):
                raise ValueError("failed initial request has inconsistent accounting")
        elif payload["calls"] != 0 or self.usage.total_tokens != 0 or payload["stopped_because"] is None:
            raise ValueError("missing initial interaction has inconsistent accounting")

    @property
    def context_sha256(self) -> str:
        return self.to_dict()["context_sha256"]

    @property
    def sha256(self) -> str:
        return self.to_dict()["sha256"]

    @property
    def interaction(self) -> Interaction | None:
        value = self.to_dict()["interaction"]
        return Interaction.from_dict(value) if value is not None else None

    @property
    def calls(self) -> int:
        return self.to_dict()["calls"]

    @property
    def usage(self) -> Usage:
        value = self.to_dict()["usage"]
        return Usage(**{key: value[key] for key in (
            "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"
        ) if key in value})

    @property
    def stopped_because(self) -> str | None:
        return self.to_dict()["stopped_because"]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "InitialCandidate":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
