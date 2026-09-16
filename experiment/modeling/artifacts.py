#!/usr/bin/env python3
"""The record written for one method run: everything needed to re-read it later.

An artifact is self-describing. It carries the method and model that produced
it, the decoding parameters and budgets it ran under, every prompt sent and every
raw response received, whether the answer parsed, what it cost in tokens and
seconds, and how the oracle evaluator scored the result. Nothing in the analysis
downstream needs to consult the runner's own configuration to interpret one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Compute the Wilson score interval for a binomial proportion.
    
    Returns (center, lower, upper).
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    z2 = z * z
    center = (k + z2 / 2) / (n + z2)
    width = z * ((k * (n - k) / n + z2 / 4) ** 0.5) / (n + z2)
    return (center, center - width, center + width)

from ..evaluation import ContractReport, Counterexample
from ..counterexamples import transition_diff

COUNTEREXAMPLE_FORMAT_VERSION = "structured_transition_diff_v3"

SCHEMA_VERSION = 2

# Contract parse outcomes.
STATUS_PARSED = "parsed"
STATUS_PARSE_FAILURE = "parse_failure"
STATUS_MISSING = "missing"

# Whole-run outcomes.
OUTCOME_EXACT = "exact"
OUTCOME_INEXACT = "inexact"
OUTCOME_PARSE_FAILURE = "parse_failure"
OUTCOME_NO_CONTRACT = "no_contract"


@dataclass(frozen=True)
class Interaction:
    """One model call, recorded whether it succeeded or not."""

    index: int
    role: str
    messages: tuple[Mapping[str, str], ...]
    request: Mapping[str, Any] | None = None
    response_text: str | None = None
    raw_response: Mapping[str, Any] | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    latency_s: float | None = None
    finish_reason: str | None = None
    error: str | None = None
    prompt_sha256: str = ""
    response_sha256: str = ""

    prompt_tokens: int | None = None
    requested_max_tokens: int | None = None
    effective_max_tokens: int | None = None
    clamp_reason: str | None = None
    context_window: int | None = None
    safety_margin: int | None = None

    def __post_init__(self) -> None:
        if not self.prompt_sha256 and self.messages:
            object.__setattr__(self, "prompt_sha256", sha256_json([dict(m) for m in self.messages]))
        if not self.response_sha256 and self.response_text is not None:
            object.__setattr__(self, "response_sha256", sha256_text(self.response_text))

    @property
    def ok(self) -> bool:
        return self.error is None and self.response_text is not None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "index": self.index,
            "role": self.role,
            "messages": [dict(message) for message in self.messages],
            "prompt_sha256": self.prompt_sha256,
            "response_sha256": self.response_sha256,
            "request": dict(self.request) if self.request is not None else None,
            "response_text": self.response_text,
            "raw_response": dict(self.raw_response) if self.raw_response is not None else None,
            "usage": dict(self.usage),
            "latency_s": self.latency_s,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }
        if self.prompt_tokens is not None:
            data["prompt_tokens"] = self.prompt_tokens
        if self.requested_max_tokens is not None:
            data["requested_max_tokens"] = self.requested_max_tokens
        if self.effective_max_tokens is not None:
            data["effective_max_tokens"] = self.effective_max_tokens
        if self.clamp_reason is not None:
            data["clamp_reason"] = self.clamp_reason
        if self.context_window is not None:
            data["context_window"] = self.context_window
        if self.safety_margin is not None:
            data["safety_margin"] = self.safety_margin
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Interaction":
        return cls(
            index=int(data["index"]),
            role=str(data["role"]),
            messages=tuple(dict(m) for m in data.get("messages", ())),
            request=dict(data["request"]) if data.get("request") is not None else None,
            response_text=data.get("response_text"),
            raw_response=dict(data["raw_response"]) if data.get("raw_response") is not None else None,
            usage=dict(data.get("usage", {})),
            latency_s=float(data["latency_s"]) if data.get("latency_s") is not None else None,
            finish_reason=data.get("finish_reason"),
            error=data.get("error"),
            prompt_sha256=str(data.get("prompt_sha256", "")),
            response_sha256=str(data.get("response_sha256", "")),
            prompt_tokens=int(data["prompt_tokens"]) if data.get("prompt_tokens") is not None else None,
            requested_max_tokens=int(data["requested_max_tokens"]) if data.get("requested_max_tokens") is not None else None,
            effective_max_tokens=int(data["effective_max_tokens"]) if data.get("effective_max_tokens") is not None else None,
            clamp_reason=str(data["clamp_reason"]) if data.get("clamp_reason") is not None else None,
            context_window=int(data["context_window"]) if data.get("context_window") is not None else None,
            safety_margin=int(data["safety_margin"]) if data.get("safety_margin") is not None else None,
        )



@dataclass(frozen=True)
class ContractRecord:
    """What became of one model answer: a contract, or a named parse failure."""

    status: str
    source: str | None = None
    spec: Mapping[str, Any] | None = None
    node_count: int | None = None
    error: str | None = None
    interaction_index: int | None = None

    @property
    def parsed(self) -> bool:
        return self.status == STATUS_PARSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "spec": self.spec,
            "node_count": self.node_count,
            "error": self.error,
            "interaction_index": self.interaction_index,
        }


    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContractRecord":
        return cls(
            status=str(data.get("status", STATUS_MISSING)),
            source=data.get("source"),
            spec=dict(data["spec"]) if data.get("spec") is not None else None,
            node_count=int(data["node_count"]) if data.get("node_count") is not None else None,
            error=data.get("error"),
            interaction_index=int(data["interaction_index"]) if data.get("interaction_index") is not None else None,
        )


def counterexample_record(counterexample: Counterexample) -> dict[str, Any]:
    """A counterexample in both readable and machine-checkable form."""
    after_state = getattr(counterexample, "after_state", None)
    before_fields = counterexample.state.to_dict()
    after_fields = after_state.to_dict() if after_state is not None else None
    changed_fields, unchanged_fields = transition_diff(before_fields, after_fields)
    return {
        "symptom": counterexample.symptom.value,
        "state": counterexample.state.describe(),
        "state_fields": before_fields,
        "before_state": before_fields,
        "after_state": after_fields,
        "action_succeeded": after_fields is not None,
        "changed_fields": changed_fields,
        "unchanged_fields": unchanged_fields,
        "detail": counterexample.detail,
    }


def report_metrics(report: ContractReport) -> dict[str, Any]:
    """The evaluator's scores, flattened for the artifact."""
    metrics = report.summary()
    metrics["sound"] = report.is_sound
    metrics["complete"] = report.is_complete
    return metrics


@dataclass(frozen=True)
class EvaluationRecord:
    """The oracle evaluator's verdict on the final contract."""

    metrics: Mapping[str, Any]
    counterexamples: tuple[Mapping[str, Any], ...]
    truncated: bool = False
    deepest_level: int = 0
    state_count: int = 0

    @property
    def exact(self) -> bool:
        return bool(self.metrics.get("exact"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "counterexamples": [dict(entry) for entry in self.counterexamples],
            "exact": self.exact,
            "truncated": self.truncated,
            "deepest_level": self.deepest_level,
            "state_count": self.state_count,
        }

    @classmethod
    def from_report(cls, report: ContractReport, enumeration_summary: dict[str, Any]) -> "EvaluationRecord":
        return cls(
            metrics=report_metrics(report),
            counterexamples=tuple(
                counterexample_record(entry) for entry in report.counterexamples
            ),
            truncated=bool(enumeration_summary.get("truncated", False)),
            deepest_level=int(enumeration_summary.get("deepest_level", 0)),
            state_count=int(enumeration_summary.get("states", 0)),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationRecord":
        return cls(
            metrics=dict(data.get("metrics", {})),
            counterexamples=tuple(dict(c) for c in data.get("counterexamples", ())),
            truncated=bool(data.get("truncated", False)),
            deepest_level=int(data.get("deepest_level", 0)),
            state_count=int(data.get("state_count", 0)),
        )


@dataclass(frozen=True)
class RunArtifact:
    """One (method, seed) run, complete enough to stand on its own."""

    method: str
    model: str
    endpoint: str
    seed: int
    decoding: Mapping[str, Any]
    budgets: Mapping[str, Any]
    sandbox: Mapping[str, Any]
    interactions: tuple[Interaction, ...]
    contract: ContractRecord
    usage: Mapping[str, int]
    latency: Mapping[str, Any]
    spend: Mapping[str, Any]
    outcome: str
    evaluation: EvaluationRecord | None = None
    rounds: tuple[Mapping[str, Any], ...] = ()
    stopped_because: str | None = None
    created_at: str = ""
    serving: Mapping[str, Any] = field(default_factory=dict)
    hashes: Mapping[str, str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def name(self) -> str:
        return f"{self.method}__seed{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run": self.name,
            "created_at": self.created_at,
            "method": self.method,
            "model": self.model,
            "endpoint": self.endpoint,
            "seed": self.seed,
            "decoding": dict(self.decoding),
            "budgets": dict(self.budgets),
            "sandbox": dict(self.sandbox),
            "serving": dict(self.serving),
            "hashes": dict(self.hashes),
            "interactions": [interaction.to_dict() for interaction in self.interactions],
            "rounds": [dict(entry) for entry in self.rounds],
            "contract": self.contract.to_dict(),
            "usage": dict(self.usage),
            "latency": dict(self.latency),
            "spend": dict(self.spend),
            "evaluation": self.evaluation.to_dict() if self.evaluation is not None else None,
            "outcome": self.outcome,
            "stopped_because": self.stopped_because,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunArtifact":
        evaluation = (
            EvaluationRecord.from_dict(data["evaluation"])
            if data.get("evaluation") is not None
            else None
        )
        return cls(
            method=str(data["method"]),
            model=str(data["model"]),
            endpoint=str(data["endpoint"]),
            seed=int(data["seed"]),
            decoding=dict(data.get("decoding", {})),
            budgets=dict(data.get("budgets", {})),
            sandbox=dict(data.get("sandbox", {})),
            interactions=tuple(
                Interaction.from_dict(entry) for entry in data.get("interactions", ())
            ),
            contract=ContractRecord.from_dict(data.get("contract", {"status": STATUS_MISSING})),
            usage=dict(data.get("usage", {})),
            latency=dict(data.get("latency", {})),
            spend=dict(data.get("spend", {})),
            outcome=str(data.get("outcome", OUTCOME_NO_CONTRACT)),
            evaluation=evaluation,
            rounds=tuple(dict(entry) for entry in data.get("rounds", ())),
            stopped_because=data.get("stopped_because"),
            created_at=str(data.get("created_at", "")),
            serving=dict(data.get("serving", {})),
            hashes=dict(data.get("hashes", {})),
            schema_version=int(data.get("schema_version", 1)),
        )

    def summary_row(self) -> dict[str, Any]:
        """The one-line view used by the aggregate summary and the console."""
        metrics = self.evaluation.metrics if self.evaluation is not None else {}
        return {
            "method": self.method,
            "seed": self.seed,
            "model": self.model,
            "outcome": self.outcome,
            "contract_status": self.contract.status,
            "exact": bool(metrics.get("exact", False)),
            "false_accepts": metrics.get("false_accepts"),
            "false_rejects": metrics.get("false_rejects"),
            "postcondition_violations": metrics.get("postcondition_violations"),
            "states_checked": metrics.get("states_checked"),
            "model_calls": self.usage.get("calls", 0),
            "total_tokens": self.usage.get("total_tokens", 0),
            "states_observed": self.spend.get("states_observed", 0),
            "oracle_feedback_queries": self.spend.get("oracle_feedback_queries", 0),
            "sampled_feedback_queries": self.spend.get("sampled_feedback_queries", 0),
            "sampled_states_checked": self.spend.get("sampled_states_checked", 0),
            "latency_s": self.latency.get("total_s", 0.0),
        }


def write_artifact(directory: str | Path, artifact: RunArtifact) -> Path:
    """Write one artifact under ``directory``, which is created if needed."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{artifact.name}.json"
    path.write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def read_artifact(source: str | Path | Mapping[str, Any]) -> RunArtifact:
    """Read one artifact from a path or dict, supporting both v1 and v2 schemas."""
    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        payload = dict(source)
    return RunArtifact.from_dict(payload)


load_artifact = read_artifact


def write_summary(
    directory: str | Path, artifacts: Sequence[RunArtifact], context: Mapping[str, Any]
) -> Path:
    """Write the aggregate index beside the individual artifacts."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    
    seen_seeds: set[tuple[str, int]] = set()
    for artifact in artifacts:
        key = (artifact.method, artifact.seed)
        if key in seen_seeds:
            raise ValueError(f"Duplicate run detected in summary: {key}")
        seen_seeds.add(key)

    path = target / "summary.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "context": dict(context),
        "runs": [artifact.summary_row() for artifact in artifacts],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path



__all__ = [
    "ContractRecord",
    "EvaluationRecord",
    "Interaction",
    "OUTCOME_EXACT",
    "OUTCOME_INEXACT",
    "OUTCOME_NO_CONTRACT",
    "OUTCOME_PARSE_FAILURE",
    "RunArtifact",
    "SCHEMA_VERSION",
    "STATUS_MISSING",
    "STATUS_PARSED",
    "STATUS_PARSE_FAILURE",
    "counterexample_record",
    "load_artifact",
    "read_artifact",
    "report_metrics",
    "write_artifact",
    "write_summary",
]
