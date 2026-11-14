#!/usr/bin/env python3
"""Model-driven contract proposal: a safe DSL, a local client, four methods.

This package is evaluator-side. It drives the sandbox through the agent-facing
API, asks a locally served model for candidate ``place_order`` contracts in a
declarative JSON DSL, and scores whatever comes back with the existing oracle
evaluator. The environment package is untouched and never imports anything here.
"""

from __future__ import annotations

from .artifacts import (
    SCHEMA_VERSION,
    ContractRecord,
    EvaluationRecord,
    Interaction,
    RunArtifact,
    write_artifact,
    write_summary,
)
from .client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    ChatClient,
    ChatResponse,
    RawResponse,
    TransportError,
    Usage,
    is_loopback,
    urllib_transport,
)
from .dsl import (
    DslError,
    ParsedContract,
    extract_json_object,
    parse_contract,
    parse_contract_text,
)
from .runner import (
    ACTIVE_CEGIS,
    ACTIVE_CEGIS_NO_BALANCE,
    ACTIVE_CEGIS_NO_COVERAGE,
    ACTIVE_CEGIS_UNIFORM,
    ASI_REPLAY,
    CONTRACTSKILL_REPAIR,
    CONTRACTSKILL_FULL_ORACLE,
    COUNTEREXAMPLE_GUIDED,
    DIRECT,
    METHODS,
    RANDOM_PROBE,
    SAMPLED_CEGIS,
    SELF_REFINE,
    SKILLCOMMIT_REPLAY,
    Budgets,
    Decoding,
    ExperimentRunner,
    RunSpec,
)

__all__ = [
    "ACTIVE_CEGIS",
    "ACTIVE_CEGIS_NO_BALANCE",
    "ACTIVE_CEGIS_NO_COVERAGE",
    "ACTIVE_CEGIS_UNIFORM",
    "ASI_REPLAY",
    "CONTRACTSKILL_REPAIR",
    "CONTRACTSKILL_FULL_ORACLE",
    "COUNTEREXAMPLE_GUIDED",
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "DIRECT",
    "METHODS",
    "RANDOM_PROBE",
    "SAMPLED_CEGIS",
    "SCHEMA_VERSION",
    "SELF_REFINE",
    "SKILLCOMMIT_REPLAY",
    "Budgets",
    "ChatClient",
    "ChatResponse",
    "ContractRecord",
    "Decoding",
    "DslError",
    "EvaluationRecord",
    "ExperimentRunner",
    "Interaction",
    "ParsedContract",
    "RawResponse",
    "RunArtifact",
    "RunSpec",
    "TransportError",
    "Usage",
    "extract_json_object",
    "is_loopback",
    "parse_contract",
    "parse_contract_text",
    "urllib_transport",
    "write_artifact",
    "write_summary",
]
