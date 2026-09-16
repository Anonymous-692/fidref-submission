"""tau-bench retail domain sandbox."""

from .audit import ClauseAudit, audit_clause_density
from .config import DEFAULT_RETAIL_CONFIG, RetailConfig
from .contracts import Contract, ContractReport, Counterexample, DefectCategory, Symptom, evaluate_contract
from .dsl import DslError, contract_json_schema, parse_contract
from .enumeration import EnumerationResult, enumerate_reachable
from .env import RetailEnv, StepResult, action_space, apply_action, initial_state, valid_actions
from .ground_truth import (
    CLAUSE_NAMES,
    clause_values,
    exchange_items_postcondition,
    exchange_items_precondition,
)
from .runner import RetailExperimentRunner, compute_protocol_sha256
from .state import ActionKind, RetailAction, RetailState

__all__ = [
    "ActionKind",
    "CLAUSE_NAMES",
    "ClauseAudit",
    "Contract",
    "ContractReport",
    "Counterexample",
    "DEFAULT_RETAIL_CONFIG",
    "DefectCategory",
    "DslError",
    "EnumerationResult",
    "RetailAction",
    "RetailConfig",
    "RetailEnv",
    "RetailExperimentRunner",
    "RetailState",
    "StepResult",
    "Symptom",
    "action_space",
    "apply_action",
    "audit_clause_density",
    "clause_values",
    "compute_protocol_sha256",
    "contract_json_schema",
    "enumerate_reachable",
    "evaluate_contract",
    "exchange_items_postcondition",
    "exchange_items_precondition",
    "initial_state",
    "parse_contract",
    "valid_actions",
]
