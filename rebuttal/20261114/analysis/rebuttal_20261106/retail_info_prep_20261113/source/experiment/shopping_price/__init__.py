"""Price-aware Shopping domain and experiment runner."""

from .audit import ClauseAudit, audit_clause_density
from .config import DEFAULT_PRICE_SHOPPING_CONFIG, PriceShoppingConfig
from .contracts import Contract, ContractReport, Counterexample, DefectCategory, Symptom, evaluate_contract
from .dsl import DslError, contract_json_schema, parse_contract_text
from .enumeration import EnumerationResult, enumerate_reachable
from .env import PriceShoppingEnv, StepResult, action_space, apply_action, initial_state, valid_actions
from .ground_truth import CLAUSE_NAMES, clause_values, place_order_postcondition, place_order_precondition
from .runner import PriceShoppingExperimentRunner, compute_protocol_sha256
from .state import ActionKind, OrderStatus, PriceShoppingAction, PriceShoppingState

__all__ = [
    "ActionKind",
    "CLAUSE_NAMES",
    "ClauseAudit",
    "Contract",
    "ContractReport",
    "Counterexample",
    "DEFAULT_PRICE_SHOPPING_CONFIG",
    "DefectCategory",
    "DslError",
    "EnumerationResult",
    "OrderStatus",
    "PriceShoppingAction",
    "PriceShoppingConfig",
    "PriceShoppingEnv",
    "PriceShoppingExperimentRunner",
    "PriceShoppingState",
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
    "initial_state",
    "parse_contract_text",
    "place_order_postcondition",
    "place_order_precondition",
    "valid_actions",
]
