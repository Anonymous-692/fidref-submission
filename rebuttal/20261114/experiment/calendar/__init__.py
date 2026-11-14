"""Controlled workspace scheduling sandbox."""

from .audit import ClauseAudit, audit_clause_density
from .config import DEFAULT_CALENDAR_CONFIG, CalendarConfig
from .contracts import Contract, ContractReport, Counterexample, DefectCategory, Symptom, evaluate_contract
from .dsl import DslError, contract_json_schema, parse_contract
from .enumeration import EnumerationResult, enumerate_reachable
from .env import CalendarEnv, StepResult, action_space, apply_action, initial_state, valid_actions
from .ground_truth import (
    CLAUSE_NAMES,
    clause_values,
    schedule_meeting_postcondition,
    schedule_meeting_precondition,
)
from .runner import CalendarExperimentRunner, compute_protocol_sha256
from .state import ActionKind, CalendarAction, CalendarState, ExternalBooking, MeetingSnapshot

__all__ = [
    "ActionKind",
    "CLAUSE_NAMES",
    "CalendarAction",
    "CalendarConfig",
    "CalendarEnv",
    "CalendarExperimentRunner",
    "CalendarState",
    "ClauseAudit",
    "Contract",
    "ContractReport",
    "Counterexample",
    "DEFAULT_CALENDAR_CONFIG",
    "DefectCategory",
    "DslError",
    "EnumerationResult",
    "ExternalBooking",
    "MeetingSnapshot",
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
    "parse_contract",
    "schedule_meeting_postcondition",
    "schedule_meeting_precondition",
    "valid_actions",
]
