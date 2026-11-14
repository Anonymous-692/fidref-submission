#!/usr/bin/env python3
"""Evaluator-side machinery: hidden ground truth, scoring, defect fixtures.

Import direction is one-way. This package depends on ``experiment.environment``
to execute actions; the environment never depends on anything here.
"""

from __future__ import annotations

from .contracts import (
    Contract,
    ContractReport,
    Counterexample,
    DefectCategory,
    Symptom,
    evaluate_all,
    evaluate_contract,
)
from .fixtures import (
    DefectFixture,
    all_fixtures,
    branch_contracts,
    faulty_merge,
    payment_branch_contract,
)
from .ground_truth import (
    GROUND_TRUTH_CLAUSES,
    VISIBILITY,
    ground_truth_contract,
    place_order_postcondition,
    place_order_precondition,
)

__all__ = [
    "Contract",
    "ContractReport",
    "Counterexample",
    "DefectCategory",
    "DefectFixture",
    "GROUND_TRUTH_CLAUSES",
    "Symptom",
    "VISIBILITY",
    "all_fixtures",
    "branch_contracts",
    "evaluate_all",
    "evaluate_contract",
    "faulty_merge",
    "ground_truth_contract",
    "payment_branch_contract",
    "place_order_postcondition",
    "place_order_precondition",
]
