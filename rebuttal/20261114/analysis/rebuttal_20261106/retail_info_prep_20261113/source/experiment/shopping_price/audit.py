#!/usr/bin/env python3
"""Clause-density audit for the price-aware Shopping applicability boundary."""

from __future__ import annotations

from dataclasses import dataclass

from .config import PriceShoppingConfig
from .ground_truth import CLAUSE_NAMES, clause_values
from .state import PriceShoppingState


@dataclass(frozen=True)
class ClauseAudit:
    clause: str
    distinguishing_states: int


def audit_clause_density(states: tuple[PriceShoppingState, ...], config: PriceShoppingConfig) -> tuple[ClauseAudit, ...]:
    counts = [0] * len(CLAUSE_NAMES)
    for state in states:
        values = clause_values(state, config)
        for index, value in enumerate(values):
            if not value and all(other for j, other in enumerate(values) if j != index):
                counts[index] += 1
    return tuple(ClauseAudit(name, count) for name, count in zip(CLAUSE_NAMES, counts))
