#!/usr/bin/env python3
"""Measure clause distinguishability over a complete reachable closure."""

from __future__ import annotations

from dataclasses import dataclass

from .config import RetailConfig
from .enumeration import EnumerationResult
from .ground_truth import CLAUSE_NAMES, clause_values


@dataclass(frozen=True)
class ClauseAudit:
    name: str
    witnesses: int
    density: float
    minimum_depth: int | None


def audit_clause_density(enumeration: EnumerationResult, config: RetailConfig) -> tuple[ClauseAudit, ...]:
    total = len(enumeration.states)
    rows = []
    for index, name in enumerate(CLAUSE_NAMES):
        witnesses = []
        for state in enumeration.states:
            values = clause_values(state, config)
            if not values[index] and all(value for j, value in enumerate(values) if j != index):
                witnesses.append(state)
        rows.append(
            ClauseAudit(
                name=name,
                witnesses=len(witnesses),
                density=(len(witnesses) / total if total else 0.0),
                minimum_depth=min((enumeration.depth_of(state) for state in witnesses), default=None),
            )
        )
    return tuple(rows)
