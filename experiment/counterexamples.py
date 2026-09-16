#!/usr/bin/env python3
"""Domain-neutral formatting and selection for observed counterexamples."""

from __future__ import annotations

from typing import Any, Hashable, Mapping, Sequence, TypeVar

S = TypeVar("S", bound=Hashable)
T = TypeVar("T")


def transition_diff(
    before: Mapping[str, Any],
    after: Mapping[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Return a deterministic field diff for one observed successful transition."""
    if after is None:
        return {}, []
    changed: dict[str, dict[str, Any]] = {}
    unchanged: list[str] = []
    for key in sorted(set(before) | set(after)):
        before_value = before.get(key)
        after_value = after.get(key)
        if before_value == after_value:
            unchanged.append(key)
        else:
            changed[key] = {"before": before_value, "after": after_value}
    return changed, unchanged


def round_robin_counterexamples(
    buckets: Mapping[S, Sequence[T]],
    symptom_order: Sequence[S],
    limit: int,
) -> tuple[T, ...]:
    """Select counterexamples round-robin so one symptom cannot starve another."""
    if limit <= 0:
        return ()
    selected: list[T] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for symptom in symptom_order:
            entries = buckets.get(symptom, ())
            if depth < len(entries):
                selected.append(entries[depth])
                added = True
                if len(selected) == limit:
                    return tuple(selected)
        if not added:
            return tuple(selected)
        depth += 1
    return tuple(selected)
