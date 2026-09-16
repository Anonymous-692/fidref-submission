#!/usr/bin/env python3
"""Agent-facing shopping sandbox: state, actions, transitions, enumeration.

Nothing in this package knows what ``place_order`` is *supposed* to do. The
declarative specification used to score a candidate lives in
``experiment.evaluation`` and must never be imported from here.
"""

from __future__ import annotations

from .config import DEFAULT_CONFIG, EnvConfig
from .enumeration import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_STATES,
    EnumerationResult,
    Transition,
    enumerate_reachable,
)
from .env import (
    EnvSnapshot,
    ShoppingEnv,
    StepResult,
    action_space,
    apply_action,
    initial_state,
    valid_actions,
)
from .state import (
    ActionKind,
    ItemCounts,
    OrderStatus,
    ShoppingAction,
    ShoppingState,
    counts_combine,
    counts_covers,
    counts_from_mapping,
    counts_get,
    counts_to_mapping,
    counts_total,
    sort_states,
)

__all__ = [
    "ActionKind",
    "DEFAULT_CONFIG",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_STATES",
    "EnumerationResult",
    "EnvConfig",
    "EnvSnapshot",
    "ItemCounts",
    "OrderStatus",
    "ShoppingAction",
    "ShoppingEnv",
    "ShoppingState",
    "StepResult",
    "Transition",
    "action_space",
    "apply_action",
    "counts_combine",
    "counts_covers",
    "counts_from_mapping",
    "counts_get",
    "counts_to_mapping",
    "counts_total",
    "enumerate_reachable",
    "initial_state",
    "sort_states",
    "valid_actions",
]
