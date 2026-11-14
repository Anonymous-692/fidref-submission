#!/usr/bin/env python3
"""Deterministic bounded BFS for tau-bench retail states."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .config import DEFAULT_RETAIL_CONFIG, RetailConfig
from .env import action_space, apply_action, initial_state
from .state import RetailAction, RetailState


@dataclass(frozen=True)
class Transition:
    source: RetailState
    action: RetailAction
    target: RetailState


@dataclass(frozen=True)
class EnumerationResult:
    states: tuple[RetailState, ...]
    transitions: tuple[Transition, ...]
    depths: tuple[tuple[RetailState, int], ...]
    truncated: bool
    deepest_level: int

    def depth_of(self, state: RetailState) -> int:
        return dict(self.depths)[state]


def enumerate_reachable(
    config: RetailConfig = DEFAULT_RETAIL_CONFIG,
    *,
    max_depth: int = 20,
    max_states: int = 15000,
) -> EnumerationResult:
    if max_depth < 0 or max_states < 1:
        raise ValueError("invalid enumeration bounds")
    origin = initial_state(config)
    depths: dict[RetailState, int] = {origin: 0}
    states: list[RetailState] = [origin]
    transitions: list[Transition] = []
    queue = deque([origin])
    truncated = False

    while queue:
        state = queue.popleft()
        depth = depths[state]
        for action in action_space(config):
            result = apply_action(state, action, config)
            if not result.ok or result.state == state:
                continue
            target = result.state
            if target in depths:
                if depth < max_depth:
                    transitions.append(Transition(state, action, target))
                continue
            if depth >= max_depth or len(states) >= max_states:
                truncated = True
                continue
            depths[target] = depth + 1
            states.append(target)
            transitions.append(Transition(state, action, target))
            queue.append(target)

    return EnumerationResult(
        tuple(states),
        tuple(transitions),
        tuple((s, depths[s]) for s in states),
        truncated,
        max(depths.values()) if depths else 0,
    )
