#!/usr/bin/env python3
"""Bounded breadth-first enumeration of reachable sandbox states."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import DEFAULT_CONFIG, EnvConfig
from .env import action_space, apply_action, initial_state
from .state import ShoppingAction, ShoppingState, sort_states

DEFAULT_MAX_DEPTH = 20
DEFAULT_MAX_STATES = 6000


@dataclass(frozen=True)
class Transition:
    """One successful edge of the reachability graph."""

    source: ShoppingState
    action: ShoppingAction
    target: ShoppingState


@dataclass(frozen=True)
class EnumerationResult:
    """States and edges found by a bounded search, in discovery order."""

    states: tuple[ShoppingState, ...]
    transitions: tuple[Transition, ...]
    depths: tuple[tuple[ShoppingState, int], ...]
    truncated: bool
    max_depth: int
    max_states: int
    deepest_level: int = field(default=0)

    def __len__(self) -> int:
        return len(self.states)

    @property
    def state_set(self) -> frozenset[ShoppingState]:
        return frozenset(self.states)

    @property
    def sorted_states(self) -> tuple[ShoppingState, ...]:
        """Canonically ordered states, for comparing two enumerations."""
        return sort_states(self.states)

    def depth_of(self, state: ShoppingState) -> int:
        for candidate, depth in self.depths:
            if candidate == state:
                return depth
        raise KeyError("state was not reached by this enumeration")

    def summary(self) -> dict[str, object]:
        return {
            "states": len(self.states),
            "transitions": len(self.transitions),
            "truncated": self.truncated,
            "max_depth": self.max_depth,
            "max_states": self.max_states,
            "deepest_level": self.deepest_level,
        }


def enumerate_reachable(
    config: EnvConfig = DEFAULT_CONFIG,
    *,
    start: ShoppingState | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_states: int = DEFAULT_MAX_STATES,
) -> EnumerationResult:
    """Breadth-first search over successful transitions from ``start``.

    The frontier is expanded in the fixed action order of ``action_space``, so
    the returned sequence, including the point at which a bound truncates the
    search, is identical on every run for the same arguments. Visited states are
    deduplicated by value, so the result never repeats a state.
    """
    if max_depth < 0:
        raise ValueError("max_depth must not be negative")
    if max_states < 1:
        raise ValueError("max_states must be at least 1")

    origin = start if start is not None else initial_state(config)
    actions = action_space(config)

    discovered: list[ShoppingState] = [origin]
    depths: dict[ShoppingState, int] = {origin: 0}
    transitions: list[Transition] = []
    queue: deque[tuple[ShoppingState, int]] = deque([(origin, 0)])
    truncated = False
    deepest_level = 0

    while queue:
        state, depth = queue.popleft()
        at_boundary = depth >= max_depth
        for action in actions:
            result = apply_action(state, action, config)
            if not result.ok or result.state == state:
                continue
            target = result.state
            if target in depths:
                if not at_boundary:
                    transitions.append(Transition(source=state, action=action, target=target))
                continue
            # A genuinely new state that a bound forbids us from keeping means
            # the returned set is a strict under-approximation.
            if at_boundary or len(discovered) >= max_states:
                truncated = True
                continue
            transitions.append(Transition(source=state, action=action, target=target))
            depths[target] = depth + 1
            deepest_level = max(deepest_level, depth + 1)
            discovered.append(target)
            queue.append((target, depth + 1))

    return EnumerationResult(
        states=tuple(discovered),
        transitions=tuple(transitions),
        depths=tuple((state, depths[state]) for state in discovered),
        truncated=truncated,
        max_depth=max_depth,
        max_states=max_states,
        deepest_level=deepest_level,
    )
