#!/usr/bin/env python3
"""Build fixed evidence that is known to expose at least one observed error.

Why this exists. A uniform pool of 48 states misses every error for these contracts about
half the time (measured: the error region is 160/11,712 = 1.37% of the closure, so a clean
pool has probability ~52%). When that happens the repair loop never runs, and a comparison
between free rewrite and restricted patch measures nothing. That is a real property of the
uniform fixed pool and is reported as such; this module supplies a *different* evidence
condition in which repair is actually exercised.

What is and is not used. A candidate "error" here is a disagreement between what the contract
claims and what the sandbox does when the action is executed: the contract accepts a state the
sandbox rejects, rejects one the sandbox accepts, or accepts a transition whose observed effect
violates its own postcondition. All of that comes from executing the action. The reference
contract is never consulted, and no particular counterexample is chosen for being helpful --
the only criterion is whether a pool exposes any error at all.

Selection bias, stated plainly. Pools are drawn from the same uniform distribution and are
accepted or rejected as whole pools. The result is uniform *conditioned on being informative*
for the initial candidate. That is a deliberate deviation from the uniform fixed pool and the
condition must not be reported as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class WitnessSearch:
    """The evidence set plus what it cost and how it was obtained."""

    states: tuple[Any, ...]
    found: bool
    pools_drawn: int
    states_executed: int
    observed_errors: int
    criterion: str = "pool exposes >=1 observed error for the shared initial candidate"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "witness_found": self.found,
            "pools_drawn": self.pools_drawn,
            "search_states_executed": self.states_executed,
            "observed_errors_in_evidence": self.observed_errors,
            "selection_criterion": self.criterion,
            "evidence_size": len(self.states),
            "notes": list(self.notes),
        }


def find_witness_pool(
    *,
    draw_pool: Callable[[int, int], Sequence[Any]],
    score: Callable[[Sequence[Any]], int],
    seed: int,
    size: int,
    max_pools: int = 12,
    seed_stride: int = 10_000,
) -> WitnessSearch:
    """Draw whole pools until one exposes an observed error, or give up.

    ``draw_pool(sub_seed, size)`` must be candidate-independent. ``score(states)`` returns the
    observed error count of the shared initial candidate on those states; it must be computed
    from execution, not from the reference.

    Giving up is a recorded outcome, not a failure: the run proceeds on the last pool drawn and
    the artifact says no witness was found.
    """
    executed = 0
    last: Sequence[Any] = ()
    for attempt in range(max_pools):
        pool = tuple(draw_pool(seed + attempt * seed_stride, size))
        executed += len(pool)
        last = pool
        errors = score(pool)
        if errors > 0:
            return WitnessSearch(states=pool, found=True, pools_drawn=attempt + 1,
                                 states_executed=executed, observed_errors=errors)
    return WitnessSearch(
        states=tuple(last), found=False, pools_drawn=max_pools, states_executed=executed,
        observed_errors=0,
        notes=[f"no pool of {size} states exposed an observed error in {max_pools} draws; "
               "the run proceeds on the last pool and enters no repair round"],
    )
