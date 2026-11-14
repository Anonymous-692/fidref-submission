#!/usr/bin/env python3
"""Contract representation and the counterexample-finding scorer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Sequence

from ..counterexamples import round_robin_counterexamples, transition_diff

from ..environment import EnvConfig, ShoppingAction, ShoppingState, apply_action

Precondition = Callable[[ShoppingState], bool]
Postcondition = Callable[[ShoppingState, ShoppingState], bool]

MAX_RECORDED_COUNTEREXAMPLES = 5


class Symptom(str, Enum):
    """What a contract got wrong, as observed against the sandbox."""

    FALSE_ACCEPT = "false_accept"
    FALSE_REJECT = "false_reject"
    POSTCONDITION_VIOLATION = "postcondition_violation"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class DefectCategory(str, Enum):
    """Why a contract is wrong; the provenance behind the symptoms."""

    NONE = "none"
    TOO_WEAK_PRECONDITION = "too_weak_precondition"
    TOO_STRONG_PRECONDITION = "too_strong_precondition"
    WRONG_POSTCONDITION = "wrong_postcondition"
    FAULTY_MERGE = "faulty_merge"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class Contract:
    """A candidate specification for one skill.

    ``precondition`` claims where the skill is applicable; ``postcondition``
    claims how the world changes when it is applied there.
    """

    name: str
    action: ShoppingAction
    precondition: Precondition
    postcondition: Postcondition
    description: str = ""

    def holds_in(self, state: ShoppingState) -> bool:
        return bool(self.precondition(state))

    def transition_holds(self, before: ShoppingState, after: ShoppingState) -> bool:
        return bool(self.postcondition(before, after))


@dataclass(frozen=True)
class Counterexample:
    """A concrete state on which the contract disagrees with the sandbox."""

    symptom: Symptom
    state: ShoppingState
    detail: str
    after_state: ShoppingState | None = None

    def to_dict(self) -> dict[str, object]:
        before_state = self.state.to_dict()
        after_state = self.after_state.to_dict() if self.after_state is not None else None
        changed_fields, unchanged_fields = transition_diff(before_state, after_state)
        return {
            "symptom": self.symptom.value,
            "state": before_state,
            "before_state": before_state,
            "after_state": after_state,
            "action_succeeded": after_state is not None,
            "changed_fields": changed_fields,
            "unchanged_fields": unchanged_fields,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ContractReport:
    """Scores for one contract over one set of states.

    ``successes`` and ``failures`` count what the *sandbox* did, over every
    state checked. ``postcondition_violations`` counts only the successes the
    contract itself claimed, so it is always at most ``successes``.
    """

    contract_name: str
    states_checked: int
    successes: int
    failures: int
    false_accepts: int
    false_rejects: int
    postcondition_violations: int
    counterexamples: tuple[Counterexample, ...]

    @property
    def symptoms(self) -> frozenset[Symptom]:
        observed = set()
        if self.false_accepts:
            observed.add(Symptom.FALSE_ACCEPT)
        if self.false_rejects:
            observed.add(Symptom.FALSE_REJECT)
        if self.postcondition_violations:
            observed.add(Symptom.POSTCONDITION_VIOLATION)
        return frozenset(observed)

    @property
    def is_sound(self) -> bool:
        """No promised-but-failing applications and no broken effect claims."""
        return not self.false_accepts and not self.postcondition_violations

    @property
    def is_complete(self) -> bool:
        """No refused applications that the sandbox would have accepted."""
        return not self.false_rejects

    @property
    def is_exact(self) -> bool:
        return self.is_sound and self.is_complete

    def summary(self) -> dict[str, object]:
        return {
            "contract": self.contract_name,
            "states_checked": self.states_checked,
            "successes": self.successes,
            "failures": self.failures,
            "false_accepts": self.false_accepts,
            "false_rejects": self.false_rejects,
            "postcondition_violations": self.postcondition_violations,
            "symptoms": sorted(symptom.value for symptom in self.symptoms),
            "exact": self.is_exact,
        }


def evaluate_contract(
    contract: Contract,
    states: Iterable[ShoppingState],
    config: EnvConfig,
    *,
    max_counterexamples: int = MAX_RECORDED_COUNTEREXAMPLES,
) -> ContractReport:
    """Score ``contract`` against the sandbox on every supplied state.

    For each state the contract's precondition is compared with what the
    sandbox actually does, which yields the false accepts and false rejects.
    The postcondition is a claim about applications the contract vouches for,
    so it is checked only where the precondition admits the state *and* the
    sandbox accepts the action. Under-claiming therefore shows up purely as a
    false reject, never as a broken effect claim.
    """
    states = tuple(states)
    successes = 0
    failures = 0
    false_accepts = 0
    false_rejects = 0
    violations = 0
    counterexample_buckets: dict[Symptom, list[Counterexample]] = {
        symptom: [] for symptom in Symptom
    }

    def record(
        symptom: Symptom,
        state: ShoppingState,
        detail: str,
        after_state: ShoppingState | None = None,
    ) -> None:
        bucket = counterexample_buckets[symptom]
        if len(bucket) < max_counterexamples:
            bucket.append(
                Counterexample(
                    symptom=symptom,
                    state=state,
                    detail=detail,
                    after_state=after_state,
                )
            )

    for state in states:
        claimed = contract.holds_in(state)
        outcome = apply_action(state, contract.action, config)
        if outcome.ok:
            successes += 1
        else:
            failures += 1

        if claimed and not outcome.ok:
            false_accepts += 1
            record(
                Symptom.FALSE_ACCEPT,
                state,
                f"contract admits the state but the sandbox refused: {outcome.error}",
            )
        elif not claimed and outcome.ok:
            false_rejects += 1
            record(
                Symptom.FALSE_REJECT,
                state,
                "contract rejects the state but the sandbox applied the action",
                outcome.state,
            )

        # An effect claim only speaks about states the contract says it covers.
        # Checking it where the precondition already refused the state would
        # charge a postcondition violation for what is really a false reject,
        # and would make any narrow-but-sound contract look unsound.
        if claimed and outcome.ok and not contract.transition_holds(state, outcome.state):
            violations += 1
            record(
                Symptom.POSTCONDITION_VIOLATION,
                state,
                "observed after-state breaks the promised postcondition",
                outcome.state,
            )

    return ContractReport(
        contract_name=contract.name,
        states_checked=len(states),
        successes=successes,
        failures=failures,
        false_accepts=false_accepts,
        false_rejects=false_rejects,
        postcondition_violations=violations,
        counterexamples=round_robin_counterexamples(
            counterexample_buckets, tuple(Symptom), max_counterexamples
        ),
    )


def evaluate_all(
    contracts: Sequence[Contract], states: Iterable[ShoppingState], config: EnvConfig
) -> tuple[ContractReport, ...]:
    """Score several contracts over one shared state set."""
    states = tuple(states)
    return tuple(evaluate_contract(contract, states, config) for contract in contracts)
