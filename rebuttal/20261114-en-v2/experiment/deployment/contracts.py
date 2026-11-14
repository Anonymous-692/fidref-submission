#!/usr/bin/env python3
"""Contract representation and counterexample-finding scorer for deployment domain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Sequence

from ..counterexamples import round_robin_counterexamples, transition_diff

from .config import DeploymentConfig
from .env import apply_action
from .state import DeploymentAction, DeploymentState

Precondition = Callable[[DeploymentState], bool]
Postcondition = Callable[[DeploymentState, DeploymentState], bool]

MAX_RECORDED_COUNTEREXAMPLES = 5


class Symptom(str, Enum):
    """What a contract got wrong, as observed against the deployment sandbox."""

    FALSE_ACCEPT = "false_accept"
    FALSE_REJECT = "false_reject"
    POSTCONDITION_VIOLATION = "postcondition_violation"

    def __str__(self) -> str:
        return self.value


class DefectCategory(str, Enum):
    """Why a contract is wrong; the provenance behind the symptoms."""

    NONE = "none"
    TOO_WEAK_PRECONDITION = "too_weak_precondition"
    TOO_STRONG_PRECONDITION = "too_strong_precondition"
    WRONG_POSTCONDITION = "wrong_postcondition"
    FAULTY_MERGE = "faulty_merge"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Contract:
    """A candidate specification for one skill in the deployment domain."""

    name: str
    action: DeploymentAction
    precondition: Precondition
    postcondition: Postcondition
    description: str = ""

    def holds_in(self, state: DeploymentState) -> bool:
        return bool(self.precondition(state))

    def transition_holds(self, before: DeploymentState, after: DeploymentState) -> bool:
        return bool(self.postcondition(before, after))


@dataclass(frozen=True)
class Counterexample:
    """A concrete state on which the contract disagrees with the sandbox."""

    symptom: Symptom
    state: DeploymentState
    detail: str
    after_state: DeploymentState | None = None

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
    """Scores for one deployment contract over one set of states."""

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
        return not self.false_accepts and not self.postcondition_violations

    @property
    def is_complete(self) -> bool:
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
    states: Iterable[DeploymentState],
    config: DeploymentConfig,
    *,
    max_counterexamples: int = MAX_RECORDED_COUNTEREXAMPLES,
) -> ContractReport:
    """Score ``contract`` against the deployment sandbox on every supplied state."""
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
        state: DeploymentState,
        detail: str,
        after_state: DeploymentState | None = None,
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
    contracts: Sequence[Contract], states: Iterable[DeploymentState], config: DeploymentConfig
) -> tuple[ContractReport, ...]:
    states = tuple(states)
    return tuple(evaluate_contract(contract, states, config) for contract in contracts)
