#!/usr/bin/env python3
"""Contract representation and evaluation for price-aware shopping domain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Sequence

from ..counterexamples import round_robin_counterexamples, transition_diff

from .config import PriceShoppingConfig
from .env import ActionKind, PriceShoppingAction, apply_action
from .state import PriceShoppingState

Precondition = Callable[[PriceShoppingState], bool]
Postcondition = Callable[[PriceShoppingState, PriceShoppingState], bool]

MAX_RECORDED_COUNTEREXAMPLES = 5


class Symptom(str, Enum):
    FALSE_ACCEPT = "false_accept"
    FALSE_REJECT = "false_reject"
    POSTCONDITION_VIOLATION = "postcondition_violation"

    def __str__(self) -> str:
        return self.value


class DefectCategory(str, Enum):
    NONE = "none"
    TOO_WEAK_PRECONDITION = "too_weak_precondition"
    TOO_STRONG_PRECONDITION = "too_strong_precondition"
    WRONG_POSTCONDITION = "wrong_postcondition"
    FAULTY_MERGE = "faulty_merge"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Contract:
    name: str
    action: PriceShoppingAction
    precondition: Precondition
    postcondition: Postcondition
    description: str = ""

    def holds_in(self, state: PriceShoppingState) -> bool:
        return bool(self.precondition(state))

    def transition_holds(self, before: PriceShoppingState, after: PriceShoppingState) -> bool:
        return bool(self.postcondition(before, after))


@dataclass(frozen=True)
class Counterexample:
    symptom: Symptom
    state: PriceShoppingState
    detail: str
    after_state: PriceShoppingState | None = None

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
    contract_name: str
    states_checked: int
    successes: int
    failures: int
    false_accepts: int
    false_rejects: int
    postcondition_violations: int
    counterexamples: tuple[Counterexample, ...]

    @property
    def exact(self) -> bool:
        return self.failures == 0 and self.is_sound and self.is_complete

    @property
    def is_sound(self) -> bool:
        return self.false_accepts == 0 and self.postcondition_violations == 0

    @property
    def is_complete(self) -> bool:
        return self.false_rejects == 0

    @property
    def is_exact(self) -> bool:
        return self.exact

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
    def primary_defect(self) -> DefectCategory:
        if self.false_accepts and self.false_rejects:
            return DefectCategory.FAULTY_MERGE
        if self.false_accepts:
            return DefectCategory.TOO_WEAK_PRECONDITION
        if self.false_rejects:
            return DefectCategory.TOO_STRONG_PRECONDITION
        if self.postcondition_violations:
            return DefectCategory.WRONG_POSTCONDITION
        return DefectCategory.NONE

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
            "exact": self.exact,
        }

    def to_dict(self) -> dict[str, object]:
        data = self.summary()
        data["primary_defect"] = self.primary_defect.value
        data["counterexamples"] = [cx.to_dict() for cx in self.counterexamples]
        return data


def evaluate_contract(
    contract: Contract,
    states: Iterable[PriceShoppingState],
    config: PriceShoppingConfig,
    max_counterexamples: int = MAX_RECORDED_COUNTEREXAMPLES,
) -> ContractReport:
    state_list = tuple(states)
    successes = 0
    failures = 0
    false_accepts = 0
    false_rejects = 0
    postcondition_violations = 0
    counterexample_buckets: dict[Symptom, list[Counterexample]] = {
        symptom: [] for symptom in Symptom
    }

    def add_cx(
        symptom: Symptom,
        state: PriceShoppingState,
        detail: str,
        after_state: PriceShoppingState | None = None,
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

    for state in state_list:
        claimed = contract.holds_in(state)
        step = apply_action(state, contract.action, config)
        if step.ok:
            successes += 1
            if claimed:
                if not contract.transition_holds(state, step.state):
                    postcondition_violations += 1
                    failures += 1
                    add_cx(
                        Symptom.POSTCONDITION_VIOLATION,
                        state,
                        "observed after-state breaks the promised postcondition",
                        step.state,
                    )
            else:
                false_rejects += 1
                failures += 1
                add_cx(
                    Symptom.FALSE_REJECT,
                    state,
                    "contract rejected the state but the sandbox successfully placed the order",
                    step.state,
                )
        else:
            if claimed:
                false_accepts += 1
                failures += 1
                add_cx(
                    Symptom.FALSE_ACCEPT,
                    state,
                    f"contract accepted the state but sandbox rejected with: {step.error}",
                )

    return ContractReport(
        contract_name=contract.name,
        states_checked=len(state_list),
        successes=successes,
        failures=failures,
        false_accepts=false_accepts,
        false_rejects=false_rejects,
        postcondition_violations=postcondition_violations,
        counterexamples=round_robin_counterexamples(
            counterexample_buckets, tuple(Symptom), max_counterexamples
        ),
    )
