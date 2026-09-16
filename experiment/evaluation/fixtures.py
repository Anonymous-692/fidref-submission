#!/usr/bin/env python3
"""Deliberately defective ``place_order`` contracts, one per defect category.

Each fixture is a contract that is wrong in a single, named way, together with
the symptoms a correct scorer must observe for it. They are the regression
targets for any future counterexample search: a search that cannot separate
these four from the ground truth is not measuring anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..environment import (
    EnvConfig,
    OrderStatus,
    ShoppingAction,
    ShoppingState,
    counts_covers,
)
from .contracts import Contract, DefectCategory, Symptom
from .ground_truth import place_order_postcondition, place_order_precondition


@dataclass(frozen=True)
class DefectFixture:
    """A defective contract plus the symptoms it must exhibit."""

    name: str
    category: DefectCategory
    contract: Contract
    expected_symptoms: frozenset[Symptom]
    rationale: str

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "category": self.category.value,
            "expected_symptoms": sorted(symptom.value for symptom in self.expected_symptoms),
            "rationale": self.rationale,
        }


def _require_two_payment_methods(config: EnvConfig) -> None:
    if len(config.valid_payment_methods) < 2:
        raise ValueError(
            "these fixtures need at least two valid payment methods to be falsifiable"
        )


def too_weak_precondition(config: EnvConfig) -> DefectFixture:
    """Forgets that an order needs a serviceable shipping address."""

    def precondition(state: ShoppingState) -> bool:
        return (
            state.logged_in
            and state.order_status is OrderStatus.NONE
            and not state.cart_is_empty
            and config.is_valid_payment(state.payment_method)
            and counts_covers(state.stock, state.cart)
        )

    contract = Contract(
        name="defect.too_weak_precondition",
        action=ShoppingAction.place_order(),
        precondition=precondition,
        postcondition=place_order_postcondition,
        description="Ground truth without the shipping-address clause.",
    )
    return DefectFixture(
        name="too_weak_precondition",
        category=DefectCategory.TOO_WEAK_PRECONDITION,
        contract=contract,
        expected_symptoms=frozenset({Symptom.FALSE_ACCEPT}),
        rationale=(
            "Admits states with a missing or unserviceable address, where the "
            "sandbox refuses to place the order."
        ),
    )


def too_strong_precondition(config: EnvConfig) -> DefectFixture:
    """Over-fits to the one payment method seen during learning."""
    _require_two_payment_methods(config)
    preferred = config.valid_payment_methods[0]

    def precondition(state: ShoppingState) -> bool:
        return place_order_precondition(state, config) and state.payment_method == preferred

    contract = Contract(
        name="defect.too_strong_precondition",
        action=ShoppingAction.place_order(),
        precondition=precondition,
        postcondition=place_order_postcondition,
        description=f"Ground truth plus an unnecessary payment_method == {preferred!r} clause.",
    )
    return DefectFixture(
        name="too_strong_precondition",
        category=DefectCategory.TOO_STRONG_PRECONDITION,
        contract=contract,
        expected_symptoms=frozenset({Symptom.FALSE_REJECT}),
        rationale=(
            f"Refuses otherwise-valid states that pay with something other than "
            f"{preferred!r}, which the sandbox accepts."
        ),
    )


def wrong_postcondition(config: EnvConfig) -> DefectFixture:
    """Gets applicability right but misdescribes the effect on stock."""

    def postcondition(before: ShoppingState, after: ShoppingState) -> bool:
        return (
            after.order_status is OrderStatus.PLACED
            and after.order_items == before.cart
            and after.cart == ()
            # Wrong: placing an order reserves the goods, so stock must drop.
            and after.stock == before.stock
        )

    contract = Contract(
        name="defect.wrong_postcondition",
        action=ShoppingAction.place_order(),
        precondition=lambda state: place_order_precondition(state, config),
        postcondition=postcondition,
        description="Ground truth precondition with a stock-preserving effect claim.",
    )
    return DefectFixture(
        name="wrong_postcondition",
        category=DefectCategory.WRONG_POSTCONDITION,
        contract=contract,
        expected_symptoms=frozenset({Symptom.POSTCONDITION_VIOLATION}),
        rationale="Claims stock is unchanged, but every placed order decrements it.",
    )


def payment_branch_contract(config: EnvConfig, method: str) -> Contract:
    """A contract learned from episodes that only ever paid with ``method``.

    On its own this is sound: it never admits a state the sandbox refuses, and
    its effect claim holds everywhere it applies. The extra ``payment_method``
    clause in the effect is an artefact of the narrow episodes it came from.
    """

    def precondition(state: ShoppingState) -> bool:
        return place_order_precondition(state, config) and state.payment_method == method

    def postcondition(before: ShoppingState, after: ShoppingState) -> bool:
        return place_order_postcondition(before, after) and after.payment_method == method

    return Contract(
        name=f"branch.place_order[{method}]",
        action=ShoppingAction.place_order(),
        precondition=precondition,
        postcondition=postcondition,
        description=f"Sound but narrow contract observed only under {method!r}.",
    )


def faulty_merge(branches: Sequence[Contract], name: str = "defect.faulty_merge") -> Contract:
    """Merge branch contracts the wrong way round.

    A union of applicability domains needs the preconditions disjoined *and*
    the effect claims disjoined. Conjoining the effect claims instead keeps
    every branch's domain-specific artefact, so the merged effect can no longer
    hold anywhere. This is the bug this fixture reproduces.
    """
    if len(branches) < 2:
        raise ValueError("a merge needs at least two branch contracts")
    action = branches[0].action
    if any(branch.action != action for branch in branches):
        raise ValueError("cannot merge contracts for different actions")
    frozen = tuple(branches)

    def precondition(state: ShoppingState) -> bool:
        return any(branch.holds_in(state) for branch in frozen)

    def postcondition(before: ShoppingState, after: ShoppingState) -> bool:
        return all(branch.transition_holds(before, after) for branch in frozen)

    return Contract(
        name=name,
        action=action,
        precondition=precondition,
        postcondition=postcondition,
        description="Disjoined preconditions with wrongly conjoined effect claims.",
    )


def faulty_merge_fixture(config: EnvConfig) -> DefectFixture:
    """Merge the per-payment-method branch contracts with the faulty operator."""
    _require_two_payment_methods(config)
    branches = [payment_branch_contract(config, method) for method in config.valid_payment_methods]
    contract = faulty_merge(branches)
    return DefectFixture(
        name="faulty_merge",
        category=DefectCategory.FAULTY_MERGE,
        contract=contract,
        expected_symptoms=frozenset({Symptom.POSTCONDITION_VIOLATION}),
        rationale=(
            "Each branch is sound alone, but conjoining their effect claims "
            "demands the order be paid by every method at once, so no successful "
            "order can satisfy the merged effect."
        ),
    )


FIXTURE_BUILDERS = (
    too_weak_precondition,
    too_strong_precondition,
    wrong_postcondition,
    faulty_merge_fixture,
)


def all_fixtures(config: EnvConfig) -> tuple[DefectFixture, ...]:
    """Every defect fixture, one per category, in a stable order."""
    return tuple(build(config) for build in FIXTURE_BUILDERS)


def branch_contracts(config: EnvConfig) -> tuple[Contract, ...]:
    """The sound branch contracts that the faulty merge operates on."""
    _require_two_payment_methods(config)
    return tuple(payment_branch_contract(config, method) for method in config.valid_payment_methods)
