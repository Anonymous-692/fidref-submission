#!/usr/bin/env python3
"""Deliberately defective ``deploy_service`` contracts, one per defect category."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import DeploymentConfig
from .contracts import Contract, DefectCategory, Symptom
from .ground_truth import deploy_service_postcondition, deploy_service_precondition
from .state import (
    DeploymentAction,
    DeploymentState,
    DeploymentStatus,
    counts_covers,
)


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


def _require_two_tiers(config: DeploymentConfig) -> None:
    if len(config.valid_cluster_tiers) < 2:
        raise ValueError(
            "these fixtures need at least two valid cluster tiers to be falsifiable"
        )


def too_weak_precondition(config: DeploymentConfig) -> DefectFixture:
    """Forgets that a deployment needs a serviceable target region."""

    def precondition(state: DeploymentState) -> bool:
        return (
            state.authenticated
            and state.deployment_status is DeploymentStatus.IDLE
            and not state.is_empty_allocation
            and config.is_valid_tier(state.cluster_tier)
            and counts_covers(state.available_quota, state.allocated_resources)
        )

    contract = Contract(
        name="defect.too_weak_precondition",
        action=DeploymentAction.deploy_service(),
        precondition=precondition,
        postcondition=deploy_service_postcondition,
        description="Ground truth without the target-region clause.",
    )
    return DefectFixture(
        name="too_weak_precondition",
        category=DefectCategory.TOO_WEAK_PRECONDITION,
        contract=contract,
        expected_symptoms=frozenset({Symptom.FALSE_ACCEPT}),
        rationale=(
            "Admits states with a missing or unserviceable region, where the "
            "sandbox refuses to deploy the service."
        ),
    )


def too_strong_precondition(config: DeploymentConfig) -> DefectFixture:
    """Over-fits to one cluster tier seen during learning."""
    _require_two_tiers(config)
    preferred = config.valid_cluster_tiers[0]

    def precondition(state: DeploymentState) -> bool:
        return deploy_service_precondition(state, config) and state.cluster_tier == preferred

    contract = Contract(
        name="defect.too_strong_precondition",
        action=DeploymentAction.deploy_service(),
        precondition=precondition,
        postcondition=deploy_service_postcondition,
        description=f"Ground truth plus an unnecessary cluster_tier == {preferred!r} clause.",
    )
    return DefectFixture(
        name="too_strong_precondition",
        category=DefectCategory.TOO_STRONG_PRECONDITION,
        contract=contract,
        expected_symptoms=frozenset({Symptom.FALSE_REJECT}),
        rationale=(
            f"Refuses otherwise-valid states that use a tier other than {preferred!r}, "
            "which the sandbox accepts."
        ),
    )


def wrong_postcondition(config: DeploymentConfig) -> DefectFixture:
    """Gets applicability right but misdescribes the effect on quota."""

    def postcondition(before: DeploymentState, after: DeploymentState) -> bool:
        return (
            after.deployment_status is DeploymentStatus.DEPLOYED
            and after.active_deployment == before.allocated_resources
            and after.allocated_resources == ()
            # Wrong: deploying consumes quota, so available quota must drop.
            and after.available_quota == before.available_quota
        )

    contract = Contract(
        name="defect.wrong_postcondition",
        action=DeploymentAction.deploy_service(),
        precondition=lambda state: deploy_service_precondition(state, config),
        postcondition=postcondition,
        description="Ground truth precondition with a quota-preserving effect claim.",
    )
    return DefectFixture(
        name="wrong_postcondition",
        category=DefectCategory.WRONG_POSTCONDITION,
        contract=contract,
        expected_symptoms=frozenset({Symptom.POSTCONDITION_VIOLATION}),
        rationale="Claims available quota is unchanged, but every deployment decrements it.",
    )


def tier_branch_contract(config: DeploymentConfig, tier: str) -> Contract:
    """A contract learned from episodes that only ever used ``tier``."""

    def precondition(state: DeploymentState) -> bool:
        return deploy_service_precondition(state, config) and state.cluster_tier == tier

    def postcondition(before: DeploymentState, after: DeploymentState) -> bool:
        return deploy_service_postcondition(before, after) and after.cluster_tier == tier

    return Contract(
        name=f"branch.deploy_service[{tier}]",
        action=DeploymentAction.deploy_service(),
        precondition=precondition,
        postcondition=postcondition,
        description=f"Sound but narrow contract observed only under {tier!r}.",
    )


def faulty_merge(branches: Sequence[Contract], name: str = "defect.faulty_merge") -> Contract:
    """Merge branch contracts the wrong way round (conjoined postconditions)."""
    if len(branches) < 2:
        raise ValueError("a merge needs at least two branch contracts")
    action = branches[0].action
    if any(branch.action != action for branch in branches):
        raise ValueError("cannot merge contracts for different actions")
    frozen = tuple(branches)

    def precondition(state: DeploymentState) -> bool:
        return any(branch.holds_in(state) for branch in frozen)

    def postcondition(before: DeploymentState, after: DeploymentState) -> bool:
        return all(branch.transition_holds(before, after) for branch in frozen)

    return Contract(
        name=name,
        action=action,
        precondition=precondition,
        postcondition=postcondition,
        description="Disjoined preconditions with wrongly conjoined effect claims.",
    )


def faulty_merge_fixture(config: DeploymentConfig) -> DefectFixture:
    _require_two_tiers(config)
    branches = [tier_branch_contract(config, tier) for tier in config.valid_cluster_tiers]
    contract = faulty_merge(branches)
    return DefectFixture(
        name="faulty_merge",
        category=DefectCategory.FAULTY_MERGE,
        contract=contract,
        expected_symptoms=frozenset({Symptom.POSTCONDITION_VIOLATION}),
        rationale=(
            "Each branch is sound alone, but conjoining their effect claims "
            "demands the service run on every cluster tier at once."
        ),
    )


FIXTURE_BUILDERS = (
    too_weak_precondition,
    too_strong_precondition,
    wrong_postcondition,
    faulty_merge_fixture,
)


def all_fixtures(config: DeploymentConfig) -> tuple[DefectFixture, ...]:
    return tuple(build(config) for build in FIXTURE_BUILDERS)
