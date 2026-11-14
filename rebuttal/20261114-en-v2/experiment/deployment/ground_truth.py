#!/usr/bin/env python3
"""Evaluator-only ground-truth contract for the ``deploy_service`` skill.

This module is the hidden reference specification for the deployment domain.
Scoring and reporting code only; the sandbox never imports it.
"""

from __future__ import annotations

from .config import DeploymentConfig
from .contracts import Contract
from .state import (
    DeploymentAction,
    DeploymentState,
    DeploymentStatus,
    counts_combine,
    counts_covers,
)

VISIBILITY = "evaluator-only"

GROUND_TRUTH_CLAUSES = (
    "the agent is authenticated",
    "no service occupies the deployment slot",
    "allocated resources are not empty",
    "the target region is serviceable",
    "the cluster tier is supported",
    "available quota covers all allocated resources",
)


def deploy_service_precondition(state: DeploymentState, config: DeploymentConfig) -> bool:
    """True exactly where the deployment sandbox accepts ``deploy_service``."""
    return (
        state.authenticated
        and state.deployment_status is DeploymentStatus.IDLE
        and not state.is_empty_allocation
        and config.is_valid_region(state.target_region)
        and config.is_valid_tier(state.cluster_tier)
        and counts_covers(state.available_quota, state.allocated_resources)
    )


def deploy_service_postcondition(before: DeploymentState, after: DeploymentState) -> bool:
    """True exactly when the observed effect matches the intended deployment effect."""
    return (
        after.deployment_status is DeploymentStatus.DEPLOYED
        and after.active_deployment == before.allocated_resources
        and after.allocated_resources == ()
        and after.available_quota == counts_combine(before.available_quota, before.allocated_resources, sign=-1)
        and after.authenticated == before.authenticated
        and after.target_region == before.target_region
        and after.cluster_tier == before.cluster_tier
    )


def ground_truth_contract(config: DeploymentConfig) -> Contract:
    """Build the reference contract bound to one deployment configuration."""
    return Contract(
        name="ground_truth.deploy_service",
        action=DeploymentAction.deploy_service(),
        precondition=lambda state: deploy_service_precondition(state, config),
        postcondition=deploy_service_postcondition,
        description="Reference specification: " + "; ".join(GROUND_TRUTH_CLAUSES),
    )
