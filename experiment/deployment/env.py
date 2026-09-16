#!/usr/bin/env python3
"""Deterministic transition rules and the agent-facing cloud deployment environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import DEFAULT_DEPLOYMENT_CONFIG, DeploymentConfig
from .state import (
    ActionKind,
    DeploymentAction,
    DeploymentState,
    DeploymentStatus,
    counts_add,
    counts_combine,
    counts_covers,
)


@dataclass(frozen=True)
class StepResult:
    """Outcome of applying one action in the deployment sandbox."""

    state: DeploymentState
    ok: bool
    action: DeploymentAction
    error: str | None = None

    def __post_init__(self) -> None:
        if self.ok and self.error is not None:
            raise ValueError("a successful step cannot carry an error")
        if not self.ok and not self.error:
            raise ValueError("a rejected step must explain why")


@dataclass(frozen=True)
class EnvSnapshot:
    """Opaque, immutable capture of everything ``restore`` needs."""

    state: DeploymentState
    step_count: int
    rejected_count: int
    config_fingerprint: tuple[Any, ...]


def initial_state(config: DeploymentConfig = DEFAULT_DEPLOYMENT_CONFIG) -> DeploymentState:
    """The seeded opening state: unauthenticated, no allocated resources, full quota."""
    return DeploymentState(
        authenticated=False,
        allocated_resources=(),
        available_quota=config.initial_quota(),
        target_region=None,
        cluster_tier=None,
        deployment_status=DeploymentStatus.IDLE,
        active_deployment=(),
    )


def action_space(config: DeploymentConfig = DEFAULT_DEPLOYMENT_CONFIG) -> tuple[DeploymentAction, ...]:
    """Every action the agent may attempt, in a fixed reproducible order."""
    actions: list[DeploymentAction] = [
        DeploymentAction.authenticate(),
        DeploymentAction.revoke_auth(),
    ]
    actions.extend(DeploymentAction.allocate_resource(res) for res in config.resource_types)
    actions.extend(DeploymentAction.release_resource(res) for res in config.resource_types)
    actions.append(DeploymentAction.clear_resources())
    actions.extend(DeploymentAction.set_target_region(region) for region in config.region_options)
    actions.append(DeploymentAction.clear_target_region())
    actions.extend(DeploymentAction.set_cluster_tier(tier) for tier in config.tier_options)
    actions.append(DeploymentAction.clear_cluster_tier())
    actions.extend(
        [
            DeploymentAction.deploy_service(),
            DeploymentAction.stop_service(),
            DeploymentAction.rollback_service(),
            DeploymentAction.clear_deployment(),
        ]
    )
    return tuple(sorted(actions, key=lambda action: action.sort_key))


def apply_action(
    state: DeploymentState,
    action: DeploymentAction,
    config: DeploymentConfig = DEFAULT_DEPLOYMENT_CONFIG,
) -> StepResult:
    """Pure transition function for the deployment environment."""
    handler = _HANDLERS[action.kind]
    return handler(state, action, config)


def _reject(state: DeploymentState, action: DeploymentAction, reason: str) -> StepResult:
    return StepResult(state=state, ok=False, action=action, error=reason)


def _accept(state: DeploymentState, action: DeploymentAction) -> StepResult:
    return StepResult(state=state, ok=True, action=action, error=None)


def _authenticate(
    state: DeploymentState, action: DeploymentAction, config: DeploymentConfig
) -> StepResult:
    if state.authenticated:
        return _reject(state, action, "already authenticated")
    return _accept(state.evolve(authenticated=True), action)


def _revoke_auth(
    state: DeploymentState, action: DeploymentAction, config: DeploymentConfig
) -> StepResult:
    if not state.authenticated:
        return _reject(state, action, "not authenticated")
    return _accept(state.evolve(authenticated=False), action)


def _allocate_resource(
    state: DeploymentState, action: DeploymentAction, config: DeploymentConfig
) -> StepResult:
    resource = action.target
    if resource not in config.resource_types:
        return _reject(state, action, f"unknown resource type {resource!r}")
    if state.allocated_total >= config.max_allocation_limit:
        return _reject(state, action, "allocation limit reached")
    return _accept(
        state.evolve(allocated_resources=counts_add(state.allocated_resources, resource, 1)),
        action,
    )


def _release_resource(
    state: DeploymentState, action: DeploymentAction, config: DeploymentConfig
) -> StepResult:
    resource = action.target
    if resource not in config.resource_types:
        return _reject(state, action, f"unknown resource type {resource!r}")
    if state.resource_quantity(resource) == 0:
        return _reject(state, action, f"{resource!r} is not allocated")
    return _accept(
        state.evolve(allocated_resources=counts_add(state.allocated_resources, resource, -1)),
        action,
    )


def _clear_resources(
    state: DeploymentState, action: DeploymentAction, config: DeploymentConfig
) -> StepResult:
    if state.is_empty_allocation:
        return _reject(state, action, "no resources allocated")
    return _accept(state.evolve(allocated_resources=()), action)


def _set_target_region(
    state: DeploymentState, action: DeploymentAction, config: DeploymentConfig
) -> StepResult:
    region = action.target
    if region not in config.region_options:
        return _reject(state, action, f"unknown region {region!r}")
    if state.target_region == region:
        return _reject(state, action, f"target region is already {region!r}")
    return _accept(state.evolve(target_region=region), action)


def _clear_target_region(
    state: DeploymentState, action: DeploymentAction, config: DeploymentConfig
) -> StepResult:
    if state.target_region is None:
        return _reject(state, action, "no target region is set")
    return _accept(state.evolve(target_region=None), action)


def _set_cluster_tier(
    state: DeploymentState, action: DeploymentAction, config: DeploymentConfig
) -> StepResult:
    tier = action.target
    if tier not in config.tier_options:
        return _reject(state, action, f"unknown cluster tier {tier!r}")
    if state.cluster_tier == tier:
        return _reject(state, action, f"cluster tier is already {tier!r}")
    return _accept(state.evolve(cluster_tier=tier), action)


def _clear_cluster_tier(
    state: DeploymentState, action: DeploymentAction, config: DeploymentConfig
) -> StepResult:
    if state.cluster_tier is None:
        return _reject(state, action, "no cluster tier is set")
    return _accept(state.evolve(cluster_tier=None), action)


def _deploy_service(
    state: DeploymentState, action: DeploymentAction, config: DeploymentConfig
) -> StepResult:
    if not state.authenticated:
        return _reject(state, action, "not authenticated")
    if state.deployment_status is not DeploymentStatus.IDLE:
        return _reject(state, action, f"a deployment is already {state.deployment_status.value}")
    if state.is_empty_allocation:
        return _reject(state, action, "no resources allocated for deployment")
    if not config.is_valid_region(state.target_region):
        return _reject(state, action, "target region is missing or unserviceable")
    if not config.is_valid_tier(state.cluster_tier):
        return _reject(state, action, "cluster tier is missing or not supported")
    if not counts_covers(state.available_quota, state.allocated_resources):
        return _reject(state, action, "insufficient cluster quota for requested resources")
    return _accept(
        state.evolve(
            allocated_resources=(),
            available_quota=counts_combine(state.available_quota, state.allocated_resources, sign=-1),
            deployment_status=DeploymentStatus.DEPLOYED,
            active_deployment=state.allocated_resources,
        ),
        action,
    )


def _stop_service(
    state: DeploymentState, action: DeploymentAction, config: DeploymentConfig
) -> StepResult:
    if state.deployment_status is not DeploymentStatus.DEPLOYED:
        return _reject(state, action, "no active deployment to stop")
    return _accept(state.evolve(deployment_status=DeploymentStatus.STOPPED), action)


def _rollback_service(
    state: DeploymentState, action: DeploymentAction, config: DeploymentConfig
) -> StepResult:
    if state.deployment_status is not DeploymentStatus.DEPLOYED:
        return _reject(state, action, "no active deployment to rollback")
    return _accept(
        state.evolve(
            available_quota=counts_combine(
                state.available_quota, state.active_deployment, sign=1
            ),
            deployment_status=DeploymentStatus.ROLLEDBACK,
            active_deployment=(),
        ),
        action,
    )


def _clear_deployment(
    state: DeploymentState, action: DeploymentAction, config: DeploymentConfig
) -> StepResult:
    if state.deployment_status not in (DeploymentStatus.STOPPED, DeploymentStatus.ROLLEDBACK):
        return _reject(state, action, "only a stopped or rolled back deployment can be cleared")
    return _accept(
        state.evolve(deployment_status=DeploymentStatus.IDLE, active_deployment=()), action
    )


_HANDLERS = {
    ActionKind.AUTHENTICATE: _authenticate,
    ActionKind.REVOKE_AUTH: _revoke_auth,
    ActionKind.ALLOCATE_RESOURCE: _allocate_resource,
    ActionKind.RELEASE_RESOURCE: _release_resource,
    ActionKind.CLEAR_RESOURCES: _clear_resources,
    ActionKind.SET_TARGET_REGION: _set_target_region,
    ActionKind.CLEAR_TARGET_REGION: _clear_target_region,
    ActionKind.SET_CLUSTER_TIER: _set_cluster_tier,
    ActionKind.CLEAR_CLUSTER_TIER: _clear_cluster_tier,
    ActionKind.DEPLOY_SERVICE: _deploy_service,
    ActionKind.STOP_SERVICE: _stop_service,
    ActionKind.ROLLBACK_SERVICE: _rollback_service,
    ActionKind.CLEAR_DEPLOYMENT: _clear_deployment,
}


class DeploymentEnv:
    """Agent-facing sandbox for the cloud service deployment domain."""

    def __init__(self, config: DeploymentConfig | None = None) -> None:
        self._config = config if config is not None else DEFAULT_DEPLOYMENT_CONFIG
        self._action_space = action_space(self._config)
        self._state = initial_state(self._config)
        self._step_count = 0
        self._rejected_count = 0

    @property
    def config(self) -> DeploymentConfig:
        return self._config

    @property
    def state(self) -> DeploymentState:
        return self._state

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def rejected_count(self) -> int:
        return self._rejected_count

    def reset(self) -> DeploymentState:
        self._state = initial_state(self._config)
        self._step_count = 0
        self._rejected_count = 0
        return self._state

    def step(self, action: DeploymentAction) -> StepResult:
        if not isinstance(action, DeploymentAction):
            raise TypeError(f"expected DeploymentAction, got {type(action).__name__}")
        result = apply_action(self._state, action, self._config)
        self._state = result.state
        self._step_count += 1
        if not result.ok:
            self._rejected_count += 1
        return result

    def action_space(self) -> tuple[DeploymentAction, ...]:
        return self._action_space

    def valid_actions(self) -> tuple[DeploymentAction, ...]:
        return valid_actions(self._state, self._config)

    def snapshot(self) -> EnvSnapshot:
        return EnvSnapshot(
            state=self._state,
            step_count=self._step_count,
            rejected_count=self._rejected_count,
            config_fingerprint=self._config.fingerprint,
        )

    def restore(self, snapshot: EnvSnapshot) -> DeploymentState:
        if not isinstance(snapshot, EnvSnapshot):
            raise TypeError(f"expected EnvSnapshot, got {type(snapshot).__name__}")
        if snapshot.config_fingerprint != self._config.fingerprint:
            raise ValueError("snapshot was taken under a different configuration")
        self._state = snapshot.state
        self._step_count = snapshot.step_count
        self._rejected_count = snapshot.rejected_count
        return self._state

    def run(self, actions: list[DeploymentAction]) -> list[StepResult]:
        return [self.step(action) for action in actions]


def valid_actions(
    state: DeploymentState, config: DeploymentConfig = DEFAULT_DEPLOYMENT_CONFIG
) -> tuple[DeploymentAction, ...]:
    return tuple(
        action for action in action_space(config) if apply_action(state, action, config).ok
    )


__all__ = [
    "DeploymentEnv",
    "EnvSnapshot",
    "StepResult",
    "action_space",
    "apply_action",
    "initial_state",
    "valid_actions",
]
