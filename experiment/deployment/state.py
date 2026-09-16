#!/usr/bin/env python3
"""Immutable state and action values for the cloud service deployment domain."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping


ResourceCounts = tuple[tuple[str, int], ...]


def counts_from_mapping(mapping: Mapping[str, int]) -> ResourceCounts:
    """Build a canonical resource multiset, dropping non-positive quantities."""
    for resource, quantity in mapping.items():
        if quantity < 0:
            raise ValueError(f"negative quantity for {resource!r}: {quantity}")
    return tuple(sorted((res, qty) for res, qty in mapping.items() if qty > 0))


def counts_to_mapping(counts: ResourceCounts) -> dict[str, int]:
    return {resource: quantity for resource, quantity in counts}


def counts_get(counts: ResourceCounts, resource: str) -> int:
    for name, quantity in counts:
        if name == resource:
            return quantity
    return 0


def counts_total(counts: ResourceCounts) -> int:
    return sum(quantity for _, quantity in counts)


def counts_add(counts: ResourceCounts, resource: str, delta: int) -> ResourceCounts:
    """Return a copy of ``counts`` with ``delta`` added to ``resource``."""
    updated = counts_to_mapping(counts)
    new_quantity = updated.get(resource, 0) + delta
    if new_quantity < 0:
        raise ValueError(f"quantity for {resource!r} would become negative")
    updated[resource] = new_quantity
    return counts_from_mapping(updated)


def counts_combine(left: ResourceCounts, right: ResourceCounts, sign: int = 1) -> ResourceCounts:
    """Return ``left + sign * right`` as a canonical multiset."""
    if sign not in (1, -1):
        raise ValueError("sign must be 1 or -1")
    combined = counts_to_mapping(left)
    for resource, quantity in right:
        new_quantity = combined.get(resource, 0) + sign * quantity
        if new_quantity < 0:
            raise ValueError(f"quantity for {resource!r} would become negative")
        combined[resource] = new_quantity
    return counts_from_mapping(combined)


def counts_covers(available: ResourceCounts, required: ResourceCounts) -> bool:
    """True when ``available`` holds at least every quantity in ``required``."""
    return all(counts_get(available, res) >= qty for res, qty in required)


class DeploymentStatus(str, Enum):
    """Lifecycle position of the service deployment slot."""

    IDLE = "idle"
    DEPLOYED = "deployed"
    STOPPED = "stopped"
    ROLLEDBACK = "rolledback"

    def __str__(self) -> str:
        return self.value


class ActionKind(Enum):
    """Every operation the deployment API exposes.

    Fixed integer order for deterministic BFS enumeration.
    """

    AUTHENTICATE = 1
    REVOKE_AUTH = 2
    ALLOCATE_RESOURCE = 3
    RELEASE_RESOURCE = 4
    CLEAR_RESOURCES = 5
    SET_TARGET_REGION = 6
    CLEAR_TARGET_REGION = 7
    SET_CLUSTER_TIER = 8
    CLEAR_CLUSTER_TIER = 9
    DEPLOY_SERVICE = 10
    STOP_SERVICE = 11
    ROLLBACK_SERVICE = 12
    CLEAR_DEPLOYMENT = 13


PARAMETERISED_KINDS = frozenset(
    {
        ActionKind.ALLOCATE_RESOURCE,
        ActionKind.RELEASE_RESOURCE,
        ActionKind.SET_TARGET_REGION,
        ActionKind.SET_CLUSTER_TIER,
    }
)


@dataclass(frozen=True)
class DeploymentAction:
    """A single deployment operation, optionally parameterised by a target."""

    kind: ActionKind
    target: str | None = None

    def __post_init__(self) -> None:
        if self.kind in PARAMETERISED_KINDS and self.target is None:
            raise ValueError(f"{self.kind.name} requires a target")
        if self.kind not in PARAMETERISED_KINDS and self.target is not None:
            raise ValueError(f"{self.kind.name} does not accept a target")

    @property
    def sort_key(self) -> tuple[int, str]:
        return (self.kind.value, self.target or "")

    @property
    def name(self) -> str:
        return self.kind.name.lower()

    def __str__(self) -> str:
        if self.target is None:
            return self.name
        return f"{self.name}({self.target})"

    def to_dict(self) -> dict[str, str | None]:
        return {"kind": self.kind.name, "target": self.target}

    @classmethod
    def authenticate(cls) -> DeploymentAction:
        return cls(ActionKind.AUTHENTICATE)

    @classmethod
    def revoke_auth(cls) -> DeploymentAction:
        return cls(ActionKind.REVOKE_AUTH)

    @classmethod
    def allocate_resource(cls, resource: str) -> DeploymentAction:
        return cls(ActionKind.ALLOCATE_RESOURCE, resource)

    @classmethod
    def release_resource(cls, resource: str) -> DeploymentAction:
        return cls(ActionKind.RELEASE_RESOURCE, resource)

    @classmethod
    def clear_resources(cls) -> DeploymentAction:
        return cls(ActionKind.CLEAR_RESOURCES)

    @classmethod
    def set_target_region(cls, region: str) -> DeploymentAction:
        return cls(ActionKind.SET_TARGET_REGION, region)

    @classmethod
    def clear_target_region(cls) -> DeploymentAction:
        return cls(ActionKind.CLEAR_TARGET_REGION)

    @classmethod
    def set_cluster_tier(cls, tier: str) -> DeploymentAction:
        return cls(ActionKind.SET_CLUSTER_TIER, tier)

    @classmethod
    def clear_cluster_tier(cls) -> DeploymentAction:
        return cls(ActionKind.CLEAR_CLUSTER_TIER)

    @classmethod
    def deploy_service(cls) -> DeploymentAction:
        return cls(ActionKind.DEPLOY_SERVICE)

    @classmethod
    def stop_service(cls) -> DeploymentAction:
        return cls(ActionKind.STOP_SERVICE)

    @classmethod
    def rollback_service(cls) -> DeploymentAction:
        return cls(ActionKind.ROLLBACK_SERVICE)

    @classmethod
    def clear_deployment(cls) -> DeploymentAction:
        return cls(ActionKind.CLEAR_DEPLOYMENT)


@dataclass(frozen=True)
class DeploymentState:
    """A complete, hashable snapshot of the cloud deployment world.

    Every field is either a scalar or a canonical tuple. Instances are
    immutable, value-comparable, and safe to share without copying.
    """

    authenticated: bool
    allocated_resources: ResourceCounts
    available_quota: ResourceCounts
    target_region: str | None
    cluster_tier: str | None
    deployment_status: DeploymentStatus
    active_deployment: ResourceCounts

    def resource_quantity(self, resource: str) -> int:
        return counts_get(self.allocated_resources, resource)

    def quota_quantity(self, resource: str) -> int:
        return counts_get(self.available_quota, resource)

    @property
    def allocated_total(self) -> int:
        return counts_total(self.allocated_resources)

    @property
    def is_empty_allocation(self) -> bool:
        return self.allocated_total == 0

    def evolve(self, **changes: object) -> DeploymentState:
        """Return a new state with the given fields replaced."""
        return dataclasses.replace(self, **changes)

    @property
    def key(self) -> tuple[object, ...]:
        """A canonical, sortable identity for the state."""
        return (
            self.authenticated,
            self.allocated_resources,
            self.available_quota,
            self.target_region or "",
            self.cluster_tier or "",
            self.deployment_status.value,
            self.active_deployment,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "authenticated": self.authenticated,
            "allocated_resources": counts_to_mapping(self.allocated_resources),
            "available_quota": counts_to_mapping(self.available_quota),
            "target_region": self.target_region,
            "cluster_tier": self.cluster_tier,
            "deployment_status": self.deployment_status.value,
            "active_deployment": counts_to_mapping(self.active_deployment),
        }

    def describe(self) -> str:
        alloc = ",".join(f"{res}x{qty}" for res, qty in self.allocated_resources) or "-"
        quota = ",".join(f"{res}x{qty}" for res, qty in self.available_quota) or "-"
        return (
            f"auth={"y" if self.authenticated else "n"} alloc={alloc} quota={quota} "
            f"region={self.target_region or "-"} tier={self.cluster_tier or "-"} "
            f"status={self.deployment_status.value}"
        )


def sort_states(states: Iterable[DeploymentState]) -> tuple[DeploymentState, ...]:
    """Order states canonically; used to compare enumerations set-wise."""
    return tuple(sorted(states, key=lambda state: state.key))
