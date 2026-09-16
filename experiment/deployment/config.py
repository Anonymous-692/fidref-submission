#!/usr/bin/env python3
"""Seeded, immutable configuration for the cloud service deployment sandbox."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .state import ResourceCounts, counts_from_mapping


@dataclass(frozen=True)
class DeploymentConfig:
    """Resource quota, limit, and domain settings for one deployment sandbox instance."""

    seed: int = 20260902
    resource_types: tuple[str, ...] = ("cpu", "ram")
    max_quota: int = 2
    max_allocation_limit: int = 2
    valid_regions: tuple[str, ...] = ("us-central", "europe-west")
    rejected_regions: tuple[str, ...] = ("unsupported-edge",)
    valid_cluster_tiers: tuple[str, ...] = ("standard", "premium")
    rejected_cluster_tiers: tuple[str, ...] = ("deprecated-v0",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "resource_types", _normalise(self.resource_types, "resource_types"))
        object.__setattr__(
            self, "valid_regions", _normalise(self.valid_regions, "valid_regions")
        )
        object.__setattr__(
            self, "rejected_regions", _normalise(self.rejected_regions, "rejected_regions", allow_empty=True)
        )
        object.__setattr__(
            self, "valid_cluster_tiers", _normalise(self.valid_cluster_tiers, "valid_cluster_tiers")
        )
        object.__setattr__(
            self,
            "rejected_cluster_tiers",
            _normalise(self.rejected_cluster_tiers, "rejected_cluster_tiers", allow_empty=True),
        )
        if self.max_quota < 1:
            raise ValueError("max_quota must be at least 1")
        if self.max_allocation_limit < 1:
            raise ValueError("max_allocation_limit must be at least 1")
        overlap_regions = set(self.valid_regions) & set(self.rejected_regions)
        if overlap_regions:
            raise ValueError(f"regions cannot be valid and rejected: {sorted(overlap_regions)}")
        overlap_tiers = set(self.valid_cluster_tiers) & set(self.rejected_cluster_tiers)
        if overlap_tiers:
            raise ValueError(f"cluster tiers cannot be valid and rejected: {sorted(overlap_tiers)}")

    @property
    def region_options(self) -> tuple[str, ...]:
        """Every region value the agent may set, valid or not."""
        return tuple(sorted(self.valid_regions + self.rejected_regions))

    @property
    def tier_options(self) -> tuple[str, ...]:
        """Every cluster tier value the agent may set, valid or not."""
        return tuple(sorted(self.valid_cluster_tiers + self.rejected_cluster_tiers))

    def is_valid_region(self, region: str | None) -> bool:
        return region is not None and region in self.valid_regions

    def is_valid_tier(self, tier: str | None) -> bool:
        return tier is not None and tier in self.valid_cluster_tiers

    def initial_quota(self) -> ResourceCounts:
        """Draw the opening cluster quota levels from the configured seed."""
        rng = random.Random(self.seed)
        return counts_from_mapping({res: rng.randint(1, self.max_quota) for res in self.resource_types})

    @property
    def fingerprint(self) -> tuple[Any, ...]:
        """Identity used to reject snapshots taken under a different setup."""
        return (
            self.seed,
            self.resource_types,
            self.max_quota,
            self.max_allocation_limit,
            self.valid_regions,
            self.rejected_regions,
            self.valid_cluster_tiers,
            self.rejected_cluster_tiers,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "resource_types": list(self.resource_types),
            "max_quota": self.max_quota,
            "max_allocation_limit": self.max_allocation_limit,
            "valid_regions": list(self.valid_regions),
            "rejected_regions": list(self.rejected_regions),
            "valid_cluster_tiers": list(self.valid_cluster_tiers),
            "rejected_cluster_tiers": list(self.rejected_cluster_tiers),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeploymentConfig:
        known = {field: payload[field] for field in _FIELD_NAMES if field in payload}
        unknown = sorted(set(payload) - set(_FIELD_NAMES) - {"description", "domain"})
        if unknown:
            raise ValueError(f"unknown configuration keys: {unknown}")
        for field in (
            "resource_types",
            "valid_regions",
            "rejected_regions",
            "valid_cluster_tiers",
            "rejected_cluster_tiers",
        ):
            if field in known:
                known[field] = tuple(known[field])
        return cls(**known)

    @classmethod
    def from_json_file(cls, path: str | Path) -> DeploymentConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


_FIELD_NAMES = (
    "seed",
    "resource_types",
    "max_quota",
    "max_allocation_limit",
    "valid_regions",
    "rejected_regions",
    "valid_cluster_tiers",
    "rejected_cluster_tiers",
)


def _normalise(values: tuple[str, ...], label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    cleaned = tuple(sorted({str(value) for value in values}))
    if "" in cleaned:
        raise ValueError(f"{label} must not contain empty strings")
    if not cleaned and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    return cleaned


DEFAULT_DEPLOYMENT_CONFIG = DeploymentConfig()
