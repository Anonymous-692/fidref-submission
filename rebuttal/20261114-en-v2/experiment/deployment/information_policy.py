"""Explicit information conditions for Deployment ablations, not launch settings."""

from __future__ import annotations

from dataclasses import dataclass
from .config import DeploymentConfig


@dataclass(frozen=True)
class InformationPolicy:
    track: str = "legacy"
    reveal_valid_sets: bool = True
    reveal_postcondition_patterns: bool = True

    def __post_init__(self) -> None:
        if self.track not in ("legacy", "pattern_only", "restricted_vp"):
            raise ValueError("unknown information-policy track")
        if type(self.reveal_valid_sets) is not bool or type(self.reveal_postcondition_patterns) is not bool:
            raise ValueError("information-policy flags must be booleans")
        if self.track != "restricted_vp" and not self.reveal_valid_sets:
            raise ValueError("hiding policy labels requires the restricted_vp track")
        if self.track == "legacy" and not self.reveal_postcondition_patterns:
            raise ValueError("pattern removal requires an explicit ablation track")

    @property
    def restricted(self) -> bool:
        return self.track == "restricted_vp"

    @property
    def condition_id(self) -> str:
        return (
            ("vplus" if self.reveal_valid_sets else "vminus")
            + "_" + ("pplus" if self.reveal_postcondition_patterns else "pminus")
        )

    def config_sets(self, legacy_sets: tuple[str, ...]) -> tuple[str, ...]:
        # Allowlist, rather than prefix filtering: future private sets stay private.
        if self.restricted:
            return ("region_options", "resource_types", "tier_options")
        return legacy_sets

    def validate_fixture(self, config: DeploymentConfig) -> None:
        if self.restricted and (
            config.region_options != ("region_a", "region_b", "region_c")
            or config.tier_options != ("tier_a", "tier_b", "tier_c")
        ):
            raise ValueError("restricted_vp requires neutral region_a/b/c and tier_a/b/c options")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": "deployment_information_v1",
            "track": self.track,
            "reveal_valid_sets": self.reveal_valid_sets,
            "reveal_postcondition_patterns": self.reveal_postcondition_patterns,
        }


LEGACY_INFORMATION_POLICY = InformationPolicy()
