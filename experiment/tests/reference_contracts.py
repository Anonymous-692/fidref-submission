"""Reference DSL fixture extracted unchanged for offline tests."""
from typing import Any

def _exact_dsl_spec() -> dict[str, Any]:
    """The reference contract represented once in the existing safe DSL."""
    return {
        "skill": "deploy_service",
        "name": "g2.cpu_gate.reference",
        "notes": "CPU gate reference contract; not a model response.",
        "precondition": {
            "op": "and",
            "args": [
                {"op": "is_true", "arg": {"var": "authenticated"}},
                {"op": "eq", "left": {"var": "deployment_status"}, "right": {"const": "idle"}},
                {"op": "not", "arg": {"op": "is_empty", "arg": {"var": "allocated_resources"}}},
                {"op": "in_set", "value": {"var": "target_region"}, "set": "valid_regions"},
                {"op": "in_set", "value": {"var": "cluster_tier"}, "set": "valid_cluster_tiers"},
                {
                    "op": "covers",
                    "available": {"var": "available_quota"},
                    "required": {"var": "allocated_resources"},
                },
            ],
        },
        "postcondition": {
            "op": "and",
            "args": [
                {"op": "eq", "left": {"var": "deployment_status", "when": "after"}, "right": {"const": "deployed"}},
                {"op": "eq", "left": {"var": "active_deployment", "when": "after"}, "right": {"var": "allocated_resources"}},
                {"op": "is_empty", "arg": {"var": "allocated_resources", "when": "after"}},
                {
                    "op": "eq",
                    "left": {"var": "available_quota", "when": "after"},
                    "right": {"op": "difference", "left": {"var": "available_quota"}, "right": {"var": "allocated_resources"}},
                },
                {"op": "unchanged", "vars": ["authenticated", "target_region", "cluster_tier"]},
            ],
        },
    }
