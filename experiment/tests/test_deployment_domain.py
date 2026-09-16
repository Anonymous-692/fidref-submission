#!/usr/bin/env python3
"""Comprehensive unit tests for the cloud service deployment domain."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from ..deployment import (
    ActionKind,
    Contract,
    DefectCategory,
    DeploymentAction,
    DeploymentConfig,
    DeploymentEnv,
    DeploymentState,
    DeploymentStatus,
    ResourceCounts,
    Symptom,
    all_fixtures,
    apply_action,
    counts_add,
    counts_combine,
    counts_covers,
    counts_from_mapping,
    counts_get,
    counts_to_mapping,
    counts_total,
    deploy_service_postcondition,
    deploy_service_precondition,
    enumerate_reachable,
    evaluate_all,
    evaluate_contract,
    faulty_merge_fixture,
    ground_truth_contract,
    initial_state,
    sort_states,
    too_strong_precondition,
    too_weak_precondition,
    valid_actions,
    wrong_postcondition,
)
from ..deployment import dsl


class DeploymentStateTest(unittest.TestCase):
    def test_counts_math(self) -> None:
        c1 = counts_from_mapping({"cpu": 2, "ram": 1})
        self.assertEqual(counts_get(c1, "cpu"), 2)
        self.assertEqual(counts_get(c1, "gpu"), 0)
        self.assertEqual(counts_total(c1), 3)

        c2 = counts_add(c1, "cpu", 1)
        self.assertEqual(counts_get(c2, "cpu"), 3)

        c3 = counts_add(c1, "ram", -1)
        self.assertEqual(counts_get(c3, "ram"), 0)
        self.assertEqual(counts_to_mapping(c3), {"cpu": 2})

        with self.assertRaises(ValueError):
            counts_add(c1, "gpu", -1)

        c_sub = counts_combine(c1, counts_from_mapping({"cpu": 1}), sign=-1)
        self.assertEqual(counts_to_mapping(c_sub), {"cpu": 1, "ram": 1})
        self.assertTrue(counts_covers(c1, c_sub))
        self.assertFalse(counts_covers(c_sub, c1))

    def test_state_immutability_and_hashing(self) -> None:
        cfg = DeploymentConfig()
        s1 = initial_state(cfg)
        self.assertFalse(s1.authenticated)
        self.assertEqual(s1.deployment_status, DeploymentStatus.IDLE)
        self.assertTrue(s1.is_empty_allocation)

        s2 = s1.evolve(authenticated=True)
        self.assertNotEqual(s1, s2)
        self.assertIn(s1, {s1})
        self.assertIn(s2, {s2})
        self.assertEqual(len({s1, s2}), 2)


class DeploymentEnvTransitionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DeploymentConfig(
            seed=42,
            resource_types=("cpu", "ram"),
            max_quota=2,
            max_allocation_limit=2,
            valid_regions=("us-central", "europe-west"),
            rejected_regions=("unsupported-edge",),
            valid_cluster_tiers=("standard", "premium"),
            rejected_cluster_tiers=("deprecated-v0",),
        )
        self.env = DeploymentEnv(self.config)

    def test_authentication_lifecycle(self) -> None:
        res = self.env.step(DeploymentAction.authenticate())
        self.assertTrue(res.ok)
        self.assertTrue(self.env.state.authenticated)

        # Re-authenticate fails
        res2 = self.env.step(DeploymentAction.authenticate())
        self.assertFalse(res2.ok)
        self.assertIn("already authenticated", res2.error or "")

        # Revoke auth
        res3 = self.env.step(DeploymentAction.revoke_auth())
        self.assertTrue(res3.ok)
        self.assertFalse(self.env.state.authenticated)

    def test_resource_allocation_limits(self) -> None:
        self.env.step(DeploymentAction.allocate_resource("cpu"))
        self.assertEqual(self.env.state.resource_quantity("cpu"), 1)
        self.env.step(DeploymentAction.allocate_resource("ram"))
        self.assertEqual(self.env.state.allocated_total, 2)

        # Exceeds max allocation limit (2)
        res = self.env.step(DeploymentAction.allocate_resource("cpu"))
        self.assertFalse(res.ok)
        self.assertIn("allocation limit reached", res.error or "")

        # Release resource
        res_rel = self.env.step(DeploymentAction.release_resource("cpu"))
        self.assertTrue(res_rel.ok)
        self.assertEqual(self.env.state.resource_quantity("cpu"), 0)

        # Clear resources
        self.env.step(DeploymentAction.clear_resources())
        self.assertTrue(self.env.state.is_empty_allocation)

    def test_deploy_service_guards_and_effects(self) -> None:
        # Step 1: Attempt deploy without auth -> rejected
        r1 = self.env.step(DeploymentAction.deploy_service())
        self.assertFalse(r1.ok)
        self.assertIn("not authenticated", r1.error or "")

        self.env.step(DeploymentAction.authenticate())

        # Step 2: Attempt deploy without allocated resources -> rejected
        r2 = self.env.step(DeploymentAction.deploy_service())
        self.assertFalse(r2.ok)
        self.assertIn("no resources allocated", r2.error or "")

        self.env.step(DeploymentAction.allocate_resource("cpu"))

        # Step 3: Attempt deploy without target region -> rejected
        r3 = self.env.step(DeploymentAction.deploy_service())
        self.assertFalse(r3.ok)
        self.assertIn("region is missing", r3.error or "")

        self.env.step(DeploymentAction.set_target_region("unsupported-edge"))

        # Step 4: Attempt deploy with unserviceable region -> rejected
        r4 = self.env.step(DeploymentAction.deploy_service())
        self.assertFalse(r4.ok)
        self.assertIn("unserviceable", r4.error or "")

        self.env.step(DeploymentAction.set_target_region("us-central"))

        # Step 5: Attempt deploy without cluster tier -> rejected
        r5 = self.env.step(DeploymentAction.deploy_service())
        self.assertFalse(r5.ok)
        self.assertIn("cluster tier is missing", r5.error or "")

        self.env.step(DeploymentAction.set_cluster_tier("standard"))

        # Step 6: Successful deployment!
        quota_before = self.env.state.available_quota
        r6 = self.env.step(DeploymentAction.deploy_service())
        self.assertTrue(r6.ok)
        self.assertEqual(self.env.state.deployment_status, DeploymentStatus.DEPLOYED)
        self.assertEqual(self.env.state.active_deployment, (("cpu", 1),))
        self.assertEqual(self.env.state.allocated_resources, ())
        self.assertEqual(
            self.env.state.available_quota,
            counts_combine(quota_before, (("cpu", 1),), sign=-1),
        )

        # Step 7: Rollback returns quota
        r7 = self.env.step(DeploymentAction.rollback_service())
        self.assertTrue(r7.ok)
        self.assertEqual(self.env.state.deployment_status, DeploymentStatus.ROLLEDBACK)
        self.assertEqual(self.env.state.available_quota, quota_before)

        # Step 8: Clear deployment returns to IDLE
        r8 = self.env.step(DeploymentAction.clear_deployment())
        self.assertTrue(r8.ok)
        self.assertEqual(self.env.state.deployment_status, DeploymentStatus.IDLE)

    def test_snapshot_and_restore(self) -> None:
        self.env.step(DeploymentAction.authenticate())
        self.env.step(DeploymentAction.allocate_resource("cpu"))
        snap = self.env.snapshot()

        self.env.step(DeploymentAction.set_target_region("us-central"))
        self.env.step(DeploymentAction.clear_resources())

        restored = self.env.restore(snap)
        self.assertEqual(restored.resource_quantity("cpu"), 1)
        self.assertIsNone(restored.target_region)
        self.assertEqual(self.env.step_count, snap.step_count)


class DeploymentEnumerationTest(unittest.TestCase):
    def test_enumeration_is_bounded_and_deterministic(self) -> None:
        config = DeploymentConfig(
            seed=1234,
            resource_types=("cpu",),
            max_quota=1,
            max_allocation_limit=1,
            valid_regions=("us-central",),
            rejected_regions=(),
            valid_cluster_tiers=("standard",),
            rejected_cluster_tiers=(),
        )
        res1 = enumerate_reachable(config, max_depth=20, max_states=1000)
        res2 = enumerate_reachable(config, max_depth=20, max_states=1000)

        self.assertFalse(res1.truncated)
        self.assertGreater(len(res1.states), 10)
        self.assertEqual(res1.states, res2.states)
        self.assertEqual(res1.transitions, res2.transitions)


class GroundTruthAndFixturesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DeploymentConfig(
            seed=2026,
            resource_types=("cpu", "ram"),
            max_quota=2,
            max_allocation_limit=2,
            valid_regions=("us-central", "europe-west"),
            rejected_regions=("unsupported-edge",),
            valid_cluster_tiers=("standard", "premium"),
            rejected_cluster_tiers=("deprecated-v0",),
        )
        self.enumeration = enumerate_reachable(self.config, max_depth=15, max_states=2000)
        self.states = self.enumeration.states

    def test_ground_truth_contract_is_exact(self) -> None:
        gt = ground_truth_contract(self.config)
        report = evaluate_contract(gt, self.states, self.config)
        self.assertTrue(report.is_exact)
        self.assertEqual(report.false_accepts, 0)
        self.assertEqual(report.false_rejects, 0)
        self.assertEqual(report.postcondition_violations, 0)
        self.assertGreater(report.successes, 0)

    def test_defect_fixtures_exhibit_expected_symptoms(self) -> None:
        fixtures = all_fixtures(self.config)
        self.assertEqual(len(fixtures), 4)

        for fix in fixtures:
            with self.subTest(fixture=fix.name):
                report = evaluate_contract(fix.contract, self.states, self.config)
                self.assertFalse(report.is_exact)
                self.assertEqual(report.symptoms, fix.expected_symptoms)


class DeploymentDslTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DeploymentConfig(
            seed=2026,
            resource_types=("cpu", "ram"),
            max_quota=2,
            max_allocation_limit=2,
            valid_regions=("us-central", "europe-west"),
            rejected_regions=("unsupported-edge",),
            valid_cluster_tiers=("standard", "premium"),
            rejected_cluster_tiers=("deprecated-v0",),
        )

    def exact_spec(self) -> dict[str, Any]:
        return {
            "skill": "deploy_service",
            "notes": "Exact reference contract in DSL",
            "precondition": {
                "op": "and",
                "args": [
                    {"op": "is_true", "arg": {"var": "authenticated"}},
                    {"op": "eq", "left": {"var": "deployment_status"}, "right": {"const": "idle"}},
                    {"op": "not", "arg": {"op": "is_empty", "arg": {"var": "allocated_resources"}}},
                    {"op": "in_set", "value": {"var": "target_region"}, "set": "valid_regions"},
                    {"op": "in_set", "value": {"var": "cluster_tier"}, "set": "valid_cluster_tiers"},
                    {"op": "covers", "available": {"var": "available_quota"}, "required": {"var": "allocated_resources"}},
                ],
            },
            "postcondition": {
                "op": "and",
                "args": [
                    {
                        "op": "eq",
                        "left": {"var": "deployment_status", "when": "after"},
                        "right": {"const": "deployed"},
                    },
                    {
                        "op": "eq",
                        "left": {"var": "active_deployment", "when": "after"},
                        "right": {"var": "allocated_resources"},
                    },
                    {"op": "is_empty", "arg": {"var": "allocated_resources", "when": "after"}},
                    {
                        "op": "eq",
                        "left": {"var": "available_quota", "when": "after"},
                        "right": {
                            "op": "difference",
                            "left": {"var": "available_quota"},
                            "right": {"var": "allocated_resources"},
                        },
                    },
                    {"op": "unchanged", "vars": ["authenticated", "target_region", "cluster_tier"]},
                ],
            },
        }

    def test_dsl_parses_and_evaluates_exact_contract(self) -> None:
        spec = self.exact_spec()
        parsed = dsl.parse_contract_text(json.dumps(spec))
        bound = parsed.bind(self.config)

        states = enumerate_reachable(self.config, max_depth=12, max_states=1000).states
        report = evaluate_contract(bound, states, self.config)
        self.assertTrue(report.is_exact)

    def test_dsl_rejects_unsafe_or_malformed_input(self) -> None:
        with self.assertRaises(dsl.DslError):
            dsl.parse_contract_text('{"precondition": {"op": "eval"}, "postcondition": {"op": "const", "value": true}}')

        with self.assertRaises(dsl.DslError):
            dsl.parse_contract_text("not json")


class DeploymentGroundTruthSeparationTest(unittest.TestCase):
    def test_env_files_never_import_ground_truth_or_contracts(self) -> None:
        env_files = [
            Path("experiment/deployment/state.py"),
            Path("experiment/deployment/config.py"),
            Path("experiment/deployment/env.py"),
            Path("experiment/deployment/enumeration.py"),
        ]
        for path in env_files:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn("ground_truth", alias.name)
                        self.assertNotIn("contracts", alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.assertNotIn("ground_truth", node.module)
                        self.assertNotIn("contracts", node.module)

    def test_ground_truth_has_evaluator_only_visibility(self) -> None:
        from ..deployment.ground_truth import VISIBILITY
        self.assertEqual(VISIBILITY, "evaluator-only")


if __name__ == "__main__":
    unittest.main()
