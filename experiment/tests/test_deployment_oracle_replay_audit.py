#!/usr/bin/env python3
"""Offline sampled-oracle replay tests; no model or network calls."""

from __future__ import annotations

import unittest

from ..deployment import dsl
from ..deployment.config import DeploymentConfig
from ..deployment.contracts import evaluate_contract
from ..deployment.oracle_replay_audit import progressive_sample, summarize
from ..deployment.runner import DeploymentExperimentRunner
from ..modeling.client import ChatClient
from .test_deployment_active_cegis import deployment_contract_spec


class OracleReplayAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = DeploymentConfig()
        cls.runner = DeploymentExperimentRunner(
            cls.config,
            ChatClient(model="offline-test"),
            max_depth=20,
            max_states=15000,
        )

    def sampled_report(self, parsed: dsl.ParsedContract, seed: int = 0):
        states = progressive_sample(
            self.runner,
            parsed,
            seed=seed,
            state_budget=48,
            query_budget=4,
            coverage=True,
        )
        return evaluate_contract(parsed.bind(self.config), states, self.config, max_counterexamples=0)

    def test_reject_everything_can_falsely_satisfy_the_48_state_sample(self) -> None:
        parsed = dsl.parse_contract_dict(
            {
                "skill": "deploy_service",
                "precondition": {"op": "const", "value": False},
                "postcondition": {"op": "const", "value": True},
            }
        )
        sampled = self.sampled_report(parsed)
        full = evaluate_contract(
            parsed.bind(self.config), self.runner.states, self.config, max_counterexamples=0
        )
        self.assertTrue(sampled.is_exact)
        self.assertEqual(sampled.successes, 0)
        self.assertFalse(full.is_exact)
        self.assertEqual(full.false_rejects, 88)

    def test_exact_dsl_contract_passes_sample_and_full_closure(self) -> None:
        parsed = dsl.parse_contract_dict(deployment_contract_spec())
        sampled = self.sampled_report(parsed)
        full = evaluate_contract(
            parsed.bind(self.config), self.runner.states, self.config, max_counterexamples=0
        )
        self.assertTrue(sampled.is_exact)
        self.assertTrue(full.is_exact)

    def test_summary_counts_replayed_and_reported_false_satisfaction(self) -> None:
        rows = [
            {
                "suite": "a",
                "contract_status": "parsed",
                "contract_sha256": "one",
                "audited": True,
                "sample_exact": True,
                "full_exact": False,
                "replay_false_satisfaction": True,
                "reported_sampled_oracle_satisfied": True,
                "reported_false_satisfaction": True,
            },
            {
                "suite": "a",
                "contract_status": "parse_failure",
                "audited": False,
            },
        ]
        summary = summarize(rows)
        self.assertEqual(summary["artifacts"], 2)
        self.assertEqual(summary["parsed_final_contracts"], 1)
        self.assertEqual(summary["replay_false_satisfaction"], 1)
        self.assertEqual(summary["reported_false_satisfaction"], 1)


if __name__ == "__main__":
    unittest.main()
