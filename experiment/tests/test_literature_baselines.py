#!/usr/bin/env python3
"""Dedicated unit tests for literature baseline adaptations.

Covers:
- Agent Skill Induction (ASI / asi_replay): single historical successful demonstration.
- SkillCommit (skillcommit_replay): proposal followed by cross-instance validation
  and conditional revision on incompatibilities.
- ContractSkill (contractskill_repair): budget-limited local replay repair.
- Fairness disclosures and prompt safety.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from .. import modeling
from ..environment import ShoppingAction, apply_action
from ..evaluation import evaluate_contract
from ..evaluation.ground_truth import GROUND_TRUTH_CLAUSES
from ..modeling import artifacts as artifacts_module
from ..modeling import prompts, runner as runner_module
from ..modeling.runner import Budgets, Decoding, ExperimentRunner, RunSpec
from .support import SMALL_CONFIG_PATH, TEST_MAX_DEPTH, TEST_MAX_STATES, small_config
from .test_model_runner import (
    EXACT_TEXT,
    SCORING,
    MockTransport,
    contract_spec,
    make_client,
    make_runner,
    spec,
)


class AsiReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.transport = MockTransport()
        cls.runner = make_runner(cls.transport)

    def setUp(self) -> None:
        self.transport.reset()

    def test_asi_replay_executes_single_historical_positive_demonstration(self) -> None:
        self.transport.reset([EXACT_TEXT])
        budgets = Budgets(state_budget=5, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("asi_replay", budgets=budgets))

        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(artifact.spend["model_calls"], 1)
        self.assertEqual(artifact.spend["states_observed"], 1)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.rounds[0]["adaptation"], "asi_replay")
        self.assertEqual(artifact.rounds[0]["evidence_policy"], "single_successful_observation")
        self.assertEqual(len(artifact.rounds[0]["observations"]), 1)
        self.assertTrue(artifact.rounds[0]["observations"][0]["ok"])

        prompt = self.transport.last_user_message(0)
        self.assertIn("historical successful execution demonstration of place_order", prompt)
        self.assertIn("place_order -> ACCEPTED", prompt)
        self.assertNotIn("REFUSED", prompt)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_EXACT)
        self.assertTrue(artifact.evaluation.exact)

    def test_asi_replay_zero_state_budget_stops_immediately(self) -> None:
        self.transport.reset([])
        budgets = Budgets(state_budget=0, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("asi_replay", budgets=budgets))
        self.assertEqual(len(self.transport.calls), 0)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_STATE_BUDGET)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_NO_CONTRACT)


class SkillCommitReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.transport = MockTransport()
        cls.runner = make_runner(cls.transport)

    def setUp(self) -> None:
        self.transport.reset()

    def test_proposal_and_early_stop_on_cross_instance_compatibility(self) -> None:
        self.transport.reset([EXACT_TEXT])
        budgets = Budgets(state_budget=5, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("skillcommit_replay", budgets=budgets))

        # Only 1 model call should be made when candidate is compatible with all replay states
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(artifact.spend["model_calls"], 1)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.spend["states_observed"], 5)

        # Round 0: Proposal
        self.assertEqual(artifact.rounds[0]["role"], "propose")
        self.assertEqual(artifact.rounds[0]["adaptation"], "skillcommit_replay")
        self.assertEqual(artifact.rounds[0]["evidence_policy"], "proposal_from_single_successful_instance")
        self.assertEqual(artifact.rounds[0]["replay_set_size"], 4)

        # Round 1: Validation with 0 incompatibilities
        self.assertEqual(artifact.rounds[1]["role"], "validate")
        self.assertTrue(artifact.rounds[1]["compatible"])
        self.assertEqual(artifact.rounds[1]["incompatibilities"], 0)
        self.assertEqual(artifact.rounds[1]["replay_states_checked"], 4)

        prompt = self.transport.last_user_message(0)
        self.assertIn("historical successful execution demonstration", prompt)
        self.assertIn("positive example", prompt)
        self.assertNotIn("REFUSED", prompt)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_COMPLETE)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_EXACT)
        self.assertTrue(artifact.evaluation.exact)

    def test_incompatibility_triggers_revision_call(self) -> None:
        # Defective candidate: adds restrictive payment requirement causing false rejects on other successful states
        defective_spec = contract_spec()
        defective_spec["precondition"]["args"].append(
            {"op": "eq", "left": {"var": "payment_method"}, "right": {"const": "card"}}
        )
        defective_text = json.dumps(defective_spec)

        self.transport.reset([defective_text, EXACT_TEXT])
        budgets = Budgets(state_budget=6, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("skillcommit_replay", budgets=budgets))

        # 2 model calls: propose, then revise
        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual([entry.role for entry in artifact.interactions], ["propose", "revise"])
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.spend["states_observed"], 6)

        # Revision round records incompatibilities
        self.assertEqual(artifact.rounds[1]["role"], "revise")
        self.assertEqual(artifact.rounds[1]["evidence_policy"], "cross_instance_replay_revision")
        self.assertGreater(artifact.rounds[1]["incompatibilities"], 0)
        self.assertTrue(artifact.rounds[1]["incompatibility_details"])

        revision_prompt = self.transport.last_user_message(1)
        self.assertIn("Cross-instance validation against additional distinct historical successful executions", revision_prompt)
        self.assertIn("false rejects", revision_prompt)
        self.assertIn("positive examples only", revision_prompt)
        self.assertNotIn("REFUSED", revision_prompt)

        self.assertEqual(artifact.stopped_because, runner_module.STOP_COMPLETE)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_EXACT)
        self.assertTrue(artifact.evaluation.exact)

    def test_incompatibility_stops_when_query_budget_is_one(self) -> None:
        defective_spec = contract_spec()
        defective_spec["precondition"]["args"].append(
            {"op": "eq", "left": {"var": "payment_method"}, "right": {"const": "card"}}
        )
        defective_text = json.dumps(defective_spec)

        self.transport.reset([defective_text])
        budgets = Budgets(state_budget=6, query_budget=1, token_budget=16000)
        artifact = self.runner.run(spec("skillcommit_replay", budgets=budgets))

        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_QUERY_BUDGET)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)

    def test_positive_only_containment_and_zero_negative_traces(self) -> None:
        states = self.runner.successful_states(0, 10)
        self.assertGreater(len(states), 0)
        for state in states:
            outcome = apply_action(state, ShoppingAction.place_order(), self.runner.config)
            self.assertTrue(outcome.ok, msg=f"State {state} was expected to succeed")

    def test_zero_state_budget_stops_immediately(self) -> None:
        self.transport.reset([])
        budgets = Budgets(state_budget=0, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("skillcommit_replay", budgets=budgets))
        self.assertEqual(len(self.transport.calls), 0)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_STATE_BUDGET)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_NO_CONTRACT)


class ContractSkillRepairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.transport = MockTransport()
        cls.runner = make_runner(cls.transport)

    def setUp(self) -> None:
        self.transport.reset()

    def test_local_replay_set_repair_flow(self) -> None:
        defective_text = json.dumps({
            "skill": "place_order",
            "precondition": {"op": "const", "value": True},
            "postcondition": {"op": "const", "value": True},
        })
        self.transport.reset([defective_text, EXACT_TEXT])
        budgets = Budgets(state_budget=8, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("contractskill_repair", budgets=budgets))

        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.spend["states_observed"], 8)
        self.assertEqual(artifact.rounds[0]["adaptation"], "contractskill_repair")
        self.assertEqual(artifact.rounds[0]["evidence_policy"], "budget_limited_observed_replay_repair")
        self.assertEqual(artifact.rounds[0]["replay_set_size"], 8)

        revision_prompt = self.transport.last_user_message(1)
        self.assertIn("budget-limited observed replay set", revision_prompt)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_COMPLETE)
        self.assertTrue(artifact.evaluation.exact)


class LiteraturePromptSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_config()

    def test_no_literature_prompt_leaks_ground_truth_clauses(self) -> None:
        dummy_obs = {"before": "state_a", "ok": True, "error": None, "after": "state_b"}
        dummy_metrics = {
            "false_accepts": 0,
            "false_rejects": 1,
            "postcondition_violations": 0,
            "states_checked": 4,
            "replay_states_checked": 4,
        }
        dummy_cx = [{"symptom": "false_reject", "state": "state_a", "detail": "detail_a"}]

        prompt_texts = [
            prompts.asi_prompt(self.config, dummy_obs),
            prompts.skillcommit_proposal_prompt(self.config, dummy_obs),
            prompts.skillcommit_prompt(self.config, [dummy_obs]),
            prompts.skillcommit_revision_prompt("{}", dummy_metrics, dummy_cx),
            prompts.contractskill_repair_prompt("{}", dummy_metrics, dummy_cx),
        ]

        for text in prompt_texts:
            for clause in GROUND_TRUTH_CLAUSES:
                self.assertNotIn(clause, text, msg=f"Prompt leaked ground truth clause: {clause!r}")


class FairnessDisclosuresTest(unittest.TestCase):
    def test_runner_methods_document_simplifications_and_scans(self) -> None:
        self.assertIn("offline experience corpus", ExperimentRunner.successful_states.__doc__ or "")
        self.assertIn("prototype simplification", ExperimentRunner.successful_states.__doc__ or "")
        self.assertIn("scans the full reachable closure", ExperimentRunner.score.__doc__ or "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
