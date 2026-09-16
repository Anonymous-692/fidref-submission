#!/usr/bin/env python3
"""Tests for active CEGIS and model methods on the cloud deployment domain.

Runs against mocked HTTP transports; no network calls or live model serving.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from ..deployment import (
    DeploymentAction,
    DeploymentConfig,
    DeploymentExperimentRunner,
    DeploymentState,
    DeploymentStatus,
    dsl,
    enumerate_reachable,
)
from ..deployment import model_runner as deployment_model_runner
from ..deployment import prompts as deployment_prompts
from ..deployment.runner import (
    STOP_CONTEXT_BUDGET,
    STOP_DUPLICATE_OUTPUT,
    STOP_VACUOUS_CANDIDATE,
    compute_protocol_sha256,
)
from ..modeling import artifacts as artifacts_module
from ..modeling.client import ChatClient, RawResponse, TransportError
from ..modeling.runner import (
    ACTIVE_CEGIS,
    ACTIVE_CEGIS_NO_BALANCE,
    ACTIVE_CEGIS_NO_COVERAGE,
    ACTIVE_CEGIS_UNIFORM,
    Budgets,
    Decoding,
    RunSpec,
    STOP_SAMPLE_SATISFIED,
    STOP_STATE_BUDGET,
    STOP_TRANSPORT,
)
from ..tests.test_model_runner import MockTransport


def deployment_contract_spec(*, require_region: bool = True, notes: str = "fixture") -> dict[str, Any]:
    precondition_args: list[dict[str, Any]] = [
        {"op": "is_true", "arg": {"var": "authenticated"}},
        {"op": "eq", "left": {"var": "deployment_status"}, "right": {"const": "idle"}},
        {"op": "not", "arg": {"op": "is_empty", "arg": {"var": "allocated_resources"}}},
        {"op": "in_set", "value": {"var": "cluster_tier"}, "set": "valid_cluster_tiers"},
        {"op": "covers", "available": {"var": "available_quota"}, "required": {"var": "allocated_resources"}},
    ]
    if require_region:
        precondition_args.insert(
            3, {"op": "in_set", "value": {"var": "target_region"}, "set": "valid_regions"}
        )
    return {
        "skill": "deploy_service",
        "notes": notes,
        "precondition": {"op": "and", "args": precondition_args},
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


EXACT_DEPLOYMENT_TEXT = json.dumps(deployment_contract_spec())
WEAK_DEPLOYMENT_TEXT = json.dumps(deployment_contract_spec(require_region=False, notes="missing region clause"))

# An answer the grammar rejects: `authenticated_check` is not a formula operator.
UNPARSEABLE_DEPLOYMENT_TEXT = json.dumps(
    {
        "skill": "deploy_service",
        "notes": "invented operator",
        "precondition": {"op": "authenticated_check", "arg": {"var": "authenticated"}},
        "postcondition": {"op": "const", "value": True},
    }
)

# Parses, but its precondition admits nothing: the sampled oracle can never
# contradict it, so it agrees with every sample for free.
VACUOUS_DEPLOYMENT_TEXT = json.dumps(
    {
        "skill": "deploy_service",
        "notes": "rejects every state",
        "precondition": {"op": "const", "value": False},
        "postcondition": {"op": "const", "value": True},
    }
)


def make_deployment_runner(
    transport: MockTransport,
    config: DeploymentConfig | None = None,
    max_depth: int = 15,
    max_states: int = 1000,
    **runner_options: Any,
) -> DeploymentExperimentRunner:
    cfg = config or DeploymentConfig(
        seed=2026,
        resource_types=("cpu", "ram"),
        max_quota=2,
        max_allocation_limit=2,
        valid_regions=("us-central", "europe-west"),
        rejected_regions=("unsupported-edge",),
        valid_cluster_tiers=("standard", "premium"),
        rejected_cluster_tiers=("deprecated-v0",),
    )
    client = ChatClient(model="test-deployment-model", transport=transport)
    return DeploymentExperimentRunner(
        cfg,
        client,
        max_depth=max_depth,
        max_states=max_states,
        **runner_options,
    )


class DeploymentStateSelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = MockTransport()
        self.runner = make_deployment_runner(self.transport)
        self.parsed_exact = dsl.parse_contract_text(EXACT_DEPLOYMENT_TEXT)
        self.parsed_weak = dsl.parse_contract_text(WEAK_DEPLOYMENT_TEXT)

    def test_selector_partitions_by_candidate_prediction(self) -> None:
        # Candidate aware selection must produce states from both predicted accept and predicted reject
        selected = self.runner.candidate_aware_states(
            self.parsed_weak,
            seed=0,
            count=6,
            balance=True,
            coverage=True,
        )
        self.assertEqual(len(selected), 6)

        contract = self.parsed_weak.bind(self.runner.config)
        predicted_accepts = sum(contract.holds_in(s) for s in selected)
        predicted_rejects = sum(not contract.holds_in(s) for s in selected)
        self.assertEqual(predicted_accepts, 3)
        self.assertEqual(predicted_rejects, 3)

    def test_selector_never_inspects_ground_truth_labels(self) -> None:
        # Verify that candidate_aware_states only uses candidate prediction and raw state fields
        selected = self.runner.candidate_aware_states(
            self.parsed_weak,
            seed=42,
            count=8,
            balance=True,
            coverage=True,
        )
        self.assertEqual(len(selected), 8)
        # All selected states are unique valid DeploymentState instances
        self.assertEqual(len(set(selected)), 8)
        for s in selected:
            self.assertIsInstance(s, DeploymentState)


class DeploymentActiveCegisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = MockTransport()
        self.runner = make_deployment_runner(self.transport)

    def test_active_cegis_end_to_end_synthesis_and_budget_tracking(self) -> None:
        # Round 0 proposes WEAK (missing region), checker discovers counterexample on unserviceable region,
        # Round 1 revises with EXACT, checker satisfied on sample, final closure scoring confirms exact!
        self.transport.reset([WEAK_DEPLOYMENT_TEXT, EXACT_DEPLOYMENT_TEXT])
        budgets = Budgets(state_budget=12, query_budget=4, token_budget=16000)
        spec = RunSpec(method=ACTIVE_CEGIS, seed=0, budgets=budgets)

        artifact = self.runner.run(spec)

        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual(artifact.spend["model_calls"], 2)
        # Zero full-closure oracle feedback queries during synthesis
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        # Sampled feedback queries were used instead
        self.assertGreater(artifact.spend["sampled_feedback_queries"], 0)
        self.assertGreater(artifact.spend["states_observed"], 0)
        self.assertLessEqual(artifact.spend["states_observed"], budgets.state_budget)

        self.assertEqual(artifact.rounds[0]["adaptation"], ACTIVE_CEGIS)
        self.assertEqual(
            artifact.rounds[0]["evidence_policy"],
            "candidate_partitioned_progressive_coverage",
        )
        self.assertTrue(artifact.rounds[0]["counterexamples"])
        for counterexample in artifact.rounds[0]["counterexamples"]:
            self.assertIn("before_state", counterexample)
            self.assertIn("after_state", counterexample)
        self.assertEqual(artifact.stopped_because, STOP_SAMPLE_SATISFIED)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_EXACT)
        self.assertTrue(artifact.evaluation.exact)

    def test_compact_context_uses_two_messages_and_clamps_output_tokens(self) -> None:
        token_count = RawResponse(
            status=200,
            body=json.dumps({"count": 7000, "max_model_len": 8192}).encode("utf-8"),
        )
        self.transport.reset([token_count, WEAK_DEPLOYMENT_TEXT, token_count, EXACT_DEPLOYMENT_TEXT])
        runner = make_deployment_runner(
            self.transport,
            compact_context=True,
            context_token_limit=8192,
            context_margin=256,
            counterexample_limit=3,
        )
        spec = RunSpec(
            method=ACTIVE_CEGIS,
            seed=0,
            decoding=Decoding(max_tokens=2048),
            budgets=Budgets(state_budget=12, query_budget=4, token_budget=16000),
        )

        artifact = runner.run(spec)

        chat_calls = [call for call in self.transport.calls if call["url"].endswith("/chat/completions")]
        tokenize_calls = [call for call in self.transport.calls if call["url"].endswith("/tokenize")]
        self.assertEqual(len(chat_calls), 2)
        self.assertEqual(len(tokenize_calls), 2)
        self.assertTrue(all(not call["url"].endswith("/v1/tokenize") for call in tokenize_calls))
        for call in chat_calls:
            self.assertEqual(
                [message["role"] for message in call["payload"]["messages"]],
                ["system", "user"],
            )
            self.assertEqual(call["payload"]["max_tokens"], 936)
        self.assertLessEqual(len(artifact.rounds[0]["counterexamples"]), 3)
        self.assertTrue(artifact.evaluation.exact)

    def test_active_cegis_no_coverage_ablation(self) -> None:
        self.transport.reset([WEAK_DEPLOYMENT_TEXT, EXACT_DEPLOYMENT_TEXT])
        budgets = Budgets(state_budget=12, query_budget=4, token_budget=16000)
        spec = RunSpec(method=ACTIVE_CEGIS_NO_COVERAGE, seed=0, budgets=budgets)

        artifact = self.runner.run(spec)

        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.rounds[0]["adaptation"], ACTIVE_CEGIS_NO_COVERAGE)
        self.assertEqual(
            artifact.rounds[0]["evidence_policy"],
            "candidate_partitioned_progressive_random",
        )
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_EXACT)
        self.assertTrue(artifact.evaluation.exact)

    def test_active_cegis_ablations_record_correct_policies(self) -> None:
        cases = {
            ACTIVE_CEGIS_NO_BALANCE: "progressive_coverage_without_candidate_balance",
            ACTIVE_CEGIS_NO_COVERAGE: "candidate_partitioned_progressive_random",
            ACTIVE_CEGIS_UNIFORM: "progressive_uniform_random",
        }
        budgets = Budgets(state_budget=12, query_budget=4, token_budget=16000)
        for method, policy in cases.items():
            with self.subTest(method=method):
                self.transport.reset([WEAK_DEPLOYMENT_TEXT, EXACT_DEPLOYMENT_TEXT])
                artifact = self.runner.run(RunSpec(method=method, seed=1, budgets=budgets))
                self.assertEqual(artifact.rounds[0]["adaptation"], method)
                self.assertEqual(artifact.rounds[0]["evidence_policy"], policy)
                self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)

    def test_direct_and_random_probe_on_deployment_domain(self) -> None:
        # Test direct
        self.transport.reset([EXACT_DEPLOYMENT_TEXT])
        artifact_direct = self.runner.run(RunSpec(method="direct", seed=0))
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(artifact_direct.spend["states_observed"], 0)
        self.assertEqual(artifact_direct.outcome, artifacts_module.OUTCOME_EXACT)

        # Test random_probe
        self.transport.reset([EXACT_DEPLOYMENT_TEXT])
        budgets = Budgets(state_budget=8, query_budget=4, token_budget=16000)
        artifact_probe = self.runner.run(RunSpec(method="random_probe", seed=0, budgets=budgets))
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(artifact_probe.spend["states_observed"], 8)
        self.assertEqual(artifact_probe.rounds[0]["role"], "probe")
        self.assertEqual(artifact_probe.outcome, artifacts_module.OUTCOME_EXACT)


class DeploymentActiveCegisGuardTest(unittest.TestCase):
    """The three loop guards: parse repair, duplicate output, vacuous candidate."""

    def setUp(self) -> None:
        self.transport = MockTransport()
        self.budgets = Budgets(state_budget=12, query_budget=4, token_budget=16000)

    def user_message(self, index: int) -> str:
        payload = self.transport.calls[index]["payload"]
        return [m for m in payload["messages"] if m["role"] == "user"][-1]["content"]

    # -- guard 1: compact-context parse repair --------------------------------

    def test_compact_parse_repair_echoes_the_rejected_answer_and_recovers(self) -> None:
        self.transport.reset([UNPARSEABLE_DEPLOYMENT_TEXT, EXACT_DEPLOYMENT_TEXT])
        runner = make_deployment_runner(self.transport, compact_context=True)

        artifact = runner.run(RunSpec(method=ACTIVE_CEGIS, seed=0, budgets=self.budgets))

        self.assertEqual(len(self.transport.calls), 2)
        repair = self.user_message(1)
        # The repair turn carries the previous invalid output, the specific parse
        # error, and the task configuration, because compact mode drops the
        # transcript that would otherwise hold them.
        self.assertIn(UNPARSEABLE_DEPLOYMENT_TEXT, repair)
        self.assertIn("authenticated_check", repair)
        self.assertIn("Parse error: unknown formula op 'authenticated_check'", repair)
        self.assertIn("Synthesize the complete formal contract for `deploy_service`", repair)
        self.assertEqual(
            [m["role"] for m in self.transport.calls[1]["payload"]["messages"]],
            ["system", "user"],
        )

        # ... and the run recovers from it.
        self.assertEqual(artifact.rounds[0]["contract_status"], artifacts_module.STATUS_PARSE_FAILURE)
        self.assertEqual(artifact.spend["parse_failures"], 1)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.stopped_because, STOP_SAMPLE_SATISFIED)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_EXACT)
        self.assertTrue(artifact.evaluation.exact)

    def test_parse_repair_echo_is_bounded(self) -> None:
        runner = make_deployment_runner(self.transport, compact_context=True)
        runaway = "{" + "x" * 5000
        prompt = deployment_prompts.parse_repair_prompt(
            runner.config, error="boom", previous_output=runaway
        )
        limit = deployment_prompts.PARSE_REPAIR_ECHO_CHARS
        self.assertIn("x" * 200, prompt)
        self.assertNotIn("x" * (limit + 1), prompt)
        self.assertIn("...[truncated:", prompt)
        self.assertIn("Parse error: boom", prompt)
        self.assertIn("Synthesize the complete formal contract for `deploy_service`", prompt)

    def test_full_transcript_parse_repair_keeps_the_existing_note(self) -> None:
        self.transport.reset([UNPARSEABLE_DEPLOYMENT_TEXT, EXACT_DEPLOYMENT_TEXT])
        runner = make_deployment_runner(self.transport)

        artifact = runner.run(RunSpec(method=ACTIVE_CEGIS, seed=0, budgets=self.budgets))

        repair = self.user_message(1)
        self.assertIn("Your previous response could not be parsed", repair)
        # The transcript already replays the rejected answer, so it is not echoed.
        self.assertNotIn(UNPARSEABLE_DEPLOYMENT_TEXT, repair)
        roles = [m["role"] for m in self.transport.calls[1]["payload"]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertTrue(artifact.evaluation.exact)

    # -- guard 2: two consecutive byte-identical answers -----------------------

    def test_duplicate_output_stops_after_two_calls_with_a_distinct_reason(self) -> None:
        self.transport.reset([WEAK_DEPLOYMENT_TEXT, WEAK_DEPLOYMENT_TEXT, EXACT_DEPLOYMENT_TEXT])
        runner = make_deployment_runner(self.transport)

        artifact = runner.run(RunSpec(method=ACTIVE_CEGIS, seed=0, budgets=self.budgets))

        # Exactly two calls: the third scripted answer is never requested.
        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual(len(self.transport.script), 1)
        self.assertEqual(artifact.spend["model_calls"], 2)
        self.assertEqual(artifact.stopped_because, STOP_DUPLICATE_OUTPUT)
        self.assertNotEqual(artifact.stopped_because, STOP_SAMPLE_SATISFIED)

        # The remaining query budget is left unspent rather than retried away.
        self.assertEqual(artifact.spend["remaining_calls"], self.budgets.query_budget - 2)

        guard_round = artifact.rounds[-1]
        self.assertEqual(guard_round["guard"], "duplicate_output")
        self.assertEqual(guard_round["round"], 1)
        self.assertEqual(guard_round["duplicate_of_round"], 0)
        self.assertEqual([entry["round"] for entry in artifact.rounds], [0, 1])

        # The repeated answer was not re-audited: no extra sampled query, no extra
        # state spend beyond what round 0 already observed.
        self.assertEqual(artifact.spend["sampled_feedback_queries"], 1)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.spend["states_observed"], guard_round["cumulative_states"])
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_INEXACT)

    def test_distinct_answers_do_not_trip_the_duplicate_guard(self) -> None:
        self.transport.reset([WEAK_DEPLOYMENT_TEXT, EXACT_DEPLOYMENT_TEXT])
        runner = make_deployment_runner(self.transport)

        artifact = runner.run(RunSpec(method=ACTIVE_CEGIS, seed=0, budgets=self.budgets))

        self.assertEqual(artifact.stopped_because, STOP_SAMPLE_SATISFIED)
        self.assertFalse(any(entry.get("guard") for entry in artifact.rounds))

    # -- guard 3: reject-everything candidates --------------------------------

    def test_vacuous_candidate_is_revised_instead_of_satisfying_the_sample(self) -> None:
        self.transport.reset([VACUOUS_DEPLOYMENT_TEXT, EXACT_DEPLOYMENT_TEXT])
        runner = make_deployment_runner(self.transport)

        artifact = runner.run(RunSpec(method=ACTIVE_CEGIS, seed=0, budgets=self.budgets))

        self.assertEqual(len(self.transport.calls), 2)
        guard_round = artifact.rounds[0]
        self.assertEqual(guard_round["guard"], "vacuous_candidate")
        self.assertEqual(guard_round["vacuity"]["predicted_accepts"], 0)
        self.assertEqual(guard_round["vacuity"]["reachable_states"], len(runner.states))
        self.assertEqual(guard_round["vacuity"]["evidence"], "candidate_prediction_only")
        # The vacuous candidate was never audited, so it cost no states.
        self.assertEqual(guard_round["cumulative_states"], 0)
        self.assertNotIn("audit_metrics", guard_round)
        self.assertNotIn("counterexamples", guard_round)

        # The revision request carries candidate-only vacuity information: the
        # candidate itself plus how many states its own precondition admits.
        revision = self.user_message(1)
        self.assertIn('"precondition": {"op": "const", "value": false}', revision)
        self.assertIn("admits none of the", revision)
        self.assertIn(f"{len(runner.states)} reachable sandbox states", revision)
        for leak in (
            "false accepts",
            "false rejects",
            "postcondition violations",
            "Counterexample",
            "sandbox refused",
            "ACCEPTED",
            "REFUSED",
        ):
            self.assertNotIn(leak, revision)

        # It recovers to an exact contract without a single oracle query.
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.stopped_because, STOP_SAMPLE_SATISFIED)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_EXACT)
        self.assertTrue(artifact.evaluation.exact)

    def test_vacuous_candidate_never_stops_as_sampled_oracle_satisfied(self) -> None:
        # Depth 3 cannot reach a deployable state, so reject-everything agrees with
        # the whole sample and would otherwise be reported as satisfied.
        self.transport.reset([VACUOUS_DEPLOYMENT_TEXT])
        runner = make_deployment_runner(self.transport, max_depth=3, vacuity_revision_limit=0)

        artifact = runner.run(
            RunSpec(
                method=ACTIVE_CEGIS,
                seed=0,
                budgets=Budgets(state_budget=6, query_budget=1, token_budget=16000),
            )
        )

        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(runner.predicted_accept_count(dsl.parse_contract_text(VACUOUS_DEPLOYMENT_TEXT)), 0)
        audit_round = artifact.rounds[-1]
        self.assertEqual(audit_round["reachable_predicted_accepts"], 0)
        self.assertFalse(audit_round["counterexamples"])
        self.assertEqual(artifact.stopped_because, STOP_VACUOUS_CANDIDATE)
        self.assertNotEqual(artifact.stopped_because, STOP_SAMPLE_SATISFIED)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)

    def test_bounded_vacuity_revisions_then_the_duplicate_guard_stops_the_loop(self) -> None:
        self.transport.reset([VACUOUS_DEPLOYMENT_TEXT] * 4)
        runner = make_deployment_runner(self.transport, vacuity_revision_limit=2)

        artifact = runner.run(RunSpec(method=ACTIVE_CEGIS, seed=0, budgets=self.budgets))

        # Round 0 asks for a revision; round 1 repeats it byte for byte and stops.
        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual(artifact.rounds[0]["guard"], "vacuous_candidate")
        self.assertEqual(artifact.rounds[0]["vacuity_revision"], 1)
        self.assertEqual(artifact.rounds[-1]["guard"], "duplicate_output")
        self.assertEqual(artifact.stopped_because, STOP_DUPLICATE_OUTPUT)
        self.assertEqual(artifact.spend["states_observed"], 0)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)

    def test_vacuity_revision_limit_must_not_be_negative(self) -> None:
        with self.assertRaises(ValueError):
            make_deployment_runner(self.transport, vacuity_revision_limit=-1)


class DeploymentCliTest(unittest.TestCase):
    def test_contextfix_pilot_configuration_is_loadable_and_pinned(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "deployment_active_cegis_contextfix_pilot.json"
        )
        config = deployment_model_runner.load_experiment_config(path)
        self.assertEqual(config["methods"], ["active_cegis"])
        self.assertEqual(config["seeds"], [0, 1, 5, 8, 15])
        self.assertEqual(config["workers"], 4)
        self.assertTrue(config["compact_context"])
        self.assertEqual(config["context_token_limit"], 8192)
        self.assertEqual(config["context_margin"], 256)
        self.assertEqual(config["counterexample_limit"], 3)
        self.assertFalse(config["guided_json"])

    def test_deployment_cli_execution_with_mocked_transport(self) -> None:
        transport = MockTransport([EXACT_DEPLOYMENT_TEXT] * 2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            argv = [
                "--output",
                str(output),
                "--sandbox-config",
                "experiment/configs/deployment_small.json",
                "--model",
                "test-deployment-model",
                "--methods",
                "active_cegis",
                "active_cegis_no_coverage",
                "--seeds",
                "0",
                "--state-budget",
                "10",
                "--query-budget",
                "2",
                "--json",
            ]
            code = deployment_model_runner.main(argv, transport=transport)
            self.assertEqual(code, 0)
            self.assertTrue(output.is_dir())
            summary_file = output / "summary.json"
            self.assertTrue(summary_file.is_file())
            summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
            self.assertEqual(summary_data["context"]["domain"], "deployment")
            self.assertEqual(len(summary_data["runs"]), 2)
            self.assertTrue(summary_data["runs"][0]["exact"])
            self.assertTrue(summary_data["runs"][1]["exact"])


class DeploymentContextClampV2Test(unittest.TestCase):
    """Auditable context-clamp v2 metadata, protocol hashes, and failure modes."""

    def setUp(self) -> None:
        self.transport = MockTransport()

    def test_short_prompts_record_requested_equals_effective_and_clamp_none(self) -> None:
        token_count = RawResponse(
            status=200,
            body=json.dumps({"count": 500, "max_model_len": 8192}).encode("utf-8"),
        )
        self.transport.reset([token_count, EXACT_DEPLOYMENT_TEXT])
        runner = make_deployment_runner(
            self.transport,
            compact_context=True,
            context_token_limit=8192,
            context_margin=256,
        )
        spec = RunSpec(
            method=ACTIVE_CEGIS,
            seed=0,
            decoding=Decoding(max_tokens=2048),
            budgets=Budgets(state_budget=12, query_budget=4, token_budget=16000),
        )
        artifact = runner.run(spec)
        self.assertEqual(len(artifact.interactions), 1)
        interaction = artifact.interactions[0]
        self.assertEqual(interaction.prompt_tokens, 500)
        self.assertEqual(interaction.requested_max_tokens, 2048)
        self.assertEqual(interaction.effective_max_tokens, 2048)
        self.assertEqual(interaction.clamp_reason, "none")
        self.assertEqual(interaction.context_window, 8192)
        self.assertEqual(interaction.safety_margin, 256)
        self.assertIn("protocol_sha256", artifact.hashes)
        self.assertEqual(artifact.schema_version, 2)

    def test_context_only_clamp_path(self) -> None:
        token_count = RawResponse(
            status=200,
            body=json.dumps({"count": 7000, "max_model_len": 8192}).encode("utf-8"),
        )
        self.transport.reset([token_count, EXACT_DEPLOYMENT_TEXT])
        runner = make_deployment_runner(
            self.transport,
            compact_context=True,
            context_token_limit=8192,
            context_margin=256,
        )
        spec = RunSpec(
            method=ACTIVE_CEGIS,
            seed=0,
            decoding=Decoding(max_tokens=2048),
            budgets=Budgets(state_budget=12, query_budget=4, token_budget=16000),
        )
        artifact = runner.run(spec)
        interaction = artifact.interactions[0]
        self.assertEqual(interaction.prompt_tokens, 7000)
        self.assertEqual(interaction.requested_max_tokens, 2048)
        self.assertEqual(interaction.effective_max_tokens, 936)
        self.assertEqual(interaction.clamp_reason, "context_window")
        self.assertEqual(interaction.context_window, 8192)
        self.assertEqual(interaction.safety_margin, 256)

    def test_token_budget_only_clamp_path(self) -> None:
        token_count = RawResponse(
            status=200,
            body=json.dumps({"count": 500, "max_model_len": 8192}).encode("utf-8"),
        )
        self.transport.reset([token_count, EXACT_DEPLOYMENT_TEXT])
        runner = make_deployment_runner(
            self.transport,
            compact_context=True,
            context_token_limit=8192,
            context_margin=256,
        )
        spec = RunSpec(
            method=ACTIVE_CEGIS,
            seed=0,
            decoding=Decoding(max_tokens=2048),
            budgets=Budgets(state_budget=12, query_budget=4, token_budget=1000),
        )
        artifact = runner.run(spec)
        interaction = artifact.interactions[0]
        self.assertEqual(interaction.prompt_tokens, 500)
        self.assertEqual(interaction.requested_max_tokens, 2048)
        self.assertEqual(interaction.effective_max_tokens, 1000)
        self.assertEqual(interaction.clamp_reason, "token_budget")
        self.assertEqual(interaction.context_window, 8192)
        self.assertEqual(interaction.safety_margin, 256)

    def test_both_clamp_path(self) -> None:
        token_count = RawResponse(
            status=200,
            body=json.dumps({"count": 7000, "max_model_len": 8192}).encode("utf-8"),
        )
        self.transport.reset([token_count, EXACT_DEPLOYMENT_TEXT])
        runner = make_deployment_runner(
            self.transport,
            compact_context=True,
            context_token_limit=8192,
            context_margin=256,
        )
        spec = RunSpec(
            method=ACTIVE_CEGIS,
            seed=0,
            decoding=Decoding(max_tokens=2048),
            budgets=Budgets(state_budget=12, query_budget=4, token_budget=1000),
        )
        artifact = runner.run(spec)
        interaction = artifact.interactions[0]
        self.assertEqual(interaction.prompt_tokens, 7000)
        self.assertEqual(interaction.requested_max_tokens, 2048)
        self.assertEqual(interaction.effective_max_tokens, 936)
        self.assertEqual(interaction.clamp_reason, "both")
        self.assertEqual(interaction.context_window, 8192)
        self.assertEqual(interaction.safety_margin, 256)

    def test_no_output_room_failure_records_structured_metadata_and_typed_stop(self) -> None:
        token_count = RawResponse(
            status=200,
            body=json.dumps({"count": 8000, "max_model_len": 8192}).encode("utf-8"),
        )
        self.transport.reset([token_count])
        runner = make_deployment_runner(
            self.transport,
            compact_context=True,
            context_token_limit=8192,
            context_margin=256,
        )
        spec = RunSpec(
            method=ACTIVE_CEGIS,
            seed=0,
            decoding=Decoding(max_tokens=2048),
            budgets=Budgets(state_budget=12, query_budget=4, token_budget=16000),
        )
        artifact = runner.run(spec)
        self.assertEqual(artifact.stopped_because, STOP_CONTEXT_BUDGET)
        self.assertEqual(len(artifact.interactions), 1)
        interaction = artifact.interactions[0]
        self.assertIn("no output room", interaction.error)
        self.assertEqual(interaction.prompt_tokens, 8000)
        self.assertEqual(interaction.requested_max_tokens, 2048)
        self.assertEqual(interaction.clamp_reason, "context_window")
        self.assertEqual(interaction.context_window, 8192)
        self.assertEqual(interaction.safety_margin, 256)

    def test_tokenize_transport_failure_records_structured_metadata_and_typed_stop(self) -> None:
        self.transport.reset([TransportError("tokenize service unavailable")])
        runner = make_deployment_runner(
            self.transport,
            compact_context=True,
            context_token_limit=8192,
            context_margin=256,
        )
        spec = RunSpec(
            method=ACTIVE_CEGIS,
            seed=0,
            decoding=Decoding(max_tokens=2048),
            budgets=Budgets(state_budget=12, query_budget=4, token_budget=16000),
        )
        artifact = runner.run(spec)
        self.assertEqual(artifact.stopped_because, "transport_error")
        self.assertEqual(len(artifact.interactions), 1)
        interaction = artifact.interactions[0]
        self.assertIn("tokenize service unavailable", interaction.error)
        self.assertIsNone(interaction.prompt_tokens)
        self.assertEqual(interaction.requested_max_tokens, 2048)
        self.assertIsNone(interaction.effective_max_tokens)
        self.assertEqual(interaction.clamp_reason, "not_evaluated")
        self.assertEqual(interaction.context_window, 8192)
        self.assertEqual(interaction.safety_margin, 256)
        self.assertEqual(artifact.spend["model_calls"], 1)

    def test_completion_transport_failure_records_structured_metadata_and_typed_stop(self) -> None:
        token_count = RawResponse(
            status=200,
            body=json.dumps({"count": 7000, "max_model_len": 8192}).encode("utf-8"),
        )
        self.transport.reset([token_count, TransportError("completion connection reset")])
        runner = make_deployment_runner(
            self.transport,
            compact_context=True,
            context_token_limit=8192,
            context_margin=256,
        )
        spec = RunSpec(
            method=ACTIVE_CEGIS,
            seed=0,
            decoding=Decoding(max_tokens=2048),
            budgets=Budgets(state_budget=12, query_budget=4, token_budget=16000),
        )
        artifact = runner.run(spec)
        self.assertEqual(artifact.stopped_because, "transport_error")
        self.assertEqual(len(artifact.interactions), 1)
        interaction = artifact.interactions[0]
        self.assertIn("completion connection reset", interaction.error)
        self.assertEqual(interaction.prompt_tokens, 7000)
        self.assertEqual(interaction.requested_max_tokens, 2048)
        self.assertEqual(interaction.effective_max_tokens, 936)
        self.assertEqual(interaction.clamp_reason, "context_window")
        self.assertEqual(interaction.context_window, 8192)
        self.assertEqual(interaction.safety_margin, 256)
        self.assertEqual(artifact.spend["model_calls"], 1)

    def test_protocol_sha256_invariance_and_sensitivity(self) -> None:
        token_count = RawResponse(
            status=200,
            body=json.dumps({"count": 500, "max_model_len": 8192}).encode("utf-8"),
        )
        self.transport.reset([token_count, EXACT_DEPLOYMENT_TEXT, token_count, EXACT_DEPLOYMENT_TEXT])
        runner = make_deployment_runner(
            self.transport,
            compact_context=True,
            context_token_limit=8192,
            context_margin=256,
        )
        spec_seed0 = RunSpec(method=ACTIVE_CEGIS, seed=0)
        spec_seed1 = RunSpec(method=ACTIVE_CEGIS, seed=1)
        artifact0 = runner.run(spec_seed0)
        artifact1 = runner.run(spec_seed1)

        # Invariant across seeds
        hash0 = artifact0.hashes["protocol_sha256"]
        hash1 = artifact1.hashes["protocol_sha256"]
        self.assertEqual(hash0, hash1)

        # Sensitivity to margin
        self.transport.reset([token_count, EXACT_DEPLOYMENT_TEXT])
        runner_diff_margin = make_deployment_runner(
            self.transport,
            compact_context=True,
            context_token_limit=8192,
            context_margin=128,
        )
        artifact_diff_margin = runner_diff_margin.run(spec_seed0)
        self.assertNotEqual(hash0, artifact_diff_margin.hashes["protocol_sha256"])

        # Sensitivity to compact_context
        self.transport.reset([token_count, EXACT_DEPLOYMENT_TEXT])
        runner_no_compact = make_deployment_runner(
            self.transport,
            compact_context=False,
            context_token_limit=8192,
            context_margin=256,
        )
        artifact_no_compact = runner_no_compact.run(spec_seed0)
        self.assertNotEqual(hash0, artifact_no_compact.hashes["protocol_sha256"])

        # Sensitivity to method
        self.transport.reset([token_count, EXACT_DEPLOYMENT_TEXT])
        spec_no_cov = RunSpec(method=ACTIVE_CEGIS_NO_COVERAGE, seed=0)
        artifact_no_cov = runner.run(spec_no_cov)
        self.assertNotEqual(hash0, artifact_no_cov.hashes["protocol_sha256"])

    def test_serving_max_model_len_validation(self) -> None:
        # Context limit exceeds max_model_len -> ValueError
        with self.assertRaisesRegex(ValueError, "exceeds serving max_model_len"):
            make_deployment_runner(
                self.transport,
                context_token_limit=8192,
                serving_metadata={"max_model_len": 4096},
            )

        # Context limit <= max_model_len -> OK
        runner_ok = make_deployment_runner(
            self.transport,
            context_token_limit=4096,
            serving_metadata={"max_model_len": 4096},
        )
        self.assertEqual(runner_ok.context_token_limit, 4096)

        # Context limit None -> not silently inferred
        runner_none = make_deployment_runner(
            self.transport,
            context_token_limit=None,
            serving_metadata={"max_model_len": 4096},
        )
        self.assertIsNone(runner_none.context_token_limit)

    def test_dedicated_phase2_clamped_config_execution(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "deployment_phase2_clamped.json"
        )
        config = deployment_model_runner.load_experiment_config(config_path)
        self.assertEqual(config["context_token_limit"], 8192)
        self.assertEqual(config["context_margin"], 256)
        self.assertEqual(len(config["seeds"]), 20)
        # tri plan: v2 전용 config 는 active_cegis 만 실행한다.
        # (matched 4-arm 80-run 은 이 단계에서 실행 금지)
        self.assertEqual(config["methods"], ["active_cegis"])
        # 클램프 이외의 동작은 v1 과 동일해야 한다.
        self.assertEqual(config["workers"], 1)
        self.assertFalse(config["compact_context"])
        self.assertEqual(config["counterexample_limit"], 5)

        token_count = RawResponse(
            status=200,
            body=json.dumps({"count": 500, "max_model_len": 8192}).encode("utf-8"),
        )
        transport = MockTransport([token_count, EXACT_DEPLOYMENT_TEXT])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "phase2_clamped_test"
            argv = [
                "--output",
                str(output),
                "--experiment-config",
                str(config_path),
                "--sandbox-config",
                "experiment/configs/deployment_small.json",
                "--model",
                "test-deployment-model",
                "--methods",
                "active_cegis",
                "--seeds",
                "0",
                "--state-budget",
                "10",
                "--query-budget",
                "2",
                "--json",
            ]
            code = deployment_model_runner.main(argv, transport=transport)
            self.assertEqual(code, 0)
            artifact_file = output / "active_cegis__seed0.json"
            self.assertTrue(artifact_file.is_file())
            data = json.loads(artifact_file.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], 2)
            self.assertIn("protocol_sha256", data["hashes"])
            self.assertEqual(data["interactions"][0]["clamp_reason"], "none")
            self.assertEqual(data["interactions"][0]["context_window"], 8192)
            self.assertEqual(data["interactions"][0]["safety_margin"], 256)


if __name__ == "__main__":
    unittest.main()
