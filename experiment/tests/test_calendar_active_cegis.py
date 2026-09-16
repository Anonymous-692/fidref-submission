#!/usr/bin/env python3
"""Tests for active CEGIS and model methods on the calendar workspace domain.

Runs against mocked HTTP transports; no network calls or live model serving.
"""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..calendar import (
    ActionKind,
    CalendarAction,
    CalendarConfig,
    CalendarExperimentRunner,
    CalendarState,
    ExternalBooking,
    MeetingSnapshot,
    dsl,
    enumerate_reachable,
    evaluate_contract,
    initial_state,
    parse_contract,
    schedule_meeting_postcondition,
    schedule_meeting_precondition,
)
from ..calendar import model_runner as calendar_model_runner
from ..calendar import prompts as calendar_prompts
from ..calendar.runner import (
    REFINEMENT_PROTOCOL_LEGACY_V2,
    REFINEMENT_PROTOCOL_SELF_CONTAINED_V3,
    STOP_CONTEXT_BUDGET,
    STOP_DUPLICATE_OUTPUT,
    STOP_QUERY_BUDGET,
    STOP_SAMPLE_SATISFIED,
    STOP_STATE_BUDGET,
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
)


class CalendarMockTransport:
    """Queued fake responses for ChatClient without HTTP calls."""

    def __init__(self, script: Sequence[Any] = ()) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def reset(self, script: Sequence[Any] = ()) -> "CalendarMockTransport":
        self.calls.clear()
        self.script = list(script)
        return self

    def __call__(
        self,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> RawResponse:
        self.calls.append({
            "url": url,
            "headers": dict(headers),
            "payload": json.loads(body.decode("utf-8")),
            "timeout": timeout,
        })
        if url.endswith("/tokenize"):
            return RawResponse(status=200, body=json.dumps({"count": 500, "max_model_len": 8192}).encode("utf-8"))

        if not self.script:
            raise TransportError("no more mock responses")
        next_resp = self.script.pop(0)
        if isinstance(next_resp, BaseException):
            raise next_resp
        if isinstance(next_resp, RawResponse):
            return next_resp
        if callable(next_resp):
            return next_resp(self.calls[-1])

        payload = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1756684800,
            "model": "test-calendar-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": str(next_resp)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
        }
        return RawResponse(status=200, body=json.dumps(payload).encode("utf-8"))


def _make_ground_truth_contract_dict() -> dict[str, Any]:
    return {
        "skill": "schedule_meeting",
        "name": "ground_truth_schedule_meeting",
        "precondition": {
            "op": "and",
            "args": [
                {"op": "is_true", "arg": {"var": "authenticated"}},
                {"op": "not", "arg": {"op": "is_true", "arg": {"var": "has_active_meeting"}}},
                {"op": "not", "arg": {"op": "is_empty", "arg": {"var": "draft_attendees"}}},
                {"op": "in_set", "value": {"var": "draft_slot"}, "set": "valid_slots"},
                {"op": "in_set", "value": {"var": "draft_room"}, "set": "valid_rooms"},
                {"op": "in_set", "value": {"var": "draft_type"}, "set": "valid_types"},
                {"op": "capacity_sufficient", "attendees": {"var": "draft_attendees"}, "room": {"var": "draft_room"}},
                {"op": "room_supports_type", "room": {"var": "draft_room"}, "type": {"var": "draft_type"}},
                {"op": "no_room_conflict", "slot": {"var": "draft_slot"}, "room": {"var": "draft_room"}},
                {"op": "no_attendee_conflict", "slot": {"var": "draft_slot"}, "attendees": {"var": "draft_attendees"}},
            ],
        },
        "postcondition": {
            "op": "and",
            "args": [
                {
                    "op": "scheduled_snapshot_matches",
                    "active_room": {"var": "active_room", "when": "after"},
                    "active_slot": {"var": "active_slot", "when": "after"},
                    "active_type": {"var": "active_type", "when": "after"},
                    "active_attendees": {"var": "active_attendees", "when": "after"},
                    "draft_room": {"var": "draft_room", "when": "before"},
                    "draft_slot": {"var": "draft_slot", "when": "before"},
                    "draft_type": {"var": "draft_type", "when": "before"},
                    "draft_attendees": {"var": "draft_attendees", "when": "before"},
                },
                {"op": "is_empty", "arg": {"var": "draft_attendees", "when": "after"}},
                {"op": "is_null", "arg": {"var": "draft_slot", "when": "after"}},
                {"op": "is_null", "arg": {"var": "draft_room", "when": "after"}},
                {"op": "is_null", "arg": {"var": "draft_type", "when": "after"}},
                {"op": "unchanged", "vars": ["authenticated", "external_booking"]},
            ],
        },
    }


class CalendarDslAndContractsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CalendarConfig()
        self.states = enumerate_reachable(self.config).states

    def test_dsl_parses_and_evaluates_exact_contract(self) -> None:
        contract_dict = _make_ground_truth_contract_dict()
        contract = parse_contract(contract_dict, self.config)
        report = evaluate_contract(contract, self.states, self.config)
        self.assertEqual(report.states_checked, 1676)
        self.assertTrue(report.exact)
        self.assertEqual(report.failures, 0)
        self.assertEqual(report.false_accepts, 0)
        self.assertEqual(report.false_rejects, 0)
        self.assertEqual(report.postcondition_violations, 0)

    def test_dsl_rejects_malformed_json_and_unsupported_ops(self) -> None:
        with self.assertRaises(dsl.DslError):
            parse_contract("not json at all", self.config)
        with self.assertRaises(dsl.DslError):
            parse_contract({"skill": "unknown_skill", "precondition": {}, "postcondition": {}}, self.config)
        with self.assertRaises(dsl.DslError):
            parse_contract({"skill": "schedule_meeting", "precondition": {"op": "unsupported_magic_op"}}, self.config)

    def test_dsl_rejects_attendee_list_as_in_set_value(self) -> None:
        with self.assertRaisesRegex(dsl.DslError, "option/text value"):
            parse_contract(
                {
                    "skill": "schedule_meeting",
                    "precondition": {
                        "op": "in_set",
                        "value": {"var": "draft_attendees", "when": "before"},
                        "set": "attendees",
                    },
                    "postcondition": {"op": "const", "value": True},
                },
                self.config,
            )

    def test_dsl_rejects_invalid_kind_combinations(self) -> None:
        # overlaps requires attendees
        with self.assertRaisesRegex(dsl.DslError, "requires attendees terms"):
            parse_contract({
                "skill": "schedule_meeting",
                "precondition": {"op": "overlaps", "left": {"var": "draft_slot"}, "right": {"var": "draft_room"}},
                "postcondition": {"op": "const", "value": True},
            }, self.config)

        # contains requires attendees container and option/text item
        with self.assertRaisesRegex(dsl.DslError, "requires attendees container"):
            parse_contract({
                "skill": "schedule_meeting",
                "precondition": {"op": "contains", "container": {"var": "draft_slot"}, "item": {"const": "morning"}},
                "postcondition": {"op": "const", "value": True},
            }, self.config)

        # capacity_sufficient requires attendees and option room
        with self.assertRaisesRegex(dsl.DslError, "requires attendees term"):
            parse_contract({
                "skill": "schedule_meeting",
                "precondition": {"op": "capacity_sufficient", "attendees": {"var": "draft_slot"}, "room": {"var": "draft_room"}},
                "postcondition": {"op": "const", "value": True},
            }, self.config)

        # room_supports_type requires option/text
        with self.assertRaisesRegex(dsl.DslError, "requires option/text terms"):
            parse_contract({
                "skill": "schedule_meeting",
                "precondition": {"op": "room_supports_type", "room": {"var": "draft_attendees"}, "type": {"var": "draft_type"}},
                "postcondition": {"op": "const", "value": True},
            }, self.config)

        # no_room_conflict requires option/text
        with self.assertRaisesRegex(dsl.DslError, "requires option/text slot and room"):
            parse_contract({
                "skill": "schedule_meeting",
                "precondition": {"op": "no_room_conflict", "slot": {"var": "draft_attendees"}, "room": {"var": "draft_room"}},
                "postcondition": {"op": "const", "value": True},
            }, self.config)

        # no_attendee_conflict requires option slot and attendees
        with self.assertRaisesRegex(dsl.DslError, "requires attendees term"):
            parse_contract({
                "skill": "schedule_meeting",
                "precondition": {"op": "no_attendee_conflict", "slot": {"var": "draft_slot"}, "attendees": {"var": "draft_room"}},
                "postcondition": {"op": "const", "value": True},
            }, self.config)

        # eq cannot compare incompatible kinds
        with self.assertRaisesRegex(dsl.DslError, "cannot compare"):
            parse_contract({
                "skill": "schedule_meeting",
                "precondition": {"op": "eq", "left": {"var": "authenticated"}, "right": {"var": "draft_slot"}},
                "postcondition": {"op": "const", "value": True},
            }, self.config)

    def test_prompt_example_is_parseable_and_not_ground_truth(self) -> None:
        example_json = calendar_prompts.example_contract_json()
        contract = parse_contract(example_json, self.config)
        report = evaluate_contract(contract, self.states, self.config)
        self.assertFalse(report.exact)
        self.assertGreater(report.failures, 0)

    def test_prompts_do_not_leak_hidden_ground_truth_logic(self) -> None:
        dp = calendar_prompts.direct_prompt(self.config)
        sp = calendar_prompts.system_prompt()
        self.assertNotIn("alpha_morning_alice", dp)
        self.assertNotIn("beta_afternoon_bob", dp)
        self.assertNotIn("clause_values", dp)
        self.assertNotIn("clause_values", sp)

    def test_v3_counterexample_prompt_is_self_contained(self) -> None:
        previous = json.dumps({
            "skill": "schedule_meeting",
            "notes": "previous-v3-candidate",
            "precondition": {"op": "is_true", "arg": {"var": "authenticated"}},
            "postcondition": {"op": "const", "value": True},
        })
        contract = parse_contract(previous, self.config)
        report = evaluate_contract(contract, self.states[:48], self.config)
        prompt = calendar_prompts.counterexample_prompt_v3(previous, report, self.config)

        self.assertIn(previous, prompt)
        self.assertIn(f"False accepts: {report.false_accepts}", prompt)
        self.assertIn(f"False rejects: {report.false_rejects}", prompt)
        self.assertIn(
            f"Postcondition violations: {report.postcondition_violations}",
            prompt,
        )
        self.assertIn("- Before:", prompt)
        self.assertIn("- After:", prompt)
        self.assertIn("Changed fields", prompt)
        self.assertIn("Unchanged fields", prompt)


class CalendarRunnerMockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CalendarConfig()
        self.gt_contract_json = json.dumps(_make_ground_truth_contract_dict())

    def _client(self, transport: CalendarMockTransport) -> ChatClient:
        return ChatClient(
            base_url="http://127.0.0.1:8000/v1",
            model="test-calendar-model",
            transport=transport,
        )

    def test_active_cegis_exact_outcome_and_zero_oracle_queries(self) -> None:
        transport = CalendarMockTransport([self.gt_contract_json])
        client = self._client(transport)
        runner = CalendarExperimentRunner(client=client, config=self.config, context_token_limit=8192)

        spec = RunSpec(
            method=ACTIVE_CEGIS,
            seed=0,
            decoding=Decoding(temperature=0.2, top_p=0.95, max_tokens=2048),
            budgets=Budgets(state_budget=48, query_budget=4, token_budget=16000),
        )
        artifact = runner.run_method(spec)

        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_EXACT)
        self.assertTrue(artifact.evaluation.exact)
        self.assertEqual(artifact.evaluation.state_count, 1676)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(len(artifact.interactions), 1)

        # Context clamp metadata recorded
        interaction = artifact.interactions[0]
        self.assertEqual(interaction.requested_max_tokens, 2048)
        self.assertEqual(interaction.effective_max_tokens, 2048)
        self.assertEqual(interaction.clamp_reason, "none")
        self.assertEqual(interaction.context_window, 8192)
        self.assertEqual(interaction.safety_margin, 256)

        # Protocol sha256 recorded in hashes
        self.assertIn("protocol_sha256", artifact.hashes)
        self.assertEqual(len(artifact.hashes["protocol_sha256"]), 64)

    def test_multi_round_active_cegis_state_budget_and_query_budget_invariants(self) -> None:
        # Four distinct inaccurate contracts that each fail on some states
        c0 = json.dumps({"skill": "schedule_meeting", "notes": "c0", "precondition": {"op": "is_true", "arg": {"var": "authenticated"}}, "postcondition": {"op": "const", "value": True}})
        c1 = json.dumps({"skill": "schedule_meeting", "notes": "c1", "precondition": {"op": "in_set", "value": {"var": "draft_slot"}, "set": "valid_slots"}, "postcondition": {"op": "const", "value": True}})
        c2 = json.dumps({"skill": "schedule_meeting", "notes": "c2", "precondition": {"op": "in_set", "value": {"var": "draft_room"}, "set": "valid_rooms"}, "postcondition": {"op": "const", "value": True}})
        c3 = json.dumps({"skill": "schedule_meeting", "notes": "c3", "precondition": {"op": "in_set", "value": {"var": "draft_type"}, "set": "valid_types"}, "postcondition": {"op": "const", "value": True}})

        transport = CalendarMockTransport([c0, c1, c2, c3])
        client = self._client(transport)
        runner = CalendarExperimentRunner(client=client, config=self.config, context_token_limit=8192)

        spec = RunSpec(
            method=ACTIVE_CEGIS,
            seed=0,
            decoding=Decoding(temperature=0.2, top_p=0.95, max_tokens=2048),
            budgets=Budgets(state_budget=48, query_budget=4, token_budget=16000),
        )
        artifact = runner.run_method(spec)

        # 1. Exactly 4 model calls made under query_budget=4
        self.assertEqual(artifact.spend["model_calls"], 4)
        self.assertEqual(artifact.spend["remaining_calls"], 0)
        self.assertEqual(artifact.stopped_because, STOP_QUERY_BUDGET)

        # 2. Total unique states observed must not exceed state_budget (48)
        observed_states = artifact.spend["states_observed"]
        self.assertLessEqual(observed_states, 48)
        self.assertEqual(observed_states, 48)  # 16 + 16 + 16 = 48
        self.assertEqual(artifact.spend["remaining_states"], 0)

        # 3. Verify disjointness of fresh states across rounds and cumulative tracking
        fresh_counts = [r["fresh_states"] for r in artifact.rounds]
        cumulative_counts = [r["cumulative_states"] for r in artifact.rounds]
        self.assertEqual(sum(fresh_counts), observed_states)
        self.assertEqual(cumulative_counts, [16, 32, 48, 48])

        # 4. Total sampled checks can exceed states_observed because audited states are re-checked
        self.assertGreater(artifact.spend["sampled_states_checked"], observed_states)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)

    def test_v3_revision_includes_previous_parsed_contract_and_changes_protocol_hash(self) -> None:
        inaccurate = json.dumps({
            "skill": "schedule_meeting",
            "notes": "previous-v3-candidate",
            "precondition": {"op": "is_true", "arg": {"var": "authenticated"}},
            "postcondition": {"op": "const", "value": True},
        })
        spec = RunSpec(
            method=ACTIVE_CEGIS,
            seed=0,
            decoding=Decoding(temperature=0.2, top_p=0.95, max_tokens=2048),
            budgets=Budgets(state_budget=48, query_budget=2, token_budget=16000),
        )

        legacy_transport = CalendarMockTransport([inaccurate, self.gt_contract_json])
        legacy_runner = CalendarExperimentRunner(
            client=self._client(legacy_transport),
            config=self.config,
            context_token_limit=8192,
            refinement_protocol=REFINEMENT_PROTOCOL_LEGACY_V2,
        )
        legacy_artifact = legacy_runner.run_method(spec)

        v3_transport = CalendarMockTransport([inaccurate, self.gt_contract_json])
        v3_runner = CalendarExperimentRunner(
            client=self._client(v3_transport),
            config=self.config,
            context_token_limit=8192,
            refinement_protocol=REFINEMENT_PROTOCOL_SELF_CONTAINED_V3,
        )
        v3_artifact = v3_runner.run_method(spec)

        legacy_revision = legacy_artifact.interactions[1].messages[-1]["content"]
        v3_revision = v3_artifact.interactions[1].messages[-1]["content"]
        self.assertNotIn(inaccurate, legacy_revision)
        self.assertIn(inaccurate, v3_revision)
        self.assertNotEqual(
            legacy_artifact.hashes["protocol_sha256"],
            v3_artifact.hashes["protocol_sha256"],
        )
        self.assertEqual(v3_artifact.spend["oracle_feedback_queries"], 0)

    def test_zero_state_budget_stops_immediately(self) -> None:
        transport = CalendarMockTransport([])
        client = self._client(transport)
        runner = CalendarExperimentRunner(client=client, config=self.config, context_token_limit=8192)

        spec = RunSpec(
            method=ACTIVE_CEGIS,
            seed=0,
            decoding=Decoding(temperature=0.2, top_p=0.95, max_tokens=2048),
            budgets=Budgets(state_budget=0, query_budget=4, token_budget=16000),
        )
        artifact = runner.run_method(spec)
        self.assertEqual(artifact.stopped_because, STOP_STATE_BUDGET)
        self.assertEqual(artifact.spend["model_calls"], 0)
        self.assertEqual(artifact.spend["states_observed"], 0)

    def test_duplicate_output_guard(self) -> None:
        inaccurate_contract = json.dumps({
            "skill": "schedule_meeting",
            "precondition": {"op": "is_true", "arg": {"var": "authenticated"}},
            "postcondition": {"op": "const", "value": True},
        })
        transport = CalendarMockTransport([
            inaccurate_contract,
            inaccurate_contract,
        ])
        client = self._client(transport)
        runner = CalendarExperimentRunner(client=client, config=self.config, context_token_limit=8192)

        spec = RunSpec(
            method=ACTIVE_CEGIS,
            seed=0,
            decoding=Decoding(temperature=0.2, top_p=0.95, max_tokens=2048),
            budgets=Budgets(state_budget=48, query_budget=4, token_budget=16000),
        )
        artifact = runner.run_method(spec)
        self.assertEqual(artifact.stopped_because, STOP_DUPLICATE_OUTPUT)

    def test_vacuous_candidate_guard(self) -> None:
        vacuous_contract_1 = json.dumps({
            "skill": "schedule_meeting",
            "notes": "attempt 1",
            "precondition": {"op": "const", "value": False},
            "postcondition": {"op": "const", "value": True},
        })
        vacuous_contract_2 = json.dumps({
            "skill": "schedule_meeting",
            "notes": "attempt 2",
            "precondition": {"op": "const", "value": False},
            "postcondition": {"op": "const", "value": True},
        })
        transport = CalendarMockTransport([
            vacuous_contract_1,
            vacuous_contract_2,
        ])
        client = self._client(transport)
        runner = CalendarExperimentRunner(client=client, config=self.config, vacuity_revision_limit=1, context_token_limit=8192)

        spec = RunSpec(
            method=ACTIVE_CEGIS,
            seed=0,
            decoding=Decoding(temperature=0.2, top_p=0.95, max_tokens=2048),
            budgets=Budgets(state_budget=48, query_budget=4, token_budget=16000),
        )
        artifact = runner.run_method(spec)
        self.assertEqual(artifact.stopped_because, STOP_VACUOUS_CANDIDATE)


class CalendarModelRunnerIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CalendarConfig()
        self.gt_contract_json = json.dumps(_make_ground_truth_contract_dict())

    def test_execute_spec_and_summary_generation_without_overwrite(self) -> None:
        transport = CalendarMockTransport([
            self.gt_contract_json,
            self.gt_contract_json,
        ])

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "calendar_pilot"
            args = argparse.Namespace(
                output=out_path,
                sandbox_config=Path("experiment/configs/calendar_default.json"),
                base_url="http://127.0.0.1:8000/v1",
                model="test-calendar-model",
                methods=["direct", "active_cegis"],
                seeds=[0],
                temperature=0.2,
                top_p=0.95,
                max_tokens=2048,
                state_budget=48,
                query_budget=2,
                token_budget=16000,
                max_depth=20,
                max_states=15000,
                timeout=60.0,
                workers=1,
                compact_context=True,
                context_token_limit=8192,
                context_margin=256,
                counterexample_limit=3,
                guided_json=False,
                api_key=None,
                allow_remote_host=False,
                json=False,
                wait_server=0.0,
            )
            summary = calendar_model_runner.execute_from_args(
                args=args,
                transport=transport,
            )
            self.assertIn("context", summary)
            self.assertIn("runs", summary)
            self.assertEqual(len(summary["runs"]), 2)
            self.assertTrue(all(r["exact"] for r in summary["runs"]))

            # Verify files on disk
            direct_file = out_path / "direct__seed0.json"
            active_file = out_path / "active_cegis__seed0.json"
            summary_file = out_path / "summary.json"
            self.assertTrue(direct_file.exists())
            self.assertTrue(active_file.exists())
            self.assertTrue(summary_file.exists())

            # Test overwrite protection: running again does not re-query mock transport
            empty_transport = CalendarMockTransport([])
            summary_rerun = calendar_model_runner.execute_from_args(
                args=args,
                transport=empty_transport,
            )
            self.assertEqual(len(summary_rerun["runs"]), 2)


if __name__ == "__main__":
    unittest.main()
