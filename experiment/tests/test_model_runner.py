#!/usr/bin/env python3
"""Tests for the model experiment runner.

Every test here runs against a mocked HTTP transport. No test opens a socket:
:class:`NoLiveServerTest` proves it by making ``urllib.request.urlopen`` raise
for the duration of a full CLI run.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from unittest import mock

from .. import model_runner, modeling
from ..environment import (
    OrderStatus,
    ShoppingState,
    apply_action,
    counts_from_mapping,
    enumerate_reachable,
)
from ..environment import ShoppingAction
from ..evaluation import Symptom, evaluate_contract
from ..modeling import artifacts as artifacts_module
from ..modeling import client as client_module
from ..modeling import dsl, prompts, runner as runner_module
from ..modeling.client import ChatClient, RawResponse, TransportError
from ..modeling.runner import Budgets, Decoding, ExperimentRunner, RunSpec
from .support import SMALL_CONFIG_PATH, TEST_MAX_DEPTH, TEST_MAX_STATES, small_config

# The scoring bounds used throughout: deep and wide enough to close the small
# configuration, so every score below is over the whole reachable set.
SCORING = {"max_depth": TEST_MAX_DEPTH, "max_states": TEST_MAX_STATES}


# --------------------------------------------------------------------------
# Fixtures: contracts in the DSL, and a transport that never touches a socket
# --------------------------------------------------------------------------


def contract_spec(*, require_address: bool = True, notes: str = "fixture") -> dict[str, Any]:
    """The reference behaviour of ``place_order``, written in the DSL.

    With ``require_address=False`` the shipping-address clause is dropped, which
    reproduces the ``too_weak_precondition`` defect: the contract then admits
    states the sandbox refuses.
    """
    precondition_args: list[dict[str, Any]] = [
        {"op": "is_true", "arg": {"var": "logged_in"}},
        {"op": "eq", "left": {"var": "order_status"}, "right": {"const": "none"}},
        {"op": "not", "arg": {"op": "is_empty", "arg": {"var": "cart"}}},
        {"op": "in_set", "value": {"var": "payment_method"}, "set": "valid_payment_methods"},
        {"op": "covers", "available": {"var": "stock"}, "required": {"var": "cart"}},
    ]
    if require_address:
        precondition_args.insert(
            4, {"op": "in_set", "value": {"var": "shipping_address"}, "set": "valid_addresses"}
        )
    return {
        "skill": "place_order",
        "notes": notes,
        "precondition": {"op": "and", "args": precondition_args},
        "postcondition": {
            "op": "and",
            "args": [
                {
                    "op": "eq",
                    "left": {"var": "order_status", "when": "after"},
                    "right": {"const": "placed"},
                },
                {
                    "op": "eq",
                    "left": {"var": "order_items", "when": "after"},
                    "right": {"var": "cart"},
                },
                {"op": "is_empty", "arg": {"var": "cart", "when": "after"}},
                {
                    "op": "eq",
                    "left": {"var": "stock", "when": "after"},
                    "right": {
                        "op": "difference",
                        "left": {"var": "stock"},
                        "right": {"var": "cart"},
                    },
                },
                {"op": "unchanged", "vars": ["logged_in", "payment_method", "shipping_address"]},
            ],
        },
    }


EXACT_TEXT = json.dumps(contract_spec())
WEAK_TEXT = json.dumps(contract_spec(require_address=False, notes="missing address clause"))
FENCED_EXACT_TEXT = "Sure, here is the contract:\n\n```json\n" + EXACT_TEXT + "\n```\n\nDone."

# An answer that tries to smuggle code through the DSL. It must be reported as a
# parse failure; nothing in it is ever executed.
UNSAFE_TEXT = json.dumps(
    {
        "skill": "place_order",
        "precondition": {"op": "python", "code": "__import__('os').system('echo pwned')"},
        "postcondition": {"op": "const", "value": True},
    }
)


def completion_body(
    text: str,
    *,
    prompt_tokens: int = 120,
    completion_tokens: int = 60,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """A minimal but realistic OpenAI-compatible chat completion payload."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1756684800,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def ok_response(text: str, **kwargs: Any) -> RawResponse:
    body = json.dumps(completion_body(text, **kwargs)).encode("utf-8")
    return RawResponse(status=200, body=body)


class MockTransport:
    """A scripted stand-in for the HTTP transport.

    It records every request it is handed and replays scripted answers. A script
    item may be a string (turned into a successful completion), a
    :class:`RawResponse`, an exception to raise, or a callable taking the decoded
    request payload.
    """

    def __init__(self, script: Sequence[Any] = ()) -> None:
        self.calls: list[dict[str, Any]] = []
        self.script: list[Any] = list(script)

    def reset(self, script: Sequence[Any] = ()) -> "MockTransport":
        self.calls.clear()
        self.script = list(script)
        return self

    @property
    def payloads(self) -> list[dict[str, Any]]:
        return [call["payload"] for call in self.calls]

    def last_user_message(self, index: int = -1) -> str:
        messages = self.payloads[index]["messages"]
        user_turns = [message for message in messages if message["role"] == "user"]
        return user_turns[-1]["content"]

    def __call__(
        self, url: str, headers: Mapping[str, str], body: bytes, timeout: float
    ) -> RawResponse:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": json.loads(body.decode("utf-8")),
                "timeout": timeout,
            }
        )
        if not self.script:
            raise AssertionError("the transport was called more often than the script allows")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, RawResponse):
            return item
        if callable(item):
            return item(self.calls[-1])
        return ok_response(item)


def make_client(transport: MockTransport, **kwargs: Any) -> ChatClient:
    options: dict[str, Any] = {"model": "test-model", "transport": transport}
    options.update(kwargs)
    return ChatClient(**options)


def make_runner(transport: MockTransport, **kwargs: Any) -> ExperimentRunner:
    return ExperimentRunner(
        small_config(),
        make_client(transport),
        config_path=str(SMALL_CONFIG_PATH),
        **SCORING,
        **kwargs,
    )


def spec(method: str, **kwargs: Any) -> RunSpec:
    budgets = kwargs.pop("budgets", Budgets(state_budget=6, query_budget=4, token_budget=16000))
    decoding = kwargs.pop("decoding", Decoding(temperature=0.0, top_p=1.0, max_tokens=512))
    return RunSpec(method=method, decoding=decoding, budgets=budgets, **kwargs)


# --------------------------------------------------------------------------
# The DSL
# --------------------------------------------------------------------------


class ContractDslTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = small_config()
        cls.states = enumerate_reachable(cls.config, **SCORING).states

    def score(self, text: str):
        parsed = dsl.parse_contract_text(text)
        return evaluate_contract(parsed.bind(self.config), self.states, self.config)

    def test_the_reference_contract_written_in_the_dsl_is_exact(self) -> None:
        report = self.score(EXACT_TEXT)
        self.assertGreater(report.successes, 0)
        self.assertGreater(report.failures, 0)
        self.assertTrue(report.is_exact, msg=report.summary())
        self.assertEqual(report.counterexamples, ())

    def test_dropping_a_clause_shows_up_as_a_false_accept(self) -> None:
        report = self.score(WEAK_TEXT)
        self.assertEqual(report.symptoms, frozenset({Symptom.FALSE_ACCEPT}))
        self.assertGreater(report.false_accepts, 0)
        self.assertEqual(report.false_rejects, 0)
        self.assertEqual(report.postcondition_violations, 0)

    def test_json_is_extracted_from_prose_and_code_fences(self) -> None:
        parsed = dsl.parse_contract_text(FENCED_EXACT_TEXT)
        self.assertEqual(parsed.spec, json.loads(EXACT_TEXT))

    def test_extraction_ignores_braces_inside_strings(self) -> None:
        text = '{"skill": "place_order", "notes": "a } brace", "precondition": ' \
               '{"op": "const", "value": true}, "postcondition": {"op": "const", "value": true}} tail'
        parsed = dsl.parse_contract_text(text)
        self.assertEqual(parsed.spec["notes"], "a } brace")

    def test_bound_contract_targets_place_order(self) -> None:
        contract = dsl.parse_contract_text(EXACT_TEXT).bind(self.config)
        self.assertEqual(contract.action, ShoppingAction.place_order())

    def test_a_precondition_may_not_read_the_after_state(self) -> None:
        spec_dict = contract_spec()
        spec_dict["precondition"] = {"op": "is_true", "arg": {"var": "logged_in", "when": "after"}}
        with self.assertRaisesRegex(dsl.DslError, "may not read"):
            dsl.parse_contract(spec_dict)

    def test_unchanged_needs_a_postcondition(self) -> None:
        spec_dict = contract_spec()
        spec_dict["precondition"] = {"op": "unchanged", "vars": ["cart"]}
        with self.assertRaisesRegex(dsl.DslError, "postcondition"):
            dsl.parse_contract(spec_dict)

    def test_unsafe_and_malformed_answers_are_rejected(self) -> None:
        cases = {
            "no json at all": "I think the precondition is that you must be logged in.",
            "unterminated object": '{"skill": "place_order", "precondition": {',
            "not json": "{this is not json}",
            "smuggled code": UNSAFE_TEXT,
            "python source": '{"precondition": "lambda s: s.logged_in", "postcondition": {"op": "const", "value": true}}',
            "unknown operator": '{"precondition": {"op": "exec", "args": []}, "postcondition": {"op": "const", "value": true}}',
            "missing postcondition": '{"skill": "place_order", "precondition": {"op": "const", "value": true}}',
            "unknown top-level key": '{"precondition": {"op": "const", "value": true}, '
            '"postcondition": {"op": "const", "value": true}, "callback": "http://x"}',
            "wrong skill": '{"skill": "drop_database", "precondition": {"op": "const", "value": true}, '
            '"postcondition": {"op": "const", "value": true}}',
            "unknown variable": '{"precondition": {"op": "is_true", "arg": {"var": "is_admin"}}, '
            '"postcondition": {"op": "const", "value": true}}',
            "unknown config set": '{"precondition": {"op": "in_set", "value": {"var": "payment_method"}, '
            '"set": "secrets"}, "postcondition": {"op": "const", "value": true}}',
            "kind mismatch": '{"precondition": {"op": "eq", "left": {"var": "cart"}, "right": {"const": 1}}, '
            '"postcondition": {"op": "const", "value": true}}',
            "term used as formula": '{"precondition": {"op": "total", "of": {"var": "cart"}}, '
            '"postcondition": {"op": "const", "value": true}}',
            "float constant": '{"precondition": {"op": "eq", "left": {"var": "cart_size"}, '
            '"right": {"const": 1.5}}, "postcondition": {"op": "const", "value": true}}',
            "negative multiset": '{"precondition": {"op": "covers", "available": {"var": "stock"}, '
            '"required": {"multiset": {"book": -1}}}, "postcondition": {"op": "const", "value": true}}',
            "contract is a list": "[1, 2, 3]",
        }
        for label, text in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(dsl.DslError):
                    dsl.parse_contract_text(text)

    def test_nesting_and_size_are_bounded(self) -> None:
        deep: dict[str, Any] = {"op": "const", "value": True}
        for _ in range(dsl.MAX_DEPTH + 2):
            deep = {"op": "not", "arg": deep}
        with self.assertRaisesRegex(dsl.DslError, "nests deeper"):
            dsl.parse_contract(
                {"precondition": deep, "postcondition": {"op": "const", "value": True}}
            )

        wide = {
            "op": "and",
            "args": [{"op": "const", "value": True} for _ in range(dsl.MAX_NODES + 5)],
        }
        with self.assertRaisesRegex(dsl.DslError, "more than"):
            dsl.parse_contract(
                {"precondition": wide, "postcondition": {"op": "const", "value": True}}
            )

    def test_a_term_without_a_value_makes_its_formula_false(self) -> None:
        # ``difference`` has no value when a line would go negative; the formula
        # must report False rather than raise in the middle of a sweep.
        parsed = dsl.parse_contract(
            {
                "precondition": {"op": "const", "value": True},
                "postcondition": {
                    "op": "eq",
                    "left": {"var": "stock", "when": "after"},
                    "right": {
                        "op": "difference",
                        "left": {"var": "stock"},
                        "right": {"var": "cart"},
                    },
                },
            }
        )
        contract = parsed.bind(self.config)
        state = ShoppingState(
            logged_in=True,
            cart=counts_from_mapping({"book": 2}),
            stock=counts_from_mapping({"book": 1}),
            payment_method="card",
            shipping_address="home",
            order_status=OrderStatus.NONE,
            order_items=(),
        )
        self.assertFalse(contract.transition_holds(state, state))

    def test_documented_operators_are_exactly_the_parsed_operators(self) -> None:
        # The prompt teaches the grammar from these tables, so they must not
        # drift away from what the parser actually accepts.
        self.assertEqual(set(dsl.FORMULA_OPS), set(dsl._FORMULA_PARSERS))
        self.assertEqual(set(dsl.TERM_OPS), set(dsl._TERM_PARSERS))
        self.assertEqual(set(dsl.TERM_LITERALS), set(dsl._TERM_LITERAL_PARSERS))
        self.assertEqual(set(dsl.STATE_VARS), set(dsl.VAR_KINDS))
        for name in dsl.CONFIG_SETS:
            self.assertIsInstance(getattr(self.config, name), tuple)


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------


class ChatClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = MockTransport()

    def test_the_default_endpoint_is_loopback_chat_completions(self) -> None:
        client = make_client(self.transport)
        self.assertEqual(client.base_url, modeling.DEFAULT_BASE_URL)
        self.assertEqual(client.endpoint, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertTrue(client_module.is_loopback(modeling.DEFAULT_BASE_URL))

    def test_a_remote_host_is_refused_unless_explicitly_allowed(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            make_client(self.transport, base_url="https://api.openai.com/v1")
        allowed = make_client(
            self.transport, base_url="https://api.openai.com/v1", allow_remote=True
        )
        self.assertFalse(client_module.is_loopback(allowed.base_url))

    def test_request_formation(self) -> None:
        client = make_client(self.transport, api_key="secret")
        self.transport.reset([EXACT_TEXT])
        response = client.complete(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            temperature=0.3,
            top_p=0.9,
            max_tokens=256,
            seed=11,
            stop=["</done>"],
        )
        self.assertEqual(len(self.transport.calls), 1)
        call = self.transport.calls[0]
        self.assertEqual(call["url"], "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(call["headers"]["Content-Type"], "application/json")
        self.assertEqual(call["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(
            call["payload"],
            {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ],
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 256,
                "stream": False,
                "seed": 11,
                "stop": ["</done>"],
            },
        )
        self.assertEqual(response.text, EXACT_TEXT)
        self.assertEqual(response.usage.total_tokens, 180)
        self.assertEqual(response.finish_reason, "stop")
        self.assertGreaterEqual(response.latency_s, 0.0)

    def test_optional_request_fields_are_omitted(self) -> None:
        client = make_client(self.transport)
        payload = client.build_request(
            [{"role": "user", "content": "hi"}], temperature=0.0, top_p=1.0, max_tokens=8
        )
        self.assertNotIn("seed", payload)
        self.assertNotIn("stop", payload)
        self.assertNotIn("Authorization", client.headers())

    def test_reasoning_controls_are_explicit_and_separately_accounted(self) -> None:
        client = make_client(
            self.transport,
            reasoning_effort="low",
            thinking_token_budget=1024,
            chat_template_kwargs={"enable_thinking": True},
            return_token_ids=True,
        )
        payload = client.build_request(
            [{"role": "user", "content": "hi"}],
            temperature=0.2,
            top_p=0.95,
            max_tokens=3072,
        )
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["thinking_token_budget"], 1024)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
        self.assertTrue(payload["return_token_ids"])

        usage = client_module.Usage.from_payload({
            "prompt_tokens": 100,
            "completion_tokens": 1500,
            "total_tokens": 1600,
            "completion_tokens_details": {"reasoning_tokens": 1026},
        })
        self.assertEqual(usage.to_dict()["reasoning_tokens"], 1026)
        self.assertEqual(usage.to_dict()["non_reasoning_completion_tokens"], 474)

    def test_invalid_reasoning_controls_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_client(self.transport, reasoning_effort="xhigh")
        with self.assertRaises(ValueError):
            make_client(self.transport, thinking_token_budget=-1)

    def test_old_vllm_guided_backend_options_are_sent_per_request(self) -> None:
        client = make_client(
            self.transport,
            structured_output_style=client_module.STRUCTURED_GUIDED_JSON,
            guided_decoding_backend="xgrammar:disable-any-whitespace",
        )
        payload = client.build_request(
            [{"role": "user", "content": "hi"}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=8,
            guided_json={"type": "object"},
        )
        self.assertEqual(payload["guided_json"], {"type": "object"})
        self.assertEqual(
            payload["guided_decoding_backend"],
            "xgrammar:disable-any-whitespace",
        )

    def test_request_formation_rejects_impossible_calls(self) -> None:
        client = make_client(self.transport)
        with self.assertRaises(ValueError):
            client.build_request([], temperature=0.0, top_p=1.0, max_tokens=8)
        with self.assertRaises(ValueError):
            client.build_request(
                [{"role": "user", "content": "hi"}], temperature=0.0, top_p=1.0, max_tokens=0
            )

    def test_latency_is_measured_with_the_injected_clock(self) -> None:
        ticks = iter([10.0, 10.25])
        client = make_client(self.transport, clock=lambda: next(ticks))
        self.transport.reset([EXACT_TEXT])
        response = client.complete([{"role": "user", "content": "hi"}])
        self.assertAlmostEqual(response.latency_s, 0.25)

    def test_unusable_responses_become_transport_errors(self) -> None:
        cases = {
            "http error": RawResponse(status=503, body=b"service unavailable"),
            "not json": RawResponse(status=200, body=b"<html>nope</html>"),
            "no choices": RawResponse(status=200, body=json.dumps({"choices": []}).encode()),
            "no message": RawResponse(
                status=200, body=json.dumps({"choices": [{"index": 0}]}).encode()
            ),
        }
        client = make_client(self.transport)
        for label, response in cases.items():
            with self.subTest(case=label):
                self.transport.reset([response])
                with self.assertRaises(TransportError):
                    client.complete([{"role": "user", "content": "hi"}])

    def test_missing_usage_counts_as_zero(self) -> None:
        body = completion_body(EXACT_TEXT)
        del body["usage"]
        self.transport.reset([RawResponse(status=200, body=json.dumps(body).encode())])
        response = make_client(self.transport).complete([{"role": "user", "content": "hi"}])
        self.assertEqual(response.usage.to_dict(), {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
        })

    def test_the_real_transport_maps_urlerror_to_transport_error(self) -> None:
        # The only test that touches urllib, and it never reaches the network:
        # urlopen is replaced by a failure.
        with mock.patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")
        ):
            with self.assertRaisesRegex(TransportError, "could not reach"):
                client_module.urllib_transport(
                    "http://127.0.0.1:8000/v1/chat/completions", {}, b"{}", 1.0
                )

    def test_urllib_is_the_default_transport(self) -> None:
        client = ChatClient(model="test-model")
        self.assertIs(client._transport, client_module.urllib_transport)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


class RepetitionPenaltyTest(unittest.TestCase):
    """`repetition_penalty` 는 설정한 실행에서만 나타나야 한다.

    기존 아티팩트의 `decoding_sha256` 이 바뀌면 provenance 대조가 깨지므로,
    설정하지 않았을 때의 해시를 실제 아티팩트 값으로 고정해 둔다.
    """

    # results/shopping_price_easy_v4_qwen14b_fullsuite_transitiondiff_v3_s20/
    #   direct__seed0.json 의 hashes.decoding_sha256 (2026-09-04 확인)
    EASY_V4_SEED0_DECODING_SHA256 = (
        "c97bd0aca759c8c6b1662f0a8d266687b354fa6fea88ed5802bcf6a8c8925106"
    )

    def test_an_unset_penalty_leaves_the_decoding_payload_and_hash_untouched(self) -> None:
        decoding = Decoding(temperature=0.2, top_p=0.95, max_tokens=2048)
        self.assertIsNone(decoding.repetition_penalty)
        self.assertEqual(
            decoding.to_dict(0),
            {"temperature": 0.2, "top_p": 0.95, "max_tokens": 2048, "seed": 0},
        )
        self.assertEqual(
            artifacts_module.sha256_json(decoding.to_dict(0)),
            self.EASY_V4_SEED0_DECODING_SHA256,
        )

    def test_a_set_penalty_is_recorded_and_changes_the_hash(self) -> None:
        plain = Decoding(temperature=0.2, top_p=0.95, max_tokens=2048)
        penalised = Decoding(
            temperature=0.2, top_p=0.95, max_tokens=2048, repetition_penalty=1.1
        )
        self.assertEqual(penalised.to_dict(0)["repetition_penalty"], 1.1)
        self.assertNotEqual(
            artifacts_module.sha256_json(plain.to_dict(0)),
            artifacts_module.sha256_json(penalised.to_dict(0)),
        )

    def test_the_request_carries_the_penalty_only_when_it_is_set(self) -> None:
        transport = MockTransport()
        client = make_client(transport)
        plain = client.build_request(
            [{"role": "user", "content": "hi"}],
            temperature=0.2,
            top_p=0.95,
            max_tokens=8,
        )
        self.assertNotIn("repetition_penalty", plain)

        penalised = client.build_request(
            [{"role": "user", "content": "hi"}],
            temperature=0.2,
            top_p=0.95,
            max_tokens=8,
            repetition_penalty=1.1,
        )
        self.assertEqual(penalised["repetition_penalty"], 1.1)

    def test_complete_forwards_the_penalty_to_the_transport(self) -> None:
        transport = MockTransport()
        client = make_client(transport)
        transport.reset([EXACT_TEXT])
        client.complete(
            [{"role": "user", "content": "hi"}],
            temperature=0.2,
            top_p=0.95,
            max_tokens=8,
            repetition_penalty=1.2,
        )
        self.assertEqual(transport.calls[0]["payload"]["repetition_penalty"], 1.2)


class PromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_config()

    def test_the_system_prompt_teaches_the_whole_grammar(self) -> None:
        text = prompts.system_prompt()
        for op in list(dsl.FORMULA_OPS) + list(dsl.TERM_OPS):
            self.assertIn(f'"{op}"', text, msg=op)
        for name in dsl.STATE_VARS:
            self.assertIn(name, text)
        for name in dsl.CONFIG_SETS:
            self.assertIn(name, text)

    def test_no_prompt_states_a_ground_truth_clause(self) -> None:
        from ..evaluation import GROUND_TRUTH_CLAUSES

        dummy_obs = {"before": "state_a", "ok": True, "error": None, "after": "state_b"}
        dummy_metrics = {
            "false_accepts": 0,
            "false_rejects": 0,
            "postcondition_violations": 0,
            "states_checked": 1,
        }
        dummy_cx = [{"symptom": "false_accept", "state": "state_a", "detail": "detail_a"}]
        texts = [
            prompts.system_prompt(),
            prompts.direct_prompt(self.config),
            prompts.critique_prompt("{}"),
            prompts.probe_prompt(self.config, [dummy_obs]),
            prompts.counterexample_prompt("{}", dummy_metrics, dummy_cx),
            prompts.asi_prompt(self.config, dummy_obs),
            prompts.skillcommit_prompt(self.config, [dummy_obs]),
            prompts.skillcommit_proposal_prompt(self.config, dummy_obs),
            prompts.skillcommit_revision_prompt("{}", dummy_metrics, dummy_cx),
            prompts.contractskill_repair_prompt("{}", dummy_metrics, dummy_cx),
        ]
        for text in texts:
            for clause in GROUND_TRUTH_CLAUSES:
                self.assertNotIn(clause, text, msg=clause)

    def test_the_briefing_describes_only_the_agent_facing_world(self) -> None:
        text = prompts.sandbox_briefing(self.config)
        self.assertIn("book", text)
        self.assertIn("card", text)
        self.assertIn("expired_card", text)
        self.assertIn("po_box", text)


# --------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------


class MethodSequenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.transport = MockTransport()
        cls.runner = make_runner(cls.transport)

    def setUp(self) -> None:
        self.transport.reset()

    def test_direct_makes_exactly_one_call_and_observes_nothing(self) -> None:
        self.transport.reset([EXACT_TEXT])
        artifact = self.runner.run(spec("direct"))
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual([entry.role for entry in artifact.interactions], ["propose"])
        self.assertEqual(artifact.spend["states_observed"], 0)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_EXACT)
        self.assertTrue(artifact.evaluation.exact)
        self.assertEqual(artifact.usage["calls"], 1)
        self.assertEqual(artifact.usage["total_tokens"], 180)

    def test_direct_prompt_carries_the_briefing_and_the_grammar(self) -> None:
        self.transport.reset([EXACT_TEXT])
        self.runner.run(spec("direct"))
        messages = self.transport.payloads[0]["messages"]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("contract DSL", messages[0]["content"])
        self.assertIn("Sandbox description", messages[1]["content"])
        self.assertEqual(self.transport.payloads[0]["model"], "test-model")

    def test_self_refine_is_one_critique_and_revision_pass(self) -> None:
        self.transport.reset([WEAK_TEXT, EXACT_TEXT])
        artifact = self.runner.run(spec("self_refine"))
        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual([entry.role for entry in artifact.interactions], ["propose", "revise"])

        revision_prompt = self.transport.last_user_message(1)
        self.assertIn("Here is the contract you proposed", revision_prompt)
        self.assertIn("Critique it", revision_prompt)
        # Self-refinement gets no evaluator feedback at all.
        self.assertNotIn("counterexample", revision_prompt.lower())
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.spend["states_observed"], 0)

        # The second answer replaces the first, so the run is scored on it.
        self.assertTrue(artifact.evaluation.exact)
        self.assertEqual(artifact.contract.interaction_index, 1)
        self.assertEqual(len(artifact.rounds), 2)

    def test_self_refine_keeps_the_earlier_contract_if_the_revision_breaks(self) -> None:
        self.transport.reset([EXACT_TEXT, "I changed my mind, no contract for you."])
        artifact = self.runner.run(spec("self_refine"))
        self.assertEqual(artifact.contract.status, artifacts_module.STATUS_PARSED)
        self.assertEqual(artifact.contract.interaction_index, 0)
        self.assertTrue(artifact.evaluation.exact)
        self.assertEqual(artifact.spend["parse_failures"], 1)
        self.assertEqual(artifact.rounds[1]["contract_status"], artifacts_module.STATUS_PARSE_FAILURE)

    def test_self_refine_asks_for_a_repair_when_the_first_answer_failed(self) -> None:
        self.transport.reset(["not a contract", EXACT_TEXT])
        artifact = self.runner.run(spec("self_refine"))
        self.assertIn("could not be parsed", self.transport.last_user_message(1))
        self.assertTrue(artifact.evaluation.exact)

    def test_random_probe_attempts_the_action_before_asking(self) -> None:
        self.transport.reset([EXACT_TEXT])
        budgets = Budgets(state_budget=5, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("random_probe", budgets=budgets))
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(artifact.spend["states_observed"], 5)
        self.assertEqual(artifact.rounds[0]["states_probed"], 5)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)

        prompt = self.transport.last_user_message(0)
        self.assertIn("You probed 5 randomly chosen reachable states", prompt)
        self.assertIn("place_order ->", prompt)
        observations = artifact.rounds[0]["observations"]
        self.assertEqual(len(observations), 5)
        for observation in observations:
            self.assertIn(observation["ok"], (True, False))
            if observation["ok"]:
                self.assertIsNotNone(observation["after"])
            else:
                self.assertIsNotNone(observation["error"])

    def test_probing_is_reproducible_for_a_seed_and_varies_across_seeds(self) -> None:
        first = self.runner.probe_states(3, 6)
        again = self.runner.probe_states(3, 6)
        other = self.runner.probe_states(4, 6)
        self.assertEqual(first, again)
        self.assertEqual(len(first), 6)
        self.assertNotEqual(first, other)

    def test_probe_observations_agree_with_the_sandbox(self) -> None:
        states = self.runner.probe_states(0, 8)
        observations = self.runner.observe(states)
        for state, observation in zip(states, observations):
            outcome = apply_action(state, ShoppingAction.place_order(), self.runner.config)
            self.assertEqual(observation["ok"], outcome.ok)
            self.assertEqual(observation["before"], state.describe())

    def test_candidate_aware_selection_is_balanced_deterministic_and_non_repeating(self) -> None:
        parsed = dsl.parse_contract_text(EXACT_TEXT)
        first = self.runner.candidate_aware_states(parsed, seed=7, count=8)
        again = self.runner.candidate_aware_states(parsed, seed=7, count=8)
        second = self.runner.candidate_aware_states(
            parsed,
            seed=7,
            count=8,
            audit_index=1,
            excluded=first,
        )
        contract = parsed.bind(self.runner.config)

        self.assertEqual(first, again)
        self.assertEqual(len(first), 8)
        self.assertEqual(sum(contract.holds_in(state) for state in first), 4)
        self.assertEqual(set(first).intersection(second), set())

    def test_candidate_selection_ablation_switches_are_deterministic(self) -> None:
        parsed = dsl.parse_contract_text(EXACT_TEXT)
        coverage_only = self.runner.candidate_aware_states(
            parsed, seed=7, count=8, balance=False, coverage=True
        )
        balanced_random = self.runner.candidate_aware_states(
            parsed, seed=7, count=8, balance=True, coverage=False
        )
        uniform = self.runner.candidate_aware_states(
            parsed, seed=7, count=8, balance=False, coverage=False
        )
        contract = parsed.bind(self.runner.config)

        self.assertEqual(
            coverage_only,
            self.runner.candidate_aware_states(
                parsed, seed=7, count=8, balance=False, coverage=True
            ),
        )
        self.assertEqual(len(coverage_only), 8)
        self.assertEqual(len(balanced_random), 8)
        self.assertEqual(len(uniform), 8)
        self.assertEqual(sum(contract.holds_in(state) for state in balanced_random), 4)
        self.assertNotEqual(coverage_only, uniform)

    def test_counterexample_guided_refines_until_the_contract_is_exact(self) -> None:
        self.transport.reset([WEAK_TEXT, EXACT_TEXT])
        artifact = self.runner.run(spec("counterexample_guided"))
        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual([entry.role for entry in artifact.interactions], ["propose", "revise"])

        revision_prompt = self.transport.last_user_message(1)
        self.assertIn("Concrete counterexamples", revision_prompt)
        self.assertIn("false_accept", revision_prompt)
        self.assertIn("before:", revision_prompt)
        self.assertIn("after:", revision_prompt)

        # Two oracle queries: one that produced the feedback, one that found the
        # revision exact and stopped the loop.
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 2)
        self.assertGreater(artifact.spend["states_observed"], 0)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_EXACT)
        self.assertTrue(artifact.evaluation.exact)
        self.assertEqual(artifact.rounds[0]["metrics"]["false_accepts"],
                         artifact.rounds[0]["metrics"]["false_accepts"])
        self.assertTrue(artifact.rounds[0]["counterexamples"])

    def test_counterexample_guided_stops_immediately_on_an_exact_first_answer(self) -> None:
        self.transport.reset([EXACT_TEXT])
        artifact = self.runner.run(spec("counterexample_guided"))
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 1)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_EXACT)

    def test_counterexample_guided_retries_after_a_parse_failure(self) -> None:
        self.transport.reset(["sorry, no JSON here", EXACT_TEXT])
        artifact = self.runner.run(spec("counterexample_guided"))
        self.assertEqual(len(self.transport.calls), 2)
        self.assertIn("could not be parsed", self.transport.last_user_message(1))
        self.assertEqual(artifact.spend["parse_failures"], 1)
        self.assertTrue(artifact.evaluation.exact)

    def test_sampled_cegis_reuses_one_budget_limited_state_sample(self) -> None:
        defective_text = json.dumps({
            "skill": "place_order",
            "precondition": {"op": "const", "value": True},
            "postcondition": {"op": "const", "value": True},
        })
        self.transport.reset([defective_text, EXACT_TEXT])
        budgets = Budgets(state_budget=8, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("sampled_cegis", budgets=budgets))

        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual(artifact.spend["states_observed"], 8)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.spend["sampled_feedback_queries"], 2)
        self.assertEqual(artifact.spend["sampled_states_checked"], 16)
        self.assertEqual(artifact.rounds[0]["sampled_oracle_size"], 8)
        self.assertEqual(
            artifact.rounds[0]["evidence_policy"],
            "fixed_budget_sampled_equivalence_oracle",
        )
        self.assertTrue(artifact.rounds[0]["counterexamples"])
        self.assertEqual(artifact.stopped_because, runner_module.STOP_SAMPLE_SATISFIED)
        self.assertTrue(artifact.evaluation.exact)

    def test_active_cegis_progressively_audits_candidate_aware_states(self) -> None:
        self.transport.reset([WEAK_TEXT, EXACT_TEXT])
        budgets = Budgets(state_budget=9, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("active_cegis", budgets=budgets))

        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual(artifact.spend["states_observed"], 9)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.spend["sampled_feedback_queries"], 3)
        self.assertEqual(artifact.spend["sampled_states_checked"], 18)
        self.assertEqual(artifact.rounds[0]["fresh_states"], 3)
        self.assertEqual(artifact.rounds[-1]["cumulative_states"], 9)
        self.assertTrue(artifact.rounds[0]["counterexamples"])
        self.assertIn("Concrete counterexamples", self.transport.last_user_message(1))
        self.assertEqual(artifact.stopped_because, runner_module.STOP_SAMPLE_SATISFIED)
        self.assertTrue(artifact.evaluation.exact)

    def test_active_cegis_ablation_methods_record_their_selection_policy(self) -> None:
        cases = {
            "active_cegis_no_balance": "progressive_coverage_without_candidate_balance",
            "active_cegis_no_coverage": "candidate_partitioned_progressive_random",
            "active_cegis_uniform": "progressive_uniform_random",
        }
        budgets = Budgets(state_budget=9, query_budget=4, token_budget=16000)
        for method, policy in cases.items():
            with self.subTest(method=method):
                self.transport.reset([WEAK_TEXT, EXACT_TEXT])
                artifact = self.runner.run(spec(method, budgets=budgets))
                self.assertEqual(artifact.rounds[0]["adaptation"], method)
                self.assertEqual(artifact.rounds[0]["evidence_policy"], policy)
                self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
                self.assertEqual(artifact.contract.status, artifacts_module.STATUS_PARSED)

    def test_asi_replay_makes_one_call_and_observes_one_successful_state(self) -> None:
        self.transport.reset([EXACT_TEXT])
        budgets = Budgets(state_budget=5, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("asi_replay", budgets=budgets))
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(artifact.spend["states_observed"], 1)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.rounds[0]["adaptation"], "asi_replay")
        self.assertEqual(artifact.rounds[0]["evidence_policy"], "single_successful_observation")
        self.assertEqual(artifact.rounds[0]["states_observed"], 1)
        self.assertEqual(artifact.rounds[0]["accepted"], 1)
        self.assertEqual(len(artifact.rounds[0]["observations"]), 1)
        self.assertTrue(artifact.rounds[0]["observations"][0]["ok"])

        prompt = self.transport.last_user_message(0)
        self.assertIn("historical successful execution demonstration of place_order", prompt)
        self.assertIn("place_order -> ACCEPTED", prompt)
        self.assertNotIn("REFUSED", prompt)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_EXACT)
        self.assertTrue(artifact.evaluation.exact)

    def test_skillcommit_replay_early_stops_on_compatible_proposal(self) -> None:
        self.transport.reset([EXACT_TEXT])
        budgets = Budgets(state_budget=3, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("skillcommit_replay", budgets=budgets))
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.spend["states_observed"], 3)
        self.assertEqual(artifact.rounds[0]["adaptation"], "skillcommit_replay")
        self.assertEqual(artifact.rounds[0]["evidence_policy"], "proposal_from_single_successful_instance")
        self.assertEqual(artifact.rounds[1]["role"], "validate")
        self.assertTrue(artifact.rounds[1]["compatible"])
        self.assertEqual(artifact.rounds[1]["incompatibilities"], 0)
        prompt = self.transport.last_user_message(0)
        self.assertIn("historical successful execution demonstration", prompt)
        self.assertIn("positive example", prompt)
        self.assertNotIn("REFUSED", prompt)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_EXACT)
        self.assertTrue(artifact.evaluation.exact)

    def test_skillcommit_replay_revises_when_incompatibilities_observed(self) -> None:
        defective_spec = contract_spec()
        defective_spec["precondition"]["args"].append(
            {"op": "eq", "left": {"var": "payment_method"}, "right": {"const": "card"}}
        )
        defective_text = json.dumps(defective_spec)
        self.transport.reset([defective_text, EXACT_TEXT])
        budgets = Budgets(state_budget=6, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("skillcommit_replay", budgets=budgets))
        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual([entry.role for entry in artifact.interactions], ["propose", "revise"])
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.spend["states_observed"], 6)
        self.assertEqual(artifact.rounds[0]["role"], "propose")
        self.assertEqual(artifact.rounds[1]["role"], "revise")
        self.assertGreater(artifact.rounds[1]["incompatibilities"], 0)
        revision_prompt = self.transport.last_user_message(1)
        self.assertIn("Cross-instance validation against additional distinct historical successful executions", revision_prompt)
        self.assertIn("false rejects", revision_prompt)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_COMPLETE)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_EXACT)
        self.assertTrue(artifact.evaluation.exact)

    def test_skillcommit_replay_respects_query_budget_when_incompatible(self) -> None:
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

    def test_contractskill_repair_uses_only_budget_limited_observed_replay(self) -> None:
        defective_text = json.dumps({
            "skill": "place_order",
            "precondition": {"op": "const", "value": True},
            "postcondition": {"op": "const", "value": True},
        })
        self.transport.reset([defective_text, EXACT_TEXT])
        budgets = Budgets(state_budget=8, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("contractskill_repair", budgets=budgets))
        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual([entry.role for entry in artifact.interactions], ["propose", "revise"])
        self.assertEqual(artifact.spend["states_observed"], 8)
        # ContractSkill repair must not use the full-closure equivalence oracle
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.rounds[0]["adaptation"], "contractskill_repair")
        self.assertEqual(artifact.rounds[0]["evidence_policy"], "budget_limited_observed_replay_repair")
        self.assertEqual(artifact.rounds[0]["replay_set_size"], 8)
        self.assertIn("replay_metrics", artifact.rounds[0])

        revision_prompt = self.transport.last_user_message(1)
        self.assertIn("budget-limited observed replay set", revision_prompt)
        self.assertIn("Observed replay violations", revision_prompt)
        self.assertTrue(artifact.rounds[0]["replay_counterexamples"])
        self.assertEqual(artifact.stopped_because, runner_module.STOP_COMPLETE)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_EXACT)
        self.assertTrue(artifact.evaluation.exact)

    def test_contractskill_repair_stops_immediately_if_initial_contract_satisfies_replay_set(self) -> None:
        self.transport.reset([EXACT_TEXT])
        budgets = Budgets(state_budget=8, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("contractskill_repair", budgets=budgets))
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_COMPLETE)
        self.assertTrue(artifact.evaluation.exact)

    def test_contractskill_repair_retries_after_a_parse_failure(self) -> None:
        self.transport.reset(["invalid json", EXACT_TEXT])
        budgets = Budgets(state_budget=8, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("contractskill_repair", budgets=budgets))
        self.assertEqual(len(self.transport.calls), 2)
        self.assertIn("could not be parsed", self.transport.last_user_message(1))
        self.assertEqual(artifact.spend["parse_failures"], 1)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 0)
        self.assertTrue(artifact.evaluation.exact)

    def test_contractskill_full_oracle_uses_contractskill_repair_prompt(self) -> None:
        self.transport.reset([WEAK_TEXT, EXACT_TEXT])
        artifact = self.runner.run(spec("contractskill_full_oracle"))

        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual(artifact.spend["oracle_feedback_queries"], 2)
        self.assertGreater(artifact.spend["states_observed"], 0)
        self.assertEqual(
            artifact.rounds[0]["evidence_policy"],
            "full_closure_contractskill_repair",
        )
        self.assertEqual(
            artifact.rounds[0]["full_closure_metrics"]["states_checked"],
            len(self.runner.states),
        )
        revision_prompt = self.transport.last_user_message(1)
        self.assertIn("full reachable-state closure", revision_prompt)
        self.assertIn("ContractSkill-style repair discipline", revision_prompt)
        self.assertNotIn("budget-limited observed replay set", revision_prompt)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_EXACT)
        self.assertTrue(artifact.evaluation.exact)

    def test_every_method_shares_the_same_decoding_and_budget_inputs(self) -> None:
        decoding = Decoding(temperature=0.7, top_p=0.8, max_tokens=321)
        budgets = Budgets(state_budget=4, query_budget=1, token_budget=9000)
        for method in modeling.METHODS:
            with self.subTest(method=method):
                self.transport.reset([EXACT_TEXT])
                artifact = self.runner.run(
                    RunSpec(method=method, seed=5, decoding=decoding, budgets=budgets)
                )
                payload = self.transport.payloads[0]
                self.assertEqual(payload["temperature"], 0.7)
                self.assertEqual(payload["top_p"], 0.8)
                self.assertEqual(payload["max_tokens"], 321)
                self.assertEqual(payload["seed"], 5)
                self.assertEqual(artifact.seed, 5)
                self.assertEqual(artifact.budgets, budgets.to_dict())
                self.assertEqual(artifact.decoding["temperature"], 0.7)
                self.assertLessEqual(artifact.spend["states_observed"], 4)


class BudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.transport = MockTransport()
        cls.runner = make_runner(cls.transport)

    def test_the_query_budget_caps_model_calls(self) -> None:
        self.transport.reset([WEAK_TEXT])
        budgets = Budgets(state_budget=6, query_budget=1, token_budget=16000)
        artifact = self.runner.run(spec("self_refine", budgets=budgets))
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_QUERY_BUDGET)
        self.assertEqual(artifact.contract.status, artifacts_module.STATUS_PARSED)
        self.assertFalse(artifact.evaluation.exact)

    def test_the_token_budget_clamps_max_tokens_and_then_stops_the_run(self) -> None:
        self.transport.reset([WEAK_TEXT])
        budgets = Budgets(state_budget=6, query_budget=4, token_budget=100)
        decoding = Decoding(temperature=0.0, top_p=1.0, max_tokens=512)
        artifact = self.runner.run(
            spec("self_refine", budgets=budgets, decoding=decoding)
        )
        # The first call may generate only what the budget still allows...
        self.assertEqual(self.transport.payloads[0]["max_tokens"], 100)
        # ...and the 180 tokens it spent leave nothing for a second call.
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_TOKEN_BUDGET)

    def test_a_zero_query_budget_produces_an_artifact_with_no_contract(self) -> None:
        self.transport.reset([])
        budgets = Budgets(state_budget=6, query_budget=0, token_budget=16000)
        artifact = self.runner.run(spec("direct", budgets=budgets))
        self.assertEqual(len(self.transport.calls), 0)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_NO_CONTRACT)
        self.assertIsNone(artifact.evaluation)
        self.assertEqual(artifact.usage["calls"], 0)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_QUERY_BUDGET)

    def test_the_state_budget_caps_probing(self) -> None:
        self.transport.reset([EXACT_TEXT])
        budgets = Budgets(state_budget=2, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("random_probe", budgets=budgets))
        self.assertEqual(artifact.rounds[0]["states_probed"], 2)
        self.assertEqual(artifact.spend["states_observed"], 2)
        self.assertEqual(artifact.spend["remaining_states"], 0)

    def test_a_zero_state_budget_leaves_random_probe_with_nothing_to_show(self) -> None:
        self.transport.reset([])
        budgets = Budgets(state_budget=0, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("random_probe", budgets=budgets))
        self.assertEqual(len(self.transport.calls), 0)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_STATE_BUDGET)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_NO_CONTRACT)

    def test_a_zero_state_budget_leaves_asi_replay_with_nothing_to_show(self) -> None:
        self.transport.reset([])
        budgets = Budgets(state_budget=0, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("asi_replay", budgets=budgets))
        self.assertEqual(len(self.transport.calls), 0)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_STATE_BUDGET)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_NO_CONTRACT)

    def test_a_zero_state_budget_leaves_skillcommit_replay_with_nothing_to_show(self) -> None:
        self.transport.reset([])
        budgets = Budgets(state_budget=0, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("skillcommit_replay", budgets=budgets))
        self.assertEqual(len(self.transport.calls), 0)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_STATE_BUDGET)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_NO_CONTRACT)

    def test_a_zero_state_budget_leaves_contractskill_repair_with_nothing_to_show(self) -> None:
        self.transport.reset([])
        budgets = Budgets(state_budget=0, query_budget=4, token_budget=16000)
        artifact = self.runner.run(spec("contractskill_repair", budgets=budgets))
        self.assertEqual(len(self.transport.calls), 0)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_STATE_BUDGET)
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_NO_CONTRACT)

    def test_the_state_budget_caps_counterexample_feedback(self) -> None:
        self.transport.reset([WEAK_TEXT, WEAK_TEXT, WEAK_TEXT])
        budgets = Budgets(state_budget=1, query_budget=8, token_budget=16000)
        artifact = self.runner.run(spec("counterexample_guided", budgets=budgets))
        self.assertEqual(len(artifact.rounds[0]["counterexamples"]), 1)
        self.assertEqual(artifact.spend["states_observed"], 1)
        # With the state budget spent there is nothing left to show, so the loop
        # stops rather than asking the same question again.
        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_STATE_BUDGET)


class TransportFailureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.transport = MockTransport()
        cls.runner = make_runner(cls.transport)

    def test_an_unreachable_endpoint_is_recorded_not_raised(self) -> None:
        self.transport.reset([TransportError("could not reach http://127.0.0.1:8000")])
        artifact = self.runner.run(spec("direct"))
        self.assertEqual(artifact.outcome, artifacts_module.OUTCOME_NO_CONTRACT)
        self.assertEqual(artifact.stopped_because, runner_module.STOP_TRANSPORT)
        self.assertEqual(len(artifact.interactions), 1)
        self.assertIn("could not reach", artifact.interactions[0].error)
        self.assertIsNone(artifact.interactions[0].response_text)
        self.assertIsNone(artifact.evaluation)


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------


REQUIRED_ARTIFACT_KEYS = (
    "schema_version",
    "run",
    "created_at",
    "method",
    "model",
    "endpoint",
    "seed",
    "decoding",
    "budgets",
    "sandbox",
    "interactions",
    "rounds",
    "contract",
    "usage",
    "latency",
    "spend",
    "evaluation",
    "outcome",
    "stopped_because",
)


class ArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.transport = MockTransport()
        cls.runner = make_runner(cls.transport)

    def run_one(self, method: str, script: Sequence[Any], **kwargs: Any):
        self.transport.reset(script)
        return self.runner.run(spec(method, **kwargs))

    def test_an_artifact_records_everything_needed_to_reread_the_run(self) -> None:
        artifact = self.run_one("counterexample_guided", [WEAK_TEXT, EXACT_TEXT], seed=7)
        payload = artifact.to_dict()
        for key in REQUIRED_ARTIFACT_KEYS:
            self.assertIn(key, payload)

        self.assertEqual(payload["method"], "counterexample_guided")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["endpoint"], "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(
            payload["decoding"],
            {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512, "seed": 7},
        )
        self.assertEqual(
            payload["budgets"],
            {"state_budget": 6, "query_budget": 4, "token_budget": 16000},
        )

        self.assertEqual(payload["sandbox"]["config"], small_config().to_dict())
        self.assertEqual(payload["sandbox"]["max_depth"], TEST_MAX_DEPTH)
        self.assertGreater(payload["sandbox"]["evaluation_states"], 0)

        # Prompts and raw responses, per call.
        self.assertEqual(len(payload["interactions"]), 2)
        for interaction in payload["interactions"]:
            self.assertTrue(interaction["messages"])
            self.assertEqual(interaction["messages"][0]["role"], "system")
            self.assertIsNotNone(interaction["response_text"])
            self.assertIn("choices", interaction["raw_response"])
            self.assertEqual(interaction["request"]["model"], "test-model")
            self.assertGreaterEqual(interaction["latency_s"], 0.0)
            self.assertEqual(interaction["usage"]["total_tokens"], 180)

        self.assertEqual(payload["contract"]["status"], "parsed")
        self.assertEqual(payload["contract"]["spec"], json.loads(EXACT_TEXT))
        self.assertGreater(payload["contract"]["node_count"], 0)

        self.assertEqual(payload["usage"], {
            "prompt_tokens": 240, "completion_tokens": 120, "total_tokens": 360, "calls": 2
        })
        self.assertEqual(len(payload["latency"]["per_call_s"]), 2)
        self.assertEqual(payload["spend"]["model_calls"], 2)
        self.assertEqual(payload["spend"]["oracle_feedback_queries"], 2)

        metrics = payload["evaluation"]["metrics"]
        for key in (
            "states_checked",
            "successes",
            "failures",
            "false_accepts",
            "false_rejects",
            "postcondition_violations",
            "symptoms",
            "exact",
            "sound",
            "complete",
        ):
            self.assertIn(key, metrics)
        self.assertTrue(payload["evaluation"]["exact"])
        self.assertEqual(payload["outcome"], "exact")

    def test_an_inexact_artifact_records_counterexamples(self) -> None:
        artifact = self.run_one(
            "direct", [WEAK_TEXT], budgets=Budgets(state_budget=6, query_budget=1, token_budget=100000)
        )
        payload = artifact.to_dict()
        self.assertEqual(payload["outcome"], "inexact")
        counterexamples = payload["evaluation"]["counterexamples"]
        self.assertTrue(counterexamples)
        for counterexample in counterexamples:
            self.assertIn(counterexample["symptom"], [symptom.value for symptom in Symptom])
            self.assertIsInstance(counterexample["state"], str)
            self.assertIn("logged_in", counterexample["state_fields"])
            self.assertEqual(counterexample["before_state"], counterexample["state_fields"])
            self.assertIn("after_state", counterexample)
            self.assertIn("changed_fields", counterexample)
            self.assertIn("unchanged_fields", counterexample)
            self.assertTrue(counterexample["detail"])

    def test_a_parse_failure_is_recorded_with_the_offending_text(self) -> None:
        artifact = self.run_one("direct", [UNSAFE_TEXT])
        payload = artifact.to_dict()
        self.assertEqual(payload["outcome"], "parse_failure")
        self.assertEqual(payload["contract"]["status"], "parse_failure")
        self.assertIsNone(payload["contract"]["spec"])
        self.assertIn("python", payload["contract"]["error"])
        self.assertEqual(payload["contract"]["source"], UNSAFE_TEXT)
        self.assertIsNone(payload["evaluation"])
        self.assertEqual(payload["spend"]["parse_failures"], 1)

    def test_artifacts_are_json_serialisable_and_written_where_asked(self) -> None:
        artifact = self.run_one("direct", [EXACT_TEXT], seed=2)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "run"
            path = artifacts_module.write_artifact(target, artifact)
            self.assertEqual(path.name, "direct__seed2.json")
            reloaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(reloaded, artifact.to_dict())

            summary_path = artifacts_module.write_summary(
                target, [artifact], {"model": "test-model"}
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["context"], {"model": "test-model"})
            self.assertEqual(summary["runs"][0]["method"], "direct")
            self.assertTrue(summary["runs"][0]["exact"])
            self.assertEqual(sorted(path.name for path in target.iterdir()),
                             ["direct__seed2.json", "summary.json"])

    def test_write_summary_rejects_duplicate_runs(self) -> None:
        artifact = self.run_one("direct", [EXACT_TEXT], seed=2)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with self.assertRaises(ValueError):
                artifacts_module.write_summary(target, [artifact, artifact], {"model": "test-model"})

    def test_the_summary_row_covers_cost_and_score(self) -> None:
        artifact = self.run_one("random_probe", [EXACT_TEXT])
        row = artifact.summary_row()
        for key in (
            "method",
            "seed",
            "model",
            "outcome",
            "contract_status",
            "exact",
            "false_accepts",
            "false_rejects",
            "postcondition_violations",
            "states_checked",
            "model_calls",
            "total_tokens",
            "states_observed",
            "oracle_feedback_queries",
            "latency_s",
        ):
            self.assertIn(key, row)


# --------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------


def run_cli(argv: Sequence[str], transport: MockTransport) -> tuple[int, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = model_runner.main(list(argv), transport=transport)
    return code, buffer.getvalue()


class CliTest(unittest.TestCase):
    def base_argv(self, output: Path) -> list[str]:
        return [
            "--output",
            str(output),
            "--sandbox-config",
            str(SMALL_CONFIG_PATH),
            "--model",
            "test-model",
            "--max-depth",
            "6",
            "--max-states",
            "400",
            "--state-budget",
            "3",
            "--query-budget",
            "2",
        ]

    def test_the_shipped_experiment_config_is_readable_and_loopback(self) -> None:
        config = model_runner.load_experiment_config(model_runner.DEFAULT_EXPERIMENT_CONFIG)
        self.assertTrue(client_module.is_loopback(config["base_url"]))
        self.assertEqual(
            config["methods"],
            ["direct", "self_refine", "random_probe", "counterexample_guided"],
        )
        self.assertTrue(set(config["methods"]).issubset(set(modeling.METHODS)))
        self.assertLessEqual(set(config), set(model_runner.CONFIG_KEYS))

    def test_a_run_writes_one_artifact_per_method_and_seed(self) -> None:
        transport = MockTransport([EXACT_TEXT, WEAK_TEXT, EXACT_TEXT])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            code, text = run_cli(
                self.base_argv(output) + ["--methods", "direct", "self_refine", "--seeds", "4"],
                transport,
            )
            self.assertEqual(code, 0, msg=text)
            self.assertIn("model contract experiment", text)
            self.assertIn("exact contracts:", text)
            written = sorted(path.name for path in output.iterdir())
            self.assertEqual(
                written, ["direct__seed4.json", "self_refine__seed4.json", "summary.json"]
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary["runs"]), 2)
            self.assertEqual(summary["context"]["model"], "test-model")
            self.assertEqual(
                summary["context"]["endpoint"], "http://127.0.0.1:8000/v1/chat/completions"
            )
            self.assertEqual(summary["context"]["seeds"], [4])
            artifact = json.loads((output / "direct__seed4.json").read_text(encoding="utf-8"))
            self.assertEqual(artifact["seed"], 4)
            self.assertEqual(artifact["sandbox"]["config_path"], str(SMALL_CONFIG_PATH))
            self.assertEqual(artifact["serving"]["request_concurrency"], 1)

    def test_json_output_is_machine_readable(self) -> None:
        transport = MockTransport([EXACT_TEXT])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            code, text = run_cli(
                self.base_argv(output) + ["--methods", "direct", "--seeds", "0", "--json"],
                transport,
            )
            self.assertEqual(code, 0, msg=text)
            payload = json.loads(text)
            self.assertEqual(payload["runs"][0]["method"], "direct")
            self.assertEqual(payload["context"]["budgets"]["query_budget"], 2)

    def test_parallel_workers_preserve_artifact_order(self) -> None:
        transport = MockTransport([EXACT_TEXT] * 4)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            code, text = run_cli(
                self.base_argv(output)
                + ["--methods", "direct", "--seeds", "0", "1", "2", "3", "--workers", "2", "--json"],
                transport,
            )
            self.assertEqual(code, 0, msg=text)
            payload = json.loads(text)
            self.assertEqual(payload["context"]["workers"], 2)
            self.assertEqual([run["seed"] for run in payload["runs"]], [0, 1, 2, 3])
            self.assertEqual(len(transport.calls), 4)
            artifact = json.loads((output / "direct__seed0.json").read_text(encoding="utf-8"))
            self.assertEqual(artifact["serving"]["request_concurrency"], 2)

    def test_workers_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            buffer = io.StringIO()
            with contextlib.redirect_stderr(buffer):
                code = model_runner.main(
                    self.base_argv(output) + ["--workers", "0"],
                    transport=MockTransport(),
                )
            self.assertEqual(code, 2)
            self.assertIn("workers must be at least 1", buffer.getvalue())

    def test_output_is_required(self) -> None:
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                model_runner.main(["--model", "test-model"], transport=MockTransport())

    def test_a_remote_base_url_is_refused(self) -> None:
        transport = MockTransport()
        with tempfile.TemporaryDirectory() as directory:
            argv = self.base_argv(Path(directory) / "artifacts") + [
                "--methods",
                "direct",
                "--seeds",
                "0",
                "--base-url",
                "https://api.openai.com/v1",
            ]
            buffer = io.StringIO()
            with contextlib.redirect_stderr(buffer):
                code = model_runner.main(argv, transport=transport)
            self.assertEqual(code, 2)
            self.assertIn("loopback", buffer.getvalue())
            self.assertEqual(transport.calls, [])

    def test_an_unreachable_endpoint_fails_the_run_but_keeps_the_artifact(self) -> None:
        transport = MockTransport([TransportError("could not reach the server")])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            argv = self.base_argv(output) + ["--methods", "direct", "--seeds", "0"]
            buffer = io.StringIO()
            with contextlib.redirect_stderr(buffer):
                code, _ = run_cli(argv, transport)
            self.assertEqual(code, 1)
            self.assertIn("unreachable", buffer.getvalue())
            artifact = json.loads((output / "direct__seed0.json").read_text(encoding="utf-8"))
            self.assertEqual(artifact["stopped_because"], "transport_error")


class NoLiveServerTest(unittest.TestCase):
    def test_a_whole_cli_run_never_opens_a_connection(self) -> None:
        transport = MockTransport([EXACT_TEXT] * len(modeling.METHODS))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            argv = [
                "--output",
                str(output),
                "--sandbox-config",
                str(SMALL_CONFIG_PATH),
                "--model",
                "test-model",
                "--max-depth",
                "6",
                "--max-states",
                "400",
                "--state-budget",
                "3",
                "--query-budget",
                "1",
                "--methods",
                *modeling.METHODS,
                "--seeds",
                "0",
            ]
            with mock.patch(
                "urllib.request.urlopen",
                side_effect=AssertionError("the tests must not touch the network"),
            ), mock.patch(
                "socket.socket",
                side_effect=AssertionError("the tests must not open a socket"),
            ):
                code, text = run_cli(argv, transport)
        self.assertEqual(code, 0, msg=text)
        self.assertEqual(len(transport.calls), len(modeling.METHODS))
        for call in transport.calls:
            self.assertEqual(call["url"], "http://127.0.0.1:8000/v1/chat/completions")


# --------------------------------------------------------------------------
# Static safety properties
# --------------------------------------------------------------------------


# Everything the modeling package may import. ``urllib`` is the whole of its
# networking, and there is no third-party dependency anywhere.
ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "argparse",
        "contextlib",
        "concurrent.futures",
        "dataclasses",
        "datetime",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "random",
        "sys",
        "time",
        "typing",
        "urllib.error",
        "urllib.parse",
        "urllib.request",
    }
)

FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile", "__import__", "breakpoint", "input"})


class SafetySourceTest(unittest.TestCase):
    """The runner must be unable to execute anything a model sends it."""

    def setUp(self) -> None:
        package = Path(modeling.__file__).resolve().parent
        self.sources = sorted(package.glob("*.py"))
        self.sources.append(Path(model_runner.__file__).resolve())

    def test_there_are_sources_to_inspect(self) -> None:
        self.assertGreaterEqual(len(self.sources), 6)

    def test_no_source_evaluates_or_executes_anything(self) -> None:
        for path in self.sources:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(
                        node.func.id, FORBIDDEN_CALLS, msg=f"{path.name} calls {node.func.id}"
                    )

    def test_only_the_standard_library_is_imported(self) -> None:
        for path in self.sources:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:  # relative: inside this project
                        continue
                    module = node.module or ""
                    if module == "experiment" or module.startswith("experiment."):
                        continue
                    names = [module]
                else:
                    continue
                for name in names:
                    self.assertIn(name, ALLOWED_IMPORTS, msg=f"{path.name} imports {name}")

    def test_the_environment_package_is_still_unaware_of_the_modeling_package(self) -> None:
        from .. import environment

        for path in sorted(Path(environment.__file__).resolve().parent.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("modeling", text, msg=path.name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
