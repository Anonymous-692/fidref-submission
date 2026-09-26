"""Offline guards for the paid API path: never reaches a network endpoint."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from experiment.modeling.hosted_api import Coordinator, HostedClient, SuitePaused


MODEL = "gpt-5.4-mini"


class Reply:
    status = 200
    headers = {"x-request-id": "test-request"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps({"model": MODEL, "choices": [{"message": {"content": '{"ok":true}'},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 10,
            "completion_tokens": 5, "total_tokens": 15}}).encode()


class HostedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.coordinator = Coordinator(self.root, "SECRET_TEST_KEY", prior_reserve=0)
        self.client = HostedClient(MODEL, self.coordinator, "initial")
        self.messages = [{"role": "user", "content": "Return JSON"}]

    def test_payload_has_no_api_seed_or_vllm_fields(self):
        body = self.client.build_request(self.messages, temperature=.2, top_p=.95,
                                        max_tokens=2048, seed=17)
        self.assertNotIn("seed", body)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["max_completion_tokens"], 2048)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertFalse(body["store"])

    def test_no_silent_schema_conversion(self):
        with self.assertRaises(ValueError):
            self.client.build_request(self.messages, temperature=.2, top_p=.95,
                                      max_tokens=2048, guided_json={"type": "object"})

    @patch("urllib.request.urlopen", return_value=Reply())
    def test_initial_shared_once_and_restart_accounting(self, network):
        first = self.client.complete(self.messages)
        spent = self.coordinator.charged
        for method in ("direct", "fixed_pool", "active", "no_balance"):
            client = HostedClient(MODEL, self.coordinator, method, "initial/0")
            self.assertEqual(client.complete(self.messages).text, first.text)
        self.assertEqual(network.call_count, 1)
        self.assertEqual(self.coordinator.charged, spent)
        restarted = Coordinator(self.root, "SECRET_TEST_KEY", prior_reserve=0)
        self.assertEqual(restarted.charged, spent)
        self.assertNotIn("SECRET_TEST_KEY", (self.root / "requests/initial/0.json").read_text())

    @patch("urllib.request.urlopen", return_value=Reply())
    def test_replay_mismatch_stops_before_request(self, network):
        self.client.complete(self.messages)
        client = HostedClient(MODEL, self.coordinator, "active", "initial/0")
        with self.assertRaises(SuitePaused):
            client.complete([{"role": "user", "content": "Different JSON request"}])
        self.assertEqual(network.call_count, 1)

    @patch("urllib.request.urlopen", return_value=Reply())
    def test_corrupted_response_is_not_replayed(self, network):
        self.client.complete(self.messages)
        path = self.root / "requests/initial/0.json"
        data = json.loads(path.read_text())
        data["response"]["choices"][0]["message"]["content"] = "changed"
        path.write_text(json.dumps(data))
        with self.assertRaises(SuitePaused):
            HostedClient(MODEL, self.coordinator, "active", "initial/0").complete(self.messages)
        self.assertEqual(network.call_count, 1)

    def test_inflight_cost_counts_toward_limit(self):
        coordinator = Coordinator(self.root, "test", limit=.012, prior_reserve=0)
        payload = self.client.build_request(self.messages, temperature=.2, top_p=.95, max_tokens=128)
        coordinator.reserve("a", MODEL, payload)
        with self.assertRaises(SuitePaused):
            coordinator.reserve("b", MODEL, payload)

    @patch("urllib.request.urlopen")
    def test_http_failure_not_retried_or_counted_as_model_failure(self, network):
        network.side_effect = urllib.error.HTTPError("https://api.openai.com", 429, "limit", {},
                                                     io.BytesIO(b'{"error":{"message":"limit"}}'))
        with self.assertRaises(SuitePaused):
            self.client.complete(self.messages)
        self.assertEqual(network.call_count, 1)
        self.assertEqual(json.loads((self.root / "requests/initial/0.json").read_text())["status"], "failed")
        with self.assertRaises(SuitePaused):
            Coordinator(self.root, "test")

    @patch("urllib.request.urlopen", side_effect=TimeoutError("timeout"))
    def test_uncertain_request_blocks_resume(self, network):
        with self.assertRaises(SuitePaused):
            self.client.complete(self.messages)
        with self.assertRaises(SuitePaused):
            Coordinator(self.root, "test")
        self.assertEqual(network.call_count, 1)

    def test_missing_initial_never_silently_regenerates(self):
        with self.assertRaises(SuitePaused):
            HostedClient(MODEL, self.coordinator, "active", "missing/0").complete(self.messages)


if __name__ == "__main__":
    unittest.main()
