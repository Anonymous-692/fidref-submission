#!/usr/bin/env python3
"""Tests that the demo entry point runs end to end and reports honestly."""

from __future__ import annotations

import contextlib
import io
import json
import unittest

from .. import run_demo
from .support import DEFAULT_CONFIG_PATH, SMALL_CONFIG_PATH, TEST_MAX_DEPTH, TEST_MAX_STATES


def run(argv: list[str]) -> tuple[int, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = run_demo.main(argv)
    return code, buffer.getvalue()


class DemoTest(unittest.TestCase):
    def test_demo_completes_on_the_small_config(self) -> None:
        code, output = run(
            [
                "--config",
                str(SMALL_CONFIG_PATH),
                "--max-depth",
                str(TEST_MAX_DEPTH),
                "--max-states",
                str(TEST_MAX_STATES),
            ]
        )
        self.assertEqual(code, 0, msg=output)
        self.assertIn("all demo checks passed", output)
        self.assertIn("scripted episode", output)
        self.assertIn("reachable states", output)
        self.assertIn("defect fixtures", output)
        self.assertNotIn("MISSED", output)

    def test_demo_completes_with_the_documented_defaults(self) -> None:
        # Exactly what `python experiment/run_demo.py` does.
        code, output = run(["--config", str(DEFAULT_CONFIG_PATH)])
        self.assertEqual(code, 0, msg=output)
        self.assertIn("all demo checks passed", output)

    def test_json_output_is_machine_readable(self) -> None:
        code, output = run(
            [
                "--config",
                str(SMALL_CONFIG_PATH),
                "--max-depth",
                str(TEST_MAX_DEPTH),
                "--max-states",
                str(TEST_MAX_STATES),
                "--json",
            ]
        )
        self.assertEqual(code, 0)
        payload = json.loads(output[: output.rindex("}") + 1])
        self.assertEqual(payload["ground_truth"]["exact"], True)
        self.assertTrue(payload["enumeration"]["reproducible"])
        self.assertTrue(payload["snapshot_probe"]["restored_exactly"])
        self.assertEqual(len(payload["fixtures"]), 4)
        for entry in payload["fixtures"]:
            self.assertTrue(entry["detected"], msg=entry["fixture"]["name"])

    def test_scripted_episode_reaches_a_placed_order(self) -> None:
        config = run_demo.EnvConfig.from_json_file(SMALL_CONFIG_PATH)
        env = run_demo.ShoppingEnv(config)
        trace = run_demo.scripted_episode(env)
        refusals = [entry for entry in trace if not entry["ok"]]
        self.assertEqual(len(refusals), 3)
        self.assertTrue(trace[-1]["ok"])
        self.assertTrue(all(entry["error"] for entry in refusals))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
