#!/usr/bin/env python3
"""Shared helpers for the sandbox unit tests."""

from __future__ import annotations

from pathlib import Path

from ..environment import EnvConfig, ShoppingAction, ShoppingEnv

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SMALL_CONFIG_PATH = CONFIG_DIR / "small.json"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.json"

# Deep and wide enough to close the small configuration completely, so tests
# can assert over the whole reachable set rather than a truncated sample.
TEST_MAX_DEPTH = 20
TEST_MAX_STATES = 4000


def small_config() -> EnvConfig:
    """The single-stock-unit configuration used by most tests."""
    return EnvConfig.from_json_file(SMALL_CONFIG_PATH)


def ready_to_order_script(config: EnvConfig) -> list[ShoppingAction]:
    """Actions that leave the sandbox one step away from a valid order."""
    return [
        ShoppingAction.login(),
        ShoppingAction.add_to_cart(config.items[0]),
        ShoppingAction.set_payment(config.valid_payment_methods[0]),
        ShoppingAction.set_address(config.valid_addresses[0]),
    ]


def ready_to_order_env(config: EnvConfig | None = None) -> ShoppingEnv:
    """An environment driven to a state where ``place_order`` succeeds."""
    config = config if config is not None else small_config()
    env = ShoppingEnv(config)
    for result in env.run(ready_to_order_script(config)):
        if not result.ok:  # pragma: no cover - guards the fixture itself
            raise AssertionError(f"setup step {result.action} failed: {result.error}")
    return env
