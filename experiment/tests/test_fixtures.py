#!/usr/bin/env python3
"""Tests for the four defect fixtures and the scoring they must trigger."""

from __future__ import annotations

import unittest

from ..counterexamples import round_robin_counterexamples, transition_diff
from ..environment import EnvConfig, ShoppingAction, enumerate_reachable
from ..evaluation import (
    Contract,
    DefectCategory,
    Symptom,
    all_fixtures,
    branch_contracts,
    evaluate_all,
    evaluate_contract,
    faulty_merge,
    ground_truth_contract,
    payment_branch_contract,
)
from ..evaluation.fixtures import DefectFixture
from .support import TEST_MAX_DEPTH, TEST_MAX_STATES, small_config


def fixture_for(config: EnvConfig, category: DefectCategory) -> DefectFixture:
    """Look a fixture up by category so tests never depend on list order."""
    for fixture in all_fixtures(config):
        if fixture.category is category:
            return fixture
    raise AssertionError(f"no fixture for {category}")


class FixtureCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_config()
        self.fixtures = all_fixtures(self.config)

    def test_every_defect_category_is_covered_exactly_once(self) -> None:
        categories = [fixture.category for fixture in self.fixtures]
        self.assertEqual(
            set(categories),
            {
                DefectCategory.TOO_WEAK_PRECONDITION,
                DefectCategory.TOO_STRONG_PRECONDITION,
                DefectCategory.WRONG_POSTCONDITION,
                DefectCategory.FAULTY_MERGE,
            },
        )
        self.assertEqual(len(categories), len(set(categories)))

    def test_fixtures_are_stable_and_described(self) -> None:
        self.assertEqual(
            [fixture.name for fixture in all_fixtures(self.config)],
            [fixture.name for fixture in self.fixtures],
        )
        for fixture in self.fixtures:
            self.assertTrue(fixture.rationale)
            self.assertTrue(fixture.contract.description)
            self.assertEqual(fixture.contract.action, ShoppingAction.place_order())
            self.assertIn("category", fixture.summary())

    def test_fixtures_need_two_valid_payment_methods(self) -> None:
        narrow = EnvConfig(valid_payment_methods=("card",))
        with self.assertRaises(ValueError):
            all_fixtures(narrow)
        with self.assertRaises(ValueError):
            branch_contracts(narrow)


class FixtureDetectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = small_config()
        cls.states = enumerate_reachable(
            cls.config, max_depth=TEST_MAX_DEPTH, max_states=TEST_MAX_STATES
        ).states

    def report_for(self, contract: Contract):
        return evaluate_contract(contract, self.states, self.config)

    def test_each_fixture_shows_exactly_its_expected_symptoms(self) -> None:
        for fixture in all_fixtures(self.config):
            with self.subTest(fixture=fixture.name):
                report = self.report_for(fixture.contract)
                self.assertEqual(report.symptoms, fixture.expected_symptoms)
                self.assertTrue(report.counterexamples)
                self.assertIn(
                    report.counterexamples[0].symptom, fixture.expected_symptoms
                )

    def test_too_weak_precondition_over_promises(self) -> None:
        fixture = fixture_for(self.config, DefectCategory.TOO_WEAK_PRECONDITION)
        report = self.report_for(fixture.contract)
        self.assertGreater(report.false_accepts, 0)
        self.assertEqual(report.false_rejects, 0)
        self.assertFalse(report.is_sound)
        self.assertTrue(report.is_complete)
        offending = report.counterexamples[0].state
        self.assertFalse(self.config.is_valid_address(offending.shipping_address))

    def test_too_strong_precondition_under_promises(self) -> None:
        fixture = fixture_for(self.config, DefectCategory.TOO_STRONG_PRECONDITION)
        report = self.report_for(fixture.contract)
        self.assertGreater(report.false_rejects, 0)
        self.assertEqual(report.false_accepts, 0)
        self.assertTrue(report.is_sound)
        self.assertFalse(report.is_complete)
        offending = report.counterexamples[0].state
        self.assertNotEqual(offending.payment_method, self.config.valid_payment_methods[0])

    def test_wrong_postcondition_misdescribes_the_effect(self) -> None:
        fixture = fixture_for(self.config, DefectCategory.WRONG_POSTCONDITION)
        report = self.report_for(fixture.contract)
        self.assertEqual(report.false_accepts, 0)
        self.assertEqual(report.false_rejects, 0)
        self.assertEqual(report.postcondition_violations, report.successes)
        self.assertFalse(report.is_sound)

    def test_faulty_merge_breaks_two_sound_branches(self) -> None:
        branches = branch_contracts(self.config)
        self.assertGreaterEqual(len(branches), 2)
        for branch in branches:
            with self.subTest(branch=branch.name):
                report = self.report_for(branch)
                # Sound on its own: never promises an application that fails,
                # never misdescribes an effect. Merely incomplete.
                self.assertEqual(report.false_accepts, 0)
                self.assertEqual(report.postcondition_violations, 0)
                self.assertTrue(report.is_sound)
                self.assertFalse(report.is_complete)

        merged = self.report_for(faulty_merge(branches))
        self.assertEqual(merged.false_accepts, 0)
        self.assertEqual(merged.false_rejects, 0)
        self.assertEqual(merged.postcondition_violations, merged.successes)
        self.assertFalse(merged.is_sound)

    def test_ground_truth_separates_from_every_fixture(self) -> None:
        truth = self.report_for(ground_truth_contract(self.config))
        self.assertTrue(truth.is_exact)
        for fixture in all_fixtures(self.config):
            with self.subTest(fixture=fixture.name):
                report = self.report_for(fixture.contract)
                self.assertFalse(report.is_exact)
                self.assertNotEqual(report.symptoms, truth.symptoms)

    def test_evaluate_all_matches_individual_scoring(self) -> None:
        contracts = [fixture.contract for fixture in all_fixtures(self.config)]
        batch = evaluate_all(contracts, self.states, self.config)
        self.assertEqual(
            [report.summary() for report in batch],
            [self.report_for(contract).summary() for contract in contracts],
        )

    def test_counterexample_records_are_serialisable(self) -> None:
        fixture = fixture_for(self.config, DefectCategory.TOO_WEAK_PRECONDITION)
        report = self.report_for(fixture.contract)
        payload = report.counterexamples[0].to_dict()
        self.assertEqual(payload["symptom"], Symptom.FALSE_ACCEPT.value)
        self.assertIn("order_status", payload["state"])
        self.assertEqual(payload["before_state"], payload["state"])
        self.assertIsNone(payload["after_state"])
        self.assertTrue(payload["detail"])

        post_report = self.report_for(
            fixture_for(self.config, DefectCategory.WRONG_POSTCONDITION).contract
        )
        post_payload = next(
            counterexample.to_dict()
            for counterexample in post_report.counterexamples
            if counterexample.symptom is Symptom.POSTCONDITION_VIOLATION
        )
        self.assertIsNotNone(post_payload["after_state"])
        self.assertTrue(post_payload["changed_fields"])
        self.assertTrue(post_payload["unchanged_fields"])

    def test_counterexample_records_are_capped(self) -> None:
        fixture = fixture_for(self.config, DefectCategory.WRONG_POSTCONDITION)
        report = evaluate_contract(
            fixture.contract, self.states, self.config, max_counterexamples=2
        )
        self.assertEqual(len(report.counterexamples), 2)
        self.assertGreater(report.postcondition_violations, 2)

    def test_counterexample_selection_balances_symptoms(self) -> None:
        buckets = {
            "fa": ["fa0", "fa1", "fa2"],
            "fr": ["fr0"],
            "pv": ["pv0"],
        }
        self.assertEqual(
            round_robin_counterexamples(buckets, ("fa", "fr", "pv"), 3),
            ("fa0", "fr0", "pv0"),
        )

    def test_transition_diff_is_observation_only(self) -> None:
        changed, unchanged = transition_diff(
            {"balance": 10, "status": "none"},
            {"balance": 10, "status": "placed"},
        )
        self.assertEqual(changed, {"status": {"before": "none", "after": "placed"}})
        self.assertEqual(unchanged, ["balance"])


class MergeOperatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_config()

    def test_merge_needs_at_least_two_branches(self) -> None:
        with self.assertRaises(ValueError):
            faulty_merge(branch_contracts(self.config)[:1])

    def test_merge_refuses_mixed_actions(self) -> None:
        branch = payment_branch_contract(self.config, "card")
        other = Contract(
            name="other",
            action=ShoppingAction.cancel_order(),
            precondition=lambda state: True,
            postcondition=lambda before, after: True,
        )
        with self.assertRaises(ValueError):
            faulty_merge([branch, other])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
