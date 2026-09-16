#!/usr/bin/env python3
"""Stage-A regression tests for the contract-repair comparison (conditions A, B, C).

Six groups, matching the handoff's stage-A gate:
  1. patch validation      legal patches apply; illegal op/path/type/cap are refused
  2. adoption              improvement adopted; regression and tie rejected, incumbent kept
  3. budget accounting     rejected patches, parse failures and validation still cost calls
  4. condition parity      same initial candidate, decoding and budget; A/B share one fixed pool
  5. oracle isolation      the loop runs without the reference; closure scoring is post hoc
  6. no regression         existing methods and hash paths untouched (checked elsewhere too)

The reference contract is used only inside this test file, never by the loop.
"""

from __future__ import annotations

import json
import unittest

from ..deployment import dsl
from ..deployment.patch import (
    DEFAULT_INSERT_NODE_CAP,
    DEFAULT_PATCH_CAP,
    PatchError,
    apply_patch,
    count_nodes,
    patch_json_schema,
)
from ..deployment.repair_core import (
    COND_FREE_ACTIVE,
    COND_FREE_POOL,
    COND_PATCH_POOL,
    COND_PATCH_TIE,
    CONDITIONS,
    PATCH_CONDITIONS,
    TIE_TOLERANT_CONDITIONS,
    WITNESS_CONDITIONS,
    STOP_POOL_CLEAN,
    STOP_QUERY_BUDGET,
    Observation,
    RepairAdapter,
    run_repair,
)

BASE = {
    "skill": "deploy_service",
    "precondition": {"op": "and", "args": [
        {"op": "is_true", "arg": {"var": "authenticated"}},
        {"op": "in_set", "value": {"var": "target_region"}, "set": "valid_regions"},
    ]},
    "postcondition": {"op": "and", "args": [
        {"op": "eq", "left": {"var": "deployment_status", "when": "after"},
         "right": {"const": "deployed"}},
        {"op": "unchanged", "vars": ["authenticated", "quota_size"]},
    ]},
}
BASE_SRC = json.dumps(BASE, sort_keys=True)


# --------------------------------------------------------------------------------------
# 1. patch validation
# --------------------------------------------------------------------------------------
class PatchValidationTest(unittest.TestCase):
    def test_legal_edits_apply(self) -> None:
        p = {"edits": [
            {"op": "add_conjunct", "scope": "precondition",
             "formula": {"op": "not", "arg": {"op": "is_empty",
                                              "arg": {"var": "allocated_resources"}}}},
            {"op": "drop_conjunct", "scope": "postcondition", "index": 1},
        ]}
        out = apply_patch(BASE_SRC, p)
        self.assertEqual(out.edits, 2)
        spec = json.loads(out.source)
        self.assertEqual(len(spec["precondition"]["args"]), 3)
        self.assertNotIn("args", spec["postcondition"])  # collapsed to the single conjunct

    def test_retarget_fixes_an_over_strong_frame_condition(self) -> None:
        """The frame list is editable, not only deletable."""
        p = {"edits": [{"op": "retarget", "scope": "postcondition", "path": ["args", 1],
                        "field": "vars", "value": ["authenticated"]}]}
        spec = json.loads(apply_patch(BASE_SRC, p).source)
        self.assertEqual(spec["postcondition"]["args"][1]["vars"], ["authenticated"])

    def test_unknown_operation_is_refused(self) -> None:
        with self.assertRaises(PatchError):
            apply_patch(BASE_SRC, {"edits": [{"op": "rewrite_everything", "scope": "precondition"}]})

    def test_bad_path_and_out_of_range_index_are_refused(self) -> None:
        with self.assertRaises(PatchError):
            apply_patch(BASE_SRC, {"edits": [{"op": "drop_conjunct", "scope": "precondition",
                                              "index": 99}]})
        with self.assertRaises(PatchError):
            apply_patch(BASE_SRC, {"edits": [{"op": "retarget", "scope": "precondition",
                                              "path": ["args", 7, "left"], "field": "var",
                                              "value": "authenticated"}]})

    def test_type_error_after_patching_is_refused(self) -> None:
        """A patch that parses as JSON but breaks the DSL type checker must not be applied."""
        p = {"edits": [{"op": "replace_conjunct", "scope": "precondition", "index": 0,
                        "formula": {"op": "is_true", "arg": {"var": "allocation_size"}}}]}
        with self.assertRaises(PatchError):
            apply_patch(BASE_SRC, p)

    def test_edit_count_cap(self) -> None:
        edits = [{"op": "negate", "scope": "precondition", "index": 0}] * (DEFAULT_PATCH_CAP + 1)
        with self.assertRaises(PatchError):
            apply_patch(BASE_SRC, {"edits": edits})

    def test_inserted_formula_size_cap(self) -> None:
        """Edit count alone is evadable: one huge insert would be a whole rewrite."""
        big = {"op": "and", "args": [
            {"op": "is_true", "arg": {"var": "authenticated"}},
            {"op": "eq", "left": {"var": "allocation_size"}, "right": {"var": "quota_size"}},
            {"op": "covers", "available": {"var": "available_quota"},
             "required": {"var": "allocated_resources"}},
        ]}
        self.assertGreater(count_nodes(big), DEFAULT_INSERT_NODE_CAP)
        with self.assertRaises(PatchError):
            apply_patch(BASE_SRC, {"edits": [{"op": "add_conjunct", "scope": "precondition",
                                              "formula": big}]})

    def test_refuses_to_empty_a_scope(self) -> None:
        one = json.dumps({"skill": "deploy_service",
                          "precondition": {"op": "is_true", "arg": {"var": "authenticated"}},
                          "postcondition": {"op": "const", "value": True}})
        with self.assertRaises(PatchError):
            apply_patch(one, {"edits": [{"op": "drop_conjunct", "scope": "precondition",
                                         "index": 0}]})

    def test_guided_schema_uses_only_string_type_values(self) -> None:
        """xgrammar rejects union types; a bad schema makes every patch call a transport error.

        The pilot lost condition B to exactly this (HTTP 400: 'type' must be a string), so the
        shape of the schema is checked here rather than only at serving time.
        """
        def offenders(node, path="$"):
            found = []
            if isinstance(node, dict):
                if "type" in node and not isinstance(node["type"], str):
                    found.append((path, node["type"]))
                for k, v in node.items():
                    found += offenders(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    found += offenders(v, f"{path}[{i}]")
            return found

        self.assertEqual(offenders(patch_json_schema()), [])

    def test_guided_schema_avoids_keywords_xgrammar_cannot_compile(self) -> None:
        """Item limits and property bounds make vLLM fall back to outlines, which then hangs.

        Measured on the serving box: the same request returned in 3.4 s without these keywords
        and timed out at 150 s with them, GPU idle throughout. The caps they expressed are
        enforced in ``apply_patch`` instead, so nothing is lost by dropping them here.
        """
        unsupported = ("minItems", "maxItems", "minimum", "maximum", "minLength", "maxLength",
                       "minProperties", "maxProperties", "pattern", "multipleOf",
                       "exclusiveMinimum", "exclusiveMaximum")
        blob = json.dumps(patch_json_schema())
        present = [k for k in unsupported if f'"{k}"' in blob]
        self.assertEqual(present, [], f"schema would fall back to outlines: {present}")

    def test_edit_cap_is_enforced_in_code_not_only_in_the_schema(self) -> None:
        """Since the schema no longer carries maxItems, the cap must hold at apply time."""
        edits = [{"op": "negate", "scope": "precondition", "index": 0}] * (DEFAULT_PATCH_CAP + 1)
        with self.assertRaises(PatchError):
            apply_patch(BASE_SRC, {"edits": edits})

    def test_path_steps_may_be_strings(self) -> None:
        """Guided decoding emits list indices as strings; they must still address the list."""
        p = {"edits": [{"op": "retarget", "scope": "postcondition", "path": ["args", "1"],
                        "field": "vars", "value": ["authenticated"]}]}
        spec = json.loads(apply_patch(BASE_SRC, p).source)
        self.assertEqual(spec["postcondition"]["args"][1]["vars"], ["authenticated"])

    def test_guided_schema_lists_exactly_the_allowed_operations(self) -> None:
        schema = patch_json_schema()
        ops = schema["properties"]["edits"]["items"]["properties"]["op"]["enum"]
        self.assertEqual(set(ops), {"add_conjunct", "drop_conjunct", "replace_conjunct",
                                    "retarget", "swap_operator", "negate"})
        # The edit cap deliberately lives in apply_patch, not in the schema: expressing it as
        # maxItems makes guided decoding fall back to outlines and hang (see the test below).
        self.assertNotIn("maxItems", schema["properties"]["edits"])


# --------------------------------------------------------------------------------------
# mock domain: D_obs is read off a table keyed by contract source
# --------------------------------------------------------------------------------------
class _Mock:
    def __init__(self, scores: dict[str, int], pool_size: int = 8) -> None:
        self.scores = scores
        self.pool_size = pool_size
        self.pool_calls: list[tuple[int, int]] = []
        self.scored: list[tuple[str, int]] = []

    def fixed_pool(self, seed: int, count: int):
        self.pool_calls.append((seed, count))
        return tuple(range(min(count, self.pool_size)))

    def select_active(self, contract, count, *, seed, audit_index, excluded=()):
        start = len(excluded)
        return tuple(range(start, start + count))


def _mk_adapter(score_table: dict[str, int], mock: _Mock) -> RepairAdapter:
    """Adapter whose observed D comes from ``score_table`` keyed by the contract source."""
    def parse(src: str):
        return dsl.parse_contract_text(src)

    def score_subset(parsed, subset):
        src = getattr(parsed, "_test_src", None) or ""
        d = score_table.get(src, 5)
        mock.scored.append((src, len(subset)))
        return Observation(states_checked=len(subset), false_accepts=d,
                           false_rejects=0, postcondition_violations=0,
                           counterexamples=[])
    def tagged_parse(src: str):
        p = parse(src)
        try:
            object.__setattr__(p, "_test_src", src)
        except Exception:  # noqa: BLE001
            p._test_src = src  # type: ignore[attr-defined]
        return p

    return RepairAdapter(
        parse=tagged_parse, score_subset=score_subset, fixed_pool=mock.fixed_pool,
        select_active=mock.select_active,
        free_rewrite_prompt=lambda s, m, c: "REWRITE",
        patch_prompt=lambda s, m, c: "PATCH",
        patch_repair_prompt=lambda e: "PATCH_REPAIR",
        parse_repair_prompt=lambda s, e: "PARSE_REPAIR",
    )


def _patch_adding(formula: dict) -> str:
    return json.dumps({"edits": [{"op": "add_conjunct", "scope": "precondition",
                                  "formula": formula}]})


IMPROVE = {"op": "is_true", "arg": {"var": "authenticated"}}
NEUTRAL = {"op": "in_set", "value": {"var": "cluster_tier"}, "set": "valid_cluster_tiers"}


class _Asker:
    def __init__(self, replies):
        self.replies = list(replies)
        self.log: list[tuple[str, str | None]] = []

    def __call__(self, role, prompt, schema_kind=None):
        self.log.append((role, schema_kind))
        return self.replies.pop(0) if self.replies else None


# --------------------------------------------------------------------------------------
# 2. adoption
# --------------------------------------------------------------------------------------
class AdoptionTest(unittest.TestCase):
    def _run(self, patch_reply: str, cand_d: int, query_budget: int = 4):
        mock = _Mock({})
        after = apply_patch(BASE_SRC, json.loads(patch_reply)).source
        table = {BASE_SRC: 4, after: cand_d}
        ask = _Asker([patch_reply])
        result = run_repair(_mk_adapter(table, mock), ask=ask, condition=COND_PATCH_POOL,
                            seed=0, state_budget=8, query_budget=query_budget,
                            initial_source=BASE_SRC)
        return result, after

    def test_strict_improvement_is_adopted(self) -> None:
        result, after = self._run(_patch_adding(IMPROVE), cand_d=2)
        self.assertEqual(result["source"], after)
        self.assertEqual(result["patches_adopted"], 1)

    def test_regression_is_rejected_and_incumbent_survives(self) -> None:
        result, _ = self._run(_patch_adding(IMPROVE), cand_d=9)
        self.assertEqual(result["source"], BASE_SRC)
        self.assertEqual(result["patches_adopted"], 0)
        self.assertEqual(result["patches_rejected"], 1)

    def test_tie_is_rejected(self) -> None:
        result, _ = self._run(_patch_adding(IMPROVE), cand_d=4)
        self.assertEqual(result["source"], BASE_SRC, "an equal-D candidate must not be adopted")
        self.assertEqual(result["patches_adopted"], 0)

    def test_free_rewrite_always_replaces(self) -> None:
        """Condition A has no adoption test: that is the difference under study."""
        mock = _Mock({})
        worse = json.dumps({**BASE, "notes": "worse"}, sort_keys=True)
        table = {BASE_SRC: 4, worse: 9}
        ask = _Asker([worse])
        result = run_repair(_mk_adapter(table, mock), ask=ask, condition=COND_FREE_POOL,
                            seed=0, state_budget=8, query_budget=4, initial_source=BASE_SRC)
        self.assertEqual(result["source"], worse)

    def test_clean_pool_stops_without_further_calls(self) -> None:
        mock = _Mock({})
        ask = _Asker([])
        result = run_repair(_mk_adapter({BASE_SRC: 0}, mock), ask=ask, condition=COND_PATCH_POOL,
                            seed=0, state_budget=8, query_budget=4, initial_source=BASE_SRC)
        self.assertEqual(result["stopped_because"], STOP_POOL_CLEAN)
        self.assertEqual(ask.log, [])


# --------------------------------------------------------------------------------------
# 3. budget accounting
# --------------------------------------------------------------------------------------
class BudgetTest(unittest.TestCase):
    def test_invalid_patch_costs_calls_and_leaves_incumbent(self) -> None:
        mock = _Mock({})
        ask = _Asker(['{"edits": [{"op": "nope", "scope": "precondition"}]}',
                      '{"edits": [{"op": "still_nope", "scope": "precondition"}]}'])
        result = run_repair(_mk_adapter({BASE_SRC: 4}, mock), ask=ask,
                            condition=COND_PATCH_POOL, seed=0, state_budget=8,
                            query_budget=4, initial_source=BASE_SRC)
        self.assertEqual(result["source"], BASE_SRC)
        self.assertGreaterEqual(result["model_calls"], 3, "proposal and repair are both charged")
        bad = [r for r in result["rounds"] if r.get("patch_valid") is False]
        self.assertTrue(bad and bad[0]["repair_attempted"])

    def test_query_budget_terminates(self) -> None:
        mock = _Mock({})
        good = _patch_adding(IMPROVE)
        after = apply_patch(BASE_SRC, json.loads(good)).source
        second = _patch_adding(NEUTRAL)
        after2 = apply_patch(after, json.loads(second)).source
        table = {BASE_SRC: 6, after: 4, after2: 3}
        ask = _Asker([good, second, good, second])
        result = run_repair(_mk_adapter(table, mock), ask=ask, condition=COND_PATCH_POOL,
                            seed=0, state_budget=8, query_budget=3, initial_source=BASE_SRC)
        self.assertEqual(result["stopped_because"], STOP_QUERY_BUDGET)
        self.assertLessEqual(result["model_calls"], 3)


# --------------------------------------------------------------------------------------
# 4. condition parity
# --------------------------------------------------------------------------------------
class ParityTest(unittest.TestCase):
    def test_ab_share_one_fixed_pool_drawn_once_and_unchanged(self) -> None:
        for condition in (COND_FREE_POOL, COND_PATCH_POOL):
            mock = _Mock({})
            reply = (_patch_adding(IMPROVE) if condition == COND_PATCH_POOL
                     else json.dumps({**BASE, "notes": "x"}, sort_keys=True))
            after = (apply_patch(BASE_SRC, json.loads(reply)).source
                     if condition == COND_PATCH_POOL else reply)
            ask = _Asker([reply])
            result = run_repair(_mk_adapter({BASE_SRC: 4, after: 1}, mock), ask=ask,
                                condition=condition, seed=7, state_budget=8, query_budget=4,
                                initial_source=BASE_SRC)
            self.assertEqual(mock.pool_calls, [(7, 8)], f"{condition}: pool drawn exactly once")
            sizes = {n for _, n in mock.scored}
            self.assertEqual(sizes, {8}, f"{condition}: every scoring uses the whole fixed pool")
            self.assertEqual(result["states_observed"], 8)

    def test_active_condition_reveals_fresh_states(self) -> None:
        mock = _Mock({})
        reply = json.dumps({**BASE, "notes": "x"}, sort_keys=True)
        ask = _Asker([reply])
        result = run_repair(_mk_adapter({BASE_SRC: 4, reply: 1}, mock), ask=ask,
                            condition=COND_FREE_ACTIVE, seed=0, state_budget=9, query_budget=4,
                            initial_source=BASE_SRC)
        self.assertEqual(mock.pool_calls, [], "condition C must not draw the fixed pool")
        self.assertGreater(result["states_observed"], 3)

    def test_candidate_and_incumbent_are_scored_on_the_same_evidence(self) -> None:
        mock = _Mock({})
        reply = json.dumps({**BASE, "notes": "x"}, sort_keys=True)
        ask = _Asker([reply])
        run_repair(_mk_adapter({BASE_SRC: 4, reply: 1}, mock), ask=ask,
                   condition=COND_FREE_ACTIVE, seed=0, state_budget=9, query_budget=4,
                   initial_source=BASE_SRC)
        last_two = mock.scored[-2:]
        self.assertEqual(last_two[0][1], last_two[1][1],
                         "incumbent and candidate must be re-scored on the same evidence size")

    def test_all_three_conditions_start_from_the_same_source_and_charge_it(self) -> None:
        for condition in (COND_FREE_POOL, COND_PATCH_POOL, COND_FREE_ACTIVE):
            mock = _Mock({})
            ask = _Asker([])
            result = run_repair(_mk_adapter({BASE_SRC: 0}, mock), ask=ask, condition=condition,
                                seed=0, state_budget=8, query_budget=4, initial_source=BASE_SRC)
            self.assertEqual(result["rounds"][0]["incumbent_sha256"],
                             _sha_of(BASE_SRC), condition)
            self.assertEqual(result["model_calls"], 1,
                             f"{condition}: the shared round-0 generation is charged once")


def _sha_of(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TieToleranceTest(unittest.TestCase):
    """B'': the only difference from B' is that an observed tie is adopted."""

    def _run(self, condition, cand_d, incumbent_d=4):
        mock = _Mock({})
        reply = _patch_adding(IMPROVE)
        after = apply_patch(BASE_SRC, json.loads(reply)).source
        ask = _Asker([reply])
        kwargs = {"evidence": tuple(range(8))} if condition in ("repair_patch_witness",
                                                                COND_PATCH_TIE) else {}
        return run_repair(_mk_adapter({BASE_SRC: incumbent_d, after: cand_d}, mock), ask=ask,
                          condition=condition, seed=0, state_budget=8, query_budget=4,
                          initial_source=BASE_SRC, **kwargs), after

    def test_tie_is_adopted_under_the_tie_tolerant_rule(self) -> None:
        result, after = self._run(COND_PATCH_TIE, cand_d=4)
        self.assertEqual(result["source"], after, "an equal-D candidate must be adopted here")
        rounds = [r for r in result["rounds"] if r.get("round", 0) > 0]
        self.assertTrue(rounds[0]["tie_adopted"])
        self.assertEqual(rounds[0]["adoption_rule"], "non_increasing_d_obs")

    def test_regression_is_still_rejected_under_the_tie_tolerant_rule(self) -> None:
        result, _ = self._run(COND_PATCH_TIE, cand_d=9)
        self.assertEqual(result["source"], BASE_SRC, "a worse candidate must still be rejected")

    def test_strict_condition_is_unchanged(self) -> None:
        result, _ = self._run("repair_patch_witness", cand_d=4)
        self.assertEqual(result["source"], BASE_SRC, "the strict rule must still reject a tie")

    def test_only_the_declared_condition_tolerates_ties(self) -> None:
        self.assertEqual(TIE_TOLERANT_CONDITIONS, (COND_PATCH_TIE,))


class DuplicateAccountingTest(unittest.TestCase):
    """The duplicate guard fires on the resulting contract, not on the patch text."""

    def test_same_patch_on_a_different_incumbent_is_not_a_duplicate(self) -> None:
        first = _patch_adding(IMPROVE)
        second = _patch_adding(NEUTRAL)
        s1 = apply_patch(BASE_SRC, json.loads(first)).source
        s2 = apply_patch(s1, json.loads(second)).source
        mock = _Mock({})
        ask = _Asker([first, second])
        result = run_repair(_mk_adapter({BASE_SRC: 6, s1: 4, s2: 2}, mock), ask=ask,
                            condition=COND_PATCH_TIE, seed=0, state_budget=8, query_budget=4,
                            initial_source=BASE_SRC, evidence=tuple(range(8)))
        rounds = [r for r in result["rounds"] if r.get("round", 0) > 0]
        self.assertEqual(len(rounds), 2)
        self.assertNotEqual(rounds[0]["applied_to_sha256"], rounds[1]["applied_to_sha256"])
        self.assertFalse(rounds[1].get("guard"))

    def test_adoption_makes_the_same_patch_text_a_different_edit(self) -> None:
        """After adoption the incumbent moved, so repeating the patch is not a repeat."""
        patch = _patch_adding(IMPROVE)
        s1 = apply_patch(BASE_SRC, json.loads(patch)).source
        s2 = apply_patch(s1, json.loads(patch)).source
        mock = _Mock({})
        ask = _Asker([patch, patch])
        result = run_repair(_mk_adapter({BASE_SRC: 6, s1: 4, s2: 3}, mock), ask=ask,
                            condition=COND_PATCH_TIE, seed=0, state_budget=8, query_budget=4,
                            initial_source=BASE_SRC, evidence=tuple(range(8)))
        rounds = [r for r in result["rounds"] if r.get("round", 0) > 0]
        self.assertTrue(rounds[1]["patch_text_seen_before"], "the text repeat must be recorded")
        self.assertNotEqual(rounds[1]["result_sha256"], rounds[0]["result_sha256"])
        self.assertIsNone(rounds[1].get("guard"), "a different result is not a duplicate")

    def test_rejected_patch_repeated_on_the_same_incumbent_trips_the_guard(self) -> None:
        """When the incumbent did not move, the same patch reproduces the same contract."""
        patch = _patch_adding(IMPROVE)
        s1 = apply_patch(BASE_SRC, json.loads(patch)).source
        mock = _Mock({})
        ask = _Asker([patch, patch])
        result = run_repair(_mk_adapter({BASE_SRC: 4, s1: 9}, mock), ask=ask,
                            condition=COND_PATCH_TIE, seed=0, state_budget=8, query_budget=4,
                            initial_source=BASE_SRC, evidence=tuple(range(8)))
        rounds = [r for r in result["rounds"] if r.get("round", 0) > 0]
        self.assertFalse(rounds[0]["adopted"], "a worse candidate is rejected even here")
        guard = [r for r in rounds if r.get("guard") == "duplicate_output"]
        self.assertTrue(guard, "the incumbent did not move, so the repeat is a duplicate")
        self.assertEqual(guard[0]["duplicate_basis"], "identical resulting contract")
        self.assertTrue(guard[0]["patch_text_seen_before"])


class AdoptionLabelTest(unittest.TestCase):
    """Every patch condition must be recognised as one, by the condition set, not by name.

    Naming a single condition in the driver mislabelled repair_patch_witness as
    "always_replace" in recorded artifacts while the loop applied the strict rule.
    """

    def test_every_patch_condition_is_in_the_patch_set(self) -> None:
        for cond in CONDITIONS:
            expected = "patch" in cond
            self.assertEqual(cond in PATCH_CONDITIONS, expected, cond)

    def test_patch_conditions_use_the_strict_rule_in_round_records(self) -> None:
        for cond in [c for c in PATCH_CONDITIONS if c not in TIE_TOLERANT_CONDITIONS]:
            mock = _Mock({})
            reply = _patch_adding(IMPROVE)
            after = apply_patch(BASE_SRC, json.loads(reply)).source
            ask = _Asker([reply])
            kwargs = {}
            if cond in WITNESS_CONDITIONS:
                kwargs["evidence"] = tuple(range(8))
            result = run_repair(_mk_adapter({BASE_SRC: 4, after: 1}, mock), ask=ask,
                                condition=cond, seed=0, state_budget=8, query_budget=4,
                                initial_source=BASE_SRC, **kwargs)
            rules = {r.get("adoption_rule") for r in result["rounds"] if r.get("round", 0) > 0}
            self.assertEqual(rules, {"strict_d_obs_decrease"}, cond)


# --------------------------------------------------------------------------------------
# 5. oracle isolation
# --------------------------------------------------------------------------------------
class OracleIsolationTest(unittest.TestCase):
    def test_loop_never_requests_closure_or_reference(self) -> None:
        """The adapter the loop receives exposes no closure scorer and no reference."""
        fields = set(RepairAdapter.__dataclass_fields__)
        self.assertNotIn("score_closure", fields)
        self.assertNotIn("reference", fields)

    def test_scoring_only_ever_sees_audited_subsets(self) -> None:
        mock = _Mock({}, pool_size=8)
        reply = _patch_adding(IMPROVE)
        after = apply_patch(BASE_SRC, json.loads(reply)).source
        ask = _Asker([reply])
        run_repair(_mk_adapter({BASE_SRC: 4, after: 1}, mock), ask=ask,
                   condition=COND_PATCH_POOL, seed=0, state_budget=8, query_budget=4,
                   initial_source=BASE_SRC)
        self.assertTrue(mock.scored)
        self.assertTrue(all(n <= 8 for _, n in mock.scored),
                        "no scoring call may exceed the observed state budget")


if __name__ == "__main__":
    unittest.main()
