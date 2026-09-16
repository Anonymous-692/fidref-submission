"""Regression tests for the post-result B16 analysis (no model calls)."""
import json
from pathlib import Path
import unittest

from analysis.b16.paired_tests import exact_mcnemar, holm, summarize, COND, CELLS


class B16StatisticsTest(unittest.TestCase):
    def test_exact_binomial_known_counts(self):
        self.assertEqual(exact_mcnemar(19, 0), 2 / 2**19)
        self.assertEqual(exact_mcnemar(7, 0), 0.015625)
        self.assertEqual(exact_mcnemar(0, 0), 1)
        self.assertEqual(exact_mcnemar(1, 1), 1)
        self.assertEqual(exact_mcnemar(0, 2), 0.5)

    def test_holm_order_and_step_down(self):
        self.assertEqual(holm([0.04, 0.01, 0.03, 1]), [0.09, 0.04, 0.09, 1])
        self.assertEqual(holm([0.01, 0.01, 1]), [0.03, 0.03, 1])

    def test_missing_seed_is_not_silently_dropped(self):
        rows = {f'{d}|{m}|{c}': {str(s): {'exact': False} for s in range(20)}
                for d, m in CELLS for c in COND}
        result = summarize(rows)
        self.assertEqual(len(result['comparisons']), 16)
        self.assertTrue(all(c['no_discordant_pairs'] for c in result['comparisons']))
        del rows[next(iter(rows))]['19']
        with self.assertRaises(ValueError):
            summarize(rows)

    def test_saved_results_and_parse_failure_denominator(self):
        path = Path(__file__).resolve().parents[2] / 'analysis/b16/rows.json'
        if not path.exists():
            self.skipTest('B16 artifacts not installed')
        rows = json.loads(path.read_text())
        result = summarize(rows)
        self.assertTrue(all(c['n'] == 20 for c in result['comparisons']))
        significant = [c for c in result['comparisons'] if c['holm_p'] < .05]
        self.assertEqual(len(significant), 4)
        self.assertTrue(all(c['model'] == 'Gemma-4-31B' and c['factor'] == 'selection' for c in significant))
        self.assertEqual(sum(r['outcome'] == 'parse_failure' for cell in rows.values() for r in cell.values()), 12)
        self.assertEqual([c['difference_in_differences_pp'] for c in result['interactions']], [0, -5, -5, 0])


if __name__ == '__main__':
    unittest.main()
