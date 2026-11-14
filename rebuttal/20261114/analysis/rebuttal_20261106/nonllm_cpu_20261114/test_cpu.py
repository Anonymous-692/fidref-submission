import copy
import itertools
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import evidence
import learner as l

HERE = Path(__file__).resolve().parent
SMOKE = l.ROOT/'results/rebuttal_info_20261113_smoke_v2/restricted_vp/vminus_pminus'


def public():
    return {'variables': dict(l.VAR_KINDS), 'statuses': ['idle', 'deployed', 'stopped', 'rolledback'],
            'region_options': ['region_a', 'region_b', 'region_c'],
            'tier_options': ['tier_a', 'tier_b', 'tier_c'], 'resource_types': ['cpu', 'ram'],
            'max_quota': 2, 'max_allocation_limit': 2}


def before(**kw):
    return {'authenticated': True, 'allocated_resources': {'cpu': 1}, 'available_quota': {'cpu': 2},
            'active_deployment': {}, 'target_region': 'region_a', 'cluster_tier': 'tier_b',
            'deployment_status': 'idle', **kw}


def success(b):
    # Synthetic identity transition, deliberately not the real deployment rule.
    return {'before': b, 'after': copy.deepcopy(b)}


def stable(r):
    return {k: v for k, v in r.items() if k != 'elapsed_seconds'}


class CpuTests(unittest.TestCase):
    def test_no_observations(self):
        r = l.synthesize({'public': public(), 'observations': []})
        self.assertEqual(r['status'], 'consistent')
        self.assertEqual(r['spec']['precondition'], l.TRUE)
        self.assertFalse(r['post_identified_from_success'])

    def test_single_class(self):
        for obs in ([success(before())], [{'before': before(), 'after': None}]):
            r = l.synthesize({'public': public(), 'observations': obs})
            self.assertEqual(r['status'], 'consistent')
            c = l.compile_contract(r['spec']['precondition'], r['spec']['postcondition'])
            self.assertEqual(c.holds_in(l.state(obs[0]['before'])), obs[0]['after'] is not None)

    def test_mixed_order_and_duplicate_invariance(self):
        obs = [success(before()), {'before': before(authenticated=False), 'after': None},
               {'before': before(deployment_status='stopped'), 'after': None}]
        a = l.synthesize({'public': public(), 'observations': obs})
        b = l.synthesize({'public': public(), 'observations': list(reversed(obs)) + obs})
        self.assertEqual(stable(a), stable(b))
        self.assertEqual(a['status'], 'consistent')

    def test_minimum_precondition_against_bruteforce(self):
        obs = [success(before()), {'before': before(authenticated=False), 'after': None},
               {'before': before(deployment_status='stopped'), 'after': None}]
        r = l.synthesize({'public': public(), 'observations': obs})
        atoms, _ = l.vocabulary(public())
        pos = l.state(obs[0]['before'])
        neg = [l.state(o['before']) for o in obs[1:]]
        masks = {}
        for atom in atoms:
            c = l.compile_contract(atom, l.TRUE)
            if c.holds_in(pos):
                mask = tuple(not c.holds_in(s) for s in neg)
                if any(mask) and mask not in masks:
                    masks[mask] = atom
        candidates = []
        for size in range(1, 3):
            for combo in itertools.combinations(masks, size):
                if all(any(mask[i] for mask in combo) for i in range(len(neg))):
                    chosen = [masks[k] for k in combo]
                    candidates.append((size, sum(l.nodes(a) for a in chosen), tuple(sorted(l.canonical(a) for a in chosen))))
        args = r['spec']['precondition']['args']
        self.assertEqual((len(args), sum(l.nodes(a) for a in args), tuple(sorted(l.canonical(a) for a in args))), min(candidates))

    def test_conflicting_observations(self):
        for bad in ({'before': before(), 'after': None}, success(before()) | {'after': before(authenticated=False)}):
            r = l.synthesize({'public': public(), 'observations': [success(before()), bad]})
            self.assertEqual(r['status'], 'inconsistent_observations')

    def test_limits(self):
        p = {'public': public(), 'observations': [success(before())]}
        self.assertEqual(l.synthesize(p, limits={'seconds': 0})['status'], 'time_limit')
        self.assertEqual(l.synthesize(p, limits={'max_steps': 0})['status'], 'search_limit')
        p['observations'].append({'before': before(authenticated=False), 'after': None})
        self.assertEqual(l.synthesize(p, limits={'max_clauses': 0})['status'], 'no_consistent_precondition')

    def test_private_input_rejected(self):
        p = public() | {'valid_regions': ['region_a']}
        with self.assertRaises(ValueError):
            l.synthesize({'public': p, 'observations': []})
        with self.assertRaises(ValueError):
            l.synthesize({'public': public(), 'observations': [], 'evaluation': {}})

    def test_all_grammar_members_parse_and_no_config_access(self):
        atoms, rhs = l.vocabulary(public())
        for atom in atoms:
            c = l.compile_contract(atom, l.TRUE)
            c.holds_in(l.state(before()))
        for field, terms in rhs.items():
            for term in terms:
                c = l.compile_contract(l.TRUE, l.eq(l.var(field, 'after'), term))
                c.transition_holds(l.state(before()), l.state(before()))
        with self.assertRaises(AssertionError):
            l.NoConfig().valid_regions

    def test_no_environment_or_evaluator_calls(self):
        with (patch('experiment.deployment.env.apply_action', side_effect=AssertionError('No environment')),
              patch('experiment.deployment.contracts.apply_action', side_effect=AssertionError('No scoring'))):
            r = l.synthesize({'public': public(), 'observations': [success(before())]})
            self.assertEqual(r['status'], 'consistent')

    def test_out_of_fragment_effect(self):
        # Fresh unknown literal is not silently learned from a target output.
        b = before(); a = before(target_region='unlisted-region')
        r = l.synthesize({'public': public(), 'observations': [{'before': b, 'after': a}]})
        self.assertEqual(r['status'], 'no_consistent_postcondition')

    def test_artifact_export_ignores_unseen_information(self):
        r = json.loads((SMOKE/'active_cegis__seed1000.json').read_text())
        original = evidence.export(r)
        r['rounds'] = [{'counterexamples': 'SECRET'}]
        r['evaluation'] = {'exact': 'SECRET'}
        r['contract'] = {'spec': 'SECRET'}
        self.assertEqual(evidence.export(r), original)
        self.assertEqual(original[0]['unique_observations'], 0)
        self.assertGreater(original[-1]['unique_observations'], 0)
        for b in original:
            for o in b['packet']['observations']:
                self.assertEqual(set(o), {'before', 'after'})
        # Extra untransmitted round must never become training evidence.
        self.assertEqual(len(original), len(r['interactions']))

    def test_prompt_policy_and_corrupt_counterexample(self):
        r = json.loads((SMOKE/'active_cegis__seed1000.json').read_text())
        s, u = [m['content'] for m in r['interactions'][0]['messages']]
        with self.assertRaises(ValueError):
            evidence.public_prompt(s, u + '\n- Valid regions: ["region_a"]')
        with self.assertRaises(ValueError):
            evidence.public_prompt(s + '\nCommon valid postcondition structural patterns:', u)
        for text in ('Concrete counterexamples:\nCounterexample 1 (false_accept):\n  Before: {}\n',
                     'Concrete counterexamples:\nCounterexample 1 (false_accept):\n  Before: ' + json.dumps(before()) + '\n  After: null\n'):
            with self.assertRaises(ValueError):
                evidence.transitions(text)


if __name__ == '__main__':
    unittest.main()
