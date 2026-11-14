"""Bounded, observation-only symbolic learner. No environment or reference calls."""
from __future__ import annotations

import heapq
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiment.deployment.dsl import VAR_KINDS, parse_contract_dict
from experiment.deployment.information_policy import InformationPolicy
from experiment.deployment.state import DeploymentState, DeploymentStatus, counts_from_mapping

POLICY = InformationPolicy('restricted_vp', False, False)
FIELDS = ('authenticated', 'allocated_resources', 'available_quota', 'active_deployment',
          'target_region', 'cluster_tier', 'deployment_status')
LIMITS = {'max_clauses': 12, 'max_expansions': 100000, 'max_steps': 1000000, 'seconds': 30.0}
TRUE = {'op': 'const', 'value': True}
FALSE = {'op': 'const', 'value': False}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def nodes(value):
    if isinstance(value, dict):
        return 1 + sum(nodes(v) for k, v in value.items() if k != 'const' and isinstance(v, (dict, list)))
    if isinstance(value, list):
        return sum(nodes(v) for v in value)
    return 0


def var(name, when='before'):
    return {'var': name, 'when': when}


def eq(left, right):
    return {'op': 'eq', 'left': left, 'right': right}


class NoConfig:
    def __getattr__(self, name):
        raise AssertionError('Private/config lookup prohibited: ' + name)


def compile_contract(pre, post):
    return parse_contract_dict({'skill': 'deploy_service', 'precondition': pre,
                                'postcondition': post}, information_policy=POLICY).bind(NoConfig())


def state(raw):
    if set(raw) != set(FIELDS) or type(raw['authenticated']) is not bool:
        raise ValueError('Malformed public state')
    values = dict(raw)
    for key in FIELDS:
        kind = VAR_KINDS[key]
        if kind == 'multiset':
            if not isinstance(values[key], dict) or any(type(v) is not int or v < 0 for v in values[key].values()):
                raise ValueError('Malformed multiset')
            values[key] = counts_from_mapping(values[key])
        elif kind == 'option' and values[key] is not None and not isinstance(values[key], str):
            raise ValueError('Malformed option')
    values['deployment_status'] = DeploymentStatus(values['deployment_status'])
    return DeploymentState(**values)


def validate_public(public):
    allowed = {'variables', 'statuses', 'region_options', 'tier_options', 'resource_types',
               'max_quota', 'max_allocation_limit'}
    if set(public) != allowed or public['variables'] != dict(VAR_KINDS):
        raise ValueError('Public vocabulary mismatch or extra fields')
    for k in ('statuses', 'region_options', 'tier_options', 'resource_types'):
        values = public[k]
        if not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values) or len(values) != len(set(values)):
            raise ValueError('Invalid public enum')
    for k in ('max_quota', 'max_allocation_limit'):
        if type(public[k]) is not int or public[k] < 1:
            raise ValueError('Invalid public bound')


def vocabulary(public):
    validate_public(public)
    typed = {}
    for kind in ('bool', 'number', 'multiset', 'text'):
        typed[kind] = [var(k) for k, t in sorted(public['variables'].items())
                       if t == kind or (kind == 'text' and t == 'option')]
    literals = {'bool': [False, True], 'number': sorted({0, 1, public['max_quota'], public['max_allocation_limit']}),
                'multiset': [{}], 'text': [None] + sorted(set(public['statuses'] + public['region_options'] + public['tier_options']))}
    atoms = [TRUE, FALSE]
    for kind, terms in typed.items():
        for a in terms:
            if kind == 'bool':
                atoms.extend([{'op': 'is_true', 'arg': a}, {'op': 'not', 'arg': {'op': 'is_true', 'arg': a}}])
            if kind == 'multiset':
                atoms.extend([{'op': 'is_empty', 'arg': a}, {'op': 'not', 'arg': {'op': 'is_empty', 'arg': a}}])
            for b in terms + [{'const': c} for c in literals[kind]]:
                if a == b:
                    continue
                ops = ('eq', 'ne', 'lt', 'le', 'gt', 'ge') if kind == 'number' else ('eq', 'ne')
                atoms.extend({'op': op, 'left': a, 'right': b} for op in ops)
        if kind == 'multiset':
            atoms.extend({'op': 'covers', 'available': a, 'required': b}
                         for a in terms for b in terms if a != b)
    atoms = sorted({canonical(a): a for a in atoms}.values(), key=lambda a: (nodes(a), canonical(a)))
    rhs = {}
    for field in FIELDS:
        kind = public['variables'][field]
        if kind == 'option':
            kind = 'text'
        terms = typed[kind] + [{'const': c} for c in literals[kind]]
        if kind == 'multiset':
            terms += [{'op': op, 'left': a, 'right': b} for op in ('difference', 'union')
                      for a in typed[kind] for b in typed[kind]]
        rhs[field] = sorted({canonical(t): t for t in terms}.values(), key=lambda t: (nodes(t), canonical(t)))
    return atoms, rhs


class LimitReached(Exception):
    pass


def synthesize(packet, *, limits=None):
    if set(packet) != {'public', 'observations'}:
        raise ValueError('Learner accepts only public vocabulary and observations')
    cap = {**LIMITS, **(limits or {})}
    if any(cap[k] < 0 for k in cap):
        raise ValueError('Negative limit')
    start = time.monotonic()
    counts = {'expansions': 0, 'steps': 0, 'atom_evaluations': 0, 'post_evaluations': 0}

    def check():
        if time.monotonic() - start >= cap['seconds']:
            raise LimitReached('time_limit')
        if counts['expansions'] >= cap['max_expansions'] or counts['steps'] >= cap['max_steps']:
            raise LimitReached('search_limit')

    def result(status, **more):
        return {'status': status, 'counts': counts, 'limits': cap,
                'elapsed_seconds': time.monotonic() - start, **more}

    seen = {}
    for obs in packet['observations']:
        if set(obs) != {'before', 'after'}:
            raise ValueError('Unexpected observation field')
        before = state(obs['before'])
        after = state(obs['after']) if obs['after'] is not None else None
        key = canonical(obs['before'])
        if key in seen and seen[key][1] != after:
            return result('inconsistent_observations')
        seen[key] = before, after
    observations = [seen[k] for k in sorted(seen)]
    positive = [(b, a) for b, a in observations if a is not None]
    negative = [b for b, a in observations if a is None]
    atoms, rhs = vocabulary(packet['public'])
    counts.update(observations=len(observations), positives=len(positive), negatives=len(negative),
                  atoms=len(atoms), rhs_terms=sum(map(len, rhs.values())))
    try:
        check()
        # Equal rejection masks are observationally equivalent for this conjunction objective.
        covers = {}
        for atom in atoms:
            check()
            evaluator = compile_contract(atom, TRUE)
            counts['atom_evaluations'] += len(observations)
            if not all(evaluator.holds_in(b) for b, _ in positive):
                continue
            mask = sum(1 << i for i, b in enumerate(negative) if not evaluator.holds_in(b))
            if mask and mask not in covers:
                covers[mask] = atom
        target = (1 << len(negative)) - 1
        heap = [(0, 0, (), 0)]
        best = {(0, 0): (0, ())}
        chosen = None
        by_key = {canonical(a): a for a in covers.values()}
        while heap:
            check()
            size, cost, keys, mask = heapq.heappop(heap)
            if best.get((mask, size)) != (cost, keys):
                continue
            counts['expansions'] += 1
            if mask == target:
                chosen = [by_key[k] for k in keys]
                break
            if size >= cap['max_clauses']:
                continue
            for bitmask, atom in sorted(covers.items()):
                check()
                counts['steps'] += 1
                newmask = mask | bitmask
                if newmask == mask:
                    continue
                newkeys = tuple(sorted(keys + (canonical(atom),)))
                rank = cost + nodes(atom), newkeys
                key = newmask, size + 1
                if key not in best or rank < best[key]:
                    best[key] = rank
                    heapq.heappush(heap, (size + 1, rank[0], newkeys, newmask))
        if chosen is None:
            return result('no_consistent_precondition')
        pre = {'op': 'and', 'args': chosen} if chosen else TRUE
        assignments, ambiguity = [], {}
        if positive:
            for field in FIELDS:
                consistent = []
                for term in rhs[field]:
                    check()
                    formula = eq(var(field, 'after'), term)
                    evaluator = compile_contract(TRUE, formula)
                    counts['post_evaluations'] += len(positive)
                    if all(evaluator.transition_holds(b, a) for b, a in positive):
                        consistent.append(term)
                ambiguity[field] = len(consistent)
                if not consistent:
                    return result('no_consistent_postcondition', field=field, ambiguity=ambiguity)
                assignments.append(eq(var(field, 'after'), consistent[0]))
        post = {'op': 'and', 'args': assignments} if assignments else TRUE
        spec = {'skill': 'deploy_service', 'precondition': pre, 'postcondition': post}
        parsed = parse_contract_dict(spec, information_policy=POLICY)
        contract = parsed.bind(NoConfig())
        assert all(contract.holds_in(b) == (a is not None) for b, a in observations)
        assert all(contract.transition_holds(b, a) for b, a in positive)
        return result('consistent', spec=spec, dsl_nodes=parsed.node_count,
                      post_identified_from_success=bool(positive), ambiguity=ambiguity)
    except LimitReached as exc:
        return result(str(exc))


if __name__ == '__main__':
    print(json.dumps(synthesize(json.load(sys.stdin)), sort_keys=True))
