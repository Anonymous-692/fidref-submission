#!/usr/bin/env python3
"""Finite Deployment pool-size queue; no changes to legacy experiment paths."""
import argparse
import copy
import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_st2x2 import (build_runner, make_session, make_ask, _adopt_final,
    RunSpec, Decoding, Budgets, ST2X2_PARTITIONED_EXHAUST, ST2X2_UNIFORM_EXHAUST)
from experiment.modeling import g2_runtime as g2
from experiment.modeling.st2x2 import run_st2x2
from experiment.modeling.st2x2_domains import deployment_adapter, taubench_adapter

VERSION = 'partial_candidate_pool_v1'
SIZES = (200, 1000, 5000, 11712)
METHODS = (ST2X2_UNIFORM_EXHAUST, ST2X2_PARTITIONED_EXHAUST)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=1, default=str))
    tmp.replace(path)


def indices(n, seed, size):
    if size > n or size < 48:
        raise ValueError('invalid pool size')
    order = list(range(n))
    random.Random(f'{VERSION}|{seed}').shuffle(order)
    return order[:size]


def domain_adapter(runner):
    return (taubench_adapter if getattr(runner, 'partial_pool_domain', 'deployment') == 'taubench'
            else deployment_adapter)(runner)


def restricted(runner, seed, size):
    ids = indices(len(runner.states), seed, size)
    local = copy.copy(runner)
    local.states = tuple(runner.states[i] for i in ids)
    adapter = domain_adapter(local)
    allowed = set(local.states)
    audits = []
    select, score = adapter.select, adapter.score_subset

    def bounded_select(*args, **kwargs):
        states = select(*args, **kwargs)
        assert set(states) <= allowed, 'selector escaped pool'
        return states

    def bounded_score(parsed, states):
        assert set(states) <= allowed, 'feedback escaped pool'
        audits.append([ids[local.states.index(s)] for s in states])
        return score(parsed, states)

    def forbidden(*args, **kwargs):
        raise AssertionError('full evaluation called during synthesis')

    return replace(adapter, select=bounded_select, score_subset=bounded_score,
                   score_closure=forbidden), ids, audits


def spec(seed, method):
    return RunSpec(method=method, seed=seed,
        decoding=Decoding(temperature=0.2, top_p=0.95, max_tokens=2048),
        budgets=Budgets(state_budget=48, query_budget=4, token_budget=16000),
        use_guided_json=True)


def build(args):
    runner, _, _ = build_runner(args.domain, args.base_url, args.model, True)
    runner.partial_pool_domain = args.domain
    runner.compact_context = True
    g2.configure(runner, 8192, 256)
    if args.domain == 'taubench':
        for path in (ROOT / 'experiment/taubench_retail').glob('*.py'):
            runner.g2_runtime['sources_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    runner.client.chat_template_kwargs = {'enable_thinking': False}
    runner.client.set_structured_output_style_from_version('0.19.1')
    runner.serving_metadata.update(vllm_version='0.19.1', dtype='bfloat16',
        tensor_parallel_size=1, max_model_len=8192, enable_thinking=False,
        enforce_eager=args.model == 'gemma-4-31b-it')
    assert len(runner.states) == (1294 if args.domain == 'taubench' else 11712)
    summary = runner.enumeration_summary
    summary = summary() if callable(summary) else summary
    assert not summary['truncated']
    return runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', default='http://127.0.0.1:18011/v1')
    ap.add_argument('--model', default='gemma-4-31b-it')
    ap.add_argument('--domain', choices=['deployment', 'taubench'], default='deployment')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    runner = build(args)
    sizes = (200, 500, 1294) if args.domain == 'taubench' else SIZES
    if args.check_only:
        fixture = 'b16_taubench_gemma_st2x2' if args.domain == 'taubench' else 'b16_deployment_gemma_st2x2'
        row = json.loads((ROOT / 'results' / fixture / 'st2x2_partitioned_exhaust__seed0.json').read_text())
        source = row['contract']['source']
        def no_call(*a, **kw):
            raise AssertionError('CPU test attempted a model call')
        for size in sizes:
            for balance in (False, True):
                adapter, ids, audits = restricted(runner, 0, size)
                assert ids == indices(len(runner.states), 0, len(runner.states))[:size]
                result = run_st2x2(adapter, ask=no_call, seed=0, state_budget=48,
                    query_budget=4, balance=balance, exhaust=True, initial_source=source)
                assert result['states_observed'] == 48
                assert result['model_calls'] == 1
                assert all(set(a) <= set(ids) for a in audits)
        print(f'CPU gate: {len(sizes)*2} cases passed; pool bounds, nesting, exhaust, no model calls', flush=True)
        return
    protocol = {'version': VERSION, 'domain': args.domain, 'sizes': sizes, 'seeds': list(range(20)),
        'methods': METHODS, 'state_budget': 48, 'query_budget': 4, 'token_budget': 16000,
        'max_tokens': 2048, 'temperature': 0.2, 'top_p': 0.95,
        'pool_policy': 'seeded uniform nested prefixes, no ground-truth stratification',
        'termination': 'exhaust', 'coverage': False,
        'partition_shortage': ('no refill when both partitions nonempty' if args.domain == 'taubench'
                              else 'fill from remaining pool'),
        'evaluation': 'full closure after synthesis only', 'runtime': runner.g2_runtime,
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'serving': runner.serving_metadata}
    manifest = args.output / 'manifest.json'
    frozen = json.loads(json.dumps(protocol))
    if manifest.exists():
        assert json.loads(manifest.read_text()) == frozen, 'manifest changed; use a new version'
    else:
        save(manifest, protocol)

    def initial(seed):
        target = args.output / 'initial' / f'seed{seed}.json'
        if target.exists():
            entry = json.loads(target.read_text()); g2.validate_initial(entry); return entry
        session = make_session(runner, spec(seed, METHODS[1]), args.domain)
        session.stopped_because = ''
        source = make_ask(session, args.domain)('synthesis', domain_adapter(runner).direct_prompt())
        inter = session.interactions[-1] if session.interactions else None
        entry = {'model': args.model, 'seed': seed, 'source': source,
            'source_sha256': hashlib.sha256(source.encode()).hexdigest() if source else None,
            'usage': dict(getattr(inter, 'usage', {}) or {}), **g2.initial_metadata(session)}
        entry['record_sha256'] = g2.record_digest(entry)
        save(target, entry)
        if entry['initial_error'] or not entry['usage_known']:
            raise RuntimeError(f'initial transport/usage failure seed {seed}; preserved, no redraw')
        print(f'initial seed={seed} saved', flush=True)
        return entry

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        entries = list(pool.map(initial, range(20)))

    def run(job):
        size, method, seed = job
        target = args.output / f'pool{size}' / f'{method}__seed{seed}.json'
        entry = entries[seed]
        if target.exists():
            old = json.loads(target.read_text())
            assert old['partial_pool']['initial_record_sha256'] == entry['record_sha256']
            return
        adapter, ids, audits = restricted(runner, seed, size)
        session = make_session(runner, spec(seed, method), args.domain)
        session.stopped_because = ''
        g2.replay_initial(session, entry)
        result = run_st2x2(adapter, ask=make_ask(session, args.domain), seed=seed,
            state_budget=48, query_budget=4, balance=method == METHODS[1], exhaust=True,
            initial_source=entry['source'])
        session.rounds.extend(result['rounds'])
        session.stopped_because = session.stopped_because or result['stopped_because']
        g2.sync_ledger(session, result)
        _adopt_final(session, adapter, result, entry)
        payload = asdict(runner._finish(session))
        payload['partial_pool'] = {'version': VERSION, 'size': size, 'indices': ids,
            'pool_sha256': g2.digest(ids), 'audited_indices_by_report': audits,
            'initial_record_sha256': entry['record_sha256'], 'initial_contract_sha256': entry['source_sha256'],
            'selection': result['selection'], 'termination': result['termination']}
        payload['g2_accounting'] = g2.accounting(session, entry, result)
        payload['hashes']['base_protocol_sha256'] = payload['hashes'].get('protocol_sha256')
        payload['hashes']['protocol_sha256'] = g2.digest({'protocol': protocol, 'size': size, 'method': method})
        save(target, payload)
        print(f'pool={size} method={method} seed={seed} saved', flush=True)
        if any(i.error for i in session.interactions):
            raise RuntimeError(f'transport failure preserved at {target}')

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Iterate futures so an exception stops submission of later pool sizes.
        for size in sizes:
            list(pool.map(run, [(size, method, seed) for seed in range(20) for method in METHODS]))
    print(f'COMPLETE {len(sizes)*40}/{len(sizes)*40}', flush=True)


if __name__ == '__main__':
    main()
