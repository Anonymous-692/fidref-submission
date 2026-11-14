"""Offline verification of the extracted rebuttal release; standard library only."""
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parent
R = 'analysis/rebuttal_20261106/'
sys.dont_write_bytecode = True

def read(path):
    return json.loads((ROOT/path).read_text())

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def score(r):
    m = (r.get('evaluation') or {}).get('metrics')
    if not m: return {'valid': 0, 'exact': 0, 'iou': 0., 'cfr': 0.}
    a, fa, fr, pv = [m[k] for k in ('successes', 'false_accepts', 'false_rejects', 'postcondition_violations')]
    assert a > 0 and min(fa, fr, pv) >= 0 and fr+pv <= a
    return {'valid': 1, 'exact': int(fa == fr == pv == 0), 'iou': 100*(a-fr)/(a+fa), 'cfr': 100*(a-fr-pv)/(a+fa)}

def cell(paths, expected):
    rows = [read(p) for p in paths]
    assert len(rows) == expected['n'] == 20
    assert {r['seed'] for r in rows} == set(range(20))
    values = [score(r) for r in rows]
    assert sum(x['exact'] for x in values) == expected['exact']
    for key in ('iou', 'cfr'):
        assert abs(mean(v[key] for v in values)-expected[key+'_mean']) < 1e-9

def main():
    manifest = read('RELEASE_MANIFEST.json')
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    assert actual == set(manifest['files']) | {'RELEASE_MANIFEST.json'}
    for rel, item in manifest['files'].items():
        p = ROOT/rel
        assert not p.is_symlink() and not Path(rel).is_absolute() and '..' not in Path(rel).parts
        raw = p.read_bytes()
        assert len(raw) == item['bytes'] and sha(raw) == item['sha256'], rel
    catalog = read('RESULT_CATALOG.json')
    assert sum(catalog['cohorts'].values()) == 661
    assert len(catalog['runs']) == 1201
    for row in catalog['runs']:
        r = read(row['path'])
        assert row['method'] == r['method'] and row['seed'] == r['seed']
        assert {k: row[k] for k in ('exact', 'valid', 'iou', 'cfr')} == score(r)
    missing = list(csv.DictReader((ROOT/(R+'reply_review_20261114/MISSING_FILES.csv')).open()))
    assert len(missing) == 542
    for item in missing:
        assert manifest['files'][item['path']]['source_sha256'] == item['sha256']
    # Rebuild 24 historical sweep cells from all 480 records.
    expected_sweep = list(csv.DictReader((ROOT/'analysis/sweep_figure_20260910/all_cells.csv').open()))
    sweep_cells = 0
    for c in expected_sweep:
        b = int(c['budget'])
        if b not in (8, 16, 24): continue
        method = {'active': 'active_cegis', 'no_cov': 'active_cegis_no_coverage',
                  'no_bal': 'active_cegis_no_balance', 'fixed': 'sampled_cegis'}[c['method']]
        folder = f"{c['domain']}_gemma4_sweep_b{b}_s20"
        if c['domain'] == 'taubench':
            folder = f'taubench_retail_gemma4_sweep_b{b}_s20'
            if c['method'] == 'fixed':
                folder = f'taubench_retail_gemma4_fixedpool_b{b}_s20'; method = 'sampled_cegis_fixed'
        exact = sum(score(read(f'results/{folder}/{method}__seed{s}.json'))['exact'] for s in range(20))
        assert exact == int(c['exact'])
        sweep_cells += 1
    assert sweep_cells == 24
    # Deployment original information conditions.
    dep = read(R+'deployment_results_20261113/summary.json')
    for c in dep['cells']:
        paths = [f"results/rebuttal_info_20261113_v2/{c['track']}/{c['condition']}/{c['method']}__seed{s}.json" for s in range(20)]
        cell(paths, c)
    # Retail replacement policy: original 120 plus disjoint 15/7 follow-ups.
    retail = read(R+'reply_draft_20261113/LATEST_INTEGRATION.json')
    assert len(retail['mappings']) == 22
    maps = {m['source']: m['replacement'] for m in retail['mappings']}
    assert len(maps) == 22
    for c in retail['cells']:
        paths = [f"results/rebuttal_retail_info_20261113_v1/{c['condition']}/{c['method']}__seed{s}.json" for s in range(20)]
        cell(paths, c['original'])
        cell([maps.get(p, p) for p in paths], c['length15_plus_budget7_policy'])
    ident = read(R+'identifiable_length3_20261114/integrated_summary.json')
    for c in ident['cells']:
        for layer in ('original', 'budget', 'latest'):
            cell([r['paths'][layer] for r in ident['rows'] if r['method'] == c['method']], c[layer])
    b24 = {}
    for model in ('14b', '32b'):
        values = [score(read(f'results/rebuttal_basic_b24_qwen{model}_20261113_v1/main/active_cegis__seed{s}/result.json')) for s in range(20)]
        b24[model] = {'exact': sum(v['exact'] for v in values), 'iou': mean(v['iou'] for v in values), 'cfr': mean(v['cfr'] for v in values)}
    assert b24['14b']['exact'] == 13 and b24['32b']['exact'] == 20
    # Replay the observation-only learner from all saved request boundaries.
    dev = ROOT/R/'nonllm_cpu_20261114'
    sys.path.insert(0, str(dev)); sys.path.insert(0, str(ROOT))
    import learner
    import evidence
    nonllm = read(R+'nonllm_main_20261114/run/manifest.json')
    boundaries = 0
    for job in nonllm['jobs']:
        packets = evidence.export(read(job['source']))
        for b in packets:
            saved = read(R+f"nonllm_main_20261114/run/learning/{job['id']}/request_{b['interaction']}.json")
            assert saved['boundary'] == b
            replay = learner.synthesize(b['packet'])
            assert {k: v for k, v in replay.items() if k != 'elapsed_seconds'} == {k: v for k, v in saved['result'].items() if k != 'elapsed_seconds'}
            boundaries += 1
    assert boundaries == 139 and len(nonllm['jobs']) == 40
    print(json.dumps({'passed': True, 'files': len(manifest['files']), 'missing_files': len(missing),
                      'result_records': len(catalog['runs']), 'new_and_followup_records': 661,
                      'sweep_cells': sweep_cells, 'deployment_cells': len(dep['cells']),
                      'retail_cells_and_policies': 12, 'identifiable_cells_and_policies': 6,
                      'b24': b24, 'symbolic_request_replays': boundaries, 'model_calls': 0}, indent=2))

if __name__ == '__main__': main()
