"""Offline reproduction entry point for the review artifact."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'


def run(script, *args):
    print(f'Running {script}', flush=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    subprocess.run([sys.executable, script, *args], cwd=ROOT, env=env, check=True)


def verify():
    manifest = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text())
    for rel, entry in manifest['files'].items():
        path = ROOT / rel
        assert path.is_file(), f'Missing file: {rel}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], rel
    rows = defaultdict(list)
    for rel in manifest['files']:
        if not rel.startswith('results/') or not rel.endswith('.json'):
            continue
        obj = json.loads((ROOT / rel).read_text())
        if not {'method', 'seed', 'outcome'} <= obj.keys():
            continue
        rows[(str(Path(rel).parent), obj['method'])].append(obj)
        metrics = (obj.get('evaluation') or {}).get('metrics')
        if metrics and 'exact' in metrics:
            fields = ('false_accepts', 'false_rejects', 'postcondition_violations')
            if all(k in metrics for k in fields):
                assert metrics['exact'] == all(metrics[k] == 0 for k in fields), rel
            if obj['outcome'] == 'exact':
                assert metrics['exact'], rel
    summary = []
    for (suite, method), records in sorted(rows.items()):
        seeds = [r['seed'] for r in records]
        assert len(seeds) == len(set(seeds)), (suite, method, 'duplicate seeds')
        summary.append({'suite': suite, 'method': method, 'n': len(records),
                        'exact': sum(r['outcome'] == 'exact' for r in records),
                        'seeds': sorted(seeds)})
    OUT.mkdir(exist_ok=True)
    (OUT / 'suite_counts.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"Verified {len(manifest['files'])} files and {sum(r['n'] for r in summary)} runs.")


def stats():
    OUT.mkdir(exist_ok=True)
    run('scripts/easy_v4_stats.py', '--out', 'outputs/shopping_stats.txt')
    run('scripts/taubench_retail_stats.py', '--out', 'outputs/retail_stats.txt')
    run('scripts/build_s17_stats.py', '--json-out', 'outputs/original_statistics.json',
        '--md-out', 'outputs/original_statistics.txt', '--tex-out', 'outputs/original_statistics.tex')
    # Execute the original same-seed factorial analysis in its package location.
    # The adapted writer puts generated tables under outputs/.
    run('analysis/b16/paired_tests.py')
    run('analysis/b10/check.py')
    followup_counts()


def followup_counts():
    """Recount descriptive follow-ups within their original conditions."""
    checked = []

    def check(label, paths, expected):
        records = [json.loads((ROOT / p).read_text()) for p in paths]
        assert len(records) == expected['n'], (label, 'count')
        assert len({r['seed'] for r in records}) == len(records), (label, 'duplicate seed')
        exact = sum(r['outcome'] == 'exact' for r in records)
        assert exact == expected['exact'], (label, exact, expected['exact'])
        checked.append({'cell': label, 'n': len(records), 'exact': exact,
                        'files': [str(p) for p in paths]})

    data = json.loads((ROOT / 'analysis/additional_models_20260910/verification.json').read_text())
    groups = defaultdict(list)
    for rel in data['input_sha256']:
        if not rel.startswith('results/') or '__seed' not in rel:
            continue
        obj = json.loads((ROOT / rel).read_text())
        groups[(obj['model'], obj['sandbox']['domain'], obj['method'])].append(rel)
    models = {'Coder': 'qwen2.5-coder-32b-instruct', 'Muse': 'muse-glimmer-30b'}
    for cell in data['cells']:
        key = (models[cell['model']], cell['domain'], cell['method'])
        check('/'.join(key), groups[key], cell)

    audit = json.loads((ROOT / 'analysis/w7_w9_audit_20260909/verification.json').read_text())
    for model, details in audit['reasoning'].items():
        for domain, methods in details['cells'].items():
            suite = f'results/{model}_reasoning_low_b1024_{domain}_s20'
            for method, expected in methods.items():
                paths = [f'{suite}/{method}__seed{s}.json' for s in range(expected['n'])]
                check(f'{model}/reasoning/{domain}/{method}', paths, expected)

    ledger = json.loads((ROOT / 'analysis/shopping_off_budget_20260909/adaptive_results.json').read_text())
    for key, expected in audit['adaptive'].items():
        model, method = key.split('/')
        paths = [r['effective'] for r in ledger if (r['model'], r['method']) == (model, method)]
        check(f'adaptive/{key}', paths, expected)

    gaps = json.loads((ROOT / 'analysis/table_gaps_20260912/verification.json').read_text())
    for cell in gaps['cells']:
        check(f"later-completion/{cell['name']}", list(cell['files']), cell)
    (OUT / 'followup_counts.json').write_text(json.dumps(checked, indent=2) + '\n')
    print(f'Reconciled {len(checked)} descriptive follow-up cells.')


def figures():
    OUT.mkdir(exist_ok=True)
    name = 'closure_trajectory'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/closure_trajectory.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.JSON_OUT = OUT / 'closure_trajectory.json'
    module.MD_OUT = OUT / 'closure_trajectory.txt'
    module.PDF_OUT = OUT / 'convergence_D.pdf'
    module.TEX_OUT = OUT / 'convergence_D.tex'
    module.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', choices=('verify', 'stats', 'figures', 'all'), default='all')
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    for mode, function in [('verify', verify), ('stats', stats), ('figures', figures)]:
        if args.mode in (mode, 'all'):
            function()


if __name__ == '__main__':
    main()
