#!/usr/bin/env python3
"""Resume the frozen gap-fill cohort with a separately recorded spending override."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_hosted_gapfill_20260926 as suite

OUT = suite.OUT
LIMIT = 12.0


def preflight():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if suite.api.digest(manifest) != suite.api.digest(suite.frozen_manifest(suite.prior_budget())):
        raise suite.api.SuitePaused('Frozen experiment source/config changed; resume blocked')
    snapshots = {}
    charged = 0.0
    for path in (OUT / 'requests').rglob('*.json'):
        item = json.loads(path.read_text())
        if (item['status'] != 'complete'
                or suite.api.digest(item['request']) != item['request_sha256']
                or suite.api.digest(item['response']) != item['response_sha256']):
            raise suite.api.SuitePaused('Unresolved or corrupted request; resume blocked')
        charged += item['cost_upper_usd']
        snapshots[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    existing = 0
    for job in manifest['protocol']['jobs']:
        path = ROOT / job['path']
        if path.exists():
            data = json.loads(path.read_text())
            assert data['hosted_api']['manifest_sha256'] == suite.api.digest(manifest)
            assert data['stopped_because'] != 'transport_error'
            snapshots[job['path']] = hashlib.sha256(path.read_bytes()).hexdigest()
            existing += 1
    return manifest, snapshots, charged, existing


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    manifest, snapshots, charged, existing = preflight()
    if args.launch:
        with (OUT / 'resume.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        suite.api.save(OUT / 'resume_launch.json', {'pid': child.pid, 'created_at': time.time(),
            'existing_runs': existing, 'remaining_runs': 940-existing, 'watchdog': False,
            'suite_upper_limit_usd': LIMIT, 'existing_upper_usd': charged})
        print(f'Resumed pid={child.pid}; existing={existing}; remaining={940-existing}; upper_limit={LIMIT}')
        return
    with (OUT / 'suite.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = {'status': 'running', 'created_at': time.time(),
            'manifest_sha256': suite.api.digest(manifest), 'existing_runs': existing,
            'old_suite_upper_limit_usd': 6.5, 'new_suite_upper_limit_usd': LIMIT,
            'existing_upper_usd': charged, 'user_reported_actual_spend_approx_usd': 2,
            'authorization': '2026-09-26 user requested removal of 6.50 admission limit; approximately 2 USD actual spend reported',
            'note': 'Only financial admission changes. Existing manifest, requests, results and scientific protocol preserved. Original paused.json is historical.',
            'resume_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'preserved_sha256': snapshots}
        suite.api.save(OUT / 'budget_override.json', record)
        coordinator = suite.RoutingCoordinator(OUT, suite.api.load_key(ROOT), suite.other.key_from_project(),
                                              limit=LIMIT, prior_reserve=0)
        try:
            runners = suite.templates(coordinator)
            assert suite.offline_gate(runners)['status'] == 'pass'
            # All shared initials exist; replay without drawing new initial candidates.
            suite.legacy.parallel(coordinator, [(suite.generate_initial, (coordinator, runners, model, d, t))
                for t in suite.TRIALS for model, domains in suite.CELLS.items() for d in domains])
            suite.legacy.parallel(coordinator, [(suite.run_one, (coordinator, runners, suite.api.digest(manifest),
                j['model'], j['domain'], j['method'], j['trial_id'])) for j in suite.jobs()])
            files = suite.verify_results(manifest)
            for relative, expected in snapshots.items():
                assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
            suite.api.save(OUT / 'completion.json', {'status': 'complete', 'runs': len(files),
                'manifest_sha256': suite.api.digest(manifest), 'cost_upper_usd': coordinator.charged,
                'budget_override': 'budget_override.json', 'preexisting_files_preserved': len(snapshots), 'files': files})
            record.update(status='complete', completed_at=time.time(), cost_upper_usd=coordinator.charged)
            print(f'COMPLETE {len(files)} runs; upper cost {coordinator.charged:.4f}', flush=True)
        except Exception as exc:
            coordinator.halted.set()
            record.update(status='paused', reason=str(exc).replace(coordinator.main.key, '[REDACTED]').replace(coordinator.free.key, '[REDACTED]'),
                          cost_upper_usd=coordinator.charged, reservations=coordinator.reservations)
            raise
        finally:
            suite.api.save(OUT / 'budget_override.json', record)


if __name__ == '__main__':
    main()
