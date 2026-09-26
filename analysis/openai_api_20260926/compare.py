"""Reaggregate the frozen 940-result cohort; no model or closure execution."""
import hashlib
import json
import math
import statistics as stats
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SUITE = OUT / 'gapfill_v1'
METHODS = ['direct', 'self_refine', 'fixed_pool', 'active_cegis', 'active_cegis_no_balance']
LABELS = dict(zip(METHODS, ['Direct', 'Self-Refine', 'Fixed-pool', 'Active', 'No-Balance']))
SUPPORT = {'deployment': (11712, 88), 'calendar': (1676, 35),
           'taubench': (1294, 14), 'shopping_price': (288, 8)}


def read(p):
    return json.loads(p.read_text())


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    completion = read(SUITE / 'completion.json')
    manifest = read(SUITE / 'manifest.json')
    assert completion['manifest_sha256'] == digest(manifest)
    assert completion['status'] == 'complete'
    assert len(completion['files']) == 940
    assert {j['path'] for j in manifest['protocol']['jobs']} == {j['path'] for j in completion['files']}
    groups = defaultdict(list)
    initials = {}
    # In the anonymous release, raw files differ only by redacted infrastructure
    # metadata; RELEASE_MANIFEST.json then maps each release hash to its source hash.
    release = ROOT / 'RELEASE_MANIFEST.json'
    release_files = read(release)['files'] if release.exists() else {}
    for item in completion['files']:
        path = ROOT / item['path']
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item['sha256']:
            entry = release_files[item['path']]
            assert entry['release_sha256'] == actual and entry['source_sha256'] == item['sha256'], item['path']
        d = read(path)
        model, domain, method, trial = (item[k] for k in ['model', 'domain', 'method', 'trial_id'])
        assert (d['model'], d['method_label'], d['trial_id']) == (model, method, trial)
        assert d['hosted_api']['manifest_sha256'] == digest(manifest)
        assert d['spend']['oracle_feedback_queries'] == 0
        first = tuple(d['hosted_api'][k] for k in ['initial_request_sha256', 'initial_response_sha256'])
        key = (model, domain, trial)
        assert key not in initials or initials[key] == first
        initials[key] = first
        m = (d.get('evaluation') or {}).get('metrics')
        parsed = (d.get('contract') or {}).get('status') == 'parsed'
        assert parsed == bool(m)
        exact = d['outcome'] == 'exact'
        assert exact == bool(m and m['exact'])
        iou = cfr = Fraction(0)
        if m:
            n, a = SUPPORT[domain]
            assert (m['states_checked'], m['successes']) == (n, a)
            fa, fr, pv = (m[k] for k in ['false_accepts', 'false_rejects', 'postcondition_violations'])
            assert 0 <= fa <= n-a and 0 <= fr <= a and 0 <= pv <= a-fr
            iou = Fraction(100 * (a-fr), a+fa)
            cfr = Fraction(100 * (a-fr-pv), a+fa)
            assert cfr == 100 * (1-Fraction(fa+fr+pv, a+fa))
            assert (cfr == 100) == exact
        else:
            assert d['outcome'] == 'parse_failure'
        assert 0 <= cfr <= iou <= 100
        groups[(model, domain, method)].append({
            'trial_id': trial, 'exact': exact, 'iou': float(iou), 'cfr': float(cfr),
            'parse_failure': not parsed, 'calls': d['usage']['calls'],
            'tokens': d['usage']['total_tokens'], 'metrics': m,
            'stop': d['stopped_because'],
            'length_requests': sum(i.get('finish_reason') == 'length' for i in d['interactions']),
            'contract_sha256': digest(json.loads(d['contract']['source'])) if parsed else None,
            'source': item})
    cells = []
    for (model, domain, method), rs in sorted(groups.items()):
        assert len(rs) == 20 and {r['trial_id'] for r in rs} == set(range(20))
        cells.append({'model': model, 'domain': domain, 'method': method, 'n': 20,
            'exact': sum(r['exact'] for r in rs),
            **{k+'_mean': stats.mean(r[k] for r in rs) for k in ['iou', 'cfr', 'calls', 'tokens']},
            'parse_failure': sum(r['parse_failure'] for r in rs),
            'stop_counts': dict(Counter(r['stop'] for r in rs)),
            'valid_error_medians': {k: stats.median(r['metrics'][k] for r in rs if r['metrics'])
                if any(r['metrics'] for r in rs) else None for k in
                ['false_accepts', 'false_rejects', 'postcondition_violations']},
            'post_violation_runs': sum(bool(r['metrics'] and r['metrics']['postcondition_violations']) for r in rs),
            'unique_valid_contracts': len({r['contract_sha256'] for r in rs if r['contract_sha256']}),
            'runs': sorted(rs, key=lambda r:r['trial_id'])})
    comparisons = []
    for model, domain, method in sorted(groups):
        if method == 'active_cegis':
            continue
        a = {r['trial_id']: r for r in groups[(model, domain, 'active_cegis')]}
        b = {r['trial_id']: r for r in groups[(model, domain, method)]}
        wins = sum(a[t]['exact'] and not b[t]['exact'] for t in a)
        losses = sum(b[t]['exact'] and not a[t]['exact'] for t in a)
        n = wins+losses
        p = min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses)+1)) / 2**n) if n else 1.
        diffs = [a[t]['cfr']-b[t]['cfr'] for t in a]
        comparisons.append({'model': model, 'domain': domain, 'control': method,
            'exact_active_only': wins, 'exact_control_only': losses, 'mcnemar_raw_p': p,
            'cfr_delta_pp': stats.mean(diffs), 'cfr_wins': sum(x>1e-9 for x in diffs),
            'cfr_losses': sum(x < -1e-9 for x in diffs), 'cfr_ties': sum(abs(x)<=1e-9 for x in diffs)})
    # This family is exploratory and declared at analysis, not pre-registered.
    previous = 0.
    for rank, row in enumerate(sorted(comparisons, key=lambda x:x['mcnemar_raw_p'])):
        previous = max(previous, min(1., (len(comparisons)-rank)*row['mcnemar_raw_p']))
        row['mcnemar_holm31_p'] = previous
    assert len(cells) == 47 and len(initials) == 320 and len(comparisons) == 31
    result = {'manifest_sha256': digest(manifest), 'n': 940, 'cells': cells, 'comparisons': comparisons,
        'definitions': {'iou': '100*(A-FR)/(A+FA)', 'cfr': '100*(A-FR-PV)/(A+FA)',
            'aggregation': 'Run-wise arithmetic mean; parse failures assigned zero; denominator 20',
            'tests': 'Exploratory exact McNemar, Active vs every co-run control: 31 comparisons with Holm. No historical family merge.'}}
    (OUT/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# API 940런 비교 집계', '', '2026-09-26 완료 코호트만 집계. 과거 결과와 합치지 않았다.', '',
        '각 셀 20회. IoU/CFR은 런별 백분율의 평균이며 파싱 실패는 0점으로 분모에 유지한다.',
        'FA/FR/PV 원본 계수 재집계이며 새 모델 호출·폐포 실행은 없다.', '',
        '| 모델 | 도메인 | 방법 | Exact /20 | IoU % | CFR % | 파싱 실패 /20 | 평균 호출 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in cells:
        lines.append(f"| {r['model']} | {r['domain']} | {LABELS[r['method']]} | {r['exact']} | {r['iou_mean']:.1f} | {r['cfr_mean']:.1f} | {r['parse_failure']} | {r['calls_mean']:.2f} |")
    lines += ['', '## 동일 초기 응답을 공유한 짝 비교', '',
        '실행 후 정한 탐색적 31비교 가족에 Holm 보정. 기존 논문 검정과 분리한다.',
        '효과 없음/동등성은 p값만으로 판정하지 않는다. 실제 호출 수는 위 표를 따른다.', '',
        '| 모델 | 도메인 | Active 대비 대조군 | Exact Active만:대조군만 | CFR 차이 %p | CFR 승:패:동률 | Holm p |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['model']} | {r['domain']} | {LABELS[r['control']]} | {r['exact_active_only']}:{r['exact_control_only']} | {r['cfr_delta_pp']:+.1f} | {r['cfr_wins']}:{r['cfr_losses']}:{r['cfr_ties']} | {r['mcnemar_holm31_p']:.4g} |")
    lines += ['', '근거: `gapfill_v1/manifest.json`, `gapfill_v1/completion.json`의 명시적 940경로.',
        '940개 파일 SHA-256, 47셀의 trial 0–19, 320개 공유 초기 응답, oracle_feedback_queries=0, 지표 범위와 Exact↔CFR100을 검사했다.',
        '재현: `.venv/bin/python analysis/openai_api_20260926/compare.py`', '']
    (OUT/'COMPARISON.txt').write_text('\n'.join(lines))
    print('PASS: 940 results / 47 cells / 31 exploratory comparisons')
    for r in comparisons:
        print(json.dumps(r))


if __name__ == '__main__':
    main()
