"""Independent table/metric/binomial checks; no model calls or artifact changes."""
import argparse
import json
import math
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--build-dir', type=Path, default=HERE/'build')
args = parser.parse_args()
x = json.loads((HERE/'results.json').read_text())
tex = (ROOT/'outputs/fidref_draft.tex').read_text()
table = tex.split('\\label{tab:ablation}',1)[1].split('\\end{table}',1)[0]
lines = [line for line in table.splitlines() if ' & ' in line][1:]
arms = ['active_cegis','active_cegis_no_balance','active_cegis_no_coverage','active_cegis_uniform']
assert len(lines) == len(x['rows']) == 18
for line,row in zip(lines,x['rows']):
    values = [int(re.search(r'\d+',s).group()) for s in line.split(' & ')[2:]]
    assert values == [row['cells'][a]['exact'] for a in arms]
    raw = {}
    for arm,st in row['comparisons'].items():
        b,c = st['active_only'],st['other_only']; n=b+c
        # Recurrence builds the binomial PMF without the aggregator's comb/tail formula.
        probabilities=[2.**(-n)]
        for k in range(n): probabilities.append(probabilities[-1]*(n-k)/(k+1))
        observed=probabilities[b]
        p=sum(v for v in probabilities if v <= observed+1e-15)
        assert math.isclose(p,st['raw_p'],abs_tol=1e-12)
        raw[arm]=p
    ordered=sorted(raw,key=raw.get)
    for index,arm in enumerate(ordered):
        adjusted=min(1.,max((3-i)*raw[ordered[i]] for i in range(index+1)))
        assert math.isclose(adjusted,row['comparisons'][arm]['holm_p'],abs_tol=1e-12)
    for cell in row['cells'].values():
        assert sorted(r['seed'] for r in cell['runs'])==list(range(20))
        ious=[];cfrs=[]
        for run in cell['runs']:
            d=json.loads((ROOT/run['path']).read_text());m=(d.get('evaluation') or {}).get('metrics')
            if run['valid']:
                tp=m['successes']-m['false_rejects']; union=tp+m['false_rejects']+m['false_accepts']
                ious.append(100*tp/union);cfrs.append(100*(tp-m['postcondition_violations'])/union)
            else: ious.append(0);cfrs.append(0)
        assert math.isclose(sum(ious)/20,cell['iou_mean'],abs_tol=1e-10)
        assert math.isclose(sum(cfrs)/20,cell['cfr_mean'],abs_tol=1e-10)
pdf=args.build_dir/'fidref_draft.pdf'
pages=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True).split('\f')
headings={}
for i,page in enumerate(pages,1):
    for heading in ('Conclusion','References'):
        pattern = r'^\s*(?:\d+\s+)?R\s*EFERENCES\s*$' if heading == 'References' else r'^.*C\s*ONCLUSION\s*$'
        if re.search(pattern,page,re.I | re.M):headings.setdefault(heading,[]).append(i)
log=(args.build_dir/'fidref_draft.log').read_text()
assert headings['Conclusion'] == [9] and headings['References'] == [10]
assert not any(s in log for s in ('Overfull','undefined','! LaTeX Error'))
print(json.dumps(dict(table_cells=72,seed_records=1440,binomial_and_holm_checks=54,
                      metric_checks=144,headings=headings,build_clean=True),indent=2))
