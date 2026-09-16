import json, pathlib, statistics as st
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
CELLS = {
    ('deployment','Gemma-4-31B'): 'results/b16_deployment_gemma_st2x2',
    ('deployment','Qwen3.8-27B'): 'results/b16_deployment_qwen38_st2x2',
    ('taubench','Gemma-4-31B'):   'results/b16_taubench_gemma_st2x2',
    ('taubench','Qwen3.8-27B'):   'results/b16_taubench_qwen38_st2x2',
}
COND = ['st2x2_partitioned_stop','st2x2_partitioned_exhaust','st2x2_uniform_stop','st2x2_uniform_exhaust']

rows = defaultdict(dict)   # (domain,model,cond) -> seed -> record
for (dom,model),rel in CELLS.items():
    for f in sorted((ROOT/rel).glob('st2x2_*__seed*.json')):
        d = json.loads(f.read_text())
        cond = d['method']; seed = d['seed']
        if cond not in COND or seed in rows[(dom,model,cond)]:
            raise ValueError(f"Unexpected condition or duplicate seed: {f}")
        if d['outcome'] not in ('exact', 'inexact', 'parse_failure'):
            raise ValueError(f"Unexpected outcome requires review: {f}")
        m = (d.get('evaluation') or {}).get('metrics') or {}
        rows[(dom,model,cond)][seed] = dict(
            exact = d.get('outcome')=='exact',
            outcome = d.get('outcome'),
            fa=m.get('false_accepts'), fr=m.get('false_rejects'),
            pv=m.get('postcondition_violations'),
            calls=(d.get('spend') or {}).get('model_calls'),
            states=(d.get('st2x2') or {}).get('states_observed_loop'),
            stop=d.get('stopped_because') or d.get('stop_reason'),
            sha=(d.get('st2x2') or {}).get('initial_contract_sha256'),
            src=(d.get('contract') or {}).get('source'),
        )

for dom,model in CELLS:
    for cond in COND:
        if set(rows[(dom,model,cond)]) != set(range(20)):
            raise ValueError(f"Expected all twenty seeds: {dom}/{model}/{cond}")
    for seed in range(20):
        hashes = {rows[(dom,model,c)][seed]['sha'] for c in COND}
        if len(hashes) != 1 or None in hashes or '' in hashes:
            raise ValueError(f"Missing or mismatched initial contract: {dom}/{model}/{seed}")

def fmt(v): return '-' if v is None else v

print(f"{'domain':11} {'model':12} {'condition':28} {'n':>3} {'exact':>6} {'calls':>6} {'states':>7} {'D med':>6}")
agg={}
for (dom,model),_ in CELLS.items():
    for c in COND:
        r = rows[(dom,model,c)]
        if not r: continue
        n=len(r); ex=sum(v['exact'] for v in r.values())
        calls=st.mean(v['calls'] for v in r.values())
        stt=st.mean(v['states'] for v in r.values() if v['states'] is not None)
        D=[ (v['fa'] or 0)+(v['fr'] or 0)+(v['pv'] or 0) for v in r.values() if v['fa'] is not None]
        agg[(dom,model,c)]=(n,ex,calls,stt,D)
        print(f"{dom:11} {model:12} {c:28} {n:3d} {ex:4d}/{n:<2d} {calls:6.2f} {stt:7.2f} {(st.median(D) if D else float('nan')):6.1f}")

# shared initial contract check
print('\n--- shared round-0 contract per (domain,model,seed) across 4 conditions ---')
bad=0
for (dom,model),_ in CELLS.items():
    for seed in range(20):
        shas={rows[(dom,model,c)].get(seed,{}).get('sha') for c in COND if seed in rows[(dom,model,c)]}
        if len(shas)>1:
            bad+=1; print(' MISMATCH',dom,model,seed,shas)
print(f'mismatches: {bad}')

# stop reasons
print('\n--- stopped_because ---')
for (dom,model),_ in CELLS.items():
    for c in COND:
        r=rows[(dom,model,c)]
        if not r: continue
        cnt=defaultdict(int)
        for v in r.values(): cnt[v['stop']]+=1
        print(f"{dom:11} {model:12} {c:28} " + ', '.join(f'{k}={v}' for k,v in sorted(cnt.items(), key=lambda x:-x[1])))

(ROOT / 'outputs').mkdir(exist_ok=True)
with (ROOT / 'outputs/b16_rows.json').open('w') as output:
    json.dump({f'{k[0]}|{k[1]}|{k[2]}':{str(s):v for s,v in r.items()} for k,r in rows.items()}, output, indent=1)
