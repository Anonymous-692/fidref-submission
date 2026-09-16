#!/usr/bin/env python3
"""B10 completion check: the gates named in B10_ORDER_NEUTRAL_PROTOCOL_2026-09-05.txt,
then the reported metrics. Reads artifacts only; contacts no model."""
import json, glob, hashlib, collections, statistics as st
from math import comb

D = 'results/taubench_retail_w2neutral_gemma4_s20'
PRE = json.load(open('B10_ORDER_NEUTRAL_PREFLIGHT_2026-09-05.json'))
METHODS = ['direct','sampled_cegis_fixed','active_cegis','active_cegis_no_coverage']
SHORT = {'direct':'Direct','sampled_cegis_fixed':'Fixed-pool','active_cegis':'Active','active_cegis_no_coverage':'No-Coverage'}

runs = {}
for f in sorted(glob.glob(D+'/*.json')):
    if f.endswith('summary.json'): continue
    a = json.load(open(f)); runs[(a['method'], a['seed'])] = a

print('=== GATE 1: 4x20 uniqueness ===')
cnt = collections.Counter(m for m,_ in runs)
print(' ', dict(cnt), '| total', len(runs), '| expected 80 ->', 'OK' if len(runs)==80 and all(cnt[m]==20 for m in METHODS) else 'FAIL')
print('  seeds 0-19 complete per method:', all(all((m,s) in runs for s in range(20)) for m in METHODS))

print('=== GATE 2: model id / endpoint ===')
print(' ', set(a['model'] for a in runs.values()), set(a['endpoint'] for a in runs.values()))

print('=== GATE 3: protocol hash == preflight (new order-neutral hashes) ===')
for m in METHODS:
    got = set(a['hashes']['protocol_sha256'] for (mm,_),a in runs.items() if mm==m)
    want = PRE['protocol_sha256'][m]
    print(f'  {m:26} unique={len(got)} match={"OK" if got=={want} else "FAIL"} {list(got)[0][:16]}')

print('=== GATE 4: shared initial prompt across the four methods ===')
first = collections.defaultdict(set)
for (m,s),a in runs.items():
    ins = a.get('interactions') or []
    if ins: first[s].add((m, ins[0]['prompt_sha256']))
bad = [s for s,v in first.items() if len({h for _,h in v})!=1]
allh = {h for v in first.values() for _,h in v}
print(f'  seeds with divergent first prompt: {len(bad)}  | distinct first-prompt hashes overall: {len(allh)}')
print(f'  matches preflight user prompt hash: n/a (artifact hashes the full request); unique = {list(allh)[0][:16] if len(allh)==1 else "MULTIPLE"}')

print('=== GATE 5: full state count / vocabulary / no truncation ===')
print(' ', set(a['sandbox']['evaluation_states'] for a in runs.values()),
      set(a['sandbox']['vocabulary_protocol'] for a in runs.values()),
      'truncated:', set(a['sandbox']['enumeration']['truncated'] for a in runs.values()))

print('=== GATE 6: oracle isolation + failure denominator ===')
orc = set(a['spend'].get('oracle_feedback_queries') for a in runs.values())
print('  oracle_feedback_queries:', orc, '-> OK' if orc=={0} else '-> FAIL')
st_ = collections.Counter(a['contract']['status'] for a in runs.values())
oc = collections.Counter(a['outcome'] for a in runs.values())
print('  contract status:', dict(st_)); print('  outcome:', dict(oc))

print()
print('=== RESULTS (n=20 per method, failures kept in the denominator) ===')
def wilson(k,n,z=1.959964):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*((p*(1-p)+z*z/(4*n))/n)**.5
    return ((c-h)/d,(c+h)/d)
rows={}
print(f"{'method':12} {'exact':>7} {'pre-complete':>13} {'FA med':>7} {'FR med':>7} {'PV med':>7} {'calls':>6} {'parsefail':>9}")
for m in METHODS:
    v=[runs[(m,s)] for s in range(20)]
    met=[(a.get('evaluation') or {}).get('metrics') or {} for a in v]
    ex=sum(a['outcome']=='exact' for a in v)
    pc=sum(1 for x in met if x.get('false_accepts')==0 and x.get('false_rejects')==0)
    fa=[x.get('false_accepts') for x in met if x.get('false_accepts') is not None]
    fr=[x.get('false_rejects') for x in met if x.get('false_rejects') is not None]
    pv=[x.get('postcondition_violations') for x in met if x.get('postcondition_violations') is not None]
    calls=st.mean(a['spend']['model_calls'] for a in v)
    pf=sum(a['contract']['status']!='parsed' for a in v)
    lo,hi=wilson(ex,20)
    rows[m]=dict(exact=ex,pc=pc,fa=fa,fr=fr,pv=pv)
    print(f"{SHORT[m]:12} {ex:2d}/20 [{lo:.2f},{hi:.2f}] {pc:2d}/20        "
          f"{(st.median(fa) if fa else -1):7.1f} {(st.median(fr) if fr else -1):7.1f} {(st.median(pv) if pv else -1):7.1f} {calls:6.2f} {pf:9d}")

print('\n=== stop reasons ===')
for m in METHODS:
    c=collections.Counter(runs[(m,s)].get('stopped_because') for s in range(20))
    print(f'  {SHORT[m]:12}', dict(c))

def mcnemar(b,c):
    n=b+c
    if n==0: return 1.0
    k=min(b,c); return min(1.0, sum(comb(n,i) for i in range(k+1))/2**n*2)
print('\n=== paired tests on precondition-complete (seed-matched) ===')
for a,b in [('active_cegis','direct'),('active_cegis','sampled_cegis_fixed'),
            ('active_cegis','active_cegis_no_coverage'),('sampled_cegis_fixed','direct')]:
    A=[(runs[(a,s)].get('evaluation') or {}).get('metrics') or {} for s in range(20)]
    B=[(runs[(b,s)].get('evaluation') or {}).get('metrics') or {} for s in range(20)]
    f=lambda x: x.get('false_accepts')==0 and x.get('false_rejects')==0
    ab=sum(1 for i in range(20) if f(A[i]) and not f(B[i]))
    ba=sum(1 for i in range(20) if f(B[i]) and not f(A[i]))
    print(f'  {SHORT[a]:12} vs {SHORT[b]:12} {ab:2d}:{ba:<2d} p={mcnemar(ab,ba):.4f}')

print('\n=== comparison with legacy W2-delivered (same fixture, legacy vocabulary) ===')
old={}
for f in glob.glob('results/taubench_retail_w2del_gemma4_s20/*.json'):
    if f.endswith('summary.json'): continue
    a=json.load(open(f)); old[(a['method'],a['seed'])]=a
for m,om in [('direct','direct'),('sampled_cegis_fixed','sampled_cegis'),
             ('active_cegis','active_cegis'),('active_cegis_no_coverage','active_cegis_no_coverage')]:
    def pcof(d,key):
        v=[d[(key,s)] for s in range(20) if (key,s) in d]
        met=[(a.get('evaluation') or {}).get('metrics') or {} for a in v]
        return sum(1 for x in met if x.get('false_accepts')==0 and x.get('false_rejects')==0), len(v)
    n_pc,n_n = pcof(runs,m); o_pc,o_n = pcof(old,om)
    print(f'  {SHORT[m]:12} neutral {n_pc}/{n_n}  vs  legacy[{om}] {o_pc}/{o_n}')
