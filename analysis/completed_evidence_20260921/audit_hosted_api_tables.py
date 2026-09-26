#!/usr/bin/env python3
"""Audit appendix hosted-API values from stored artifacts only."""
import hashlib, json, statistics
from collections import Counter
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
METHODS=['direct','sampled_cegis','sampled_cegis_fixed','fixed_pool','active_cegis','active_cegis_no_balance']
LABEL={'direct':'Direct','sampled_cegis':'Fixed-pool','sampled_cegis_fixed':'Fixed-pool','fixed_pool':'Fixed-pool','active_cegis':'Active','active_cegis_no_balance':'No-Balance'}
EXPECTED_SUCCESSES={'deployment':88,'taubench_retail':14,'calendar':35,'shopping_price':8}
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(a,fa,fr,pv=0): return float(Fraction(a-fr-pv,a+fa)*100) if a+fa else 0.
def summarize(paths,domain=None):
 rows=[read(p) for p in paths]; groups={}
 for r in rows:
  dom=domain or r['sandbox']['domain']; key=(r['model'],dom,LABEL.get(r['method'],r['method']))
  groups.setdefault(key,[]).append(r)
 out=[]
 for (model,dom,method),rs in sorted(groups.items()):
  ious=[];cfrs=[];valid=0; states=set(); successes=set()
  for r in rs:
   m=((r.get('evaluation') or {}).get('metrics'))
   parsed=(r.get('contract') or {}).get('status')=='parsed'
   assert (r['outcome']=='exact') == bool(m and m.get('exact')), (r.get('run'),r['outcome'],m)
   if parsed:
    assert m, r.get('run'); valid+=1; a=m['successes']; states.add(m['states_checked']);successes.add(a)
    assert a==EXPECTED_SUCCESSES[dom],(dom,a); assert 0<=m['false_rejects']<=a; assert 0<=m['postcondition_violations']<=a-m['false_rejects']
    ious.append(metric(a,m['false_accepts'],m['false_rejects'])); cfrs.append(metric(a,m['false_accepts'],m['false_rejects'],m['postcondition_violations']))
   else:
    assert r['outcome']=='parse_failure'; ious.append(0.);cfrs.append(0.)
  out.append({'model':model,'domain':dom,'method':method,'n':len(rs),'exact':sum(r['outcome']=='exact' for r in rs),'parse_failure':sum(r['outcome']=='parse_failure' for r in rs),'valid_evaluation_n':valid,'states_checked_values':sorted(states),'successes_values':sorted(successes),'iou_mean':round(sum(ious)/len(rs),10),'cfr_mean':round(sum(cfrs)/len(rs),10),'outcomes':dict(Counter(r['outcome'] for r in rs)),'api_seed_values':sorted({str((r.get('hosted_api') or {}).get('api_seed')) for r in rs})})
 return out
def paths(rel): return sorted((ROOT/rel).rglob('*.json'))
def main():
 suites={
  'original':summarize(paths('results/openai_json_mode_v1_20260916')),
  'nano_original':summarize(paths('results/openai_nano_json_mode_v1_20260916')),
  'deployment_extension':summarize(paths('results/openai_deployment_extension_v1_20260916')),
  'terra_flex':summarize(paths('results/openai_terra_flex_v1_20260917')),
  'mini_other':summarize(paths('results/openai_mini_all_bench_v1_20260917')),
  'nano_other':summarize(paths('results/openai_nano_all_bench_v1_20260917')),
  'reasoning':summarize(paths('results/openai_mini_shopping_reasoning_v1_20260917')),
  'common_initial':summarize(paths('results/openai_common_initial_v1_20260916'),'deployment'),
 }
 # reasoning arms and common-initial source-model are not represented by method alone.
 reasoning=[]
 for arm in ('legacy_off','default_off','default_low'):
  ps=sorted((ROOT/'results/openai_mini_shopping_reasoning_v1_20260917').rglob(f'{arm}__trial*.json')); row=summarize(ps)[0]; row['arm']=arm
  ds=[read(p)['diagnostics'] for p in ps]; row['final_has_pre_order_status_none']=sum(d['final']['has_pre_order_status_none'] for d in ds); reasoning.append(row)
 common=[]
 for revision_dir in sorted((ROOT/'results/openai_common_initial_v1_20260916').iterdir()):
  if not revision_dir.is_dir(): continue
  for initial_dir in sorted(revision_dir.iterdir()):
   ps=sorted(initial_dir.glob('*.json'))
   common.append({'initial_source_model':initial_dir.name,'revision_model':revision_dir.name,'n':len(ps),'exact':sum(read(p)['outcome']=='exact' for p in ps),'parse_failure':sum(read(p)['outcome']=='parse_failure' for p in ps)})
 retail_paths=[p for rel in ('results/openai_json_mode_v1_20260916','results/openai_nano_json_mode_v1_20260916') for p in (ROOT/rel).glob('*/taubench/*.json')]
 retail_files=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'model':read(p)['model'],'method':read(p)['method'],'trial_id':read(p).get('trial_id'),'outcome':read(p)['outcome'],'evaluation_exact':bool(((read(p).get('evaluation') or {}).get('metrics') or {}).get('exact',False))} for p in sorted(retail_paths)]
 # actual service tiers retained in raw API responses, when supplied.
 tiers={}
 for name,rel in [('original','results/openai_json_mode_v1_20260916'),('nano_original','results/openai_nano_json_mode_v1_20260916'),('deployment_extension','results/openai_deployment_extension_v1_20260916'),('terra_flex','results/openai_terra_flex_v1_20260917')]:
  vals=[]
  for p in paths(rel):
   for i in read(p).get('interactions',[]):
    rr=i.get('raw_response') or {}
    if rr.get('service_tier') is not None: vals.append(rr['service_tier'])
  tiers[name]=dict(Counter(vals))
 # Values asserted by appendix, kept explicit so mismatch fails loudly.
 expected={
  'deployment_original':{('gpt-5.4-mini','Direct'):0,('gpt-5.4-mini','Fixed-pool'):0,('gpt-5.4-mini','Active'):0,('gpt-5.4-mini','No-Balance'):1,('gpt-5.6-luna','Direct'):1,('gpt-5.6-luna','Fixed-pool'):3,('gpt-5.6-luna','Active'):2,('gpt-5.6-luna','No-Balance'):3,('gpt-5.4-nano','Direct'):0,('gpt-5.4-nano','Fixed-pool'):2,('gpt-5.4-nano','Active'):12,('gpt-5.4-nano','No-Balance'):0},
  'deployment_extension':{('gpt-5.4-mini','Direct'):0,('gpt-5.4-mini','Fixed-pool'):1,('gpt-5.4-mini','Active'):0,('gpt-5.4-mini','No-Balance'):0,('gpt-5.6-luna','Direct'):0,('gpt-5.6-luna','Fixed-pool'):4,('gpt-5.6-luna','Active'):5,('gpt-5.6-luna','No-Balance'):0,('gpt-5.4-nano','Direct'):0,('gpt-5.4-nano','Fixed-pool'):2,('gpt-5.4-nano','Active'):19,('gpt-5.4-nano','No-Balance'):2},
  'terra':{('gpt-5.6-terra','Direct'):0,('gpt-5.6-terra','Fixed-pool'):1,('gpt-5.6-terra','Active'):20,('gpt-5.6-terra','No-Balance'):3},
 }
 checks=[]
 def check(block,rows,domain,n):
  got={(r['model'],r['method']):r['exact'] for r in rows if r['domain']==domain}
  checks.append({'block':block,'expected':{f'{k[0]}|{k[1]}':v for k,v in expected[block].items()},'got':{f'{k[0]}|{k[1]}':v for k,v in got.items()},'pass':got==expected[block] and all(r['n']==n for r in rows if r['domain']==domain)})
 check('deployment_original',suites['original']+suites['nano_original'],'deployment',20);check('deployment_extension',suites['deployment_extension'],'deployment',40);check('terra',suites['terra_flex'],'deployment',20)
 out={'artifact_counts':{k:sum(r['n'] for r in v) for k,v in suites.items()},'suite_summaries':suites,'reasoning_by_arm':reasoning,'common_initial_cells':common,'service_tiers_in_raw_responses':tiers,'appendix_exact_checks':checks,'retail_artifacts':{'n':len(retail_files),'files':retail_files},'notes':['Exact uses outcome==exact with every artifact in denominator.','IoU/CFR assign zero to runs without evaluation, matching suite aggregation code.','No new full-closure evaluation was performed.']}
 (OUT/'hosted_api_raw_audit.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'artifact_counts':out['artifact_counts'],'checks':checks,'reasoning':reasoning,'common':common,'tiers':tiers,'retail_n':len(retail_files)},indent=2))
if __name__=='__main__': main()
