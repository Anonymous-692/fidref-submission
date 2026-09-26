#!/usr/bin/env python3
"""Read-only CPU audit of completed evidence; never contacts a model/API."""
import hashlib, json, math, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def mcnemar(b, c):
    n=b+c
    return min(1.0, 2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.0
def holm(rows):
    running=0.0; n=len(rows)
    for i,r in enumerate(sorted(rows,key=lambda x:x['p_raw'])):
        running=max(running,min(1.0,r['p_raw']*(n-i))); r['p_holm']=running; r['significant_holm_05']=running<.05

def scorers():
    from experiment.modeling.client import ChatClient
    client=ChatClient(model='offline',base_url='http://127.0.0.1:1/v1',timeout=1,allow_remote=False)
    from experiment.deployment.config import DeploymentConfig
    from experiment.deployment.runner import DeploymentExperimentRunner
    from experiment.deployment import dsl as ddsl
    from experiment.deployment.contracts import evaluate_contract as deval
    dcfg=DeploymentConfig.from_json_file(ROOT/'experiment/configs/deployment_default.json')
    dr=DeploymentExperimentRunner(dcfg,client,max_depth=20,max_states=15000)
    def dep(src):
        x=deval(ddsl.parse_contract_text(src).bind(dcfg),dr.states,dcfg,max_counterexamples=0)
        return {'states_checked':len(dr.states),'false_accepts':x.false_accepts,'false_rejects':x.false_rejects,'postcondition_violations':x.postcondition_violations,'exact':x.is_exact}
    from experiment.taubench_retail.enumeration import RetailConfig
    from experiment.taubench_retail.runner import RetailExperimentRunner
    from experiment.taubench_retail.dsl import parse_contract as tparse
    raw=read(ROOT/'experiment/configs/taubench_retail_default.json')
    tcfg=RetailConfig(**{k:(tuple(v) if isinstance(v,list) else v) for k,v in raw.items()})
    tr=RetailExperimentRunner(client,config=tcfg,max_depth=20,max_states=5000,compact_context=True,context_token_limit=8192,counterexample_limit=0,refinement_protocol='self_contained_v3')
    def tau(src):
        x=tr.score(tparse(src,tcfg)); return {'states_checked':len(tr.states),'false_accepts':x.false_accepts,'false_rejects':x.false_rejects,'postcondition_violations':x.postcondition_violations,'exact':x.is_exact}
    return {'deployment':dep,'taubench':tau}

def audit_pool(rel,domain,sizes):
    base=ROOT/rel; score=scorers()[domain]; cache={}; rows=[]; issues=[]; serialization_defects=[]; provenance_notes=[]; manifests={}
    for modeldir in ('gemma','qwen38'):
        m=read(base/modeldir/'manifest.json'); manifests[modeldir]=m
        expected={'sizes':sizes,'seeds':list(range(20)),'methods':['st2x2_uniform_exhaust','st2x2_partitioned_exhaust'],'state_budget':48,'query_budget':4,'token_budget':16000,'max_tokens':2048,'temperature':.2,'top_p':.95,'termination':'exhaust','coverage':False,'evaluation':'full closure after synthesis only'}
        for k,v in expected.items():
            if m.get(k)!=v: issues.append(f'{modeldir} manifest {k}: {m.get(k)!r} != {v!r}')
        if sha(ROOT/'scripts/run_partial_candidate_pool.py')!=m['driver_sha256']: provenance_notes.append(f'{modeldir} current driver hash differs from frozen manifest')
        source_hash_matches={p:(ROOT/p).is_file() and sha(ROOT/p)==h for p,h in m['runtime']['sources_sha256'].items()}
        for p,ok in source_hash_matches.items():
            if not ok: issues.append(f'{modeldir} source hash mismatch: {p}')
        initials={}
        for seed in range(20):
            p=base/modeldir/'initial'/f'seed{seed}.json'; d=read(p); initials[seed]=d
            if d['seed']!=seed: issues.append(f'{p}: seed mismatch')
            if d.get('source') and hashlib.sha256(d['source'].encode()).hexdigest()!=d.get('source_sha256'): issues.append(f'{p}: source hash mismatch')
        for size in sizes:
            pools={}
            for method in m['methods']:
                pools[method]={}
                for seed in range(20):
                    p=base/modeldir/f'pool{size}'/f'{method}__seed{seed}.json'; d=read(p)
                    pools[method][seed]=d
                    stat=(d.get('contract') or {}).get('status'); met=(d.get('evaluation') or {}).get('metrics') or {}
                    exact=bool(met.get('exact',False)); source=(d.get('contract') or {}).get('source')
                    rec=None
                    if stat=='parsed' and source:
                        try:
                            if source not in cache: cache[source]=score(source)
                            rec=cache[source]
                            for k in ('states_checked','false_accepts','false_rejects','postcondition_violations','exact'):
                                if rec[k]!=met.get(k): issues.append(f'{p}: rescore {k} {rec[k]} != stored {met.get(k)}')
                        except Exception as e:
                            # run_st2x2 retains the previous parsed incumbent when the final
                            # response fails parsing, while returning the invalid raw source.
                            recovered=None
                            for origin,candidate in reversed([('initial',initials[seed].get('source'))]+[(f'interaction:{i["index"]}:{i["role"]}',i.get('response_text')) for i in d['interactions']]):
                                if not candidate: continue
                                try:
                                    if candidate not in cache: cache[candidate]=score(candidate)
                                    recovered=(origin,cache[candidate]); break
                                except Exception: pass
                            if recovered is None:
                                issues.append(f'{p}: CPU rescore failed with no recoverable incumbent: {e}')
                            else:
                                origin,rec=recovered
                                mismatch=[k for k in ('states_checked','false_accepts','false_rejects','postcondition_violations','exact') if rec[k]!=met.get(k)]
                                if mismatch: issues.append(f'{p}: recovered incumbent mismatch: {mismatch}')
                                else: serialization_defects.append({'path':str(p.relative_to(ROOT)),'invalid_serialized_source_error':str(e),'recovered_incumbent':origin})
                    checks=[d['seed']==seed,d['method']==method,d['budgets']=={'state_budget':48,'query_budget':4,'token_budget':16000},d['spend']['oracle_feedback_queries']==0,d['partial_pool']['size']==size,len(d['partial_pool']['indices'])==size,d['partial_pool']['pool_sha256']==digest(d['partial_pool']['indices']),d['partial_pool']['initial_record_sha256']==initials[seed]['record_sha256'],d['g2_accounting']['usage_complete']]
                    if not all(checks): issues.append(f'{p}: protocol/budget/pool/initial/oracle/usage check failed')
                    rows.append({'model':modeldir,'size':size,'method':method,'seed':seed,'outcome':d['outcome'],'parse_status':stat,'stored_exact':exact,'rescored_exact':None if rec is None else rec['exact'],'oracle_feedback_queries':d['spend']['oracle_feedback_queries']})
            # nesting and shared pools/methods
            for seed in range(20):
                if pools[m['methods'][0]][seed]['partial_pool']['indices']!=pools[m['methods'][1]][seed]['partial_pool']['indices']: issues.append(f'{modeldir}/pool{size}/seed{seed}: methods do not share pool')
        for seed in range(20):
            seq=[read(base/modeldir/f'pool{s}'/f'{m["methods"][0]}__seed{seed}.json')['partial_pool']['indices'] for s in sizes]
            if any(seq[i+1][:len(seq[i])]!=seq[i] for i in range(len(seq)-1)): issues.append(f'{modeldir}/seed{seed}: pools not nested prefixes')
    sums=[]; comps=[]
    by=defaultdict(dict)
    for r in rows: by[(r['model'],r['size'],r['method'])][r['seed']]=r
    for k,v in sorted(by.items()): sums.append({'model':k[0],'size':k[1],'method':k[2],'n':len(v),'exact':sum(x['stored_exact'] for x in v.values()),'parse_status':dict(Counter(x['parse_status'] for x in v.values()))})
    for model in ('gemma','qwen38'):
        for size in sizes:
            u=by[(model,size,'st2x2_uniform_exhaust')]; p=by[(model,size,'st2x2_partitioned_exhaust')]
            b=sum(p[s]['stored_exact'] and not u[s]['stored_exact'] for s in range(20)); c=sum(u[s]['stored_exact'] and not p[s]['stored_exact'] for s in range(20))
            comps.append({'model':model,'size':size,'partitioned_only':b,'uniform_only':c,'paired_difference':(b-c)/20,'p_raw':mcnemar(b,c)})
    holm(comps)
    return {'path':rel,'domain':domain,'artifacts':len(rows),'initials':40,'issues':issues,'serialization_defects':serialization_defects,'provenance_notes':provenance_notes,'all_checks_passed':not issues,'summaries':sums,'comparisons':comps,'parse_status':dict(Counter(r['parse_status'] for r in rows)),'stored_outcomes':dict(Counter(r['outcome'] for r in rows)),'independent_cpu_rescore':{'attempted_parsed':sum(r['parse_status']=='parsed' for r in rows),'matched':sum(r['parse_status']=='parsed' and r['stored_exact']==r['rescored_exact'] for r in rows),'recovered_from_prior_incumbent':len(serialization_defects),'unique_contracts':len(cache)},'rows':rows}

def audit_api():
    suites=[('json_mode_v1','results/openai_json_mode_v1_20260916',320),('nano_json_mode_v1','results/openai_nano_json_mode_v1_20260916',160),('deployment_extension_v1','results/openai_deployment_extension_v1_20260916',480),('common_initial_v1','results/openai_common_initial_v1_20260916',180),('terra_flex_v1','results/openai_terra_flex_v1_20260917',80),('nano_all_bench_v1','results/openai_nano_all_bench_v1_20260917',160),('mini_all_bench_v1','results/openai_mini_all_bench_v1_20260917',160),('mini_shopping_reasoning_v1','results/openai_mini_shopping_reasoning_v1_20260917',60)]
    out=[]; total=0; issues=[]
    for name,rel,expected in suites:
        files=list((ROOT/rel).rglob('*.json')); total+=len(files)
        comps=list((ROOT/'analysis/openai_api_20260916'/name).rglob('completion.json'))
        mans=list((ROOT/'analysis/openai_api_20260916'/name).rglob('manifest.json'))
        declared=sum(read(p)['runs'] for p in comps)
        manifest_digests={digest(read(m)) for m in mans}
        hash_ok=all(read(c)['manifest_sha256'] in manifest_digests for c in comps)
        api_seeds=[]
        for m in mans:
            proto=read(m)['protocol']; dec=proto.get('decoding',{}); api_seeds.append(dec.get('api_seed','not-declared'))
        item={'suite':name,'result_files':len(files),'completion_runs':declared,'expected':expected,'completion_statuses':[read(p)['status'] for p in comps],'manifest_hashes_match':hash_ok,'api_seed_fields':api_seeds}
        if len(files)!=expected or declared!=expected or not hash_ok or any(x!='complete' for x in item['completion_statuses']): issues.append(item)
        out.append(item)
    return {'total_result_files':total,'expected_total':1600,'all_checks_passed':total==1600 and not issues,'issues':issues,'suites':out,'scope_note':'Counts and completion/manifest linkage only; published reports are retained for result interpretation and statistics. API sampling seed is distinct from local trial/state-selection IDs.'}

def audit_calendar():
    base=ROOT/'results/qwen14b_calendar_controls_20260917_s20'; files=[p for p in base.glob('*.json') if p.name!='summary.json']; rows=[read(p) for p in files]; issues=[]
    keys={(d['method'],d['seed']) for d in rows}; expected={(m,s) for m in ('random_probe','sampled_cegis') for s in range(20)}
    if keys!=expected: issues.append('method/seed grid mismatch')
    for p,d in zip(files,rows):
        if d['budgets']!={'state_budget':48,'query_budget':4,'token_budget':16000} or d['decoding']!={'temperature':.2,'top_p':.95,'max_tokens':2048,'seed':d['seed']}: issues.append(f'{p}: condition mismatch')
        if d['spend']['oracle_feedback_queries']!=0: issues.append(f'{p}: nonzero oracle feedback')
    summary=read(base/'summary.json'); summary_keys={(r['method'],r['seed']) for r in summary['runs']}
    if summary_keys!=keys: issues.append('summary/run grid mismatch')
    # Exact byte/content duplication against pre-existing calendar cohorts.
    other=[]
    for od in ROOT.glob('results/qwen14b_calendar*'):
        if od==base: continue
        hashes={sha(p):str(p.relative_to(ROOT)) for p in od.glob('*.json') if p.name!='summary.json'}
        for p in files:
            if sha(p) in hashes: other.append({'new':str(p.relative_to(ROOT)),'existing':hashes[sha(p)]})
    return {'artifacts':len(files),'all_checks_passed':not issues,'issues':issues,'parse_status':dict(Counter((d.get('contract') or {}).get('status') for d in rows)),'outcomes':dict(Counter(d['outcome'] for d in rows)),'exact_by_method':{m:sum(d['outcome']=='exact' for d in rows if d['method']==m) for m in ('random_probe','sampled_cegis')},'byte_identical_existing_artifacts':other,'duplicate_judgment':'No byte-identical prior artifact' if not other else 'Contains byte-identical prior artifacts; do not double count'}

def main():
    result={'retail_partial_pool':audit_pool('results/partial_candidate_pool_tau_v1_20260917','taubench',[200,500,1294]),'deployment_partial_pool':audit_pool('results/partial_candidate_pool_v1_20260917','deployment',[200,1000,5000,11712]),'openai_api':audit_api(),'qwen14b_calendar_controls':audit_calendar()}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{x:v[x] for x in v if x in ('artifacts','all_checks_passed','issues','total_result_files','expected_total','parse_status','outcomes','exact_by_method','duplicate_judgment','comparisons')} for k,v in result.items()},indent=2))
if __name__=='__main__': main()
