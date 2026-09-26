#!/usr/bin/env python3
"""Recover the valid incumbent used by the seven retail final evaluations."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from experiment.modeling.client import ChatClient
from experiment.taubench_retail.enumeration import RetailConfig
from experiment.taubench_retail.runner import RetailExperimentRunner
from experiment.taubench_retail.dsl import parse_contract

def read(p): return json.loads(Path(p).read_text())
raw=read(ROOT/'experiment/configs/taubench_retail_default.json')
cfg=RetailConfig(**{k:(tuple(v) if isinstance(v,list) else v) for k,v in raw.items()})
client=ChatClient(model='offline',base_url='http://127.0.0.1:1/v1',timeout=1,allow_remote=False)
runner=RetailExperimentRunner(client,config=cfg,max_depth=20,max_states=5000,compact_context=True,context_token_limit=8192,counterexample_limit=0,refinement_protocol='self_contained_v3')
verification=read(ROOT/'analysis/completed_evidence_20260921/verification.json')
paths=[Path(x.split(': CPU rescore failed')[0]) for x in verification['retail_partial_pool']['issues'] if 'CPU rescore failed' in x]
out=[]
for p in paths:
    d=read(p); initial=read(p.parents[1]/'initial'/f"seed{d['seed']}.json")['source']
    candidates=[{'origin':'initial','source':initial}]+[{'origin':f"interaction:{i['index']}:{i['role']}",'source':i.get('response_text')} for i in d['interactions']]
    parsed=[]
    for c in candidates:
        try:
            contract=parse_contract(c['source'],cfg); rep=runner.score(contract)
            parsed.append({**c,'parseable':True,'metrics':{'states_checked':len(runner.states),'false_accepts':rep.false_accepts,'false_rejects':rep.false_rejects,'postcondition_violations':rep.postcondition_violations,'exact':rep.is_exact}})
        except Exception as e: parsed.append({**c,'parseable':False,'error':str(e)})
    incumbent=next(x for x in reversed(parsed) if x['parseable'])
    stored=d['evaluation']['metrics']; fields=('states_checked','false_accepts','false_rejects','postcondition_violations','exact')
    out.append({'path':str(p.relative_to(ROOT)),'stopped_because':d['stopped_because'],'finish_reason':d['interactions'][-1]['finish_reason'],'serialized_contract_status':d['contract']['status'],'serialized_source_origin':parsed[-1]['origin'],'serialized_source_parseable':parsed[-1]['parseable'],'last_valid_incumbent_origin':incumbent['origin'],'incumbent_matches_stored_evaluation':all(incumbent['metrics'][k]==stored[k] for k in fields),'incumbent_metrics':incumbent['metrics'],'stored_metrics':{k:stored[k] for k in fields},'candidate_trace':[{k:v for k,v in x.items() if k!='source'} for x in parsed]})
(ROOT/'analysis/completed_evidence_20260921/retail_incumbent_trace.json').write_text(json.dumps({'runs':out,'all_incumbents_match':all(x['incumbent_matches_stored_evaluation'] for x in out)},indent=2)+'\n')
print(json.dumps({'runs':len(out),'all_incumbents_match':all(x['incumbent_matches_stored_evaluation'] for x in out),'origins':[(x['path'],x['last_valid_incumbent_origin']) for x in out]},indent=2))
