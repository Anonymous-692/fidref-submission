"""Independent re-scoring: reload every B16 artifact, re-parse its stored contract
source, and re-evaluate it over the full closure from a freshly enumerated sandbox.
Compare against the metrics the run recorded. No model is contacted."""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiment.modeling.client import ChatClient

CELLS = {
    ('deployment','results/b16_deployment_gemma_st2x2'),
    ('deployment','results/b16_deployment_qwen38_st2x2'),
    ('taubench','results/b16_taubench_gemma_st2x2'),
    ('taubench','results/b16_taubench_qwen38_st2x2'),
}

def build(domain):
    client = ChatClient(model='offline', base_url='http://127.0.0.1:1/v1', timeout=1.0, allow_remote=False)
    if domain == 'taubench':
        from experiment.taubench_retail.enumeration import RetailConfig
        from experiment.taubench_retail.runner import RetailExperimentRunner
        raw = json.loads((ROOT/'experiment/configs/taubench_retail_default.json').read_text())
        cfg = RetailConfig(**{k:(tuple(v) if isinstance(v,list) else v) for k,v in raw.items()})
        r = RetailExperimentRunner(client, config=cfg, max_depth=20, max_states=5000,
                                   compact_context=True, context_token_limit=8192,
                                   counterexample_limit=3, refinement_protocol='self_contained_v3')
        from experiment.taubench_retail.dsl import parse_contract
        def score(src):
            c = parse_contract(src, cfg)
            rep = r.score(c)
            return rep, len(r.states)
        return score
    from experiment.deployment.config import DeploymentConfig
    from experiment.deployment.runner import DeploymentExperimentRunner
    from experiment.deployment import dsl
    from experiment.deployment.contracts import evaluate_contract
    cfg = DeploymentConfig.from_json_file(ROOT/'experiment/configs/deployment_default.json')
    r = DeploymentExperimentRunner(cfg, client, max_depth=20, max_states=15000)
    def score(src):
        parsed = dsl.parse_contract_text(src)
        rep = evaluate_contract(parsed.bind(cfg), r.states, cfg, max_counterexamples=3)
        return rep, len(r.states)
    return score

total = ok = skipped = 0
mismatches = []
for domain in ('deployment','taubench'):
    score = build(domain)
    dirs = [d for dom,d in CELLS if dom==domain]
    for rel in sorted(dirs):
        for f in sorted((ROOT/rel).glob('*.json')):
            d = json.loads(f.read_text()); total += 1
            src = (d.get('contract') or {}).get('source')
            rec = (d.get('evaluation') or {}).get('metrics') or {}
            if (d.get('contract') or {}).get('status') != 'parsed':
                skipped += 1; continue
            try:
                rep, nstates = score(src)
            except Exception as exc:
                mismatches.append((str(f), 'reparse-failed', str(exc)[:100])); continue
            got = dict(false_accepts=rep.false_accepts, false_rejects=rep.false_rejects,
                       postcondition_violations=rep.postcondition_violations,
                       states_checked=nstates)
            want = {k: rec.get(k) for k in got}
            exact_got = (got['false_accepts']==0 and got['false_rejects']==0
                         and got['postcondition_violations']==0)
            if got != want or exact_got != (d.get('outcome')=='exact'):
                mismatches.append((str(f.relative_to(ROOT)), want, got, d.get('outcome'), exact_got))
            else:
                ok += 1
    print(f'{domain}: done', flush=True)

print(f'\nartifacts={total}  re-scored-and-matched={ok}  skipped(parse-failure)={skipped}  mismatches={len(mismatches)}')
for m in mismatches[:10]: print('  MISMATCH', m)
