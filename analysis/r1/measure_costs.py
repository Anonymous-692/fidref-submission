#!/usr/bin/env python3
"""R1: separate the cost terms that the single symbol B is currently read as covering.

Three distinct quantities, measured rather than asserted:
  (1) enumeration      offline BFS over the reachable set: how many action invocations it
                       costs to build the closure once, and how many of those are the target skill.
  (2) in-loop audit     what a run actually spends: unique feedback states (the B ledger) versus
                       target-skill executions, which repeat because each round re-scores the
                       cumulative audited set.
  (3) offline scoring   the evaluator's closure pass, one target execution per reachable state,
                       charged to no method.
"""
import json, sys, importlib, collections, glob, statistics as st
sys.path.insert(0, '.')

DOMAINS = [
    ('Shopping (price), full',        'experiment.shopping_price', 'shopping_price_default.json', 'PLACE_ORDER'),
    ('Shopping (price), reduced',     'experiment.shopping_price', 'shopping_price_moderate_v3.json',            'PLACE_ORDER'),
    ('Shopping (price), relaxed',     'experiment.shopping_price', 'shopping_price_easy_v4.json',                'PLACE_ORDER'),
    ('Shopping (price), identifiable','experiment.shopping_price', 'shopping_price_easy_v5.json',                'PLACE_ORDER'),
    ('Shopping (basic)',              'experiment.environment',    None,                          'PLACE_ORDER'),
    ('Deployment',                    'experiment.deployment',     'deployment_default.json',     'DEPLOY_SERVICE'),
    ('Calendar',                      'experiment.calendar',       'calendar_default.json',       'SCHEDULE_MEETING'),
    ('tau-bench retail',              'experiment.taubench_retail','taubench_retail_default.json','EXCHANGE_ITEMS'),
    ('tau-bench retail (W2)',         'experiment.taubench_retail','taubench_retail_w2del.json',  'EXCHANGE_ITEMS'),
]

def load_config(mod, cfgname):
    """Use each domain's own config loader, the same path the runners take."""
    cfgmod = importlib.import_module(mod + '.config')
    cls = next(getattr(cfgmod, n) for n in dir(cfgmod)
               if n.endswith('Config') and hasattr(getattr(cfgmod, n), '__dataclass_fields__'))
    if cfgname is None:
        return next(getattr(cfgmod, n) for n in dir(cfgmod) if n.startswith('DEFAULT_'))
    path = 'experiment/configs/' + cfgname
    for loader in ('from_json_file', 'from_json'):
        if hasattr(cls, loader):
            return getattr(cls, loader)(path)
    raw = json.load(open(path))
    raw.pop('description', None)
    fields = set(cls.__dataclass_fields__)
    def tup(v):
        return tuple(tup(x) for x in v) if isinstance(v, list) else v
    return cls(**{k: tup(v) for k, v in raw.items() if k in fields})

print(f"{'domain':32} {'states':>7} {'transitions':>11} {'BFS actions':>12} {'BFS target':>11} {'per state':>9}")
rows = {}
for label, mod, cfgname, target in DOMAINS:
    enum = importlib.import_module(mod + '.enumeration')
    statemod = importlib.import_module(mod + '.state')
    kind = getattr(statemod.ActionKind, target, None)
    orig = enum.apply_action
    counter = collections.Counter()
    def counted(state, action, config, _orig=orig, _c=counter, _k=kind):
        _c['all'] += 1
        if _k is not None and getattr(action, 'kind', None) == _k:
            _c['target'] += 1
        return _orig(state, action, config)
    enum.apply_action = counted
    try:
        cfg = load_config(mod, cfgname)
        res = enum.enumerate_reachable(cfg, max_depth=20, max_states=15000)
    except Exception as exc:
        enum.apply_action = orig
        print(f'{label:32} SKIP ({type(exc).__name__}: {str(exc)[:60]})')
        continue
    enum.apply_action = orig
    n = len(res.states)
    rows[label] = dict(states=n, transitions=len(res.transitions),
                       bfs_all=counter['all'], bfs_target=counter['target'])
    print(f"{label:32} {n:7d} {len(res.transitions):11d} {counter['all']:12d} "
          f"{counter['target']:11d} {counter['all']/n:9.1f}")

json.dump(rows, open('analysis/r1/enumeration_costs.json','w'), indent=1)

print('\n=== in-loop audit: unique feedback states vs target executions (from artifacts) ===')
CELLS = {
  'Deployment / Gemma-4-31B':   'results/gemma4_deployment_active_cegis_s20',
  'Deployment / Gemma-4-31B (rest)': 'results/gemma4_deployment_final_v3_rest_s20',
  'Deployment / Qwen3.8-27B':   'results/qwen38_deployment_final_v3_s20',
  'tau-bench / Gemma-4-31B':    'results/taubench_retail_gemma4_fullsuite_v3_s20',
  'tau-bench / Qwen3.8-27B':    'results/taubench_retail_qwen38_fullsuite_v3_s20',
}
for label, D in CELLS.items():
    for method in ('active_cegis','sampled_cegis'):
        uniq, execs = [], []
        for f in glob.glob(f'{D}/{method}__seed*.json'):
            a = json.load(open(f))
            audits = [r for r in (a.get('rounds') or []) if 'report' in r or 'audit' in r]
            cum = [r.get('cumulative_states') or (r.get('report') or {}).get('states_checked')
                   for r in audits]
            cum = [c for c in cum if c]
            if not cum: continue
            uniq.append(max(cum)); execs.append(sum(cum))
        if uniq:
            print(f'  {label:36} {method:15} unique {st.mean(uniq):5.1f}  '
                  f'target executions {st.mean(execs):6.1f}  ratio {st.mean(execs)/st.mean(uniq):.2f}x  (n={len(uniq)})')
