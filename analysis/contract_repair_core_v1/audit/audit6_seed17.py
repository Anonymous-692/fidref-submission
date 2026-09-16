#!/usr/bin/env python3
"""Audit 6: reconcile seed 17. The offline audit predicted closure D 188 -> 88 for the round-1
patch; the tie-tolerant run adopted that same patch yet finished at 188.

Replays the run's own trajectory, scoring the FULL closure after every adopted round, so the
prediction (a statement about round 1 only) can be told apart from the final outcome (the
result of every adoption). Offline; nothing is returned to a model. Artifacts unmodified.
"""
import json, glob, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiment.deployment import dsl
from experiment.deployment.config import DeploymentConfig
from experiment.deployment.contracts import evaluate_contract
from experiment.deployment.patch import apply_patch
from experiment.deployment.runner import DeploymentExperimentRunner
from experiment.modeling.client import ChatClient

cfg = DeploymentConfig.from_json_file(ROOT / "experiment/configs/deployment_default.json")
runner = DeploymentExperimentRunner(
    cfg, ChatClient(model="offline", base_url="http://127.0.0.1:1/v1", timeout=1.0,
                    allow_remote=False), max_depth=20, max_states=15000)

def score(src):
    r = evaluate_contract(dsl.parse_contract_text(src).bind(cfg), runner.states, cfg,
                          max_counterexamples=0)
    return {"FA": r.false_accepts, "FR": r.false_rejects, "PV": r.postcondition_violations,
            "D": r.false_accepts + r.false_rejects + r.postcondition_violations,
            "exact": bool(r.is_exact)}

INIT = {g["seed"]: g for g in json.loads(
    (ROOT / "analysis/contract_repair_core_v1/initial_s20_qwen14b.json").read_text())["generated"]}
SEED = 17
out = {"seed": SEED, "runs": {}}
for label, path, cond in (
        ("B'  strict", "results/deployment_qwen14b_repair_core_v1_s20", "repair_patch_witness"),
        ("B'' tie",    "results/deployment_qwen14b_repair_tie_v1",      "repair_patch_tie")):
    art = json.load(open(f"{ROOT/path}/{cond}__seed{SEED}.json"))
    incumbent = INIT[SEED]["source"]
    steps = [{"round": 0, "closure": score(incumbent), "note": "shared initial contract"}]
    for rd in art["rounds"]:
        if rd.get("round", 0) == 0:
            continue
        raw = rd.get("patch_raw")
        step = {"round": rd["round"], "adopted": bool(rd.get("adopted")),
                "tie_adopted": rd.get("tie_adopted"), "observed_delta": rd.get("d_obs_delta"),
                "guard": rd.get("guard"), "patch_text_seen_before": rd.get("patch_text_seen_before")}
        if raw is None:
            steps.append(step); continue
        try:
            cand = apply_patch(incumbent, raw).source
        except Exception as exc:  # noqa: BLE001
            step["error"] = str(exc)[:120]; steps.append(step); continue
        step["closure_if_applied"] = score(cand)
        if rd.get("adopted"):
            incumbent = cand
        step["closure_incumbent_after_round"] = score(incumbent)
        steps.append(step)
    out["runs"][label] = {"final_closure_recorded": {
        k: art["evaluation"]["metrics"][v] for k, v in
        (("FA", "false_accepts"), ("FR", "false_rejects"), ("PV", "postcondition_violations"))},
        "final_closure_replayed": score(incumbent),
        "final_source_matches_artifact": incumbent == art["contract"]["source"],
        "stopped_because": art["stopped_because"], "steps": steps}

for label, d in out["runs"].items():
    print(f"== {label}  stop={d['stopped_because']}")
    for s in d["steps"]:
        ca = s.get("closure_if_applied"); inc = s.get("closure_incumbent_after_round")
        base = s.get("closure")
        if base:
            print(f"   r{s['round']}: 초기 계약 D={base['D']}")
            continue
        print(f"   r{s['round']}: adopted={s['adopted']} tie={s['tie_adopted']} "
              f"obs_delta={s['observed_delta']} guard={s['guard']} dupText={s['patch_text_seen_before']}")
        if ca:  print(f"        이 패치를 적용하면 D={ca['D']} (FA={ca['FA']} FR={ca['FR']} PV={ca['PV']})")
        if inc: print(f"        라운드 후 incumbent D={inc['D']}")
    print(f"   최종 replay D={d['final_closure_replayed']['D']} / 아티팩트 기록 "
          f"D={sum(d['final_closure_recorded'].values())} / 소스 일치={d['final_source_matches_artifact']}")
    print()
p = ROOT / "analysis/contract_repair_core_v1/audit/audit6_seed17.json"
p.write_text(json.dumps(out, indent=1))
print(f"wrote {p.relative_to(ROOT)}")
