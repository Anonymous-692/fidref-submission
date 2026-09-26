#!/usr/bin/env python3
"""Approved 940-run API gap fill; shared initials and one cross-project cost guard."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.modeling import hosted_api as api
from experiment.modeling.client import ChatResponse, Usage
from experiment.modeling.runner import Budgets, Decoding, RunSpec
from scripts import run_hosted_api as legacy
from scripts import run_hosted_mini_all_bench as other
from scripts.run_hosted_terra_flex import FlexClient, FlexCoordinator, MODEL

OUT = ROOT / "analysis/openai_api_20260926/gapfill_v1"
RESULTS = ROOT / "results/openai_gapfill_v1_20260926"
METHODS = ("direct", "self_refine", "fixed_pool", "active_cegis", "active_cegis_no_balance")
DOMAINS = ("deployment", "taubench", "calendar", "shopping_price")
BRIDGE = ("self_refine", "active_cegis")
CELLS = {
    MODEL: {d: BRIDGE if d == "deployment" else METHODS for d in DOMAINS},
    "gpt-5.6-luna": {d: BRIDGE if d in ("deployment", "taubench") else METHODS for d in DOMAINS},
    "gpt-5.4-mini": {d: BRIDGE for d in DOMAINS},
    "gpt-5.4-nano": {d: BRIDGE for d in DOMAINS},
}
api.RATES["gpt-5.4-nano"] = (0.20, 1.25)
TRIALS = range(20)
PRIOR = ("json_mode_v1", "nano_json_mode_v1", "deployment_extension_v1/main",
         "deployment_extension_v1/free", "common_initial_v1/main", "common_initial_v1/free",
         "terra_flex_v1", "nano_all_bench_v1", "mini_all_bench_v1", "mini_shopping_reasoning_v1")
TOTAL_LIMIT = 15.0
UNTRACKED_PREFLIGHT_RESERVE = 0.25


def tier(model):
    return "flex" if model == MODEL else "default"


def client_for(model, coordinator, label, initial=None):
    return (FlexClient if model == MODEL else api.HostedClient)(model, coordinator, label, initial)


class ProjectTransport:
    """Separate credentials, shared admission state; no mutable global key switching."""
    def __init__(self, owner, key):
        self.owner, self.key = owner, key

    def __getattr__(self, name):
        return getattr(self.owner, name)

    @property
    def charged(self):
        return self.owner.charged

    @charged.setter
    def charged(self, value):
        self.owner.charged = value


class RoutingCoordinator(api.Coordinator):
    def __init__(self, root, main_key, free_key, **kwargs):
        super().__init__(root, main_key, **kwargs)
        self.main = ProjectTransport(self, main_key)
        self.free = ProjectTransport(self, free_key)

    def call(self, relative, payload):
        model = payload["model"]
        if model not in CELLS or payload.get("service_tier") != tier(model):
            self.stop("unexpected model/tier")
        transport = self.free if model in ("gpt-5.4-mini", "gpt-5.4-nano") else self.main
        invoke = FlexCoordinator.call if model == MODEL else api.Coordinator.call
        return invoke(transport, relative, payload)


def native_method(domain, method):
    if method == "fixed_pool":
        return "sampled_cegis_fixed" if domain == "taubench" else "sampled_cegis"
    return method


def spec_for(domain, method, trial, *, offline=False):
    return RunSpec(method=native_method(domain, method), seed=trial,
        decoding=Decoding(temperature=0.2, top_p=0.95, max_tokens=2048),
        budgets=Budgets(state_budget=48, query_budget=1 if offline or method == "direct" else 4,
                        token_budget=16000), use_guided_json=False)


def prior_budget():
    base = ROOT / "analysis/openai_api_20260916"
    spent = 0.0
    provenance = {}
    for relative in PRIOR:
        directory = base / relative
        manifest = json.loads((directory / "manifest.json").read_text())
        completion = json.loads((directory / "completion.json").read_text())
        assert completion["status"] == "complete"
        assert api.digest(manifest) == completion["manifest_sha256"]
        for path in (directory / "requests").rglob("*.json"):
            item = json.loads(path.read_text())
            if item["status"] != "complete":
                raise api.SuitePaused("unresolved prior request; audit before spending")
            spent += item["cost_upper_usd"]
        provenance[relative] = completion["manifest_sha256"]
    limit = min(6.5, TOTAL_LIMIT - spent - UNTRACKED_PREFLIGHT_RESERVE)
    if limit <= 0:
        raise api.SuitePaused("no conservative research allocation remains")
    return {"prior_upper_usd": spent, "preflight_reserve_usd": UNTRACKED_PREFLIGHT_RESERVE,
            "new_suite_limit_usd": limit, "total_authorized_usd": TOTAL_LIMIT,
            "prior_manifest_hashes": provenance,
            "note": "Conservative research ledger, not a live account balance or bill"}


def configure():
    legacy.MODELS = (MODEL,)
    legacy.DOMAINS = ("deployment", "taubench")
    other.MODEL = MODEL
    # Two tested domain constructors are reused; no existing source is edited.


def templates(coordinator):
    configure()
    runners = legacy.templates(coordinator)
    extra = other.templates(coordinator)
    runners.update(extra)
    return runners


def messages(runner, domain):
    return (other.initial_messages(runner, domain) if domain in ("calendar", "shopping_price")
            else legacy.initial_messages(runner, domain))


def path_for(model, domain, method, trial):
    return RESULTS / model / domain / f"{method}__trial{trial}.json"


def initial_name(model, domain, trial):
    return legacy.initial_name(model, domain, trial)


def jobs():
    return [{"model": model, "domain": d, "method": m, "trial_id": t,
             "path": str(path_for(model, d, m, t).relative_to(ROOT))}
            for t in TRIALS for model, domains in CELLS.items()
            for d, methods in domains.items() for m in methods]


def frozen_manifest(budget):
    source_files = sorted((ROOT / "experiment").rglob("*.py"))
    source_files += [Path(__file__), Path(legacy.__file__), Path(other.__file__),
                     ROOT / "scripts/run_hosted_terra_flex.py"]
    source_files += [ROOT / f"experiment/configs/{name}.json" for name in
                     ("deployment_default", "taubench_retail_default", "calendar_default", "shopping_price_easy_v4")]
    return {"protocol": {
        "version": "hosted_gapfill_20260926_v1", "models": list(CELLS), "cells": CELLS,
        "trials": list(TRIALS), "n": 940, "fresh_initials": 320,
        "decoding": {"reasoning_effort": "none", "temperature": 0.2, "top_p": 0.95,
                     "max_completion_tokens": 2048, "response_format": "json_object",
                     "service_tier_by_model": {m: tier(m) for m in CELLS}, "api_seed": None},
        "projects": {m: "free" if m in ("gpt-5.4-mini", "gpt-5.4-nano") else "main" for m in CELLS},
        "budgets": {"states": 48, "queries": 4, "total_tokens": 16000, "direct_queries": 1},
        "self_refine": "Existing domain implementations; Deployment/Calendar/Shopping typically 2 calls, Retail up to 4; parse repair consumes Q; actual spend recorded",
        "token_budget_policy": "Existing runner total-usage stop; final call may cross 16000; no outcome-selected cap changes",
        "termination": {"deployment_active": "exhaust", "taubench_active": "stop", "calendar_active": "stop", "shopping_price_active": "stop"},
        "cohort": "Fresh shared initials within each model/domain/trial; historical results not substituted",
        "statistics": "New descriptive cohort. Report Exact/IoU/CFR, validity and costs; no historical Holm-family merge",
        "oracle": "Full closure used only for final evaluation; zero oracle-feedback queries required",
        "execution": {"per_model_concurrency": 16, "workers": 32, "rpm": 300, "tpm": 1400000,
                      "timeout_seconds": {"flex": 900, "default": 180}, "watchdog": False,
                      "retry": "Only explicit Flex resource_unavailable, up to 5; no standard fallback or uncertain replay"},
        "budget_admission": budget,
        "rates_upper_per_million": {m: api.RATES[m] for m in CELLS},
        "jobs": jobs(),
    }, "source_sha256": {str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_files}}


class OfflineClient(api.HostedClient):
    def __init__(self, model):
        super().__init__(model, None, "offline")
        self.requests = []

    def build_request(self, messages, **kwargs):
        request = super().build_request(messages, **kwargs)
        request["service_tier"] = tier(self.model)
        return request

    def complete(self, messages, **kwargs):
        allowed = {k: v for k, v in kwargs.items() if k in
                   {"temperature", "top_p", "max_tokens", "seed", "stop", "guided_json", "repetition_penalty"}}
        options = {"temperature": 0.2, "top_p": 0.95, "max_tokens": 2048, **allowed}
        request = self.build_request(messages, **options)
        self.requests.append(request)
        # Intentionally invalid DSL: exercises the failure denominator without model access.
        return ChatResponse(text='{"offline":true}', usage=Usage(), latency_s=0, status=200,
                            request=request, raw={}, finish_reason="stop")


def offline_gate(runners):
    checked = []
    for model, domains in CELLS.items():
      for domain, methods in domains.items():
        for method in methods:
            runner = copy.copy(runners[domain])
            runner.serving_metadata = {"provider": "offline-test"}
            client = runner.client = OfflineClient(model)
            fn = getattr(runner, "run", None) or runner.run_method
            data = fn(spec_for(domain, method, 0, offline=True)).to_dict()
            expected = client.build_request(messages(runner, domain), temperature=0.2, top_p=0.95, max_tokens=2048)
            assert client.requests == [expected], (domain, method, "shared initial mismatch")
            assert "seed" not in expected and expected["service_tier"] == tier(model)
            assert data["spend"]["oracle_feedback_queries"] == 0
            checked.append([model, domain, method])
    assert len(checked) == 47
    return {"status": "pass", "model_calls": 0, "shared_initial_request_checks": checked}


def generate_initial(coordinator, runners, model, domain, trial):
    client = client_for(model, coordinator, f"{model}/{domain}/trial{trial}/initial")
    response = client.complete(messages(runners[domain], domain))
    print(f"initial {model} {domain} trial={trial} finish={response.finish_reason}", flush=True)


def run_one(coordinator, runners, manifest_hash, model, domain, method, trial):
    target = path_for(model, domain, method, trial)
    if target.exists():
        assert json.loads(target.read_text())["hosted_api"]["manifest_sha256"] == manifest_hash
        return
    runner = copy.copy(runners[domain])
    initial = initial_name(model, domain, trial)
    runner.client = client_for(model, coordinator, f"{model}/{domain}/trial{trial}/{method}", initial)
    runner.serving_metadata = {"provider": "OpenAI", "endpoint": runner.client.endpoint,
        "reasoning_effort": "none", "response_format": "json_object", "service_tier": tier(model)}
    fn = getattr(runner, "run", None) or runner.run_method
    data = fn(spec_for(domain, method, trial)).to_dict()
    if data["spend"]["oracle_feedback_queries"] != 0:
        raise api.SuitePaused("oracle isolation gate failed")
    first = json.loads((OUT / "requests" / (initial + ".json")).read_text())
    data.update(trial_id=trial, local_sampling_seed=trial, method_label=method,
                run=f"{method}__trial{trial}")
    data["decoding"].pop("seed", None)
    data["decoding"].update(api_seed=None, reasoning_effort="none", response_format="json_object", service_tier=tier(model))
    data["hashes"]["local_runner_protocol_sha256"] = data["hashes"]["protocol_sha256"]
    data["hashes"]["decoding_sha256"] = api.digest(data["decoding"])
    data["hashes"]["protocol_sha256"] = api.digest({"manifest": manifest_hash, "domain": domain,
        "method": method, "local": data["hashes"]["local_runner_protocol_sha256"]})
    data["hosted_api"] = {"manifest_sha256": manifest_hash, "api_seed": None,
        "initial_request_sha256": first["request_sha256"], "initial_response_sha256": api.digest(first["response"]),
        "initial_journal": "requests/" + initial + ".json",
        "initial_call_accounting": "One logical call in each arm, charged once per shared initial",
        "cost_source": "Durable request journal; do not sum duplicated initial usage across arms"}
    api.save(target, data)
    print(f"result {model} {domain} {method} trial={trial} {data['outcome']}", flush=True)


def verify_results(manifest):
    files = []
    initials = {}
    for job in manifest["protocol"]["jobs"]:
        path = ROOT / job["path"]
        data = json.loads(path.read_text())
        assert data["trial_id"] == job["trial_id"] and data["method_label"] == job["method"]
        assert data["model"] == job["model"]
        assert data["hosted_api"]["manifest_sha256"] == api.digest(manifest)
        assert data["spend"]["oracle_feedback_queries"] == 0
        key = (job["model"], job["domain"], job["trial_id"])
        hashes = [data["hosted_api"][k] for k in ("initial_request_sha256", "initial_response_sha256")]
        assert key not in initials or initials[key] == hashes
        initials[key] = hashes
        files.append({**job, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for path in (OUT / "requests").rglob("*.json"):
        item = json.loads(path.read_text())
        assert item["status"] == "complete"
        assert item["request"].get("service_tier") == item["response"].get("service_tier") == tier(item["request"]["model"])
        assert api.digest(item["request"]) == item["request_sha256"]
        assert api.digest(item["response"]) == item["response_sha256"]
        assert "seed" not in item["request"]
        assert (item["response"]["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0) == 0
    assert len(files) == 940 and len(initials) == 320
    assert {ROOT / f["path"] for f in files} == set(RESULTS.glob("*/*/*.json"))
    return files


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--offline-check", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    budget = prior_budget()
    if not any((args.offline_check, args.launch, args.run)):
        print(json.dumps({"runs": 940, "cells": CELLS, "budget": budget}, indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if args.offline_check:
        api.save(OUT / "offline_gate.json", offline_gate(templates(None)))
        print("OFFLINE GATE PASS: 47 cells, shared first requests, zero API calls")
        return
    if args.launch:
        gate = json.loads((OUT / "offline_gate.json").read_text())
        assert gate["status"] == "pass"
        with (OUT / "run.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run"],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        api.save(OUT / "launch.json", {"pid": child.pid, "created_at": time.time(),
            "planned_runs": 940, "watchdog": False, "log": str((OUT / "run.log").relative_to(ROOT))})
        print(f"launched pid={child.pid}, 940 runs; no watchdog")
        return
    with (OUT / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = frozen_manifest(budget)
        manifest_path = OUT / "manifest.json"
        if manifest_path.exists():
            assert api.digest(json.loads(manifest_path.read_text())) == api.digest(manifest), "frozen source/config changed"
        else:
            api.save(manifest_path, manifest)
        coordinator = RoutingCoordinator(OUT, api.load_key(ROOT), other.key_from_project(),
                                         limit=budget["new_suite_limit_usd"], prior_reserve=0)
        try:
            runners = templates(coordinator)
            api.save(OUT / "offline_gate.json", offline_gate(runners))
            for model in CELLS:
                smoke = client_for(model, coordinator, f"smoke/{model}").complete(
                    [{"role": "user", "content": 'Return only JSON: {"ok": true}'}], max_tokens=128)
                assert json.loads(smoke.text) == {"ok": True} and smoke.finish_reason == "stop"
                assert smoke.usage.reasoning_tokens in (None, 0)
                print(f"smoke PASS: {model} {tier(model)}", flush=True)
            # Interleave domains within a fixed, predeclared queue; never select on results.
            legacy.parallel(coordinator, [(generate_initial, (coordinator, runners, model, d, t))
                for t in TRIALS for model, domains in CELLS.items() for d in domains])
            legacy.parallel(coordinator, [(run_one, (coordinator, runners, api.digest(manifest),
                j["model"], j["domain"], j["method"], j["trial_id"])) for j in jobs()])
            files = verify_results(manifest)
            api.save(OUT / "completion.json", {"status": "complete", "runs": len(files),
                "manifest_sha256": api.digest(manifest), "cost_upper_usd": coordinator.charged,
                "files": files})
            print(f"COMPLETE {len(files)} runs; upper cost ${coordinator.charged:.4f}", flush=True)
        except Exception as exc:
            coordinator.halted.set()
            api.save(OUT / "paused.json", {"status": "paused", "error_type": type(exc).__name__,
                "reason": str(exc).replace(coordinator.main.key, "[REDACTED]").replace(coordinator.free.key, "[REDACTED]"),
                "cost_upper_usd": coordinator.charged, "reservations": coordinator.reservations})
            raise


if __name__ == "__main__":
    main()
