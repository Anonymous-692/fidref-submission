"""Offline, candidate-independent successor fault injection for G1."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import subprocess
import sys
from collections import Counter
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiment.modeling.artifacts import sha256_json


def plain(value):
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def canonical(value):
    return json.dumps(plain(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2) + "\n")


def mutants(after, pools):
    """One-field typed replacements; no candidate/reference access."""
    output, invalid = [], Counter()
    for f in dataclasses.fields(after):
        old = getattr(after, f.name)
        values = {canonical(v): v for v in pools[f.name]}
        if isinstance(old, bool):
            values[canonical(not old)] = not old
        elif type(old) in (int, float):
            for value in (old - 1, old + 1):
                if value >= 0:
                    values[canonical(value)] = value
        seen = set()
        for _, value in sorted(values.items()):
            if value == old:
                continue
            try:
                changed = dataclasses.replace(after, **{f.name: value})
            except (TypeError, ValueError):
                invalid[f.name] += 1
                continue
            key = canonical(changed)
            if key in seen or changed == after:
                continue
            assert [x.name for x in dataclasses.fields(after)
                    if getattr(after, x.name) != getattr(changed, x.name)] == [f.name]
            seen.add(key)
            output.append((f.name, changed))
    return output, dict(invalid)


def check(fn, *args):
    try:
        return bool(fn(*args)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def measure(pre, post, transitions, errors):
    accepted = []
    normal_pass = normal_errors = pre_errors = 0
    for before, after in transitions:
        ok, err = check(pre, before)
        pre_errors += int(err is not None)
        accepted.append(ok is True)
        if ok is True:
            passed, err = check(post, before, after)
            normal_pass += int(passed is True)
            normal_errors += int(err is not None)
    fields = {}
    witnesses = {}
    for i, field, wrong, reference_rejects, reachable in errors:
        if not accepted[i]:
            continue
        c = fields.setdefault(field, Counter())
        passed, err = check(post, transitions[i][0], wrong)
        c["tested"] += 1
        c["rejected"] += int(passed is False)
        c["accepted"] += int(passed is True)
        c["errors"] += int(err is not None)
        c["reachable_tested"] += int(reachable)
        c["reachable_rejected"] += int(reachable and passed is False)
        c["reference_invalid_tested"] += int(reference_rejects)
        c["reference_invalid_rejected"] += int(reference_rejects and passed is False)
        if passed is True and field not in witnesses:
            witnesses[field] = {"transition_index": i, "before": plain(transitions[i][0]),
                                "correct_after": plain(transitions[i][1]), "wrong_after": plain(wrong),
                                "reference_rejects": reference_rejects, "reachable_output": reachable}
    total = Counter()
    for counts in fields.values():
        total.update(counts)
    n = sum(accepted)
    return {"normal_total": len(transitions), "normal_eligible": n,
            "pre_excluded": len(transitions) - n - pre_errors, "pre_errors": pre_errors,
            "normal_pass": normal_pass, "normal_errors": normal_errors,
            "normal_pass_rate": normal_pass / n if n else None,
            "mutation_counts": dict(total),
            "mutation_rejection_rate": total["rejected"] / total["tested"] if total["tested"] else None,
            "by_field": {k: dict(v) for k, v in fields.items()}, "accepted_witnesses": witnesses}


def adapter(domain, sandbox):
    package = f"experiment.{domain}"
    config_module = importlib.import_module(package + ".config")
    env = importlib.import_module(package + ".env")
    dsl = importlib.import_module(package + ".dsl")
    truth = importlib.import_module(package + ".ground_truth")
    enumeration = importlib.import_module(package + ".enumeration")
    raw = sandbox["config"]
    if domain == "deployment":
        cfg = config_module.DeploymentConfig.from_dict(raw)
        action = env.DeploymentAction.deploy_service()
        parse = lambda spec: dsl.parse_contract_dict(spec).bind(cfg)
        reference = truth.deploy_service_postcondition
    elif domain == "calendar":
        cfg = config_module.CalendarConfig.from_mapping(raw)
        action = env.CalendarAction.make(env.ActionKind.SCHEDULE_MEETING)
        parse = lambda spec: dsl.parse_contract(spec, cfg)
        reference = truth.schedule_meeting_postcondition
    elif domain == "shopping_price":
        cfg = config_module.PriceShoppingConfig.from_mapping(raw)
        action = env.PriceShoppingAction.place_order()
        parse = lambda spec: dsl.parse_contract(spec).bind(cfg)
        reference = lambda b, a: truth.place_order_postcondition(b, a, cfg)
    elif domain == "taubench_retail":
        cfg = config_module.RetailConfig.from_mapping(raw)
        action = env.RetailAction.make(env.ActionKind.EXCHANGE_ITEMS)
        parse = lambda spec: dsl.parse_contract(spec, cfg).bind(cfg)
        reference = lambda b, a: truth.exchange_items_postcondition(b, a, cfg)
    else:
        raise ValueError(domain)
    if domain == "calendar":
        serialized = plain(cfg)
        for name in ("room_capacities", "room_video_support", "external_bookings"):
            serialized[name] = {k: plain(v) for k, v in getattr(cfg, name)}
    else:
        serialized = cfg.to_dict()
    config_compatibility = {}
    if domain == "taubench_retail" and "deliverable_orders" not in raw:
        assert serialized.pop("deliverable_orders") == []
        config_compatibility["deliverable_orders"] = "legacy missing key: empty list; no deliver actions"
    assert serialized == raw, (domain, "config round-trip mismatch")
    closure = enumeration.enumerate_reachable(cfg, max_depth=sandbox["max_depth"], max_states=sandbox["max_states"])
    assert not closure.truncated and len(closure.states) == sandbox["evaluation_states"]
    outcomes = [(s, env.apply_action(s, action, cfg)) for s in closure.states]
    transitions = [(s, r.state) for s, r in outcomes if r.ok]
    assert transitions
    pools = {f.name: sorted({getattr(s, f.name) for s in closure.states}, key=canonical)
             for f in dataclasses.fields(transitions[0][1])}
    reachable = {canonical(s) for s in closure.states}
    errors, invalid, corpus = [], Counter(), []
    for i, (before, after) in enumerate(transitions):
        assert reference(before, after), (domain, "reference rejects normal transition", i)
        variants, bad = mutants(after, pools)
        invalid.update(bad)
        for field, wrong in variants:
            is_rejected = not reference(before, wrong)
            is_reachable = canonical(wrong) in reachable
            errors.append((i, field, wrong, is_rejected, is_reachable))
            corpus.append({"transition_index": i, "field": field, "wrong_after": plain(wrong),
                           "reference_rejects": is_rejected, "reachable_output": is_reachable})
    covered = {e[1] for e in errors}
    body = {"domain": domain, "sandbox": sandbox, "config_compatibility": config_compatibility,
            "states": len(closure.states),
            "normal_transitions": [dict(before=plain(b), after=plain(a)) for b, a in transitions],
            "mutants": corpus, "constructor_rejections": dict(invalid),
            "untested_fields": sorted(set(pools) - covered),
            "constant_true": measure(lambda _: True, lambda b, a: True, transitions, errors),
            "reference": measure(lambda _: True, reference, transitions, errors)}
    assert body["constant_true"]["normal_pass_rate"] == 1
    assert body["constant_true"]["mutation_rejection_rate"] == 0
    return parse, outcomes, transitions, errors, body


def select_inputs():
    manifest = json.loads((ROOT / "analysis/cell_manifest_2026-09-04.json").read_text())
    selected = []
    seen = set()
    for key, cell in manifest["cells"].items():
        if cell["table"] not in {"tab:easyv4", "tab:other", "pending_taubench"} or cell["method"] != "active_cegis":
            continue
        directories = cell["dir"] if isinstance(cell["dir"], list) else [cell["dir"]]
        records = []
        for directory in directories:
            for path in sorted((ROOT / "results" / directory).glob("*.json")):
                record = json.loads(path.read_text())
                if record.get("method") != "active_cegis":
                    continue
                assert path not in seen, (key, path)
                seen.add(path)
                for field in ("sandbox_config_sha256", "protocol_sha256", "budgets_sha256"):
                    assert record["hashes"][field] == cell[field], (key, path, field)
                assert sha256_json(record["sandbox"]["config"]) == record["hashes"]["sandbox_config_sha256"]
                records.append((path, record))
        assert sorted(r["seed"] for _, r in records) == cell["seeds"], key
        assert len(records) == cell["n_runs"], key
        assert sum(original_exact(r) for _, r in records) == cell["exact"], key
        selected.extend((key, cell["model"], path, record) for path, record in records)
    assert selected
    return selected


def original_exact(record):
    evaluation = record["evaluation"]
    if evaluation is None:
        assert record["outcome"] in {"parse_failure", "transport_error", "no_contract"}, record["outcome"]
        return False
    return evaluation["exact"]


def contract_spec(record):
    if record["contract"]["spec"] is not None:
        return record["contract"]["spec"]
    text = record["contract"]["source"].strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines)
    return json.loads(text)


def main():
    selected = select_inputs()
    inputs = {str(p.relative_to(ROOT)): file_hash(p) for _, _, p, _ in selected}
    source_paths = {*OUT.glob("*.py"), OUT / "PLAN.txt", ROOT / "analysis/cell_manifest_2026-09-04.json"}
    for domain in {r["sandbox"]["domain"] for _, _, _, r in selected}:
        source_paths.update((ROOT / "experiment" / domain).glob("*.py"))
    source_paths.add(ROOT / "experiment/modeling/artifacts.py")
    source_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(source_paths)}
    adapters, unique_scores, rows = {}, {}, []
    for cell, model, path, record in selected:
        domain = record["sandbox"]["domain"]
        config_id = record["hashes"]["sandbox_config_sha256"]
        cohort = domain + "_" + config_id[:12]
        row = {"cell": cell, "model": model, "seed": record["seed"], "domain": domain,
               "path": str(path.relative_to(ROOT)), "cohort": cohort,
               "original_exact": original_exact(record), "outcome": record["outcome"], "hashes": record["hashes"]}
        if not row["original_exact"]:
            row["status"] = "excluded_nonexact"
            rows.append(row)
            continue
        if cohort not in adapters:
            print("Building independent corpus:", cohort, flush=True)
            adapters[cohort] = adapter(domain, record["sandbox"])
            write_json(OUT / f"corpus_{cohort}.json", adapters[cohort][4])
        parse, outcomes, transitions, errors, corpus = adapters[cohort]
        assert record["sandbox"] == corpus["sandbox"], (path, "sandbox mismatch")
        spec = contract_spec(record)
        score_id = cohort + "_" + digest(spec)
        if score_id not in unique_scores:
            contract = parse(spec)
            fa = fr = pv = 0
            for before, outcome in outcomes:
                accepted = contract.holds_in(before)
                fa += int(accepted and not outcome.ok)
                fr += int(not accepted and outcome.ok)
                pv += int(accepted and outcome.ok and not contract.transition_holds(before, outcome.state))
            assert (fa, fr, pv) == (0, 0, 0), (path, "exact replay mismatch", fa, fr, pv)
            score = measure(contract.holds_in, contract.transition_holds, transitions, errors)
            assert score["normal_pass_rate"] == 1 and score["pre_excluded"] == 0
            unique_scores[score_id] = {"spec": spec, "replay": {"FA": fa, "FR": fr, "PV": pv}, **score}
        row.update(status="audited", score_id=score_id)
        rows.append(row)
    for name, expected in {**inputs, **source_hashes}.items():
        assert file_hash(ROOT / name) == expected, ("input/source changed during audit", name)
    results = {"schema_version": 1, "selection_runs": len(rows),
               "audited_runs": sum(r["status"] == "audited" for r in rows),
               "rows": rows, "scores": unique_scores,
               "corpora": {k: {f: v for f, v in a[4].items() if f not in {"mutants", "normal_transitions"}}
                           for k, a in adapters.items()}}
    write_json(OUT / "results.json", results)
    outputs = [OUT / "results.json", *sorted(OUT.glob("corpus_*.json"))]
    write_json(OUT / "manifest.json", {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                                      "python": sys.version, "inputs": inputs, "sources": source_hashes,
                                      "outputs": {p.name: file_hash(p) for p in outputs},
                                      "input_and_source_unchanged": True, "model_calls": 0})
    print(f"Audited {results['audited_runs']}/{len(rows)} runs; {len(unique_scores)} unique contracts", flush=True)


if __name__ == "__main__":
    main()
