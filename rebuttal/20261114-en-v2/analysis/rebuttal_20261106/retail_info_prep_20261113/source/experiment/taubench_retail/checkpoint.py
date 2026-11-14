"""Sequential durable Retail suite. In-flight uncertainty always blocks resend."""
from __future__ import annotations

import copy
import fcntl
import json
from pathlib import Path
import threading

from ..deployment.checkpoint import CallJournal, ResumeError, atomic_json, read_sealed, seal
from ..modeling.artifacts import sha256_json

METHODS = ("direct", "sampled_cegis_fixed", "active_cegis")


class RetailSuite:
    def __init__(self, runner, specs, output, *, resume=False):
        self.runner, self.specs = runner, tuple(specs)
        self.output, self.resume = Path(output), resume
        self.meta = self.output / ".checkpoint"
        self.stop_requested = threading.Event()
        self.lock = None
        self.initials, self.completed = {}, {}
        self.by_name = {self.name(s): s for s in self.specs}
        self.by_seed = {s.seed: s for s in self.specs}
        if not self.specs or len(self.by_name) != len(self.specs) or runner.information_policy is None:
            raise ResumeError("unique explicit information suite required")
        for s in self.specs:
            if s.method not in METHODS or type(s.seed) is not int or s.seed < 0:
                raise ResumeError("invalid method or seed")
            if runner.initial_context(s) != runner.initial_context(self.by_seed[s.seed]):
                raise ResumeError("shared initial conditions disagree")
        self.manifest = {"version": "retail_resume_v1", "matrix": [[s.method, s.seed] for s in self.specs],
                         "contexts": {str(k): runner.initial_context(s) for k, s in self.by_seed.items()},
                         "protocols": {n: runner.compute_protocol_hash(s) for n, s in self.by_name.items()},
                         "timeout": getattr(runner.client, "timeout", None)}

    @staticmethod
    def name(spec):
        return f"{spec.method}__seed{spec.seed}"

    def identity(self, spec, initial=None):
        return sha256_json({"context": self.runner.initial_context(spec),
                            "method": spec.method if initial is not None else "initial",
                            "initial": initial["sha256"] if initial is not None else None})

    def __enter__(self):
        if self.resume:
            if not (self.meta / "manifest.json").is_file():
                raise ResumeError("resume requires a manifest")
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            self.meta.mkdir()
        self.lock = (self.meta / "lock").open("a+b")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.resume:
                if read_sealed(self.meta / "manifest.json") != self.manifest:
                    raise ResumeError("source/configuration/matrix changed")
            else:
                atomic_json(self.meta / "manifest.json", seal(self.manifest))
            self.validate_existing()
            return self
        except BaseException:
            self.lock.close()
            raise

    def __exit__(self, *args):
        self.lock.close()

    def validate_existing(self):
        for p in (self.output / "initial").glob("*.json"):
            names = {f"trial_{seed}.json": seed for seed in self.by_seed}
            if p.name not in names:
                raise ResumeError("unexpected initial")
            seed = names[p.name]
            item = json.loads(p.read_text())
            self.runner.validate_initial(self.by_seed[seed], item)
            if not (self.meta / "calls" / f"initial_{seed}" / "call_0.json").is_file():
                raise ResumeError("initial request journal missing")
            self.initials[seed] = item
        for folder in (self.meta / "calls").glob("*"):
            if folder.name.startswith("initial_"):
                try:
                    identity = self.identity(self.by_seed[int(folder.name[8:])])
                except (ValueError, KeyError) as exc:
                    raise ResumeError("unexpected initial journal") from exc
            else:
                spec = self.by_name.get(folder.name)
                if spec is None or spec.seed not in self.initials:
                    raise ResumeError("orphaned run journal")
                identity = self.identity(spec, self.initials[spec.seed])
            files = list(folder.glob("*.json"))
            if {p.name for p in files} != {f"call_{i}.json" for i in range(len(files))}:
                raise ResumeError("missing journal slot")
            for p in files:
                data = read_sealed(p)
                if data.get("identity") != identity or data.get("status") != "completed":
                    raise ResumeError("uncertain/mismatched request; resend forbidden")
        for p in (self.meta / "results").glob("*.json"):
            if p.stem not in self.by_name:
                raise ResumeError("unexpected result")
            receipt = read_sealed(p)
            data = receipt["artifact"]
            folder = self.meta / "calls" / p.stem
            journals = {f.name: sha256_json(read_sealed(f)) for f in folder.glob("*.json")}
            if receipt["journals"] != journals:
                raise ResumeError("completed request journal missing or changed")
            spec = self.by_name[p.stem]
            initial = self.initials.get(spec.seed)
            if initial is None or (data["method"], data["seed"], data["model"], data["endpoint"]) != (
                    spec.method, spec.seed, self.runner.client.model, self.runner.client.endpoint):
                raise ResumeError("result identity mismatch")
            if (data["hashes"].get("initial_candidate_sha256") != initial["sha256"]
                    or data["hashes"].get("protocol_sha256") != self.runner.compute_protocol_hash(spec)
                    or data["decoding"] != spec.decoding.to_dict(spec.seed)
                    or data["budgets"] != spec.budgets.to_dict()):
                raise ResumeError("result provenance mismatch")
            public = self.output / p.name
            if public.exists() and json.loads(public.read_text()) != data:
                raise ResumeError("public result changed")
            self.completed[p.stem] = data
        for p in self.output.glob("*__seed*.json"):
            if p.stem not in self.completed:
                raise ResumeError("result without completion receipt")

    def execute(self):
        for name, data in self.completed.items():
            if not (self.output / f"{name}.json").exists():
                atomic_json(self.output / f"{name}.json", data)
        for seed, spec in self.by_seed.items():
            if seed in self.initials:
                continue
            child = copy.copy(self.runner)
            child.journal = CallJournal(self.meta / "calls" / f"initial_{seed}", self.identity(spec), self.stop_requested)
            initial = child.generate_initial(spec)
            child.validate_initial(spec, initial)
            atomic_json(self.output / "initial" / f"trial_{seed}.json", initial)
            self.initials[seed] = initial
        for spec in self.specs:
            name = self.name(spec)
            if name in self.completed:
                continue
            if self.stop_requested.is_set():
                raise InterruptedError("stopped before next run")
            child = copy.copy(self.runner)
            child.initial_candidate = self.initials[spec.seed]
            child.journal = CallJournal(self.meta / "calls" / name, self.identity(spec, child.initial_candidate), self.stop_requested)
            data = child.run(spec).to_dict()
            folder = self.meta / "calls" / name
            journals = {f.name: sha256_json(read_sealed(f)) for f in folder.glob("*.json")}
            atomic_json(self.meta / "results" / f"{name}.json", seal({"artifact": data, "journals": journals}))
            atomic_json(self.output / f"{name}.json", data)
            self.completed[name] = data
            print(f"completed {name}: {data['outcome']}", flush=True)
            if data["stopped_because"] in ("transport_error", "usage_unavailable", "token_budget_overrun"):
                raise ResumeError("infrastructure failure preserved; inspect before continuing")
        rows = [self.completed[self.name(s)] for s in self.specs]
        summary = {"version": "retail_suite_summary_v1", "count": len(rows),
                   "results_sha256": {self.name(s): sha256_json(r) for s, r in zip(self.specs, rows)}}
        path = self.output / "summary.json"
        if path.exists():
            if json.loads(path.read_text()) != summary:
                raise ResumeError("summary mismatch")
        else:
            atomic_json(path, summary)
        return rows
