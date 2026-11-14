"""Durable, opt-in information-suite checkpoints. Never retry an uncertain request."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..modeling.artifacts import Interaction, RunArtifact, sha256_json
from ..modeling.client import Usage
from .initial_candidate import InitialCandidate


class ResumeError(ValueError):
    pass


class SuiteIncomplete(RuntimeError):
    pass


def atomic_json(path: Path, payload, *, replace: bool = False) -> None:
    """Publish a whole file atomically; existing scientific artifacts are immutable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".pending-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # atomic exclusive creation; never overwrite a result
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def seal(payload):
    return {"payload": payload, "sha256": sha256_json(payload)}


def read_sealed(path: Path):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["sha256"] != sha256_json(record["payload"]):
            raise ValueError("content hash mismatch")
        return record["payload"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ResumeError(f"invalid checkpoint: {path}: {exc}") from exc


def usage_from_dict(value):
    return Usage(**{key: value[key] for key in (
        "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"
    ) if key in value})


class CallJournal:
    """Replay completed ask() transitions, not model-generated contracts or scores."""

    def __init__(self, folder: Path, identity: str, stop_requested=None):
        self.folder = folder
        self.identity = identity
        self.cursor = 0
        self.replayed_calls = 0
        self.replayed_tokens = 0
        self.replayed_latency_s = 0.0
        self.stop_requested = stop_requested

    def invoke(self, session, role, user_text, execute):
        path = self.folder / f"call_{self.cursor}.json"
        self.cursor += 1
        before = sha256_json({
            "identity": self.identity, "role": role, "user_text": user_text,
            "messages": session.messages,
            "interactions": [item.to_dict() for item in session.interactions],
            "spend": session.ledger.spend(), "usage": session.ledger.usage.to_dict(),
            "stopped_because": session.stopped_because,
        })
        if path.exists():
            record = read_sealed(path)
            if record.get("identity") != self.identity or record.get("before_sha256") != before:
                raise ResumeError(f"request replay mismatch: {path}")
            if record.get("status") != "completed":
                raise ResumeError(f"uncertain request; automatic resend forbidden: {path}")
            after = record["after"]
            self.replayed_calls += after["calls"] - session.ledger.calls
            self.replayed_tokens += after["usage"]["total_tokens"] - session.ledger.usage.total_tokens
            self.replayed_latency_s += sum(
                item.get("latency_s") or 0.0 for item in after["interactions"][len(session.interactions):]
            )
            session.messages = after["messages"]
            session.interactions = [Interaction.from_dict(item) for item in after["interactions"]]
            session.ledger.calls = after["calls"]
            session.ledger.usage = usage_from_dict(after["usage"])
            session.stopped_because = after["stopped_because"]
            returned = record["returned_index"]
            return None if returned is None else session.interactions[returned]

        if self.stop_requested is not None and self.stop_requested.is_set():
            raise InterruptedError("suite stopping before a new request")
        pending = {"status": "pending", "identity": self.identity, "before_sha256": before}
        atomic_json(path, seal(pending))
        # Exceptions/process death leave the pending marker intact. A request may
        # have reached the server even when no response was recorded locally.
        result = execute()
        completed = {**pending, "status": "completed", "after": {
            "messages": session.messages,
            "interactions": [item.to_dict() for item in session.interactions],
            "calls": session.ledger.calls, "usage": session.ledger.usage.to_dict(),
            "stopped_because": session.stopped_because,
        }, "returned_index": None if result is None else result.index}
        atomic_json(path, seal(completed), replace=True)
        return result


class InformationSuite:
    def __init__(self, runner, specs, output: Path, *, workers: int, resume: bool):
        self.runner = runner
        self.specs = tuple(specs)
        self.output = output
        self.workers = workers
        self.resume = resume
        self.meta = output / ".checkpoint"
        self.lock = None
        self.stop_requested = threading.Event()
        self.initials = {}
        self.completed = {}
        self.by_seed = {spec.seed: spec for spec in self.specs}
        self.by_name = {self.name(spec): spec for spec in self.specs}
        if workers < 1:
            raise ResumeError("workers must be positive")
        if not specs or len(self.by_name) != len(specs):
            raise ResumeError("suite requires unique method/trial identities")
        for spec in self.specs:
            runner._check_information_spec(spec)
            if runner.initial_context(spec) != runner.initial_context(self.by_seed[spec.seed]):
                raise ResumeError("methods within a trial must share the same initial context")
        self.manifest = {
            "version": "information_resume_v1", "workers": workers,
            "timeout": getattr(runner.client, "timeout", None),
            "matrix": [[spec.method, spec.seed] for spec in specs],
            "contexts": {str(seed): runner.initial_context(spec) for seed, spec in self.by_seed.items()},
            "resume_policy": "reuse_initials_results_and_recorded_calls; reject_uncertain_requests",
        }

    @staticmethod
    def name(spec):
        return f"{spec.method}__seed{spec.seed}"

    def __enter__(self):
        if self.resume:
            if not (self.meta / "manifest.json").is_file():
                raise ResumeError("resume requires an existing checkpoint manifest; no automatic legacy import")
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            self.meta.mkdir()
        self.lock = (self.meta / "lock").open("a+b")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.resume:
                if read_sealed(self.meta / "manifest.json") != self.manifest:
                    raise ResumeError("resume settings/source/matrix differ from the original manifest")
            else:
                atomic_json(self.meta / "manifest.json", seal(self.manifest))
            self._validate_existing()
            return self
        except BaseException:
            self.lock.close()
            self.lock = None
            raise

    def __exit__(self, *args):
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def _identity(self, spec, initial=None):
        return sha256_json({"context": self.runner.initial_context(spec),
                            "method": spec.method if initial is not None else "shared_initial",
                            "initial": initial.sha256 if initial is not None else None})

    def _validate_existing(self):
        # Validate every existing slot before any missing candidate is generated.
        for path in (self.output / "initial").glob("*.json"):
            expected = {f"trial_{seed}.json": seed for seed in self.by_seed}
            if path.name not in expected:
                raise ResumeError(f"unexpected initial candidate: {path}")
            seed = expected[path.name]
            initial = InitialCandidate.load(path)
            initial.validate(self.runner.initial_context(self.by_seed[seed]))
            self.initials[seed] = initial
        for folder in (self.meta / "calls").glob("*"):
            if folder.name.startswith("initial_"):
                try:
                    identity = self._identity(self.by_seed[int(folder.name.removeprefix("initial_"))])
                except (KeyError, ValueError) as exc:
                    raise ResumeError(f"unexpected initial journal: {folder}") from exc
            else:
                spec = self.by_name.get(folder.name)
                if spec is None or spec.seed not in self.initials:
                    raise ResumeError(f"unexpected or orphaned trial journal: {folder}")
                identity = self._identity(spec, self.initials[spec.seed])
            files = list(folder.glob("*.json"))
            if {path.name for path in files} != {f"call_{index}.json" for index in range(len(files))}:
                raise ResumeError(f"missing or unexpected request journal slots: {folder}")
            for path in files:
                record = read_sealed(path)
                if record.get("identity") != identity:
                    raise ResumeError(f"request journal identity mismatch: {path}")
                if record.get("status") != "completed":
                    raise ResumeError(f"uncertain request; automatic resend forbidden: {path}")
        for path in (self.meta / "results").glob("*.json"):
            name = path.stem
            if name not in self.by_name:
                raise ResumeError(f"unexpected completed result: {path}")
            data = read_sealed(path)
            spec = self.by_name[name]
            initial = self.initials.get(spec.seed)
            if initial is None:
                raise ResumeError(f"completed result is missing its original initial candidate: {path}")
            artifact = RunArtifact.from_dict(data)
            if (artifact.method != spec.method or artifact.seed != spec.seed
                    or artifact.model != self.runner.client.model
                    or artifact.decoding != spec.decoding.to_dict(spec.seed)
                    or artifact.budgets != spec.budgets.to_dict()
                    or artifact.hashes.get("initial_context_sha256") != initial.context_sha256
                    or artifact.hashes.get("initial_candidate_sha256") != initial.sha256):
                raise ResumeError(f"completed result identity mismatch: {path}")
            public = self.output / f"{name}.json"
            if public.exists() and json.loads(public.read_text()) != data:
                raise ResumeError(f"completed result differs from its checkpoint: {public}")
            self.completed[name] = artifact
        for path in self.output.glob("*__seed*.json"):
            if path.stem not in self.completed:
                raise ResumeError(f"result has no validated completion checkpoint: {path}")

    def _error(self, phase, target, exc):
        atomic_json(self.meta / "errors" / f"{uuid.uuid4().hex}.json", seal({
            "phase": phase, "target": target, "exception": type(exc).__name__, "message": str(exc),
        }))

    def execute(self):
        # Recover publication interrupted after durable completion, with no scoring/model call.
        for name, artifact in self.completed.items():
            path = self.output / f"{name}.json"
            if not path.exists():
                atomic_json(path, read_sealed(self.meta / "results" / f"{name}.json"))
        for seed, spec in self.by_seed.items():
            if seed in self.initials:
                continue
            journal = CallJournal(self.meta / "calls" / f"initial_{seed}", self._identity(spec),
                                  self.stop_requested)
            try:
                initial = self.runner.prepare_initial_candidate(spec, journal=journal)
                atomic_json(self.output / "initial" / f"trial_{seed}.json", initial.to_dict())
                self.initials[seed] = initial
            except BaseException as exc:
                self._error("initial", str(seed), exc)
                raise

        def execute_one(spec):
            name = self.name(spec)
            initial = self.initials[spec.seed]
            journal = CallJournal(self.meta / "calls" / name, self._identity(spec, initial),
                                  self.stop_requested)
            try:
                if self.stop_requested.is_set():
                    raise InterruptedError("suite stopping before a new trial")
                artifact = self.runner.run(spec, initial_candidate=initial, journal=journal)
                # Receipt first: if publication is interrupted, resume restores these exact bytes' data.
                atomic_json(self.meta / "results" / f"{name}.json", seal(artifact.to_dict()))
                atomic_json(self.output / f"{name}.json", artifact.to_dict())
                return artifact
            except BaseException as exc:
                self._error("trial", name, exc)
                raise

        pending = [spec for spec in self.specs if self.name(spec) not in self.completed]
        errors = []
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="information_suite") as pool:
            futures = {pool.submit(execute_one, spec): spec for spec in pending}
            try:
                for future in as_completed(futures):
                    try:
                        artifact = future.result()
                        self.completed[artifact.name] = artifact
                    except Exception as exc:
                        errors.append(exc)  # drain all finished work; never discard a sibling's result
            except BaseException:
                self.stop_requested.set()
                for future in futures:
                    future.cancel()
                raise
        if errors:
            raise SuiteIncomplete(f"{len(errors)} trial(s) incomplete; preserved results and request journals")
        return [self.completed[self.name(spec)] for spec in self.specs], self.initials
