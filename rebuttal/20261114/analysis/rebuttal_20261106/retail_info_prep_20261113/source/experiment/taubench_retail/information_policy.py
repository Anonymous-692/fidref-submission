"""Retail V-only information ablation; private fixture remains evaluator-side.

The public view is a bijection over identifiers, not a new transition system.
Both conditions retain domain helper predicates and effect examples (P+).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from ..modeling.artifacts import sha256_json
from . import dsl, prompts
from .config import RetailConfig
from .state import SEMANTIC_FIELDS_PROTOCOL

CONDITIONS = ("vplus_pplus", "vminus_pplus")
PUBLIC_SETS = ("order_options", "item_options", "new_item_options", "payment_method_options")
PRIVATE_SETS = tuple(x for x in dsl.CONFIG_SETS if x not in PUBLIC_SETS)
VERSION = "retail_vonly_alias_view_v1"


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "taubench_retail").rglob("*.py"))
    files += [root / p for p in ("modeling/runner.py", "modeling/client.py",
                                "modeling/artifacts.py", "modeling/g2_runtime.py", "counterexamples.py",
                                "deployment/checkpoint.py", "deployment/initial_candidate.py")]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


@dataclass(frozen=True)
class InformationPolicy:
    condition: str
    config: RetailConfig

    def __post_init__(self):
        if self.condition not in CONDITIONS:
            raise ValueError("unsupported Retail information condition")

    @cached_property
    def aliases(self) -> dict[str, str]:
        groups = {
            "order": set(self.config.order_options),
            "item": set(self.config.item_options) | set(self.config.new_item_options),
            "payment": set(self.config.payment_method_options),
        }
        result = {}
        for group, values in groups.items():
            # Determined by the full option universe, never valid/rejected membership.
            ordered = sorted(values, key=lambda v: hashlib.sha256(f"retail-public-v1|{group}|{v}".encode()).hexdigest())
            for index, value in enumerate(ordered):
                if value in result:
                    raise ValueError("option namespaces overlap")
                result[value] = f"{group}_{index + 1:02d}"
        for name in dsl.state_var_kinds(SEMANTIC_FIELDS_PROTOCOL):
            for native in ("#W1", "#W2"):
                if native not in result:
                    raise ValueError("this preparation supports the original W1 fixture only")
                prefix = f"order_{native[1:].lower()}_"
                if name.startswith(prefix):
                    result[name] = name.replace(prefix, result[native] + "_", 1)
        if len(set(result.values())) != len(result):
            raise ValueError("public identifier collision")
        return result

    def encode(self, value: Any) -> Any:
        return self._map(value, self.aliases)

    def decode(self, value: Any) -> Any:
        return self._map(value, {v: k for k, v in self.aliases.items()})

    @staticmethod
    def _map(value: Any, mapping: dict[str, str]) -> Any:
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, dict):
            return {mapping.get(k, k): InformationPolicy._map(v, mapping) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [InformationPolicy._map(v, mapping) for v in value]
        return value

    def render(self, text: str) -> str:
        mapping = {**self.aliases, **{x: "unavailable_set" for x in PRIVATE_SETS}}
        # W1/W2 in explanatory prose refer to the same public order identifiers.
        mapping.update({"W1": self.aliases["#W1"], "W2": self.aliases["#W2"]})
        pattern = r"(?<![\w#])(?:" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r")(?!\w)"
        return re.sub(pattern, lambda m: mapping[m.group()], text)

    def system_prompt(self) -> str:
        base = prompts.system_prompt(SEMANTIC_FIELDS_PROTOCOL)
        lines = [line for line in base.splitlines() if line not in {f"- `{s}`" for s in PRIVATE_SETS}]
        lines += ["", "Only the listed full option sets are available to in_set.",
                  "Use public identifiers in all constants and state variable references.",
                  "Domain helper predicates and effect patterns are supplied background knowledge."]
        return self.render("\n".join(lines))

    def direct_prompt(self) -> str:
        lines = []
        for line in prompts.direct_prompt(self.config, SEMANTIC_FIELDS_PROTOCOL).splitlines():
            if line.startswith(("- Valid ", "- Rejected ")):
                continue
            lines.append(line)
            if line == "Task configuration:":
                for name in PUBLIC_SETS:
                    values = sorted(self.encode(list(getattr(self.config, name))))
                    lines.append(f"- {name}: {json.dumps(values)}")
                if self.condition == "vplus_pplus":
                    for name in PRIVATE_SETS:
                        # Labels disclosed as literal lists; private handles remain unavailable.
                        label = name.replace("_", " ")
                        values = sorted(self.encode(list(getattr(self.config, name))))
                        lines.append(f"- {label}: {json.dumps(values)}")
        return self.render("\n".join(lines))

    def schema(self) -> dict[str, Any]:
        schema = dsl.contract_json_schema(SEMANTIC_FIELDS_PROTOCOL)
        schema["title"] = "RetailInformationVOnlyV1"
        schema["description"] = "Public identifier view; restricted parser validates nested operators and set names."
        return schema

    def parse(self, source: str | dict[str, Any]):
        # Reject nonfinite values, excessive nesting, private handles and original IDs
        # before the legacy compiler can read fixture attributes or format an error.
        try:
            if isinstance(source, str):
                if len(source) > dsl.MAX_SOURCE_CHARS:
                    raise ValueError("size")
                text = source.strip()
                if text.startswith("```"):
                    lines = text.splitlines()[1:]
                    if lines and lines[-1].strip() == "```":
                        lines.pop()
                    text = "\n".join(lines)
                data = json.loads(text, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
            else:
                data = source
            if not isinstance(data, dict) or set(data) - set(dsl.CONTRACT_KEYS):
                raise ValueError("root")
            self._validate(data)
            return dsl.parse_contract(self.decode(data), self.config, vocabulary_protocol=SEMANTIC_FIELDS_PROTOCOL)
        except (dsl.DslError, ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
            raise dsl.DslError("Invalid public contract: check JSON, listed operators, public identifiers and option sets.") from None

    def _validate(self, value: Any, depth: int = 0):
        if depth > 48:
            raise ValueError("depth")
        if isinstance(value, str):
            if value in self.aliases or value in PRIVATE_SETS:
                raise ValueError("private name")
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("nonfinite")
        elif isinstance(value, dict):
            if "set" in value and value["set"] not in PUBLIC_SETS:
                raise ValueError("set")
            for key, child in value.items():
                self._validate(key, depth + 1)
                self._validate(child, depth + 1)
        elif isinstance(value, (list, tuple)):
            for child in value:
                self._validate(child, depth + 1)
        elif value is not None and not isinstance(value, (int, bool)):
            raise ValueError("value")

    def features(self, state, config=None):
        public = self.encode(state.to_dict(vocabulary_protocol=SEMANTIC_FIELDS_PROTOCOL))
        return tuple((key, json.dumps(value, sort_keys=True)) for key, value in sorted(public.items()))

    def metadata(self) -> dict[str, Any]:
        return {"version": VERSION, "condition": self.condition,
                "alias_sha256": sha256_json(self.aliases), "public_sets": list(PUBLIC_SETS),
                "selector_features": "public_raw_state_v1", "postcondition_patterns": "retained",
                "domain_helpers": "retained_supplied_background_knowledge",
                "vocabulary_protocol": SEMANTIC_FIELDS_PROTOCOL}
