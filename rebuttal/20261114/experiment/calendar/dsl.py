#!/usr/bin/env python3
"""A safe, declarative JSON DSL for candidate ``schedule_meeting`` contracts.

Model output is data, never code. Parsed into small closures without eval/exec.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import CalendarConfig
from .contracts import Contract
from .state import (
    ActionKind,
    CalendarAction,
    CalendarState,
    ExternalBooking,
    MeetingSnapshot,
    canonical_attendees,
)

SKILL = "schedule_meeting"

MAX_NODES = 200
MAX_DEPTH = 12
MAX_SOURCE_CHARS = 40000

CONTRACT_KEYS = ("skill", "name", "notes", "precondition", "postcondition")
REQUIRED_CONTRACT_KEYS = ("precondition", "postcondition")

SCOPE_PRE = "precondition"
SCOPE_POST = "postcondition"


class DslError(ValueError):
    """Model output is not a well-formed, in-grammar contract."""


class _Invalid:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<invalid>"


INVALID = _Invalid()


@dataclass(frozen=True)
class Binding:
    before: CalendarState
    after: CalendarState
    config: CalendarConfig


Term = Callable[[Binding], Any]
Formula = Callable[[Binding], bool]


_VAR_READERS: dict[str, Callable[[CalendarState], Any]] = {
    "authenticated": lambda state: state.authenticated,
    "has_active_meeting": lambda state: state.active_meeting is not None,
    "has_external_booking": lambda state: state.external_booking is not None,
    "draft_attendees": lambda state: list(state.draft_attendees),
    "draft_attendees_count": lambda state: len(state.draft_attendees),
    "draft_slot": lambda state: state.draft_slot,
    "draft_room": lambda state: state.draft_room,
    "draft_type": lambda state: state.draft_type,
    "external_room": lambda state: state.external_booking.room if state.external_booking else None,
    "external_slot": lambda state: state.external_booking.slot if state.external_booking else None,
    "external_attendees": lambda state: list(state.external_booking.attendees) if state.external_booking else [],
    "external_attendees_count": lambda state: len(state.external_booking.attendees) if state.external_booking else 0,
    "active_room": lambda state: state.active_meeting.room if state.active_meeting else None,
    "active_slot": lambda state: state.active_meeting.slot if state.active_meeting else None,
    "active_type": lambda state: state.active_meeting.meeting_type if state.active_meeting else None,
    "active_attendees": lambda state: list(state.active_meeting.attendees) if state.active_meeting else [],
}

_VAR_KINDS: dict[str, str] = {
    "authenticated": "bool",
    "has_active_meeting": "bool",
    "has_external_booking": "bool",
    "draft_attendees": "attendees",
    "draft_attendees_count": "number",
    "draft_slot": "option",
    "draft_room": "option",
    "draft_type": "option",
    "external_room": "option",
    "external_slot": "option",
    "external_attendees": "attendees",
    "external_attendees_count": "number",
    "active_room": "option",
    "active_slot": "option",
    "active_type": "option",
    "active_attendees": "attendees",
}

STATE_VARS: tuple[str, ...] = tuple(sorted(_VAR_READERS))
VAR_KINDS: Mapping[str, str] = dict(_VAR_KINDS)

CONFIG_SETS: tuple[str, ...] = (
    "attendees",
    "valid_slots",
    "rejected_slots",
    "slot_options",
    "valid_rooms",
    "rejected_rooms",
    "room_options",
    "valid_types",
    "rejected_types",
    "type_options",
)

_TEXTUAL = frozenset({"text", "option", "null"})

FORMULA_OPS: dict[str, str] = {
    "const": '{"op": "const", "value": true} — a literal truth value.',
    "and": '{"op": "and", "args": [f, ...]} — every listed formula holds.',
    "or": '{"op": "or", "args": [f, ...]} — at least one listed formula holds.',
    "not": '{"op": "not", "arg": f} — the formula does not hold.',
    "is_true": '{"op": "is_true", "arg": t} — a boolean term is true.',
    "is_null": '{"op": "is_null", "arg": t} — an optional term is unset (null).',
    "is_empty": '{"op": "is_empty", "arg": t} — an attendees list has no members.',
    "eq": '{"op": "eq", "left": t, "right": t} — two terms of the same kind are equal.',
    "ne": '{"op": "ne", "left": t, "right": t} — two terms of the same kind differ.',
    "lt": '{"op": "lt", "left": t, "right": t} — a number is strictly smaller.',
    "le": '{"op": "le", "left": t, "right": t} — a number is smaller or equal.',
    "gt": '{"op": "gt", "left": t, "right": t} — a number is strictly greater.',
    "ge": '{"op": "ge", "left": t, "right": t} — a number is greater or equal.',
    "in_set": '{"op": "in_set", "value": t, "set": "<config set>"} — member of set.',
    "overlaps": '{"op": "overlaps", "left": t, "right": t} — two attendee lists share at least one attendee.',
    "intersects": '{"op": "intersects", "left": t, "right": t} — alias for overlaps.',
    "contains": '{"op": "contains", "container": t, "item": t} — list contains item.',
    "capacity_sufficient": '{"op": "capacity_sufficient", "attendees": t, "room": t} — attendees count <= room capacity.',
    "room_supports_type": '{"op": "room_supports_type", "room": t, "type": t} — room supports meeting type (e.g. video_conf).',
    "no_room_conflict": '{"op": "no_room_conflict", "slot": t, "room": t} — draft slot & room do not conflict with external booking.',
    "no_attendee_conflict": '{"op": "no_attendee_conflict", "slot": t, "attendees": t} — draft slot & attendees do not conflict with external booking.',
    "scheduled_snapshot_matches": '{"op": "scheduled_snapshot_matches", "active_room": t, "active_slot": t, "active_type": t, "active_attendees": t, "draft_room": t, "draft_slot": t, "draft_type": t, "draft_attendees": t} — active meeting matches scheduled draft.',
    "unchanged": '{"op": "unchanged", "vars": ["..."]} — postcondition variables unchanged.',
}

TERM_OPS: dict[str, str] = {
    "var": '{"var": "<name>", "when": "before"|"after"} — read a state variable.',
    "const": '{"const": ...} — literal number, string, boolean, null, or attendees list.',
    "count": '{"op": "count", "arg": t} — count of attendees in a list.',
    "room_capacity": '{"op": "room_capacity", "room": t} — look up capacity of room from config.',
    "room_supports_video": '{"op": "room_supports_video", "room": t} — check if room supports video from config.',
}


def _config_set(name: str, config: CalendarConfig) -> set[str]:
    if name == "attendees":
        return set(config.attendees)
    if name == "valid_slots":
        return set(config.valid_slots)
    if name == "rejected_slots":
        return set(config.rejected_slots)
    if name == "slot_options":
        return set(config.slot_options)
    if name == "valid_rooms":
        return set(config.valid_rooms)
    if name == "rejected_rooms":
        return set(config.rejected_rooms)
    if name == "room_options":
        return set(config.room_options)
    if name == "valid_types":
        return set(config.valid_types)
    if name == "rejected_types":
        return set(config.rejected_types)
    if name == "type_options":
        return set(config.type_options)
    raise DslError(f"unknown config set: {name!r}; choose from {list(CONFIG_SETS)}")


class _Counter:
    __slots__ = ("nodes",)

    def __init__(self) -> None:
        self.nodes = 0

    def tick(self) -> None:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise DslError(f"contract exceeds {MAX_NODES} AST nodes")


def _read_var(var_name: str, when: str, binding: Binding) -> Any:
    state = binding.before if when == "before" else binding.after
    reader = _VAR_READERS.get(var_name)
    if reader is None:
        return INVALID
    return reader(state)


def _compile_term(node: Any, scope: str, depth: int, counter: _Counter) -> tuple[Term, str]:
    counter.tick()
    if depth > MAX_DEPTH:
        raise DslError(f"AST exceeds maximum depth of {MAX_DEPTH}")

    if not isinstance(node, dict):
        raise DslError(f"expected object for term, got {type(node).__name__}: {node!r}")

    if "var" in node:
        var_name = node["var"]
        if not isinstance(var_name, str) or var_name not in _VAR_READERS:
            raise DslError(f"unknown state variable {var_name!r}; choose from {STATE_VARS}")
        when = node.get("when", "before")
        if when not in ("before", "after"):
            raise DslError(f"'when' must be 'before' or 'after', got {when!r}")
        if when == "after" and scope != SCOPE_POST:
            raise DslError("when: 'after' is only valid inside postconditions")
        kind = _VAR_KINDS[var_name]

        def var_term(binding: Binding, v=var_name, w=when) -> Any:
            return _read_var(v, w, binding)

        return var_term, kind

    if "const" in node:
        val = node["const"]
        if isinstance(val, bool):
            kind = "bool"
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            val = int(val)
            kind = "number"
        elif isinstance(val, str):
            kind = "option"
        elif val is None:
            kind = "null"
        elif isinstance(val, (list, tuple)) and all(isinstance(x, str) for x in val):
            val = canonical_attendees(val)
            kind = "attendees"
        else:
            raise DslError(f"unsupported constant value: {val!r}")

        def const_term(binding: Binding, c=val) -> Any:
            return c

        return const_term, kind

    op = node.get("op")
    if not isinstance(op, str):
        raise DslError(f"term object must contain 'var', 'const', or 'op': {node!r}")

    if op == "count":
        arg_node = node.get("arg")
        if arg_node is None:
            raise DslError("'count' requires 'arg'")
        arg_term, arg_kind = _compile_term(arg_node, scope, depth + 1, counter)
        if arg_kind not in ("attendees", "null"):
            raise DslError(f"'count' requires attendees list, got {arg_kind}")

        def count_term(binding: Binding) -> int:
            val = arg_term(binding)
            if val is None or val is INVALID:
                return 0
            return len(val)

        return count_term, "number"

    if op == "room_capacity":
        room_node = node.get("room")
        if room_node is None:
            raise DslError("'room_capacity' requires 'room'")
        room_term, room_kind = _compile_term(room_node, scope, depth + 1, counter)
        if room_kind not in _TEXTUAL:
            raise DslError(f"'room_capacity' requires option/text room term, got {room_kind}")

        def cap_term(binding: Binding) -> int | None:
            r = room_term(binding)
            if r is None or r is INVALID:
                return None
            return binding.config.room_capacity(str(r))

        return cap_term, "number"

    if op == "room_supports_video":
        room_node = node.get("room")
        if room_node is None:
            raise DslError("'room_supports_video' requires 'room'")
        room_term, room_kind = _compile_term(room_node, scope, depth + 1, counter)
        if room_kind not in _TEXTUAL:
            raise DslError(f"'room_supports_video' requires option/text room term, got {room_kind}")

        def video_term(binding: Binding) -> bool:
            r = room_term(binding)
            if r is None or r is INVALID:
                return False
            return binding.config.room_supports_video(str(r))

        return video_term, "bool"

    raise DslError(f"unknown term operator: {op!r}; choose from {list(TERM_OPS)}")


def _compile_formula(node: Any, scope: str, depth: int, counter: _Counter) -> Formula:
    counter.tick()
    if depth > MAX_DEPTH:
        raise DslError(f"AST exceeds maximum depth of {MAX_DEPTH}")

    if not isinstance(node, dict):
        raise DslError(f"expected object for formula, got {type(node).__name__}: {node!r}")

    op = node.get("op")
    if not isinstance(op, str):
        raise DslError(f"formula object must contain 'op': {node!r}")

    if op == "const":
        val = node.get("value")
        if not isinstance(val, bool):
            raise DslError(f"'const' formula requires boolean 'value', got {val!r}")
        return lambda binding, v=val: v

    if op in ("and", "or"):
        args = node.get("args")
        if not isinstance(args, list) or not args:
            raise DslError(f"'{op}' requires non-empty list 'args'")
        compiled_args = [_compile_formula(arg, scope, depth + 1, counter) for arg in args]
        if op == "and":
            return lambda binding: all(f(binding) for f in compiled_args)
        return lambda binding: any(f(binding) for f in compiled_args)

    if op == "not":
        arg = node.get("arg")
        if arg is None:
            raise DslError("'not' requires 'arg'")
        compiled_arg = _compile_formula(arg, scope, depth + 1, counter)
        return lambda binding: not compiled_arg(binding)

    if op == "is_true":
        arg = node.get("arg")
        if arg is None:
            raise DslError("'is_true' requires 'arg'")
        term, kind = _compile_term(arg, scope, depth + 1, counter)
        if kind != "bool":
            raise DslError(f"'is_true' requires boolean term, got {kind}")
        return lambda binding: term(binding) is True

    if op == "is_null":
        arg = node.get("arg")
        if arg is None:
            raise DslError("'is_null' requires 'arg'")
        term, _ = _compile_term(arg, scope, depth + 1, counter)
        return lambda binding: term(binding) is None

    if op == "is_empty":
        arg = node.get("arg")
        if arg is None:
            raise DslError("'is_empty' requires 'arg'")
        term, kind = _compile_term(arg, scope, depth + 1, counter)
        if kind not in ("attendees", "null"):
            raise DslError(f"'is_empty' requires attendees term, got {kind}")
        return lambda binding: (not term(binding)) if term(binding) is not None else True

    if op in ("eq", "ne", "lt", "le", "gt", "ge"):
        left_node = node.get("left")
        right_node = node.get("right")
        if left_node is None or right_node is None:
            raise DslError(f"'{op}' requires 'left' and 'right'")
        left_term, left_kind = _compile_term(left_node, scope, depth + 1, counter)
        right_term, right_kind = _compile_term(right_node, scope, depth + 1, counter)

        if op in ("lt", "le", "gt", "ge"):
            if left_kind != "number" or right_kind != "number":
                raise DslError(f"'{op}' requires number comparisons, got {left_kind} and {right_kind}")
            if op == "lt":
                return lambda binding: (l := left_term(binding)) is not None and (r := right_term(binding)) is not None and l < r
            if op == "le":
                return lambda binding: (l := left_term(binding)) is not None and (r := right_term(binding)) is not None and l <= r
            if op == "gt":
                return lambda binding: (l := left_term(binding)) is not None and (r := right_term(binding)) is not None and l > r
            if op == "ge":
                return lambda binding: (l := left_term(binding)) is not None and (r := right_term(binding)) is not None and l >= r

        if left_kind == "attendees" or right_kind == "attendees":
            if left_kind not in ("attendees", "null") or right_kind not in ("attendees", "null"):
                raise DslError(f"'{op}' cannot compare attendees with {left_kind if left_kind != 'attendees' else right_kind}")
            if op == "eq":
                return lambda binding: tuple(sorted(left_term(binding) or ())) == tuple(sorted(right_term(binding) or ()))
            return lambda binding: tuple(sorted(left_term(binding) or ())) != tuple(sorted(right_term(binding) or ()))

        if (left_kind in _TEXTUAL and right_kind in _TEXTUAL) or (left_kind == right_kind):
            if op == "eq":
                return lambda binding: left_term(binding) == right_term(binding)
            return lambda binding: left_term(binding) != right_term(binding)

        raise DslError(f"'{op}' cannot compare incompatible kinds: {left_kind} and {right_kind}")

    if op == "in_set":
        val_node = node.get("value")
        set_name = node.get("set")
        if val_node is None or not isinstance(set_name, str):
            raise DslError("'in_set' requires 'value' and string 'set'")
        val_term, val_kind = _compile_term(val_node, scope, depth + 1, counter)
        if val_kind not in _TEXTUAL:
            raise DslError(
                f"'in_set' requires an option/text value, got {val_kind}"
            )
        if set_name not in CONFIG_SETS:
            raise DslError(f"unknown config set {set_name!r}; choose from {CONFIG_SETS}")
        return lambda binding: (v := val_term(binding)) is not None and v in _config_set(set_name, binding.config)

    if op in ("overlaps", "intersects"):
        left_node = node.get("left")
        right_node = node.get("right")
        if left_node is None or right_node is None:
            raise DslError(f"'{op}' requires 'left' and 'right'")
        left_term, left_kind = _compile_term(left_node, scope, depth + 1, counter)
        right_term, right_kind = _compile_term(right_node, scope, depth + 1, counter)
        if left_kind not in ("attendees", "null") or right_kind not in ("attendees", "null"):
            raise DslError(f"'{op}' requires attendees terms, got {left_kind} and {right_kind}")
        return lambda binding: bool(set(left_term(binding) or ()) & set(right_term(binding) or ()))

    if op == "contains":
        container_node = node.get("container")
        item_node = node.get("item")
        if container_node is None or item_node is None:
            raise DslError("'contains' requires 'container' and 'item'")
        cont_term, cont_kind = _compile_term(container_node, scope, depth + 1, counter)
        item_term, item_kind = _compile_term(item_node, scope, depth + 1, counter)
        if cont_kind not in ("attendees", "null"):
            raise DslError(f"'contains' requires attendees container, got {cont_kind}")
        if item_kind not in _TEXTUAL:
            raise DslError(f"'contains' requires option/text item, got {item_kind}")
        return lambda binding: (it := item_term(binding)) is not None and it in (cont_term(binding) or ())

    if op == "capacity_sufficient":
        att_node = node.get("attendees")
        room_node = node.get("room")
        if att_node is None or room_node is None:
            raise DslError("'capacity_sufficient' requires 'attendees' and 'room'")
        att_term, att_kind = _compile_term(att_node, scope, depth + 1, counter)
        room_term, room_kind = _compile_term(room_node, scope, depth + 1, counter)
        if att_kind not in ("attendees", "null"):
            raise DslError(f"'capacity_sufficient' requires attendees term, got {att_kind}")
        if room_kind not in _TEXTUAL:
            raise DslError(f"'capacity_sufficient' requires option/text room term, got {room_kind}")

        def cap_ok(binding: Binding) -> bool:
            r = room_term(binding)
            if r is None or r is INVALID:
                return False
            cap = binding.config.room_capacity(str(r))
            if cap is None:
                return False
            atts = att_term(binding) or ()
            return len(atts) <= cap

        return cap_ok

    if op == "room_supports_type":
        room_node = node.get("room")
        type_node = node.get("type")
        if room_node is None or type_node is None:
            raise DslError("'room_supports_type' requires 'room' and 'type'")
        room_term, room_kind = _compile_term(room_node, scope, depth + 1, counter)
        type_term, type_kind = _compile_term(type_node, scope, depth + 1, counter)
        if room_kind not in _TEXTUAL or type_kind not in _TEXTUAL:
            raise DslError(f"'room_supports_type' requires option/text terms, got {room_kind} and {type_kind}")

        def type_ok(binding: Binding) -> bool:
            t = type_term(binding)
            r = room_term(binding)
            if t != "video_conf":
                return True
            if r is None or r is INVALID:
                return False
            return binding.config.room_supports_video(str(r))

        return type_ok

    if op == "no_room_conflict":
        slot_node = node.get("slot")
        room_node = node.get("room")
        if slot_node is None or room_node is None:
            raise DslError("'no_room_conflict' requires 'slot' and 'room'")
        slot_term, slot_kind = _compile_term(slot_node, scope, depth + 1, counter)
        room_term, room_kind = _compile_term(room_node, scope, depth + 1, counter)
        if slot_kind not in _TEXTUAL or room_kind not in _TEXTUAL:
            raise DslError(f"'no_room_conflict' requires option/text slot and room terms, got {slot_kind} and {room_kind}")

        def no_r_conf(binding: Binding) -> bool:
            ext = binding.before.external_booking
            if ext is None:
                return True
            s = slot_term(binding)
            r = room_term(binding)
            return not (ext.slot == s and ext.room == r)

        return no_r_conf

    if op == "no_attendee_conflict":
        slot_node = node.get("slot")
        att_node = node.get("attendees")
        if slot_node is None or att_node is None:
            raise DslError("'no_attendee_conflict' requires 'slot' and 'attendees'")
        slot_term, slot_kind = _compile_term(slot_node, scope, depth + 1, counter)
        att_term, att_kind = _compile_term(att_node, scope, depth + 1, counter)
        if slot_kind not in _TEXTUAL:
            raise DslError(f"'no_attendee_conflict' requires option/text slot term, got {slot_kind}")
        if att_kind not in ("attendees", "null"):
            raise DslError(f"'no_attendee_conflict' requires attendees term, got {att_kind}")

        def no_att_conf(binding: Binding) -> bool:
            ext = binding.before.external_booking
            if ext is None:
                return True
            s = slot_term(binding)
            atts = att_term(binding) or ()
            return not (ext.slot == s and bool(set(ext.attendees) & set(atts)))

        return no_att_conf

    if op == "scheduled_snapshot_matches":
        if scope != SCOPE_POST:
            raise DslError("'scheduled_snapshot_matches' is only valid in postcondition")

        def snapshot_matches(binding: Binding) -> bool:
            before = binding.before
            after = binding.after
            if after.active_meeting is None:
                return False
            act = after.active_meeting
            expected = MeetingSnapshot(
                room=before.draft_room or "",
                slot=before.draft_slot or "",
                meeting_type=before.draft_type or "",
                attendees=before.draft_attendees,
            )
            return act == expected

        return snapshot_matches

    if op == "unchanged":
        if scope != SCOPE_POST:
            raise DslError("'unchanged' is only valid in postcondition")
        vars_list = node.get("vars")
        if not isinstance(vars_list, list) or not vars_list:
            raise DslError("'unchanged' requires non-empty list 'vars'")
        for v in vars_list:
            if v not in _VAR_READERS and v not in ("external_booking", "active_meeting"):
                raise DslError(f"unknown variable in unchanged: {v!r}")

        def unchanged_formula(binding: Binding, vl=tuple(vars_list)) -> bool:
            before = binding.before
            after = binding.after
            for var_name in vl:
                if var_name == "external_booking":
                    if before.external_booking != after.external_booking:
                        return False
                elif var_name == "active_meeting":
                    if before.active_meeting != after.active_meeting:
                        return False
                else:
                    reader = _VAR_READERS[var_name]
                    if reader(before) != reader(after):
                        return False
            return True

        return unchanged_formula

    raise DslError(f"unknown formula operator: {op!r}; choose from {list(FORMULA_OPS)}")


def parse_contract(source: str | Mapping[str, Any], config: CalendarConfig | None = None) -> Contract:
    """Parse JSON or dict into an executable Calendar Contract."""
    if isinstance(source, str):
        if len(source) > MAX_SOURCE_CHARS:
            raise DslError(f"contract source exceeds {MAX_SOURCE_CHARS} characters")
        text = source.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DslError(f"invalid JSON: {exc}") from exc
    elif isinstance(source, Mapping):
        data = dict(source)
    else:
        raise DslError(f"expected str or Mapping, got {type(source).__name__}")

    if not isinstance(data, dict):
        raise DslError(f"contract root must be a JSON object, got {type(data).__name__}")

    for key in REQUIRED_CONTRACT_KEYS:
        if key not in data:
            raise DslError(f"missing required key: {key!r}")

    skill = data.get("skill", SKILL)
    if skill != SKILL:
        raise DslError(f"expected skill {SKILL!r}, got {skill!r}")

    name = str(data.get("name", "candidate_schedule_meeting"))
    description = str(data.get("notes", ""))

    counter = _Counter()
    pre_fn = _compile_formula(data["precondition"], SCOPE_PRE, 1, counter)
    post_fn = _compile_formula(data["postcondition"], SCOPE_POST, 1, counter)

    effective_config = config or CalendarConfig()

    def precondition(state: CalendarState) -> bool:
        binding = Binding(before=state, after=state, config=effective_config)
        return bool(pre_fn(binding))

    def postcondition(before: CalendarState, after: CalendarState) -> bool:
        binding = Binding(before=before, after=after, config=effective_config)
        return bool(post_fn(binding))

    action = CalendarAction.make(ActionKind.SCHEDULE_MEETING)
    return Contract(
        name=name,
        action=action,
        precondition=precondition,
        postcondition=postcondition,
        description=description,
    )


def contract_json_schema() -> dict[str, Any]:
    """JSON Schema for guided decoding."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "CalendarContract",
        "type": "object",
        "required": ["skill", "precondition", "postcondition"],
        "properties": {
            "skill": {"type": "string", "const": SKILL},
            "name": {"type": "string"},
            "notes": {"type": "string"},
            "precondition": {"type": "object"},
            "postcondition": {"type": "object"},
        },
        "additionalProperties": False,
    }
