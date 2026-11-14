#!/usr/bin/env python3
"""Immutable values for the controlled workspace scheduling domain."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


Attendees = tuple[str, ...]


def canonical_attendees(values: Iterable[str]) -> Attendees:
    items = tuple(values)
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError("attendees must be non-empty strings")
    if len(set(items)) != len(items):
        raise ValueError("attendees must be unique")
    return tuple(sorted(items))


@dataclass(frozen=True)
class ExternalBooking:
    room: str
    slot: str
    attendees: Attendees

    def __post_init__(self) -> None:
        object.__setattr__(self, "attendees", canonical_attendees(self.attendees))

    def to_dict(self) -> dict[str, object]:
        return {"room": self.room, "slot": self.slot, "attendees": list(self.attendees)}


@dataclass(frozen=True)
class MeetingSnapshot:
    room: str
    slot: str
    meeting_type: str
    attendees: Attendees

    def __post_init__(self) -> None:
        object.__setattr__(self, "attendees", canonical_attendees(self.attendees))

    def to_dict(self) -> dict[str, object]:
        return {"room": self.room, "slot": self.slot, "meeting_type": self.meeting_type, "attendees": list(self.attendees)}


class ActionKind(Enum):
    LOGIN = 1
    LOGOUT = 2
    ADD_ATTENDEE = 3
    REMOVE_ATTENDEE = 4
    CLEAR_ATTENDEES = 5
    SET_SLOT = 6
    CLEAR_SLOT = 7
    SET_ROOM = 8
    CLEAR_ROOM = 9
    SET_MEETING_TYPE = 10
    CLEAR_MEETING_TYPE = 11
    SET_EXTERNAL_BOOKING = 12
    CLEAR_EXTERNAL_BOOKING = 13
    SCHEDULE_MEETING = 14
    CANCEL_MEETING = 15
    LOAD_ACTIVE_MEETING = 16


PARAMETERISED_KINDS = frozenset(
    {
        ActionKind.ADD_ATTENDEE,
        ActionKind.REMOVE_ATTENDEE,
        ActionKind.SET_SLOT,
        ActionKind.SET_ROOM,
        ActionKind.SET_MEETING_TYPE,
        ActionKind.SET_EXTERNAL_BOOKING,
    }
)


@dataclass(frozen=True)
class CalendarAction:
    kind: ActionKind
    target: str | None = None

    def __post_init__(self) -> None:
        if self.kind in PARAMETERISED_KINDS and self.target is None:
            raise ValueError(f"{self.kind.name} requires a target")
        if self.kind not in PARAMETERISED_KINDS and self.target is not None:
            raise ValueError(f"{self.kind.name} does not accept a target")

    @property
    def sort_key(self) -> tuple[int, str]:
        return (self.kind.value, self.target or "")

    @classmethod
    def make(cls, kind: ActionKind, target: str | None = None) -> CalendarAction:
        return cls(kind, target)


@dataclass(frozen=True)
class CalendarState:
    authenticated: bool
    draft_attendees: Attendees
    draft_slot: str | None
    draft_room: str | None
    draft_type: str | None
    external_booking: ExternalBooking | None
    active_meeting: MeetingSnapshot | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "draft_attendees", canonical_attendees(self.draft_attendees))

    def evolve(self, **changes: object) -> CalendarState:
        return dataclasses.replace(self, **changes)

    @property
    def key(self) -> tuple[object, ...]:
        external = self.external_booking
        active = self.active_meeting
        return (
            self.authenticated,
            self.draft_attendees,
            self.draft_slot or "",
            self.draft_room or "",
            self.draft_type or "",
            (external.room, external.slot, external.attendees) if external else ("", "", ()),
            (active.room, active.slot, active.meeting_type, active.attendees)
            if active
            else ("", "", "", ()),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "authenticated": self.authenticated,
            "draft_attendees": list(self.draft_attendees),
            "draft_slot": self.draft_slot,
            "draft_room": self.draft_room,
            "draft_type": self.draft_type,
            "external_booking": self.external_booking.to_dict() if self.external_booking else None,
            "active_meeting": self.active_meeting.to_dict() if self.active_meeting else None,
        }

    def describe(self) -> str:
        ext_str = f"{self.external_booking.room}@{self.external_booking.slot}" if self.external_booking else "none"
        act_str = f"{self.active_meeting.room}@{self.active_meeting.slot}" if self.active_meeting else "none"
        return f"auth={self.authenticated}, draft_att={list(self.draft_attendees)}, draft_slot={self.draft_slot}, draft_room={self.draft_room}, draft_type={self.draft_type}, ext={ext_str}, active={act_str}"


def sort_states(states: Iterable[CalendarState]) -> tuple[CalendarState, ...]:
    return tuple(sorted(states, key=lambda state: state.key))
