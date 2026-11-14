#!/usr/bin/env python3
"""Evaluator-only applicability predicate for ``schedule_meeting``."""

from __future__ import annotations

from .config import CalendarConfig
from .state import CalendarState, MeetingSnapshot

VISIBILITY = "evaluator-only"

CLAUSE_NAMES = (
    "authenticated", "no_active_meeting", "nonempty_attendees", "valid_slot",
    "valid_room", "valid_type", "room_capacity", "room_type_support",
    "no_room_slot_conflict", "no_attendee_slot_conflict",
)


def clause_values(state: CalendarState, config: CalendarConfig) -> tuple[bool, ...]:
    external = state.external_booking
    capacity = config.room_capacity(state.draft_room)
    same_slot = bool(external and external.slot == state.draft_slot)
    return (
        state.authenticated,
        state.active_meeting is None,
        bool(state.draft_attendees),
        state.draft_slot in config.valid_slots,
        state.draft_room in config.valid_rooms,
        state.draft_type in config.valid_types,
        capacity is not None and len(state.draft_attendees) <= capacity,
        state.draft_type != "video_conf" or config.room_supports_video(state.draft_room),
        not (same_slot and external is not None and external.room == state.draft_room),
        not (same_slot and external is not None and bool(set(external.attendees).intersection(state.draft_attendees))),
    )


def schedule_meeting_precondition(state: CalendarState, config: CalendarConfig) -> bool:
    return all(clause_values(state, config))


def schedule_meeting_postcondition(before: CalendarState, after: CalendarState) -> bool:
    expected = MeetingSnapshot(
        room=before.draft_room or "",
        slot=before.draft_slot or "",
        meeting_type=before.draft_type or "",
        attendees=before.draft_attendees,
    )
    return (
        after.active_meeting == expected
        and after.draft_attendees == ()
        and after.draft_slot is None
        and after.draft_room is None
        and after.draft_type is None
        and after.external_booking == before.external_booking
        and after.authenticated == before.authenticated
    )
