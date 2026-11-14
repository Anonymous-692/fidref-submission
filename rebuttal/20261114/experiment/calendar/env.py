#!/usr/bin/env python3
"""Pure transition rules for the controlled workspace scheduling domain."""

from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_CALENDAR_CONFIG, CalendarConfig
from .state import ActionKind, CalendarAction, CalendarState, MeetingSnapshot, canonical_attendees


@dataclass(frozen=True)
class StepResult:
    state: CalendarState
    ok: bool
    action: CalendarAction
    error: str | None = None

    def __post_init__(self) -> None:
        if self.ok and self.error is not None:
            raise ValueError("a successful step cannot carry an error")
        if not self.ok and not self.error:
            raise ValueError("a rejected step must explain why")


def initial_state(config: CalendarConfig = DEFAULT_CALENDAR_CONFIG) -> CalendarState:
    return CalendarState(False, (), None, None, None, None, None)


def action_space(config: CalendarConfig = DEFAULT_CALENDAR_CONFIG) -> tuple[CalendarAction, ...]:
    actions = [CalendarAction.make(ActionKind.LOGIN), CalendarAction.make(ActionKind.LOGOUT)]
    actions += [CalendarAction.make(ActionKind.ADD_ATTENDEE, value) for value in config.attendees]
    actions += [CalendarAction.make(ActionKind.REMOVE_ATTENDEE, value) for value in config.attendees]
    actions.append(CalendarAction.make(ActionKind.CLEAR_ATTENDEES))
    actions += [CalendarAction.make(ActionKind.SET_SLOT, value) for value in config.slot_options]
    actions.append(CalendarAction.make(ActionKind.CLEAR_SLOT))
    actions += [CalendarAction.make(ActionKind.SET_ROOM, value) for value in config.room_options]
    actions.append(CalendarAction.make(ActionKind.CLEAR_ROOM))
    actions += [CalendarAction.make(ActionKind.SET_MEETING_TYPE, value) for value in config.type_options]
    actions.append(CalendarAction.make(ActionKind.CLEAR_MEETING_TYPE))
    actions += [CalendarAction.make(ActionKind.SET_EXTERNAL_BOOKING, name) for name, _ in config.external_bookings]
    actions.append(CalendarAction.make(ActionKind.CLEAR_EXTERNAL_BOOKING))
    actions += [
        CalendarAction.make(ActionKind.SCHEDULE_MEETING),
        CalendarAction.make(ActionKind.CANCEL_MEETING),
        CalendarAction.make(ActionKind.LOAD_ACTIVE_MEETING),
    ]
    return tuple(sorted(actions, key=lambda action: action.sort_key))


def _reject(state: CalendarState, action: CalendarAction, reason: str) -> StepResult:
    return StepResult(state, False, action, reason)


def _accept(state: CalendarState, action: CalendarAction, **changes: object) -> StepResult:
    return StepResult(state.evolve(**changes), True, action)


def _draftable(state: CalendarState, action: CalendarAction) -> StepResult | None:
    if not state.authenticated:
        return _reject(state, action, "not authenticated")
    if state.active_meeting is not None:
        return _reject(state, action, "an active meeting already exists")
    return None


def _schedule_meeting_accepts(state: CalendarState, config: CalendarConfig) -> bool:
    external = state.external_booking
    room_conflict = bool(external and external.slot == state.draft_slot and external.room == state.draft_room)
    attendee_conflict = bool(
        external
        and external.slot == state.draft_slot
        and set(external.attendees).intersection(state.draft_attendees)
    )
    capacity = config.room_capacity(state.draft_room)
    return (
        state.authenticated
        and state.active_meeting is None
        and bool(state.draft_attendees)
        and state.draft_slot in config.valid_slots
        and state.draft_room in config.valid_rooms
        and state.draft_type in config.valid_types
        and capacity is not None
        and len(state.draft_attendees) <= capacity
        and (state.draft_type != "video_conf" or config.room_supports_video(state.draft_room))
        and not room_conflict
        and not attendee_conflict
    )


def apply_action(state: CalendarState, action: CalendarAction, config: CalendarConfig = DEFAULT_CALENDAR_CONFIG) -> StepResult:
    kind = action.kind
    if kind is ActionKind.LOGIN:
        return _reject(state, action, "already authenticated") if state.authenticated else _accept(state, action, authenticated=True)
    if kind is ActionKind.LOGOUT:
        return _reject(state, action, "not authenticated") if not state.authenticated else _accept(state, action, authenticated=False)
    if kind is ActionKind.CANCEL_MEETING:
        if not state.authenticated:
            return _reject(state, action, "not authenticated")
        if state.active_meeting is None:
            return _reject(state, action, "no active meeting")
        return _accept(state, action, active_meeting=None)
    if kind is ActionKind.LOAD_ACTIVE_MEETING:
        if not state.authenticated:
            return _reject(state, action, "not authenticated")
        if state.active_meeting is None:
            return _reject(state, action, "no active meeting")
        meeting = state.active_meeting
        return _accept(
            state,
            action,
            draft_attendees=meeting.attendees,
            draft_slot=meeting.slot,
            draft_room=meeting.room,
            draft_type=meeting.meeting_type,
        )
    blocked = _draftable(state, action)
    if blocked is not None:
        return blocked
    if kind is ActionKind.ADD_ATTENDEE:
        if action.target not in config.attendees:
            return _reject(state, action, "unknown attendee")
        if action.target in state.draft_attendees:
            return _reject(state, action, "attendee already added")
        if len(state.draft_attendees) >= config.max_attendees:
            return _reject(state, action, "attendee limit reached")
        return _accept(state, action, draft_attendees=canonical_attendees((*state.draft_attendees, action.target)))
    if kind is ActionKind.REMOVE_ATTENDEE:
        if action.target not in state.draft_attendees:
            return _reject(state, action, "attendee is not in the draft")
        return _accept(state, action, draft_attendees=tuple(x for x in state.draft_attendees if x != action.target))
    if kind is ActionKind.CLEAR_ATTENDEES:
        return _reject(state, action, "draft attendees are empty") if not state.draft_attendees else _accept(state, action, draft_attendees=())
    if kind is ActionKind.SET_SLOT:
        if action.target not in config.slot_options:
            return _reject(state, action, "unknown slot")
        return _reject(state, action, "slot already selected") if state.draft_slot == action.target else _accept(state, action, draft_slot=action.target)
    if kind is ActionKind.CLEAR_SLOT:
        return _reject(state, action, "no slot selected") if state.draft_slot is None else _accept(state, action, draft_slot=None)
    if kind is ActionKind.SET_ROOM:
        if action.target not in config.room_options:
            return _reject(state, action, "unknown room")
        return _reject(state, action, "room already selected") if state.draft_room == action.target else _accept(state, action, draft_room=action.target)
    if kind is ActionKind.CLEAR_ROOM:
        return _reject(state, action, "no room selected") if state.draft_room is None else _accept(state, action, draft_room=None)
    if kind is ActionKind.SET_MEETING_TYPE:
        if action.target not in config.type_options:
            return _reject(state, action, "unknown meeting type")
        return _reject(state, action, "meeting type already selected") if state.draft_type == action.target else _accept(state, action, draft_type=action.target)
    if kind is ActionKind.CLEAR_MEETING_TYPE:
        return _reject(state, action, "no meeting type selected") if state.draft_type is None else _accept(state, action, draft_type=None)
    if kind is ActionKind.SET_EXTERNAL_BOOKING:
        booking = config.external_booking_for(action.target)
        if booking is None:
            return _reject(state, action, "unknown external booking")
        return _reject(state, action, "external booking already selected") if state.external_booking == booking else _accept(state, action, external_booking=booking)
    if kind is ActionKind.CLEAR_EXTERNAL_BOOKING:
        return _reject(state, action, "no external booking") if state.external_booking is None else _accept(state, action, external_booking=None)
    if kind is ActionKind.SCHEDULE_MEETING:
        if not _schedule_meeting_accepts(state, config):
            return _reject(state, action, "draft meeting is not schedulable")
        snapshot = MeetingSnapshot(state.draft_room or "", state.draft_slot or "", state.draft_type or "", state.draft_attendees)
        return _accept(state, action, active_meeting=snapshot, draft_attendees=(), draft_slot=None, draft_room=None, draft_type=None)
    raise AssertionError(f"unhandled action kind: {kind}")


def valid_actions(state: CalendarState, config: CalendarConfig = DEFAULT_CALENDAR_CONFIG) -> tuple[CalendarAction, ...]:
    return tuple(action for action in action_space(config) if apply_action(state, action, config).ok)


class CalendarEnv:
    """Small stateful wrapper over the pure transition function."""

    def __init__(self, config: CalendarConfig | None = None) -> None:
        self._config = config or DEFAULT_CALENDAR_CONFIG
        self._state = initial_state(self._config)
        self._step_count = 0
        self._rejected_count = 0

    @property
    def state(self) -> CalendarState:
        return self._state

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def rejected_count(self) -> int:
        return self._rejected_count

    def reset(self) -> CalendarState:
        self._state = initial_state(self._config)
        self._step_count = 0
        self._rejected_count = 0
        return self._state

    def step(self, action: CalendarAction) -> StepResult:
        if not isinstance(action, CalendarAction):
            raise TypeError(f"expected CalendarAction, got {type(action).__name__}")
        result = apply_action(self._state, action, self._config)
        self._state = result.state
        self._step_count += 1
        self._rejected_count += int(not result.ok)
        return result
