#!/usr/bin/env python3
"""Core gates for the controlled calendar environment."""

from __future__ import annotations

import unittest
from pathlib import Path

from ..calendar import (
    ActionKind, CalendarAction, CalendarConfig, apply_action, audit_clause_density,
    action_space, enumerate_reachable, initial_state, schedule_meeting_postcondition,
    schedule_meeting_precondition,
)


class CalendarDomainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CalendarConfig()

    def test_rejection_does_not_mutate_state(self) -> None:
        state = initial_state(self.config)
        result = apply_action(state, CalendarAction.make(ActionKind.SCHEDULE_MEETING), self.config)
        self.assertFalse(result.ok)
        self.assertIs(result.state, state)

    def test_successful_schedule_and_cancel(self) -> None:
        state = initial_state(self.config)
        actions = (
            CalendarAction.make(ActionKind.LOGIN),
            CalendarAction.make(ActionKind.ADD_ATTENDEE, "alice"),
            CalendarAction.make(ActionKind.SET_SLOT, "afternoon"),
            CalendarAction.make(ActionKind.SET_ROOM, "room_alpha"),
            CalendarAction.make(ActionKind.SET_MEETING_TYPE, "video_conf"),
            CalendarAction.make(ActionKind.SCHEDULE_MEETING),
        )
        for action in actions:
            before = state
            result = apply_action(state, action, self.config)
            self.assertTrue(result.ok, result.error)
            state = result.state
        self.assertIsNotNone(state.active_meeting)
        self.assertEqual(state.draft_attendees, ())
        self.assertTrue(schedule_meeting_postcondition(before, state))
        cancelled = apply_action(state, CalendarAction.make(ActionKind.CANCEL_MEETING), self.config)
        self.assertTrue(cancelled.ok)
        self.assertIsNone(cancelled.state.active_meeting)

    def test_ground_truth_matches_environment_over_full_closure(self) -> None:
        enumeration = enumerate_reachable(self.config)
        self.assertFalse(enumeration.truncated)
        target = CalendarAction.make(ActionKind.SCHEDULE_MEETING)
        for state in enumeration.states:
            self.assertEqual(
                schedule_meeting_precondition(state, self.config),
                apply_action(state, target, self.config).ok,
            )

    def test_closure_and_clause_density_gates(self) -> None:
        first = enumerate_reachable(self.config)
        second = enumerate_reachable(self.config)
        self.assertEqual(first.states, second.states)
        self.assertFalse(first.truncated)
        self.assertEqual(len(first.states), 1676)
        self.assertEqual(len(action_space(self.config)), 25)
        rows = audit_clause_density(first, self.config)
        self.assertEqual(len(rows), 10)
        for row in rows:
            self.assertGreater(row.witnesses, 0, row.name)
            self.assertIsNotNone(row.minimum_depth, row.name)
        relational = {row.name: row for row in rows}
        self.assertLessEqual(relational["no_room_slot_conflict"].minimum_depth or 99, 6)
        self.assertLessEqual(relational["no_attendee_slot_conflict"].minimum_depth or 99, 6)

    def test_room_and_attendee_conflicts_are_independently_falsifiable(self) -> None:
        def build(*, attendee: str, room: str) -> object:
            state = initial_state(self.config)
            for action in (
                CalendarAction.make(ActionKind.LOGIN),
                CalendarAction.make(ActionKind.ADD_ATTENDEE, attendee),
                CalendarAction.make(ActionKind.SET_SLOT, "morning"),
                CalendarAction.make(ActionKind.SET_ROOM, room),
                CalendarAction.make(ActionKind.SET_MEETING_TYPE, "in_person"),
                CalendarAction.make(ActionKind.SET_EXTERNAL_BOOKING, "alpha_morning_alice"),
            ):
                state = apply_action(state, action, self.config).state
            return state

        room_only = build(attendee="bob", room="room_alpha")
        attendee_only = build(attendee="alice", room="room_beta")
        self.assertFalse(schedule_meeting_precondition(room_only, self.config))
        self.assertFalse(schedule_meeting_precondition(attendee_only, self.config))

    def test_default_json_matches_default_config(self) -> None:
        loaded = CalendarConfig.from_json("experiment/configs/calendar_default.json")
        self.assertEqual(loaded, self.config)

    def test_agent_facing_environment_does_not_import_evaluator_ground_truth(self) -> None:
        source = Path("experiment/calendar/env.py").read_text(encoding="utf-8")
        self.assertNotIn("import ground_truth", source)
        self.assertNotIn("from .ground_truth", source)


if __name__ == "__main__":
    unittest.main()
