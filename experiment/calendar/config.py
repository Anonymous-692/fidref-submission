#!/usr/bin/env python3
"""Finite configuration for the controlled workspace scheduling domain."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .state import ExternalBooking


def _strings(values: tuple[str, ...], name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    normalised = tuple(values)
    if not allow_empty and not normalised:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value for value in normalised):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"{name} must not contain duplicates")
    return normalised


@dataclass(frozen=True)
class CalendarConfig:
    attendees: tuple[str, ...] = ("alice", "bob")
    max_attendees: int = 2
    valid_slots: tuple[str, ...] = ("morning", "afternoon")
    rejected_slots: tuple[str, ...] = ("midnight",)
    valid_rooms: tuple[str, ...] = ("room_alpha", "room_beta")
    rejected_rooms: tuple[str, ...] = ("deprecated_room",)
    valid_types: tuple[str, ...] = ("in_person", "video_conf")
    rejected_types: tuple[str, ...] = ("unsupported_type",)
    room_capacities: tuple[tuple[str, int], ...] = (
        ("deprecated_room", 2),
        ("room_alpha", 2),
        ("room_beta", 1),
    )
    room_video_support: tuple[tuple[str, bool], ...] = (
        ("deprecated_room", True),
        ("room_alpha", True),
        ("room_beta", False),
    )
    external_bookings: tuple[tuple[str, ExternalBooking], ...] = (
        ("alpha_morning_alice", ExternalBooking("room_alpha", "morning", ("alice",))),
        ("beta_afternoon_bob", ExternalBooking("room_beta", "afternoon", ("bob",))),
    )

    def __post_init__(self) -> None:
        for name in (
            "attendees", "valid_slots", "rejected_slots", "valid_rooms",
            "rejected_rooms", "valid_types", "rejected_types",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name, allow_empty=name.startswith("rejected")))
        if self.max_attendees < 1 or self.max_attendees > len(self.attendees):
            raise ValueError("max_attendees must be between 1 and the attendee count")
        if set(self.valid_slots) & set(self.rejected_slots):
            raise ValueError("valid and rejected slots must be disjoint")
        if set(self.valid_rooms) & set(self.rejected_rooms):
            raise ValueError("valid and rejected rooms must be disjoint")
        if set(self.valid_types) & set(self.rejected_types):
            raise ValueError("valid and rejected meeting types must be disjoint")
        rooms = set(self.room_options)
        if {name for name, _ in self.room_capacities} != rooms:
            raise ValueError("room_capacities must cover every room option exactly once")
        if {name for name, _ in self.room_video_support} != rooms:
            raise ValueError("room_video_support must cover every room option exactly once")
        if any(capacity < 0 for _, capacity in self.room_capacities):
            raise ValueError("room capacities must not be negative")
        booking_names = [name for name, _ in self.external_bookings]
        if len(set(booking_names)) != len(booking_names):
            raise ValueError("external booking names must be unique")
        for _, booking in self.external_bookings:
            if booking.room not in self.valid_rooms or booking.slot not in self.valid_slots:
                raise ValueError("external bookings must use valid rooms and slots")
            if not set(booking.attendees).issubset(self.attendees):
                raise ValueError("external bookings contain an unknown attendee")

    @property
    def slot_options(self) -> tuple[str, ...]:
        return self.valid_slots + self.rejected_slots

    @property
    def room_options(self) -> tuple[str, ...]:
        return self.valid_rooms + self.rejected_rooms

    @property
    def type_options(self) -> tuple[str, ...]:
        return self.valid_types + self.rejected_types

    def room_capacity(self, room: str | None) -> int | None:
        return dict(self.room_capacities).get(room) if room is not None else None

    def room_supports_video(self, room: str | None) -> bool:
        return bool(dict(self.room_video_support).get(room, False))

    def external_booking_for(self, name: str | None) -> ExternalBooking | None:
        return dict(self.external_bookings).get(name) if name is not None else None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CalendarConfig:
        external = tuple(
            (
                str(name),
                ExternalBooking(str(value["room"]), str(value["slot"]), tuple(value["attendees"])),
            )
            for name, value in data.get("external_bookings", {}).items()
        )
        return cls(
            attendees=tuple(data.get("attendees", cls.attendees)),
            max_attendees=int(data.get("max_attendees", cls.max_attendees)),
            valid_slots=tuple(data.get("valid_slots", cls.valid_slots)),
            rejected_slots=tuple(data.get("rejected_slots", cls.rejected_slots)),
            valid_rooms=tuple(data.get("valid_rooms", cls.valid_rooms)),
            rejected_rooms=tuple(data.get("rejected_rooms", cls.rejected_rooms)),
            valid_types=tuple(data.get("valid_types", cls.valid_types)),
            rejected_types=tuple(data.get("rejected_types", cls.rejected_types)),
            room_capacities=tuple(sorted((str(k), int(v)) for k, v in data.get("room_capacities", dict(cls.room_capacities)).items())),
            room_video_support=tuple(sorted((str(k), bool(v)) for k, v in data.get("room_video_support", dict(cls.room_video_support)).items())),
            external_bookings=external or cls.external_bookings,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> CalendarConfig:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


DEFAULT_CALENDAR_CONFIG = CalendarConfig()
