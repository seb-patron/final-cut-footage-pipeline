"""Trip settings for the Final Cut footage pipeline.

:func:`load_trip` reads a trip folder's ``project.toml`` into a
:class:`TripSettings` object. Only the five known top-level keys (``name``,
``timezone``, ``start_date``, ``clock_offsets`` and ``places``) are consumed;
each other top-level key produces exactly one :class:`UserWarning` and loading
still succeeds. Layout paths are recorded without creating or checking them.
"""

import datetime
import re
import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

KNOWN_KEYS = ("name", "timezone", "start_date", "clock_offsets", "places")
REQUIRED_KEYS = ("name", "timezone", "start_date")

_CLOCK_OFFSET_RE = re.compile(r"([+-]?)([0-9]+):([0-5][0-9]):([0-5][0-9])")


class TripSettingsError(ValueError):
    """Raised when a trip folder's project.toml cannot be loaded."""


@dataclass(frozen=True)
class TripSettings:
    """Settings loaded from a trip folder's project.toml."""

    folder: Path
    name: str
    timezone: str
    start_date: datetime.date
    clock_offsets: dict[str, int] = field(default_factory=dict)
    places: list = field(default_factory=list)
    project_toml: Path | None = None
    outline: Path | None = None
    memo: Path | None = None
    transcripts: Path | None = None
    shots: Path | None = None
    edit: Path | None = None
    contact_sheets: Path | None = None
    pipeline: Path | None = None
    scratch: Path | None = None


def parse_clock_offset(text: str) -> int:
    """Parse an optional-sign HH:MM:SS clock offset to signed seconds."""
    if not isinstance(text, str):
        raise ValueError(f"bad clock offset: {text!r}")
    match = _CLOCK_OFFSET_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"bad clock offset: {text!r}")
    sign, hours, minutes, seconds = match.groups()
    total = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return -total if sign == "-" else total


def load_trip(folder: str | Path) -> TripSettings:
    """Load ``<folder>/project.toml`` into a :class:`TripSettings`."""
    folder = Path(folder)
    project_toml = folder / "project.toml"
    try:
        with open(project_toml, "rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError:
        raise TripSettingsError(f"missing project.toml: {project_toml} not found") from None
    except tomllib.TOMLDecodeError as exc:
        raise TripSettingsError(f"invalid project.toml: {exc}") from None

    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise TripSettingsError(
            "missing required project.toml keys: " + ", ".join(missing)
        )

    name = data["name"]
    if not isinstance(name, str):
        raise TripSettingsError("project.toml key 'name' must be a string")
    timezone = data["timezone"]
    if not isinstance(timezone, str):
        raise TripSettingsError("project.toml key 'timezone' must be a string")
    start_date = data["start_date"]
    if not isinstance(start_date, datetime.date):
        raise TripSettingsError("project.toml key 'start_date' must be a TOML date")

    raw_offsets = data.get("clock_offsets", {})
    if not isinstance(raw_offsets, dict):
        raise TripSettingsError("project.toml key 'clock_offsets' must be a [clock_offsets] table")
    clock_offsets = {}
    for camera, value in raw_offsets.items():
        try:
            clock_offsets[camera] = parse_clock_offset(value)
        except ValueError:
            raise TripSettingsError(
                f"bad clock offset for camera {camera!r}: {value!r}"
            ) from None

    places = data.get("places", [])
    if not isinstance(places, list):
        raise TripSettingsError("project.toml key 'places' must be a [[places]] array")

    for key in data:
        if key not in KNOWN_KEYS:
            warnings.warn(f"unknown project.toml key: {key}", UserWarning, stacklevel=2)

    return TripSettings(
        folder=folder,
        name=name,
        timezone=timezone,
        start_date=start_date,
        clock_offsets=clock_offsets,
        places=places,
        project_toml=project_toml,
        outline=folder / "outline.md",
        memo=folder / "memo",
        transcripts=folder / "transcripts",
        shots=folder / "shots",
        edit=folder / "edit",
        contact_sheets=folder / "contact-sheets",
        pipeline=folder / ".pipeline",
        scratch=folder / ".scratch",
    )
