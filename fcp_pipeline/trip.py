"""Trip settings for the Final Cut footage pipeline.

:func:`load_trip` reads a trip folder's ``project.toml`` into a
:class:`TripSettings` object. The file follows the editor's project layout
contract: ``title`` at the top level (``status``, ``created``, ``dates`` and
``youtube`` are accepted but unused); ``[final_cut]`` with ``library`` and
``events`` and ``[media]`` with ``footage``, ``photos`` and ``assets``,
recorded as given; and the pipeline-owned ``[pipeline]`` table holding
``timezone``, ``start_date``, ``clock_offsets`` and ``places``. Each other
top-level key or table, and each unknown key inside ``[pipeline]``, produces
exactly one :class:`UserWarning` and loading still succeeds. Layout paths are
recorded without creating or checking them.
"""

import datetime
import re
import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

#: Pipeline-owned keys: they belong under ``[pipeline]``, never at top level.
TOP_LEVEL_PIPELINE_KEYS = ("timezone", "start_date", "clock_offsets", "places")
#: Accepted top-level keys and tables; anything else warns exactly once.
KNOWN_TOP_LEVEL_KEYS = (
    "title",
    "status",
    "created",
    "dates",
    "youtube",
    "final_cut",
    "media",
    "pipeline",
)
#: Keys the pipeline owns inside ``[pipeline]``; anything else warns exactly once.
KNOWN_PIPELINE_KEYS = ("timezone", "start_date", "clock_offsets", "places")

_CLOCK_OFFSET_RE = re.compile(r"([+-]?)([0-9]+):([0-5][0-9]):([0-5][0-9])")


class TripSettingsError(ValueError):
    """Raised when a trip folder's project.toml cannot be loaded."""


@dataclass(frozen=True)
class TripSettings:
    """Settings loaded from a trip folder's project.toml."""

    folder: Path
    title: str
    timezone: str
    start_date: datetime.date
    clock_offsets: dict[str, int] = field(default_factory=dict)
    places: list = field(default_factory=list)
    library: str = ""
    events: list = field(default_factory=list)
    footage: list = field(default_factory=list)
    photos: list = field(default_factory=list)
    assets: str = ""
    project_toml: Path | None = None
    outline: Path | None = None
    memo: Path | None = None
    transcripts: Path | None = None
    shots: Path | None = None
    edit: Path | None = None
    contact_sheets: Path | None = None
    pipeline: Path | None = None
    scratch: Path | None = None

    @property
    def name(self) -> str:
        """Backwards-compatible alias for :attr:`title`."""
        return self.title


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

    for key in TOP_LEVEL_PIPELINE_KEYS:
        if key in data:
            raise TripSettingsError(
                f"project.toml key {key!r} belongs under [pipeline], not at the top level"
            )

    missing = []
    if "title" not in data:
        missing.append("title")
    pipeline = data.get("pipeline", {})
    if not isinstance(pipeline, dict):
        raise TripSettingsError("project.toml table '[pipeline]' must be a table")
    for key in ("timezone", "start_date"):
        if key not in pipeline:
            missing.append(f"[pipeline].{key}")
    if missing:
        raise TripSettingsError(
            "missing required project.toml keys: " + ", ".join(missing)
        )

    title = data["title"]
    if not isinstance(title, str):
        raise TripSettingsError("project.toml key 'title' must be a string")
    timezone = pipeline["timezone"]
    if not isinstance(timezone, str):
        raise TripSettingsError("project.toml key '[pipeline].timezone' must be a string")
    start_date = pipeline["start_date"]
    if type(start_date) is not datetime.date:
        raise TripSettingsError(
            "project.toml key '[pipeline].start_date' must be a TOML local date (YYYY-MM-DD)"
        )

    raw_offsets = pipeline.get("clock_offsets", {})
    if not isinstance(raw_offsets, dict):
        raise TripSettingsError("project.toml key '[pipeline].clock_offsets' must be a table")
    clock_offsets = {}
    for camera, value in raw_offsets.items():
        try:
            clock_offsets[camera] = parse_clock_offset(value)
        except ValueError:
            raise TripSettingsError(
                f"bad clock offset for camera {camera!r}: {value!r}"
            ) from None

    places = pipeline.get("places", [])
    if not isinstance(places, list):
        raise TripSettingsError("project.toml key '[pipeline].places' must be an array")

    final_cut = data.get("final_cut", {})
    if not isinstance(final_cut, dict):
        raise TripSettingsError("project.toml table '[final_cut]' must be a table")
    library = final_cut.get("library", "")
    if not isinstance(library, str):
        raise TripSettingsError("project.toml key '[final_cut].library' must be a string")
    events = final_cut.get("events", [])
    if not isinstance(events, list):
        raise TripSettingsError("project.toml key '[final_cut].events' must be an array")

    media = data.get("media", {})
    if not isinstance(media, dict):
        raise TripSettingsError("project.toml table '[media]' must be a table")
    footage = media.get("footage", [])
    if not isinstance(footage, list):
        raise TripSettingsError("project.toml key '[media].footage' must be an array")
    photos = media.get("photos", [])
    if not isinstance(photos, list):
        raise TripSettingsError("project.toml key '[media].photos' must be an array")
    assets = media.get("assets", "")
    if not isinstance(assets, str):
        raise TripSettingsError("project.toml key '[media].assets' must be a string")

    for key in data:
        if key not in KNOWN_TOP_LEVEL_KEYS:
            warnings.warn(f"unknown project.toml key: {key}", UserWarning, stacklevel=2)
    for key in pipeline:
        if key not in KNOWN_PIPELINE_KEYS:
            warnings.warn(
                f"unknown project.toml key: [pipeline].{key}", UserWarning, stacklevel=2
            )

    return TripSettings(
        folder=folder,
        title=title,
        timezone=timezone,
        start_date=start_date,
        clock_offsets=clock_offsets,
        places=places,
        library=library,
        events=events,
        footage=footage,
        photos=photos,
        assets=assets,
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
