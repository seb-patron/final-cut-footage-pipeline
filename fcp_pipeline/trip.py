"""Trip settings for the Final Cut footage pipeline.

:func:`load_trip` reads a trip folder's ``project.toml`` into a
:class:`TripSettings` object. The file follows the editor's project layout
contract: ``title`` at the top level (``status``, ``created``, ``dates`` and
``youtube`` are accepted but unused); ``[final_cut]`` with ``library`` and
``events`` and ``[media]`` with ``footage``, ``photos`` and ``assets``,
recorded as given; and the pipeline-owned ``[pipeline]`` table holding
``timezone``, ``start_date``, ``clock_offsets``, ``places``, ``offline`` and
the optional ``[pipeline.ai]`` provider setting. Each other
top-level key or table, and each unknown key inside ``[pipeline]``, produces
exactly one :class:`UserWarning` and loading still succeeds. Layout paths are
recorded without creating or checking them.
"""

import datetime
import math
import re
import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

#: Pipeline-owned keys: they belong under ``[pipeline]``, never at top level.
TOP_LEVEL_PIPELINE_KEYS = ("timezone", "start_date", "clock_offsets", "places", "offline", "ai", "transcribe")
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
KNOWN_PIPELINE_KEYS = ("timezone", "start_date", "clock_offsets", "places", "offline", "ai", "transcribe")
#: Keys accepted inside the ``[pipeline.transcribe]`` table.
TRANSCRIBE_KEYS = ("model",)
#: AI step tables that may appear under ``[pipeline.ai]``.
AI_STEP_TABLES = ("tag", "photos", "memo", "organize", "roughcut")
#: Transcription tables under ``[pipeline.ai]``; they never inherit [pipeline.ai].
AI_AUDIO_TABLES = ("memo_audio", "vo_audio")
#: Keys accepted directly under ``[pipeline.ai]`` besides the step tables.
AI_TOP_KEYS = ("provider", "model", "max_output_tokens", "prices")
#: Keys accepted inside each per-step table under ``[pipeline.ai]``.
AI_STEP_KEYS = ("provider", "model", "max_output_tokens")
#: Services accepted as an AI step provider.
AI_PROVIDERS = ("openai", "anthropic")
#: Services accepted as a transcription provider.
AI_AUDIO_PROVIDERS = ("openai", "local")


class TripSettingsError(ValueError):
    """Raised when a trip folder's project.toml cannot be loaded."""


def _ai_error(key: str, what: str) -> TripSettingsError:
    """Build a settings error naming a ``[pipeline.ai...]`` key."""
    return TripSettingsError(f"project.toml key '{key}' {what}")


def _check_ai_provider(key: str, value) -> str:
    if not isinstance(value, str) or value not in AI_PROVIDERS:
        raise _ai_error(key, "must be \"openai\" or \"anthropic\"")
    return value


def _check_ai_audio_provider(key: str, value) -> str:
    if not isinstance(value, str) or value not in AI_AUDIO_PROVIDERS:
        raise _ai_error(key, "must be \"openai\" or \"local\"")
    return value


def _check_ai_model(key: str, value) -> str:
    if not isinstance(value, str) or not value:
        raise _ai_error(key, "must be a non-empty string")
    return value


def _check_ai_tokens(key: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _ai_error(key, "must be a positive integer")
    return value


def _check_ai_price(key: str, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _ai_error(key, "must be a non-negative number")
    if not math.isfinite(value) or value < 0:
        raise _ai_error(key, "must be a non-negative number")
    return value


def _validate_ai_table(raw) -> dict:
    """Validate the ``[pipeline.ai]`` table into plain data.

    Unknown keys and unknown step tables each produce exactly one
    :class:`UserWarning` and loading still succeeds; type or value
    problems raise :class:`TripSettingsError` naming the key.
    """
    if not isinstance(raw, dict):
        raise TripSettingsError(
            "project.toml table '[pipeline.ai]' must be a table"
        )
    out: dict = {}
    if "provider" in raw:
        out["provider"] = _check_ai_provider(
            "[pipeline.ai].provider", raw["provider"]
        )
    if "model" in raw:
        out["model"] = _check_ai_model("[pipeline.ai].model", raw["model"])
    if "max_output_tokens" in raw:
        out["max_output_tokens"] = _check_ai_tokens(
            "[pipeline.ai].max_output_tokens", raw["max_output_tokens"]
        )
    if "prices" in raw:
        prices = raw["prices"]
        if not isinstance(prices, dict):
            raise TripSettingsError(
                "project.toml table '[pipeline.ai].prices' must be a table"
            )
        out_prices: dict = {}
        for model_id, entry in prices.items():
            base = f"[pipeline.ai.prices.{model_id!r}]"
            if not isinstance(entry, dict):
                raise TripSettingsError(
                    f"project.toml table '{base}' must be a table"
                )
            checked: dict = {}
            for rate_key in ("input_per_million", "output_per_million"):
                if rate_key not in entry:
                    raise _ai_error(
                        f"{base}.{rate_key}", "is required"
                    )
                checked[rate_key] = _check_ai_price(
                    f"{base}.{rate_key}", entry[rate_key]
                )
            for key in entry:
                if key not in ("input_per_million", "output_per_million"):
                    warnings.warn(
                        f"unknown project.toml key: {base}.{key}",
                        UserWarning,
                        stacklevel=2,
                    )
            out_prices[model_id] = checked
        out["prices"] = out_prices
    for step in AI_STEP_TABLES:
        if step not in raw:
            continue
        table = raw[step]
        base = f"[pipeline.ai.{step}]"
        if not isinstance(table, dict):
            raise TripSettingsError(
                f"project.toml table '{base}' must be a table"
            )
        checked_step: dict = {}
        if "provider" in table:
            checked_step["provider"] = _check_ai_provider(
                f"{base}.provider", table["provider"]
            )
        if "model" in table:
            checked_step["model"] = _check_ai_model(
                f"{base}.model", table["model"]
            )
        if "max_output_tokens" in table:
            checked_step["max_output_tokens"] = _check_ai_tokens(
                f"{base}.max_output_tokens", table["max_output_tokens"]
            )
        for key in table:
            if key not in AI_STEP_KEYS:
                warnings.warn(
                    f"unknown project.toml key: {base}.{key}",
                    UserWarning,
                    stacklevel=2,
                )
        out[step] = checked_step
    for step in AI_AUDIO_TABLES:
        if step not in raw:
            continue
        table = raw[step]
        base = f"[pipeline.ai.{step}]"
        if not isinstance(table, dict):
            raise TripSettingsError(
                f"project.toml table '{base}' must be a table"
            )
        checked_audio: dict = {}
        if "provider" in table:
            checked_audio["provider"] = _check_ai_audio_provider(
                f"{base}.provider", table["provider"]
            )
        if "model" in table:
            checked_audio["model"] = _check_ai_model(
                f"{base}.model", table["model"]
            )
        if "max_output_tokens" in table:
            checked_audio["max_output_tokens"] = _check_ai_tokens(
                f"{base}.max_output_tokens", table["max_output_tokens"]
            )
        for key in table:
            if key not in ("provider", "model", "max_output_tokens"):
                warnings.warn(
                    f"unknown project.toml key: {base}.{key}",
                    UserWarning,
                    stacklevel=2,
                )
        out[step] = checked_audio
    for key in raw:
        if key not in AI_TOP_KEYS and key not in AI_STEP_TABLES + AI_AUDIO_TABLES:
            warnings.warn(
                f"unknown project.toml key: [pipeline.ai].{key}",
                UserWarning,
                stacklevel=2,
            )
    return out

def _validate_transcribe_table(raw) -> str | None:
    """Validate the ``[pipeline.transcribe]`` table into a model id or None.

    An absent table gives None. A table that is not a table, or whose
    ``model`` is not a non-empty string, raises
    :class:`TripSettingsError` naming ``[pipeline.transcribe].model``;
    any other key inside it warns once with :class:`UserWarning`.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TripSettingsError(
            "project.toml key '[pipeline.transcribe].model' must be a "
            "non-empty string (the [pipeline.transcribe] table is not a table)"
        )
    model = raw.get("model")
    if not isinstance(model, str) or not model:
        raise TripSettingsError(
            "project.toml key '[pipeline.transcribe].model' must be a "
            "non-empty string"
        )
    for key in raw:
        if key not in TRANSCRIBE_KEYS:
            warnings.warn(
                f"unknown project.toml key: [pipeline.transcribe].{key}",
                UserWarning,
                stacklevel=2,
            )
    return model


_CLOCK_OFFSET_RE = re.compile(r"([+-]?)([0-9]+):([0-5][0-9]):([0-5][0-9])")


@dataclass(frozen=True)
class TripSettings:
    """Settings loaded from a trip folder's project.toml."""

    folder: Path
    title: str
    timezone: str
    start_date: datetime.date
    clock_offsets: dict[str, int] = field(default_factory=dict)
    places: list = field(default_factory=list)
    offline: bool = False
    ai: dict = field(default_factory=dict)
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
    transcribe_model: str | None = None

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

    offline = pipeline.get("offline", False)
    if not isinstance(offline, bool):
        raise TripSettingsError(
            "project.toml key '[pipeline].offline' must be a boolean"
        )

    ai = _validate_ai_table(pipeline.get("ai", {}))

    transcribe_model = _validate_transcribe_table(pipeline.get("transcribe"))

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
        offline=offline,
        ai=ai,
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
        transcribe_model=transcribe_model,
    )
