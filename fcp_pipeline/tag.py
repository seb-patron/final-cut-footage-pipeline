"""Tag one clip from its contact sheet through the AI provider setting.

For each catalog clip that has a current contact sheet, this step sends
that one sheet image plus one metadata line, the clip's slate text, the
first 300 words of its transcript and the trip vocabulary through
:func:`fcp_pipeline.ai.ask` with step name ``"tag"``, checks the
structured answer and stores it as catalog tags with source ``"ai"``.
Only the contact-sheet image is ever sent, never a footage file, a
video or a full-resolution frame. Importing this module runs no tool,
opens no database and reads no file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from fcp_pipeline import ai
from fcp_pipeline import catalog as catalog_module
from fcp_pipeline.net import OfflineError, set_offline
from fcp_pipeline.sheets import read_manifest
from fcp_pipeline.trip import TripSettingsError, load_trip

#: Step name used for the AI call, the run record and the usage log.
STEP = "tag"
#: Confidence at or above this value stores a plain tag; below it a check.
DEFAULT_CUTOFF = 0.75
#: How many transcript words the model sees.
TRANSCRIPT_WORDS = 300
#: Largest contact sheet accepted, in pixels.
MAX_SHEET_SIZE = (1536, 864)
#: Most values kept per vocabulary folder; the first in sorted order win.
MAX_VOCABULARY_VALUES = 200
#: Largest absolute value any AI answer number may carry. Times are
#: seconds within one clip and confidences run 0 to 1, so anything
#: wider is not a real measurement. Wider integers are rejected before
#: float conversion (which would raise OverflowError) and floats are
#: checked before scaling to milliseconds (which would overflow).
_MAX_ANSWER_NUMBER = 1_000_000
#: Longest free-text value the vocabulary takes from the "Slate"
#: folder. Slate values are sentences a person wrote or that vision
#: read off the picture; only short ones work as repeatable labels.
_MAX_SLATE_VOCAB_CHARS = 80
#: Whole-clip folders written by this step.
SHOT_FOLDER = "shot"
NOTE_FOLDER = "note"
TODO_FOLDER = "todo"
HINT_FOLDER = "outline-hint"

#: Folders that never enter the shown vocabulary (this step owns them).
_OWN_FOLDERS = frozenset({SHOT_FOLDER, NOTE_FOLDER, TODO_FOLDER, HINT_FOLDER})

#: Tag sources this step reads slate text from.
_SLATE_SOURCES = ("slate", "vision-text")

#: Marker phrases stripped from slate and transcript text before sending.
_MARKERS = (
    "BEGIN SLATE",
    "END SLATE",
    "BEGIN TRANSCRIPT",
    "END TRANSCRIPT",
)


class TagError(ValueError):
    """Raised for bad input to the tag step."""


@dataclass(frozen=True)
class TagEntry:
    """What the tag step did with one catalog clip."""

    path: str
    asset_id: str
    status: str
    reason: str | None = None
    stored: int = 0
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TagResult:
    """Every entry of one tag run, in catalog id order."""

    entries: tuple[TagEntry, ...] = ()

    @property
    def tagged(self) -> int:
        """How many clips were tagged in this run."""
        return sum(1 for entry in self.entries if entry.status == "tagged")

    @property
    def skipped(self) -> int:
        """How many clips already had ai tags and were left alone."""
        return sum(1 for entry in self.entries if entry.status == "skipped")

    @property
    def stale(self) -> int:
        """How many rows no longer matched their file."""
        return sum(1 for entry in self.entries if entry.status == "stale")

    @property
    def no_sheet(self) -> int:
        """How many clips had no contact sheet for their current state."""
        return sum(1 for entry in self.entries if entry.status == "no-sheet")

    @property
    def failed(self) -> int:
        """How many clips could not be tagged."""
        return sum(1 for entry in self.entries if entry.status == "failed")

    @property
    def would_tag(self) -> int:
        """How many clips a dry run would tag."""
        return sum(
            1 for entry in self.entries if entry.status == "would-tag"
        )

    @property
    def outcome(self) -> str:
        """The run-record line, like ``"tagged 3, skipped 1, ..."``."""
        if self.would_tag:
            first = f"would-tag {self.would_tag}"
        else:
            first = f"tagged {self.tagged}"
        return (
            f"{first}, skipped {self.skipped}, stale {self.stale}, "
            f"no-sheet {self.no_sheet}, failed {self.failed}"
        )


def answer_schema() -> dict:
    """Return the tag answer schema, read from next to this module."""
    path = Path(__file__).resolve().parent / "tag_schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def first_words(text: str, limit: int) -> str:
    """Keep the first ``limit`` whitespace-separated words of ``text``."""
    if not isinstance(text, str):
        raise TagError(f"bad text {text!r}: must be a string")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise TagError(
            f"bad limit {limit!r}: must be a non-negative integer"
        )
    return " ".join(text.split()[:limit])


def jpeg_size(data: bytes) -> tuple[int, int]:
    """Return the ``(width, height)`` of JPEG ``data``.

    A small walk over the JPEG markers to the first start-of-frame
    marker; anything that is not a JPEG raises :class:`TagError`.
    """
    if not isinstance(data, bytes) or len(data) < 4:
        raise TagError("bad sheet bytes: not a JPEG image")
    if data[0] != 0xFF or data[1] != 0xD8:
        raise TagError("bad sheet bytes: not a JPEG image")
    standalone = {0xD8, 0xD9, 0x01} | set(range(0xD0, 0xD8))
    frame_markers = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    pos = 2
    size = len(data)
    try:
        while pos + 3 < size:
            if data[pos] != 0xFF:
                raise TagError("bad sheet bytes: not a JPEG image")
            while pos < size and data[pos] == 0xFF:
                pos += 1
            if pos >= size:
                break
            marker = data[pos]
            pos += 1
            if marker in standalone:
                continue
            if pos + 1 >= size:
                break
            length = struct.unpack(">H", data[pos : pos + 2])[0]
            if length < 2 or pos + length > size:
                raise TagError("bad sheet bytes: not a JPEG image")
            if marker in frame_markers:
                if length < 7:
                    raise TagError("bad sheet bytes: not a JPEG image")
                height = struct.unpack(">H", data[pos + 3 : pos + 5])[0]
                width = struct.unpack(">H", data[pos + 5 : pos + 7])[0]
                if width <= 0 or height <= 0:
                    raise TagError("bad sheet bytes: not a JPEG image")
                return (width, height)
            pos += length
    except (IndexError, struct.error):
        raise TagError("bad sheet bytes: not a JPEG image") from None
    raise TagError("bad sheet bytes: not a JPEG image")


def snap_range(
    start: Fraction, end: Fraction, edges: object
) -> tuple[Fraction, Fraction]:
    """Snap ``(start, end)`` outwards to the nearest tile edges.

    ``start`` moves down to the largest edge at or below it and ``end``
    moves up to the smallest edge at or above it, so a stored range
    always covers what the model named.
    """
    points = sorted(set(edges))  # type: ignore[arg-type]
    below = [edge for edge in points if edge <= start]
    above = [edge for edge in points if edge >= end]
    snapped_start = below[-1] if below else start
    snapped_end = above[0] if above else end
    return (snapped_start, snapped_end)


def _clean_markers(text: str) -> str:
    """Remove the BEGIN/END marker phrases from untrusted ``text``."""
    cleaned = str(text)
    for marker in _MARKERS:
        cleaned = cleaned.replace(marker, "")
    return cleaned


def build_vocabulary(
    catalog: object, settings: object, extra: object = None
) -> dict[str, tuple[str, ...]]:
    """Return ``folder -> sorted values`` shown to the model.

    The vocabulary holds the ``name`` of every places entry that is a
    table with a non-empty string name (folder ``"Place"``, the same
    name the days_places step writes), the distinct non-empty camera
    values in the catalog (folder ``"Gear"``), the distinct
    ``(folder, value)`` of every existing catalog tag whose source is
    not ``"ai"`` and whose folder is not one this step owns, and the
    optional ``extra`` mapping of folder to string values. The
    ``"Slate"`` folder only contributes short values (at most
    :data:`_MAX_SLATE_VOCAB_CHARS` characters); its longer free text
    never enters the vocabulary. Each folder keeps at most
    :data:`MAX_VOCABULARY_VALUES` values, the first in sorted order.
    """
    folders: dict[str, set[str]] = {}
    for entry in getattr(settings, "places", None) or ():
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str) and name:
            folders.setdefault("Place", set()).add(name)
    rows = catalog.list_media()  # type: ignore[union-attr]
    for row in rows:
        camera = getattr(row, "camera", None)
        if isinstance(camera, str) and camera:
            folders.setdefault("Gear", set()).add(camera)
    for row in rows:
        for tag in catalog.list_tags(row.id):  # type: ignore[union-attr]
            folder = getattr(tag, "folder", None)
            value = getattr(tag, "value", None)
            if not isinstance(folder, str) or not isinstance(value, str):
                continue
            if getattr(tag, "source", None) == "ai":
                continue
            if folder in _OWN_FOLDERS:
                continue
            if (
                folder == "Slate"
                and len(value) > _MAX_SLATE_VOCAB_CHARS
            ):
                continue
            folders.setdefault(folder, set()).add(value)
    if extra is not None:
        if not isinstance(extra, dict):
            raise TagError(
                f"bad extra vocabulary {extra!r}: must map folders to values"
            )
        for folder, values in extra.items():
            if not isinstance(folder, str):
                raise TagError(
                    f"bad vocabulary folder {folder!r}: must be a string"
                )
            if not isinstance(values, (list, tuple)) or any(
                not isinstance(value, str) for value in values
            ):
                raise TagError(
                    f"bad vocabulary values for folder {folder!r}: "
                    "must be a list of strings"
                )
            folders.setdefault(folder, set()).update(values)
    return {
        folder: tuple(sorted(values)[:MAX_VOCABULARY_VALUES])
        for folder, values in sorted(folders.items())
    }


def _check_vocabulary_mapping(vocabulary: object) -> dict[str, tuple[str, ...]]:
    """Normalize an explicit ``vocabulary`` mapping, or raise TagError."""
    if not isinstance(vocabulary, dict):
        raise TagError(
            f"bad vocabulary {vocabulary!r}: must map folders to values"
        )
    checked: dict[str, tuple[str, ...]] = {}
    for folder, values in vocabulary.items():
        if not isinstance(folder, str):
            raise TagError(
                f"bad vocabulary folder {folder!r}: must be a string"
            )
        if not isinstance(values, (list, tuple)) or any(
            not isinstance(value, str) for value in values
        ):
            raise TagError(
                f"bad vocabulary values for folder {folder!r}: "
                "must be a list of strings"
            )
        checked[folder] = tuple(values)
    return checked


def build_instructions(
    metadata: str,
    slate: str,
    transcript: str,
    vocabulary: dict[str, tuple[str, ...]],
    tiles: object,
) -> str:
    """Fill the tag prompt and return the instructions string for ask."""
    path = Path(__file__).resolve().parent / "prompts" / "tag.md"
    prompt = path.read_text(encoding="utf-8")
    vocabulary_lines = [
        f"{folder}: {', '.join(values)}"
        for folder, values in sorted(vocabulary.items())
    ]
    tile_times = [Fraction(moment) for moment in tiles]  # type: ignore[union-attr]
    tile_texts = [f"{float(moment):.1f}" for moment in tile_times]
    prompt = prompt.replace("{{METADATA}}", str(metadata))
    prompt = prompt.replace("{{SLATE}}", str(slate))
    prompt = prompt.replace("{{TRANSCRIPT}}", str(transcript))
    prompt = prompt.replace("{{VOCABULARY}}", "\n".join(vocabulary_lines))
    prompt = prompt.replace("{{TILES}}", ", ".join(tile_texts))
    return prompt


def metadata_line(row: object) -> str:
    """Build the one metadata line for a catalog clip row.

    ``camera=<name>; duration=<seconds with 3 decimals>s; day=<n>;
    place=<name>; captured=<UTC text>`` with every unknown part left
    out, never a file path or file name.
    """
    parts: list[str] = []
    camera = getattr(row, "camera", None)
    if isinstance(camera, str) and camera:
        parts.append(f"camera={camera}")
    duration = getattr(row, "duration", None)
    if isinstance(duration, Fraction):
        parts.append(f"duration={float(duration):.3f}s")
    day = getattr(row, "day", None)
    if isinstance(day, int) and not isinstance(day, bool):
        parts.append(f"day={day}")
    place = getattr(row, "place", None)
    if isinstance(place, str) and place:
        parts.append(f"place={place}")
    captured = getattr(row, "capture_time", None)
    if isinstance(captured, str) and captured:
        parts.append(f"captured={captured}")
    return "; ".join(parts)


def slate_text(catalog: object, media_id: int) -> str:
    """Return the clip's slate text from its slate and vision-text tags.

    The distinct values in tag id order, joined with ``"; "``; empty
    when the clip has none.
    """
    seen: list[str] = []
    for tag in catalog.list_tags(media_id):  # type: ignore[union-attr]
        if getattr(tag, "source", None) not in _SLATE_SOURCES:
            continue
        value = " ".join(_clean_markers(getattr(tag, "value", None) or "").split())
        if value and value not in seen:
            seen.append(value)
    return "; ".join(seen)


def transcript_text(catalog: object, media_id: int) -> tuple[str, str | None]:
    """Return ``(text, note)`` for the clip's first ``.txt`` transcript.

    The text is the first :meth:`list_transcripts` record whose path
    ends in ``".txt"``, read as UTF-8 and cut to
    :data:`TRANSCRIPT_WORDS` words. A missing record, a missing file or
    unreadable text gives an empty transcript and a note, never a
    failure.
    """
    for record in catalog.list_transcripts(media_id):  # type: ignore[union-attr]
        path = getattr(record, "path", None)
        if not isinstance(path, str) or not path.endswith(".txt"):
            continue
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except (OSError, ValueError):
            return (
                "",
                "no transcript text could be read; sent an empty transcript",
            )
        return (
            first_words(_clean_markers(raw), TRANSCRIPT_WORDS),
            None,
        )
    return ("", "no transcript text on file; sent an empty transcript")


def _manifest_tile_times(manifest: dict) -> list[Fraction]:
    """Return the stamp times from a parsed manifest as Fractions."""
    times: list[Fraction] = []
    tiles = manifest.get("tiles", [])
    if isinstance(tiles, list):
        for entry in tiles:
            if not isinstance(entry, dict):
                continue
            try:
                times.append(Fraction(str(entry["time"])))
            except (KeyError, ValueError, ZeroDivisionError, TypeError):
                continue
    return times


def _manifest_edges(manifest: dict, duration: Fraction) -> list[Fraction]:
    """Return the sorted distinct snap edges: 0, duration, tile times.

    Tile times outside ``[0, duration]`` are left out, so snapping a
    corrupt manifest can never store a range past the clip.
    """
    edges: set[Fraction] = {Fraction(0), duration}
    tiles = manifest.get("tiles", [])
    if isinstance(tiles, list):
        for entry in tiles:
            if not isinstance(entry, dict):
                continue
            try:
                moment = Fraction(str(entry["time"]))
            except (KeyError, ValueError, ZeroDivisionError, TypeError):
                continue
            if Fraction(0) <= moment <= duration:
                edges.add(moment)
    return sorted(edges)


def _answer_number(value: object, number: int, label: str) -> float | None:
    """Return an answer number as a float safe to scale to milliseconds.

    Booleans, non-numbers and non-finite floats return ``None``: the
    caller drops that range with a note. Integers and finite floats
    wider than :data:`_MAX_ANSWER_NUMBER` are never real clip times
    or confidences, so they raise :class:`ai.AiAnswerError` instead,
    rejecting the whole answer the way a schema mismatch does. The
    bound is checked before any float conversion or multiplication,
    so a huge integer can never raise OverflowError and a huge float
    can never overflow to infinity when scaled.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, int):
        if abs(value) > _MAX_ANSWER_NUMBER:
            raise ai.AiAnswerError(
                f"AI answer for step {STEP!r} failed checks: "
                f"range {number}: invalid answer ({label} out of range)"
            )
        return float(value)
    if not math.isfinite(value):
        return None
    if abs(value) > _MAX_ANSWER_NUMBER:
        raise ai.AiAnswerError(
            f"AI answer for step {STEP!r} failed checks: "
            f"range {number}: invalid answer ({label} out of range)"
        )
    return value


def check_answer(
    answer: dict,
    vocabulary: dict[str, tuple[str, ...]],
    duration: Fraction,
    edges: list[Fraction],
) -> tuple[list[tuple[str, str, float, Fraction, Fraction]], list[str]]:
    """Check a validated answer; return ``(kept, notes)``.

    ``kept`` holds ``(folder, value, confidence, start, duration)``
    with exact :class:`Fraction` times already snapped outwards to the
    tile edges and deduplicated by ``(folder, start, duration)``
    (higher confidence wins, first wins a tie). Every drop appends one
    note naming the range number or the tag and the rule. A number too
    huge to be a real time or confidence raises
    :class:`ai.AiAnswerError`, rejecting the whole answer the way a
    schema mismatch does. Nothing here is retried: only a
    schema-failing answer is retried, inside ask.
    """
    kept: list[tuple[str, str, float, Fraction, Fraction]] = []
    notes: list[str] = []
    ranges = answer.get("ranges", [])
    if not isinstance(ranges, list):
        notes.append("ranges: dropped (not a list of ranges)")
        return (kept, notes)
    for number, entry in enumerate(ranges, start=1):
        if not isinstance(entry, dict):
            notes.append(f"range {number}: dropped (not a range)")
            continue
        confidence = _answer_number(
            entry.get("confidence"), number, "confidence"
        )
        if confidence is None or not 0 <= confidence <= 1:
            notes.append(
                f"range {number}: dropped (confidence out of range)"
            )
            continue
        start_raw = _answer_number(entry.get("start"), number, "start")
        end_raw = _answer_number(entry.get("end"), number, "end")
        if start_raw is None or end_raw is None:
            notes.append(f"range {number}: dropped (outside the clip)")
            continue
        start = Fraction(round(start_raw * 1000), 1000)
        end = Fraction(round(end_raw * 1000), 1000)
        if start < 0 or end > duration or end <= start:
            notes.append(f"range {number}: dropped (outside the clip)")
            continue
        tags = entry.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        survivors: list[tuple[str, str]] = []
        for tag in tags:
            if not isinstance(tag, dict):
                notes.append(f"range {number}: dropped a tag (not a tag)")
                continue
            folder = tag.get("folder")
            value = tag.get("value")
            if not isinstance(folder, str) or not isinstance(value, str):
                notes.append(
                    f"range {number}: dropped a tag (not a folder/value pair)"
                )
                continue
            allowed = vocabulary.get(folder)
            if allowed is None or value not in allowed:
                notes.append(
                    f"range {number} tag {folder}/{value}: "
                    "dropped (not in the trip vocabulary)"
                )
                continue
            survivors.append((folder, value))
        if not survivors:
            notes.append(
                f"range {number}: dropped (no tags in the trip vocabulary)"
            )
            continue
        snapped_start, snapped_end = snap_range(start, end, edges)
        for folder, value in survivors:
            kept.append(
                (
                    folder,
                    value,
                    float(confidence),
                    snapped_start,
                    snapped_end - snapped_start,
                )
            )
    deduped: dict[tuple[str, Fraction, Fraction], tuple[str, str, float, Fraction, Fraction]] = {}
    origins: dict[tuple[str, Fraction, Fraction], int] = {}
    for at, item in enumerate(kept):
        key = (item[0], item[3], item[4])
        current = deduped.get(key)
        if current is None or item[2] > current[2]:
            deduped[key] = item
            origins[key] = at
    dropped = [item for at, item in enumerate(kept) if origins[(item[0], item[3], item[4])] != at]
    for item in dropped:
        notes.append(
            f"tag {item[0]}/{item[1]}: dropped (duplicate in the same range)"
        )
    ordered = sorted(deduped.values(), key=lambda item: (item[3], item[4], item[0]))
    return (ordered, notes)


def _load_sheet_bytes(
    settings: object, manifest: dict
) -> tuple[bytes | None, str | None]:
    """Return ``(data, failure)`` for the manifest's contact sheet.

    The sheet name must stay inside ``.pipeline/sheets/``; its bytes
    must hash to the manifest checksum and parse as a JPEG no larger
    than :data:`MAX_SHEET_SIZE`. A failure reason means nothing is sent.
    """
    name = manifest.get("sheet")
    if not isinstance(name, str) or not name:
        return (None, "contact sheet manifest has no sheet name")
    pipeline = Path(os.fspath(getattr(settings, "pipeline")))
    sheets_dir = pipeline / "sheets"
    sheet_path = sheets_dir / name
    try:
        if sheet_path.resolve().parent != sheets_dir.resolve():
            return (
                None,
                f"contact sheet {name!r} escapes the sheets folder",
            )
    except OSError as exc:
        return (None, f"contact sheet {name!r} could not be read: {exc}")
    try:
        data = sheet_path.read_bytes()
    except OSError as exc:
        return (None, f"contact sheet {name!r} could not be read: {exc}")
    expected = manifest.get("sheet_sha256")
    if (
        not isinstance(expected, str)
        or hashlib.sha256(data).hexdigest() != expected
    ):
        return (
            None,
            f"contact sheet {name!r} does not match its manifest; "
            "run the sheets step again",
        )
    try:
        width, height = jpeg_size(data)
    except TagError as exc:
        return (None, f"contact sheet {name!r} is not a readable JPEG: {exc}")
    limit_width, limit_height = MAX_SHEET_SIZE
    if width > limit_width or height > limit_height:
        return (
            None,
            f"contact sheet {name!r} is {width}x{height}, "
            f"over the {limit_width}x{limit_height} limit",
        )
    return (data, None)


def _store_records(
    catalog: object, media_id: int, records: list[object]
) -> tuple[int, list[str]]:
    """Store new records through put_tag, honouring locked slots.

    Before each write the existing tag in the same
    ``(folder, start, duration)`` slot is looked up: a locked or
    deleted tag, or one from any source other than ``"ai"``, is never
    replaced. Returns ``(stored, notes)``.
    """
    stored = 0
    notes: list[str] = []
    existing = {
        (tag.folder, tag.start, tag.duration): tag
        for tag in catalog.list_tags(media_id)  # type: ignore[union-attr]
    }
    for record in records:
        slot = (record.folder, record.start, record.duration)  # type: ignore[union-attr]
        current = existing.get(slot)
        if current is not None and (
            current.status in ("locked", "deleted")
            or current.source != "ai"
        ):
            notes.append(
                f"kept the existing {record.folder} tag "  # type: ignore[union-attr]
                f"(source {current.source}, status {current.status}); "
                "locked, deleted and higher-source tags are never changed"
            )
            continue
        if catalog.put_tag(record):  # type: ignore[union-attr]
            stored += 1
        else:
            notes.append(
                f"kept the existing {record.folder} tag; "  # type: ignore[union-attr]
                "the catalog refused the replacement"
            )
    return (stored, notes)


def _whole_clip_records(answer: dict, media_id: int) -> list[object]:
    """Build the whole-clip records (shot, note, to-do, outline hint)."""
    records: list[object] = []
    shot = answer.get("shot")
    if isinstance(shot, str) and shot:
        records.append(
            catalog_module.TagRecord(
                media_id=media_id,
                folder=SHOT_FOLDER,
                value=shot,
                start=None,
                duration=None,
                source="ai",
                confidence=None,
                status="plain",
            )
        )
    description = answer.get("description")
    if isinstance(description, str) and description.strip():
        records.append(
            catalog_module.TagRecord(
                media_id=media_id,
                folder=NOTE_FOLDER,
                value=description,
                start=None,
                duration=None,
                source="ai",
                confidence=None,
                status="plain",
            )
        )
    problems = answer.get("problems", [])
    if isinstance(problems, list):
        cleaned = [
            problem.strip()
            for problem in problems
            if isinstance(problem, str) and problem.strip()
        ]
        if cleaned:
            records.append(
                catalog_module.TagRecord(
                    media_id=media_id,
                    folder=TODO_FOLDER,
                    value="; ".join(cleaned),
                    start=None,
                    duration=None,
                    source="ai",
                    confidence=None,
                    status="plain",
                )
            )
    hint = answer.get("outline_hint")
    if isinstance(hint, str) and hint.strip():
        records.append(
            catalog_module.TagRecord(
                media_id=media_id,
                folder=HINT_FOLDER,
                value=hint,
                start=None,
                duration=None,
                source="ai",
                confidence=None,
                status="check",
            )
        )
    return records


def _tag_one_clip(
    catalog: object,
    settings: object,
    row: object,
    vocabulary: dict[str, tuple[str, ...]],
    cutoff: float,
    schema: dict,
    backends: object,
) -> TagEntry:
    """Tag one clip row; ask failures become ``"failed"``, never raised."""
    path = os.fspath(getattr(row, "path"))
    asset_id = getattr(row, "asset_id")
    duration = getattr(row, "duration")
    manifest = read_manifest(settings, row)
    if manifest is None:
        return TagEntry(
            path=path,
            asset_id=asset_id,
            status="no-sheet",
            reason=(
                "no contact sheet for this clip's current state; "
                "run the sheets step first"
            ),
        )
    data, failure = _load_sheet_bytes(settings, manifest)
    if data is None:
        assert failure is not None
        return TagEntry(
            path=path, asset_id=asset_id, status="failed", reason=failure
        )
    slate = slate_text(catalog, row.id)  # type: ignore[union-attr]
    transcript, transcript_note = transcript_text(catalog, row.id)  # type: ignore[union-attr]
    notes: list[str] = []
    if transcript_note is not None:
        notes.append(transcript_note)
    instructions = build_instructions(
        metadata_line(row),
        slate,
        transcript,
        vocabulary,
        _manifest_tile_times(manifest),
    )
    image = ai.Image(data=data, media_type="image/jpeg")
    try:
        answer = ai.ask(
            settings,
            STEP,
            instructions,
            schema,
            images=[image],
            backends=backends,
        )
    except (ai.AiConfigError, OfflineError):
        raise
    except Exception as exc:
        return TagEntry(
            path=path, asset_id=asset_id, status="failed", reason=str(exc)
        )
    edges = _manifest_edges(manifest, duration)
    try:
        kept, check_notes = check_answer(answer, vocabulary, duration, edges)
    except ai.AiAnswerError as exc:
        return TagEntry(
            path=path, asset_id=asset_id, status="failed", reason=str(exc)
        )
    notes.extend(check_notes)
    records: list[object] = []
    for folder, value, confidence, start, span in kept:
        records.append(
            catalog_module.TagRecord(
                media_id=row.id,  # type: ignore[union-attr]
                folder=folder,
                value=value,
                start=start,
                duration=span,
                source="ai",
                confidence=confidence,
                status="plain" if confidence >= cutoff else "check",
            )
        )
    records.extend(_whole_clip_records(answer, row.id))  # type: ignore[union-attr]
    stored, store_notes = _store_records(catalog, row.id, records)  # type: ignore[union-attr]
    notes.extend(store_notes)
    return TagEntry(
        path=path,
        asset_id=asset_id,
        status="tagged",
        stored=stored,
        notes=tuple(notes),
    )


def tag_catalog(
    catalog: object,
    settings: object,
    *,
    vocabulary: object = None,
    cutoff: object = DEFAULT_CUTOFF,
    retag: bool = False,
    dry_run: bool = False,
    backends: object = None,
) -> TagResult:
    """Tag every catalog clip that has a current contact sheet.

    Clips are visited in catalog id order. A row with no duration, or
    whose file changed since the scan, is ``"stale"``; a row with no
    manifest for its current state is ``"no-sheet"``; a clip that
    already holds an ``"ai"`` tag is ``"skipped"`` unless ``retag`` is
    true; with ``dry_run`` an eligible clip is ``"would-tag"`` and
    nothing is called, read for writing or recorded. Otherwise the
    sheet, slate, transcript and vocabulary go through
    :func:`fcp_pipeline.ai.ask` with step ``"tag"`` and the checked
    answer is stored. A bad answer retried once still failing, or any
    other per-clip failure, makes that clip ``"failed"`` and the run
    goes on; a missing model or an offline switch stops the run with
    ``"stopped: <ExceptionClass>"`` and propagates.
    """
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, (int, float))
        or not math.isfinite(cutoff)
        or not 0 <= cutoff <= 1
    ):
        raise TagError(
            f"bad cutoff {cutoff!r}: must be a number from 0 to 1"
        )
    if vocabulary is None:
        shown = build_vocabulary(catalog, settings)
    else:
        shown = _check_vocabulary_mapping(vocabulary)
    schema = answer_schema()
    rows = [
        row
        for row in catalog.list_media()  # type: ignore[union-attr]
        if getattr(row, "kind", None) == "clip"
    ]
    entries: list[TagEntry] = []
    if dry_run:
        for row in rows:
            path = os.fspath(getattr(row, "path"))
            asset_id = getattr(row, "asset_id")
            if getattr(row, "duration", None) is None or not catalog.unchanged(path):  # type: ignore[union-attr]
                entries.append(
                    TagEntry(
                        path=path,
                        asset_id=asset_id,
                        status="stale",
                        reason=(
                            "changed since the scan; run the scan again"
                        ),
                    )
                )
                continue
            if read_manifest(settings, row) is None:
                entries.append(
                    TagEntry(
                        path=path,
                        asset_id=asset_id,
                        status="no-sheet",
                        reason=(
                            "no contact sheet for this clip's current state; "
                            "run the sheets step first"
                        ),
                    )
                )
                continue
            if not retag and any(
                getattr(tag, "source", None) == "ai"
                for tag in catalog.list_tags(row.id)  # type: ignore[union-attr]
            ):
                entries.append(
                    TagEntry(
                        path=path,
                        asset_id=asset_id,
                        status="skipped",
                        reason=(
                            "already has ai tags; "
                            "use --retag to tag it again"
                        ),
                    )
                )
                continue
            entries.append(
                TagEntry(path=path, asset_id=asset_id, status="would-tag")
            )
        return TagResult(entries=tuple(entries))
    run_id = catalog.start_run("tag", dry_run=False)  # type: ignore[union-attr]
    try:
        for row in rows:
            path = os.fspath(getattr(row, "path"))
            asset_id = getattr(row, "asset_id")
            if getattr(row, "duration", None) is None or not catalog.unchanged(path):  # type: ignore[union-attr]
                entries.append(
                    TagEntry(
                        path=path,
                        asset_id=asset_id,
                        status="stale",
                        reason=(
                            "changed since the scan; run the scan again"
                        ),
                    )
                )
                continue
            if read_manifest(settings, row) is None:
                entries.append(
                    TagEntry(
                        path=path,
                        asset_id=asset_id,
                        status="no-sheet",
                        reason=(
                            "no contact sheet for this clip's current state; "
                            "run the sheets step first"
                        ),
                    )
                )
                continue
            if not retag and any(
                getattr(tag, "source", None) == "ai"
                for tag in catalog.list_tags(row.id)  # type: ignore[union-attr]
            ):
                entries.append(
                    TagEntry(
                        path=path,
                        asset_id=asset_id,
                        status="skipped",
                        reason=(
                            "already has ai tags; "
                            "use --retag to tag it again"
                        ),
                    )
                )
                continue
            entries.append(
                _tag_one_clip(
                    catalog,
                    settings,
                    row,
                    shown,
                    float(cutoff),
                    schema,
                    backends,
                )
            )
    except (ai.AiConfigError, OfflineError) as exc:
        outcome = f"stopped: {type(exc).__name__}"
        try:
            catalog.finish_run(run_id, outcome)  # type: ignore[union-attr]
        except (catalog_module.CatalogError, OSError):
            pass
        raise
    except Exception as exc:
        try:
            catalog.finish_run(run_id, f"failed: {exc}")  # type: ignore[union-attr]
        except (catalog_module.CatalogError, OSError):
            pass
        raise
    result = TagResult(entries=tuple(entries))
    catalog.finish_run(run_id, result.outcome)  # type: ignore[union-attr]
    return result


def _read_vocabulary_file(path: str) -> dict[str, list[str]]:
    """Read a ``--vocabulary`` JSON file, or raise TagError naming it."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TagError(
            f"bad vocabulary file {path}: could not be read ({exc})"
        ) from None
    if not isinstance(raw, dict):
        raise TagError(
            f"bad vocabulary file {path}: "
            "must be an object whose values are arrays of strings"
        )
    for folder, values in raw.items():
        if not isinstance(folder, str) or not isinstance(values, list):
            raise TagError(
                f"bad vocabulary file {path}: "
                "must be an object whose values are arrays of strings"
            )
        for value in values:
            if not isinstance(value, str):
                raise TagError(
                    f"bad vocabulary file {path}: "
                    "must be an object whose values are arrays of strings"
                )
    return raw


def main(argv: list[str] | None = None) -> int:
    """Tag one trip's clips from their contact sheets; exit code."""
    parser = argparse.ArgumentParser(
        prog="fcp_pipeline.tag",
        description=(
            "Tag each catalog clip from its contact sheet through "
            "the [pipeline.ai.tag] provider setting."
        ),
    )
    parser.add_argument(
        "--trip",
        required=True,
        help="Trip folder holding project.toml.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_CUTOFF,
        help="Confidence at or above this value stays plain (default 0.75).",
    )
    parser.add_argument(
        "--retag",
        action="store_true",
        help="Tag clips that already hold ai tags again.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be tagged without calling or writing.",
    )
    parser.add_argument(
        "--vocabulary",
        default=None,
        help="JSON file mapping extra folders to arrays of values.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Refuse every network call before anything is sent.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Submit every tag-eligible clip's contact sheet as "
            "provider batch jobs, then return."
        ),
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help=(
            "Fetch finished batch job results and store them as tags."
        ),
    )
    args = parser.parse_args(argv)
    if args.batch and args.collect:
        print(
            "error: --batch and --collect cannot be used together",
            file=sys.stderr,
        )
        return 2
    if args.collect and args.dry_run:
        print(
            "error: --dry-run cannot be used with --collect",
            file=sys.stderr,
        )
        return 2
    if args.offline:
        set_offline(True)
    try:
        settings = load_trip(args.trip)
    except TripSettingsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        extra = (
            _read_vocabulary_file(args.vocabulary)
            if args.vocabulary is not None
            else None
        )
    except TagError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        catalog = catalog_module.Catalog.open(settings)
    except (catalog_module.CatalogError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.batch or args.collect:
        from fcp_pipeline import tag_batch as tag_batch_module

        try:
            result = tag_batch_module.run(args, settings, catalog, extra)
        except OfflineError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
        except ai.AiConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except TagError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except tag_batch_module.TagBatchError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except (catalog_module.CatalogError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        finally:
            catalog.close()
        for entry in result.entries:
            if entry.status in ("failed", "stale", "no-sheet", "expired"):
                if entry.reason:
                    print(f"{entry.status}: {entry.path}: {entry.reason}")
            elif entry.status == "pending":
                if entry.reason:
                    print(f"pending: {entry.path}: {entry.reason}")
            elif entry.status == "running":
                print(f"running: {entry.path}: {entry.reason}")
            elif entry.status == "would-submit":
                print(f"would-submit: {entry.path}")
            for note in entry.notes:
                print(f"note: {entry.path}: {note}")
        for note in result.notes:
            print(f"note: {note}")
        for job in result.jobs:
            print(
                f"batch: {job.provider} {job.batch_id} "
                f"({job.count} clips)"
            )
        print(result.outcome)
        return 0
    try:
        try:
            vocabulary = (
                build_vocabulary(catalog, settings, extra=extra)
                if extra is not None
                else None
            )
            result = tag_catalog(
                catalog,
                settings,
                vocabulary=vocabulary,
                cutoff=args.cutoff,
                retag=args.retag,
                dry_run=args.dry_run,
            )
        except OfflineError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
        except ai.AiConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except TagError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except (catalog_module.CatalogError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    finally:
        catalog.close()
    for entry in result.entries:
        if entry.status in ("failed", "stale", "no-sheet") and entry.reason:
            print(f"{entry.status}: {entry.path}: {entry.reason}")
        for note in entry.notes:
            print(f"note: {entry.path}: {note}")
    print(result.outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
