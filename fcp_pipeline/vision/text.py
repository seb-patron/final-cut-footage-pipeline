"""Read phone-screen slates off contact sheets with Apple Vision.

For every ``clip`` row in the catalog that already has a contact sheet,
this step reads the sheet's drawn tiles (the picture the sheets step
saved, never the footage) with a small local Swift helper that runs
Apple's accurate text reader on each tile. Confident text that reads
as a slate becomes whole-clip catalog tags with source
``"vision-text"``: the slate text plus Day, Place, Shot, Stock and
Camera. A spoken slate, metadata, GPS and editor tags, and anything
locked or deleted, always win over these; an AI tag does not.

Everything runs on this Mac and offline: the helper uploads nothing,
this module imports no network code, and nothing ever contacts,
launches or scripts Final Cut Pro or Motion. Off macOS, or without
``swift``, the step says so; a Vision failure is an error, never
"no text". Importing this module runs no tool, opens no database and
touches no file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from fcp_pipeline import catalog
from fcp_pipeline.sheets import read_manifest
from fcp_pipeline.slates import (
    CAMERA_FOLDER,
    DAY_FOLDER,
    MAX_SLATE_CHARS,
    PLACE_FOLDER,
    SHOT_FOLDER,
    SLATE_FOLDER,
    STOCK_FOLDER,
    SlatePart,
    build_vocabulary,
    find_slate,
    normalize,
    parse_slate,
)
from fcp_pipeline.trip import TripSettingsError, load_trip

__all__ = [
    "CAMERA_FOLDER",
    "DAY_FOLDER",
    "ENGINE",
    "GRID",
    "HELPER",
    "HELPER_TIMEOUT",
    "MAX_SLATE_CHARS",
    "MIN_CONFIDENCE",
    "PLACE_FOLDER",
    "SHEET_SIZE",
    "SHOT_FOLDER",
    "SLATE_FOLDER",
    "SOURCE",
    "STAMP_MAX_X",
    "STAMP_MAX_Y",
    "STEP",
    "STOCK_FOLDER",
    "SlateReading",
    "TextLine",
    "TileRef",
    "TileSelectionError",
    "TileText",
    "VisionError",
    "VisionTextEntry",
    "VisionTextResult",
    "VisionUnavailable",
    "confident_lines",
    "main",
    "parse_helper_output",
    "read_sheet_text",
    "select_tiles",
    "slate_from_tile",
    "tag_vision_text",
    "vision_unavailable_reason",
]

#: Step name recorded in the catalog's runs table.
STEP = "vision-text"

#: Tag source this step writes (ranked just above "ai").
SOURCE = "vision-text"

#: Text reader the helper runs.
ENGINE = "Apple Vision VNRecognizeTextRequest"

#: Lowest helper confidence kept as read text.
MIN_CONFIDENCE = 0.5

#: Top-left corner of a tile, as fractions, where the sheets step
#: stamps each tile's time; a digits-and-"s" line ending in here is
#: the stamp, not slate text.
STAMP_MAX_X = 0.35
STAMP_MAX_Y = 0.20

#: Contact-sheet layout this step reads: a 3x3 grid of 512x288 tiles
#: in a 1536x864 sheet.
GRID = (3, 3)
TILE_SIZE = (512, 288)
SHEET_SIZE = (1536, 864)

#: The Swift helper next to this file.
HELPER = Path(__file__).with_name("read_text.swift")

#: How long one helper call may take before it is abandoned, in seconds.
HELPER_TIMEOUT = 300

#: A stamp line: digits with one optional decimal, suffixed with "s".
_STAMP_RE = re.compile(r"^\d+(\.\d)?s$")

#: A comma-free "Day N" with more text after it on the same line gets
#: a full stop split off, so "SLATE DAY 3 B-ROLL" reads as two pieces.
_DAY_SPLIT_RE = re.compile(r"(?i)\b(day\s+\d{1,3})\s+(?=\S)")

#: An exact "N/D" fraction, as the sheets manifest stores tile times.
_FRACTION_RE = re.compile(r"-?\d+/\d+")

#: Folders that count as recognized slate parts; a tile whose slate
#: parses to none of these stores nothing.
_RECOGNIZED_FOLDERS = (DAY_FOLDER, SHOT_FOLDER, STOCK_FOLDER, CAMERA_FOLDER)


class VisionError(RuntimeError):
    """Raised for any failure of Vision, the helper or its output."""


class VisionUnavailable(RuntimeError):
    """Raised when this Mac cannot run Vision here (not a Vision failure)."""


class TileSelectionError(ValueError):
    """Raised for a sheets manifest that cannot be used."""


@dataclass(frozen=True)
class TextLine:
    """One line the helper read: text, confidence and tile box.

    The box is ``(x, y, width, height)`` as fractions of the tile,
    with the origin at the top left.
    """

    text: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class TileText:
    """Every line the helper read on one tile, in helper order."""

    index: int
    lines: tuple[TextLine, ...]


@dataclass(frozen=True)
class TileRef:
    """One drawn tile to read: its grid cell and exact time."""

    index: int
    time: Fraction
    column: int
    row: int


@dataclass(frozen=True)
class SlateReading:
    """One tile's confident lines read as a slate."""

    tile: int
    text: str
    parts: tuple[SlatePart, ...]
    confidence: float


@dataclass(frozen=True)
class VisionTextEntry:
    """What the step did with one clip."""

    path: str
    asset_id: str
    status: str
    reason: str | None = None
    stored: int = 0
    text: str | None = None
    tile_time: Fraction | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class VisionTextResult:
    """Every entry of one run, in catalog order."""

    entries: tuple[VisionTextEntry, ...] = ()

    @property
    def tagged(self) -> int:
        """How many clips got vision-text tags."""
        return sum(1 for entry in self.entries if entry.status == "tagged")

    @property
    def no_slate(self) -> int:
        """How many clips hold no slate text on any tile."""
        return sum(1 for entry in self.entries if entry.status == "no-slate")

    @property
    def no_sheet(self) -> int:
        """How many clips hold no contact sheet to read."""
        return sum(1 for entry in self.entries if entry.status == "no-sheet")

    @property
    def stale(self) -> int:
        """How many rows no longer matched their file."""
        return sum(1 for entry in self.entries if entry.status == "stale")

    @property
    def failed(self) -> int:
        """How many clips could not be read."""
        return sum(1 for entry in self.entries if entry.status == "failed")

    @property
    def outcome(self) -> str:
        """One summary line for the run."""
        return (
            f"tagged {self.tagged}, no-slate {self.no_slate}, "
            f"no-sheet {self.no_sheet}, stale {self.stale}, "
            f"failed {self.failed}"
        )


def vision_unavailable_reason() -> str | None:
    """Return None when Vision can run, else one sentence saying why not."""
    if sys.platform != "darwin":
        return "Apple Vision needs macOS"
    if shutil.which("swift") is None:
        return "swift is not on PATH (install Xcode or the Command Line Tools)"
    if not HELPER.is_file():
        return "the Vision helper file is missing"
    return None


def _is_number(value: object) -> bool:
    """Whether a decoded JSON value is a real number (never a bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_helper_output(text: object, tile_count: int) -> list[TileText]:
    """Check decoded helper stdout and return one TileText per tile.

    Raises :class:`VisionError` when the output is not one JSON object,
    its engine is not :data:`ENGINE`, its tiles are not exactly indexes
    ``0`` to ``tile_count - 1`` in order, or a line's text, confidence
    or box is not shaped like one. Empty lines are "no text" only after
    all of that passed.
    """
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VisionError(
                f"bad Vision output: stdout is not UTF-8: {exc}"
            ) from None
    if not isinstance(text, str):
        raise VisionError(
            f"bad Vision output: expected text, got {type(text).__name__}"
        )
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise VisionError(f"bad Vision output: not one JSON object: {exc}") from None
    if not isinstance(document, dict):
        raise VisionError(
            "bad Vision output: expected one JSON object, "
            f"got {type(document).__name__}"
        )
    if document.get("engine") != ENGINE:
        raise VisionError(
            f"bad Vision output: engine {document.get('engine')!r} "
            f"is not {ENGINE!r}"
        )
    raw_tiles = document.get("tiles")
    if not isinstance(raw_tiles, list) or len(raw_tiles) != tile_count:
        found = len(raw_tiles) if isinstance(raw_tiles, list) else raw_tiles
        raise VisionError(
            f"bad Vision output: expected {tile_count} tiles, got {found!r}"
        )
    tiles: list[TileText] = []
    for position, entry in enumerate(raw_tiles):
        if (
            not isinstance(entry, dict)
            or type(entry.get("index")) is not int
            or entry.get("index") != position
        ):
            raise VisionError(
                "bad Vision output: tiles are not exactly indexes "
                f"0 to {tile_count - 1} in order "
                f"(tile {position} is {entry!r})"
            )
        raw_lines = entry.get("lines")
        if not isinstance(raw_lines, list):
            raise VisionError(
                f"bad Vision output: tile {position} has no lines list"
            )
        lines: list[TextLine] = []
        for line in raw_lines:
            if not isinstance(line, dict):
                raise VisionError(
                    f"bad Vision output: tile {position} holds "
                    f"a line that is not an object: {line!r}"
                )
            line_text = line.get("text")
            confidence = line.get("confidence")
            box = line.get("box")
            if not isinstance(line_text, str):
                raise VisionError(
                    f"bad Vision output: tile {position} holds "
                    f"a line whose text is not a string: {line_text!r}"
                )
            if not _is_number(confidence) or not 0 <= confidence <= 1:
                raise VisionError(
                    f"bad Vision output: tile {position} holds "
                    f"a line whose confidence is not 0 to 1: {confidence!r}"
                )
            if (
                not isinstance(box, list)
                or len(box) != 4
                or not all(_is_number(value) for value in box)
            ):
                raise VisionError(
                    f"bad Vision output: tile {position} holds "
                    f"a line whose box is not four numbers: {box!r}"
                )
            lines.append(
                TextLine(
                    text=line_text,
                    confidence=float(confidence),
                    box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                )
            )
        tiles.append(TileText(index=position, lines=tuple(lines)))
    return tiles


def _run_helper(
    sheet: str, tile_count: int, cache_path: str, runner: object
) -> list[TileText]:
    """Run the helper once and check its output, raising VisionError."""
    call = runner  # type: ignore[operator]
    command = [
        "swift",
        "-module-cache-path",
        cache_path,
        os.fspath(HELPER),
        os.fspath(Path(os.fspath(sheet)).resolve()),
        str(GRID[0]),
        str(GRID[1]),
        str(tile_count),
    ]
    try:
        completed = call(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=HELPER_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise VisionError(
            f"Vision helper timed out after {HELPER_TIMEOUT}s for {sheet}: {exc}"
        ) from None
    except OSError as exc:
        raise VisionError(f"Vision helper could not run for {sheet}: {exc}") from None
    if getattr(completed, "returncode", 1) != 0:
        error = getattr(completed, "stderr", b"")
        if isinstance(error, bytes):
            error_text = error.decode("utf-8", "replace")
        else:
            error_text = str(error)
        stripped = error_text.strip().splitlines()
        detail = stripped[-1].strip() if stripped else "no error output"
        raise VisionError(
            f"Vision helper failed for {sheet} "
            f"(exit {completed.returncode}): {detail}"
        )
    output = getattr(completed, "stdout", b"")
    return parse_helper_output(output, tile_count)


def read_sheet_text(
    sheet: object, tile_count: int, *, runner: object = None, cache_dir: object = None
) -> list[TileText]:
    """Read every tile of a contact-sheet image with the Vision helper.

    Returns one :class:`TileText` per tile in index order; a tile whose
    lines are all empty still returns its (empty) entry. Raises
    :class:`VisionUnavailable` before running anything when Vision
    cannot run here, and :class:`VisionError`, never empty tiles, for
    any helper or output failure. ``runner`` defaults to
    :func:`subprocess.run` and is called with a list of arguments (never
    a shell): ``swift -module-cache-path <cache> <helper> <sheet> 3 3
    <tiles>``.
    """
    reason = vision_unavailable_reason()
    if reason is not None:
        raise VisionUnavailable(reason)
    call = subprocess.run if runner is None else runner
    sheet_path = os.fspath(sheet)
    if cache_dir is not None:
        return _run_helper(sheet_path, tile_count, os.fspath(cache_dir), call)
    with tempfile.TemporaryDirectory() as fresh:
        return _run_helper(sheet_path, tile_count, fresh, call)


def select_tiles(manifest: object) -> list[TileRef]:
    """Return one TileRef per drawn tile of a parsed sheets manifest.

    Pure: runs no tool and opens no file. Raises
    :class:`TileSelectionError` for a manifest whose version is not 1,
    whose grid, tile size or sheet size differ from this step's layout,
    whose tiles list is empty or longer than nine, or whose entries'
    index values are not ``0, 1, 2 ...`` in order or whose time is not
    an ``N/D`` fraction. Unused (black) cells are never read.
    """
    if not isinstance(manifest, dict):
        raise TileSelectionError(
            f"bad sheets manifest {manifest!r}: must be an object"
        )
    if type(manifest.get("version")) is not int or manifest.get("version") != 1:
        raise TileSelectionError(
            f"bad sheets manifest version {manifest.get('version')!r}: "
            "expected 1"
        )
    if list(manifest.get("grid") or []) != [GRID[0], GRID[1]]:
        raise TileSelectionError(
            f"bad sheets manifest grid {manifest.get('grid')!r}: "
            f"expected [{GRID[0]}, {GRID[1]}]"
        )
    if list(manifest.get("tile_size") or []) != [TILE_SIZE[0], TILE_SIZE[1]]:
        raise TileSelectionError(
            f"bad sheets manifest tile_size {manifest.get('tile_size')!r}: "
            f"expected [{TILE_SIZE[0]}, {TILE_SIZE[1]}]"
        )
    if list(manifest.get("sheet_size") or []) != [SHEET_SIZE[0], SHEET_SIZE[1]]:
        raise TileSelectionError(
            f"bad sheets manifest sheet_size {manifest.get('sheet_size')!r}: "
            f"expected [{SHEET_SIZE[0]}, {SHEET_SIZE[1]}]"
        )
    raw_tiles = manifest.get("tiles")
    if not isinstance(raw_tiles, list) or not raw_tiles:
        raise TileSelectionError(
            "bad sheets manifest tiles: must be a non-empty list"
        )
    if len(raw_tiles) > GRID[0] * GRID[1]:
        raise TileSelectionError(
            f"bad sheets manifest tiles: {len(raw_tiles)} entries "
            f"is more than {GRID[0] * GRID[1]}"
        )
    refs: list[TileRef] = []
    for position, entry in enumerate(raw_tiles):
        if (
            not isinstance(entry, dict)
            or type(entry.get("index")) is not int
            or entry.get("index") != position
        ):
            raise TileSelectionError(
                "bad sheets manifest tiles: index values are not "
                f"0, 1, 2 ... in order (entry {position} is {entry!r})"
            )
        raw_time = entry.get("time")
        if (
            not isinstance(raw_time, str)
            or _FRACTION_RE.fullmatch(raw_time) is None
        ):
            raise TileSelectionError(
                f"bad sheets manifest time {raw_time!r} on tile {position}: "
                'must be an "N/D" fraction'
            )
        try:
            moment = Fraction(raw_time)
        except (ValueError, ZeroDivisionError) as exc:
            raise TileSelectionError(
                f"bad sheets manifest time {raw_time!r} on tile {position}: "
                f"must be an N/D fraction: {exc}"
            ) from None
        refs.append(
            TileRef(
                index=position,
                time=moment,
                column=position % GRID[0],
                row=position // GRID[0],
            )
        )
    return refs


def confident_lines(tile: TileText) -> list[TextLine]:
    """Keep a tile's confident, non-stamp lines in reading order.

    Drops blank lines, lines below :data:`MIN_CONFIDENCE`, and the
    sheets step's time stamp (digits-and-"s" text ending inside the
    tile's top-left corner); the same text elsewhere on the tile is
    kept. Orders the rest top to bottom, then left to right.
    """
    kept: list[TextLine] = []
    for line in tile.lines:
        stripped = line.text.strip()
        if not stripped:
            continue
        if line.confidence < MIN_CONFIDENCE:
            continue
        x, y, width, height = line.box
        if (
            _STAMP_RE.match(stripped) is not None
            and x + width <= STAMP_MAX_X
            and y + height <= STAMP_MAX_Y
        ):
            continue
        kept.append(line)
    kept.sort(key=lambda kept_line: (round(kept_line.box[1], 2), kept_line.box[0]))
    return kept


def slate_from_tile(tile: TileText, vocabulary: object) -> SlateReading | None:
    """Turn a tile's confident lines into one slate reading, or None.

    The lines are joined with ``". "`` after a comma-free "Day N" is
    split off from text after it on the same line, and the joined text
    is read with the existing slate parser. None means the tile holds
    no slate: nothing confident, no "Slate" word, or no recognized day,
    shot, stock or camera part. It never invents text.
    """
    lines = confident_lines(tile)
    if not lines:
        return None
    split = [
        _DAY_SPLIT_RE.sub(r"\1. ", line.text.strip()) for line in lines
    ]
    joined = ". ".join(split)
    if not normalize(joined):
        return None
    found = find_slate(
        [
            catalog.TranscriptSegmentRecord(
                idx=0, start=Fraction(0), stop=Fraction(0), text=joined
            )
        ],
        vocabulary,
    )
    if found is None:
        return None
    text = found[0][:MAX_SLATE_CHARS]
    parts = parse_slate(text, vocabulary)
    if not any(
        part.folder in _RECOGNIZED_FOLDERS and part.status == "plain"
        for part in parts
    ):
        return None
    confidence = min(line.confidence for line in lines)
    return SlateReading(
        tile=tile.index,
        text=text,
        parts=tuple(parts),
        confidence=float(confidence),
    )


def _sheet_error(manifest: object, sheets_dir: Path) -> tuple[str | None, str | None]:
    """Check a manifest's sheet file, returning (reason, path).

    ``(None, path)`` means the sheet is a bare ``.jpg`` name for a
    regular, non-symlink file inside ``sheets_dir`` whose bytes hash to
    the manifest's checksum; otherwise the reason names the file and no
    tool has run.
    """
    document = manifest if isinstance(manifest, dict) else {}
    name = document.get("sheet")
    if (
        not isinstance(name, str)
        or not name
        or not name.endswith(".jpg")
        or "/" in name
        or "\\" in name
        or name in (".", "..")
        or ".." in Path(name).parts
        or Path(name).name != name
    ):
        return (f"bad contact sheet {name!r}: must be a bare .jpg file name", None)
    candidate = sheets_dir / name
    if candidate.is_symlink():
        return (
            f"bad contact sheet {name}: {candidate} is a symlink",
            None,
        )
    if not candidate.is_file():
        return (
            f"bad contact sheet {name}: {candidate} is not a regular file",
            None,
        )
    try:
        resolved = candidate.resolve()
        wanted = sheets_dir.resolve()
    except OSError as exc:
        return (f"bad contact sheet {name}: cannot resolve {candidate}: {exc}", None)
    if resolved.parent != wanted:
        return (
            f"bad contact sheet {name}: {candidate} resolves outside "
            f"the sheets folder",
            None,
        )
    try:
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    except OSError as exc:
        return (f"bad contact sheet {name}: cannot read {candidate}: {exc}", None)
    if digest != document.get("sheet_sha256"):
        return (
            f"bad contact sheet {name}: {candidate} does not match "
            "its manifest; run the sheets step again",
            None,
        )
    return (None, os.fspath(candidate))


def _outranks_vision(source: str) -> bool:
    """Whether an existing tag source beats a vision-text tag."""
    return catalog.SOURCE_RANK.index(source) > catalog.SOURCE_RANK.index(SOURCE)


def _tag_clip(
    catalog_handle: object,
    settings: object,
    row: object,
    vocabulary: object,
    *,
    runner: object,
    cache_dir: str,
) -> VisionTextEntry:
    """Tag one clip's phone-screen slate, returning its entry.

    A :class:`VisionError` or :class:`VisionUnavailable` from the helper
    is not caught here: it stops the whole run at once.
    """
    path = os.fspath(getattr(row, "path"))
    asset_id = getattr(row, "asset_id")
    if getattr(row, "duration", None) is None or not catalog_handle.unchanged(path):  # type: ignore[union-attr]
        return VisionTextEntry(
            path=path,
            asset_id=asset_id,
            status="stale",
            reason="changed since the scan; run the scan again",
        )
    manifest = read_manifest(settings, row)
    if manifest is None:
        return VisionTextEntry(
            path=path,
            asset_id=asset_id,
            status="no-sheet",
            reason="no contact sheet for this clip's current state; "
            "run the sheets step first",
        )
    try:
        refs = select_tiles(manifest)
    except TileSelectionError as exc:
        return VisionTextEntry(
            path=path, asset_id=asset_id, status="failed", reason=str(exc)
        )
    sheets_dir = (
        Path(os.fspath(getattr(settings, "folder"))) / ".pipeline" / "sheets"
    )
    problem, sheet_path = _sheet_error(manifest, sheets_dir)
    if problem is not None or sheet_path is None:
        return VisionTextEntry(
            path=path,
            asset_id=asset_id,
            status="failed",
            reason=problem,
        )
    tiles = read_sheet_text(
        sheet_path, len(refs), runner=runner, cache_dir=cache_dir
    )
    by_index = {tile.index: tile for tile in tiles}
    winner: SlateReading | None = None
    winner_time: Fraction | None = None
    notes: list[str] = []
    for ref in refs:
        reading = slate_from_tile(by_index[ref.index], vocabulary)
        if reading is None:
            continue
        if winner is None:
            winner = reading
            winner_time = ref.time
        else:
            notes.append(f"second slate on tile {reading.tile} ignored")
    if winner is None or winner_time is None:
        return VisionTextEntry(
            path=path,
            asset_id=asset_id,
            status="no-slate",
            reason="no slate text on any tile",
        )
    planned: list[SlatePart] = [
        *winner.parts,
        SlatePart(SLATE_FOLDER, winner.text, "plain"),
    ]
    stored = 0
    slots = {
        (tag.folder, tag.start, tag.duration): tag
        for tag in catalog_handle.list_tags(row.id)  # type: ignore[union-attr]
    }
    for part in planned:
        old = slots.get((part.folder, None, None))
        if old is not None and (
            old.status in ("locked", "deleted") or _outranks_vision(old.source)
        ):
            notes.append(f"{part.folder} kept {old.source} tag")
            continue
        if catalog_handle.put_tag(  # type: ignore[union-attr]
            catalog.TagRecord(
                media_id=row.id,
                folder=part.folder,
                value=part.value,
                start=None,
                duration=None,
                source=SOURCE,
                confidence=winner.confidence,
                status=part.status,
            )
        ):
            stored += 1
    return VisionTextEntry(
        path=path,
        asset_id=asset_id,
        status="tagged",
        stored=stored,
        text=winner.text,
        tile_time=winner_time,
        notes=tuple(notes),
    )


def tag_vision_text(
    catalog_handle: object,
    settings: object,
    *,
    vocabulary: object = None,
    runner: object = None,
    cache_dir: object = None,
) -> VisionTextResult:
    """Read every clip's contact sheet and store slate text as tags.

    Visits media rows with kind ``"clip"`` in id order. Stale rows and
    rows with no sheet are reported and left alone; one helper call
    reads each other clip's sheet, and the first tile in index order
    that reads as a slate wins. Every tag is whole-clip with source
    ``"vision-text"`` and goes in through ``catalog.put_tag`` only; a
    tag held by a locked or deleted tag, or by a higher-ranked source,
    is kept and noted. A :class:`VisionError` or
    :class:`VisionUnavailable` stops the run at once: the run record is
    finished as ``"failed: <message>"``, clips already tagged stay
    tagged, and the error propagates.
    """
    reason = vision_unavailable_reason()
    if reason is not None:
        raise VisionUnavailable(reason)
    vocab = (
        build_vocabulary(catalog_handle, settings)
        if vocabulary is None
        else vocabulary
    )
    call = subprocess.run if runner is None else runner
    run_id = catalog_handle.start_run(STEP, dry_run=False)  # type: ignore[union-attr]
    entries: list[VisionTextEntry] = []
    scratch: tempfile.TemporaryDirectory | None = None
    try:
        if cache_dir is None:
            scratch = tempfile.TemporaryDirectory()
            shared = scratch.name
        else:
            shared = os.fspath(cache_dir)
        for row in catalog_handle.list_media():  # type: ignore[union-attr]
            if getattr(row, "kind", None) != "clip":
                continue
            entries.append(
                _tag_clip(
                    catalog_handle,
                    settings,
                    row,
                    vocab,
                    runner=call,
                    cache_dir=shared,
                )
            )
        result = VisionTextResult(entries=tuple(entries))
        catalog_handle.finish_run(run_id, result.outcome)  # type: ignore[union-attr]
        return result
    except Exception as exc:
        try:
            catalog_handle.finish_run(run_id, f"failed: {exc}")  # type: ignore[union-attr]
        except Exception:
            pass
        raise
    finally:
        if scratch is not None:
            scratch.cleanup()


def main(argv: list[str] | None = None) -> int:
    """Read phone-screen slates for one trip folder; return an exit code."""
    parser = argparse.ArgumentParser(
        prog="fcp_pipeline.vision.text",
        description=(
            "Read phone-screen slates off contact sheets with Apple Vision."
        ),
    )
    parser.add_argument(
        "--trip",
        required=True,
        help="Trip folder holding project.toml.",
    )
    args = parser.parse_args(argv)
    reason = vision_unavailable_reason()
    if reason is not None:
        print(f"error: {reason}", file=sys.stderr)
        return 3
    try:
        settings = load_trip(args.trip)
    except (TripSettingsError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        catalog_handle = catalog.Catalog.open(settings)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    result: VisionTextResult | None = None
    code = 0
    try:
        result = tag_vision_text(catalog_handle, settings)
    except VisionUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 3
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 1
    finally:
        catalog_handle.close()
    if result is None:
        return code
    for entry in result.entries:
        if entry.status in ("stale", "no-sheet", "failed") and entry.reason is not None:
            print(f"{entry.status}: {entry.path}: {entry.reason}")
    for entry in result.entries:
        for note in entry.notes:
            print(f"note: {entry.path}: {note}")
    print(result.outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
