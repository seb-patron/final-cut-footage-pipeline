"""One contact sheet per catalog clip.

For every ``clip`` row in the catalog this step picks up to nine tile
times (nine even slices, plus ffmpeg scene changes thinned evenly), pulls
those frames with ffmpeg, and composes a 3x3 grid of 1536x864 pixels with
each 512x288 tile stamped with its source time in seconds. The sheet is
saved as a JPEG under ``.pipeline/sheets/`` with a small manifest holding
the tile times, both through the safe write layer.

The same clip gives the same sheet: tile times are exact fractions, the
stamp is drawn by this module (no font, no drawtext filter), and every
ffmpeg run uses one thread with bitexact flags. Unchanged clips are
skipped without running ffmpeg. Footage is only read: each clip is
opened as an ffmpeg input, never written beside. Nothing leaves the
machine: no network module is imported and no other program is launched
or scripted. Importing this module runs no tool, opens no database and
touches no file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from fcp_pipeline.catalog import Catalog, CatalogError
from fcp_pipeline.safe_write import SafeWriter, SafeWriteError
from fcp_pipeline.trip import TripSettingsError, load_trip

__all__ = [
    "SheetEntry",
    "SheetError",
    "SheetsResult",
    "build_sheets",
    "main",
    "read_manifest",
    "stamp_png",
    "stamp_text",
    "tile_times",
]

#: The sheet is a 3x3 grid of 512x288 tiles.
GRID_COLUMNS = 3
GRID_ROWS = 3
TILE_WIDTH = 512
TILE_HEIGHT = 288
SHEET_WIDTH = GRID_COLUMNS * TILE_WIDTH
SHEET_HEIGHT = GRID_ROWS * TILE_HEIGHT

#: Most tiles one sheet holds; more candidates are thinned evenly.
MAX_TILES = 9

#: Clips shorter than this get a single tile at their middle.
SHORT_CLIP_SECONDS = Fraction(3, 1)

#: Scene candidates must sit this far inside the clip, and two kept
#: scene times must sit at least this far apart (first wins).
SCENE_MARGIN = Fraction(1, 2)

#: Scene detector: one ffmpeg run over the whole clip; a frame whose
#: score is above 0.4 opens a new shot.
SCENE_FILTER = "scale=320:-2,select='gt(scene,0.4)',showinfo"

#: Size of the stamp dots and gaps, in pixels (see stamp_png).
_STAMP_DOT = 4
_STAMP_GAP = 4
_STAMP_PAD = 6

#: How long one ffmpeg call may take before it is abandoned.
_SHEETS_TIMEOUT = 300

#: Matches one ``pts_time:<seconds>`` value in showinfo output.
_PTS_TIME_RE = re.compile(rb"pts_time\s*:\s*([0-9]+(?:\.[0-9]+)?)")

#: Characters the stamp may hold: digits, a decimal point and "s".
_STAMP_CHARS = frozenset("0123456789.s")

#: 5x7 dot-matrix glyphs for every stamp character, one "0"/"1" string
#: per row, written out as data (no font file, no drawtext filter).
_GLYPHS: dict[str, tuple[str, ...]] = {
    "0": (
        "01110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "01110",
    ),
    "1": (
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ),
    "2": (
        "01110",
        "10001",
        "00001",
        "00010",
        "00100",
        "01000",
        "11111",
    ),
    "3": (
        "11110",
        "00001",
        "00001",
        "01110",
        "00001",
        "00001",
        "11110",
    ),
    "4": (
        "00010",
        "00110",
        "01010",
        "10010",
        "11111",
        "00010",
        "00010",
    ),
    "5": (
        "11111",
        "10000",
        "11110",
        "00001",
        "00001",
        "00001",
        "11110",
    ),
    "6": (
        "01110",
        "10000",
        "10000",
        "11110",
        "10001",
        "10001",
        "01110",
    ),
    "7": (
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "01000",
        "01000",
    ),
    "8": (
        "01110",
        "10001",
        "10001",
        "01110",
        "10001",
        "10001",
        "01110",
    ),
    "9": (
        "01110",
        "10001",
        "10001",
        "01111",
        "00001",
        "00001",
        "01110",
    ),
    ".": (
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "01100",
        "01100",
    ),
    "s": (
        "00000",
        "00000",
        "01110",
        "10000",
        "01110",
        "00001",
        "11110",
    ),
}

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class SheetError(ValueError):
    """Raised for bad sheet values and for ffmpeg failures.

    The message names the path, value or tool call and the rule it
    broke, never footage contents.
    """


def _fraction_text(value: Fraction) -> str:
    """Render an exact fraction as ``N/D`` text, as the catalog stores it."""
    return f"{value.numerator}/{value.denominator}"


def tile_times(
    duration: Fraction, scene_times: object = ()
) -> list[Fraction]:
    """Return up to nine ascending, distinct tile times for a clip.

    ``duration`` is the clip length in seconds as a :class:`Fraction`;
    ``scene_times`` holds scene-change times as :class:`Fractions`. A
    clip under three seconds gets one tile at its middle; otherwise the
    even candidates are the centres of nine equal slices,
    ``duration x (2i + 1) / 18`` for ``i`` from 0 to 8. Scene times less
    than half a second inside the clip are kept, and a scene time
    within half a second of an earlier kept one is dropped (first
    wins). While more than nine candidates remain, the non-scene
    candidate closest to its nearest neighbour goes (ties drop the
    later one); when only scene candidates are left over nine, they are
    thinned to the nine at positions ``(k x (m - 1) + 4) // 8``.
    """
    if isinstance(duration, bool) or not isinstance(duration, (Fraction, int)):
        raise SheetError(
            f"bad duration {duration!r}: must be a Fraction, never a float"
        )
    length = duration if isinstance(duration, Fraction) else Fraction(duration, 1)
    if length <= 0:
        raise SheetError(
            f"bad duration {length!r}: must be above zero"
        )
    if length < SHORT_CLIP_SECONDS:
        return [length / 2]
    even = [length * (2 * i + 1) / 18 for i in range(MAX_TILES)]
    moments: list[Fraction] = []
    for raw in scene_times:  # type: ignore[union-attr]
        if isinstance(raw, bool) or not isinstance(raw, (Fraction, int)):
            raise SheetError(
                f"bad scene time {raw!r}: must be a Fraction, never a float"
            )
        moments.append(raw if isinstance(raw, Fraction) else Fraction(raw))
    kept: list[Fraction] = []
    for moment in sorted(moments):
        if moment < SCENE_MARGIN or moment > length - SCENE_MARGIN:
            continue
        if kept and moment - kept[-1] < SCENE_MARGIN:
            continue
        kept.append(moment)
    scene_set = set(kept)
    current = sorted(set(even) | scene_set)
    while len(current) > MAX_TILES:
        plain = [time for time in current if time not in scene_set]
        if not plain:
            break
        indexed = {time: at for at, time in enumerate(current)}

        def nearest(time: Fraction) -> Fraction:
            at = indexed[time]
            best: Fraction | None = None
            if at > 0:
                best = time - current[at - 1]
            if at < len(current) - 1:
                gap = current[at + 1] - time
                if best is None or gap < best:
                    best = gap
            assert best is not None
            return best

        victim = min(plain, key=lambda time: (nearest(time), -time))
        current.remove(victim)
    if len(current) > MAX_TILES:
        scenes = sorted(scene_set)
        count = len(scenes)
        current = [scenes[(k * (count - 1) + 4) // 8] for k in range(MAX_TILES)]
    return current


def stamp_text(time: Fraction) -> str:
    """Return the stamp for a tile time, like ``"31.0s"``.

    The seconds are rounded half up to one decimal and suffixed with
    ``"s"``; there is never a colon.
    """
    if isinstance(time, bool) or not isinstance(time, (Fraction, int)):
        raise SheetError(
            f"bad time {time!r}: must be a Fraction, never a float"
        )
    tenths = math.floor(Fraction(time) * 10 + Fraction(1, 2))
    sign = "-" if tenths < 0 else ""
    whole = abs(tenths)
    return f"{sign}{whole // 10}.{whole % 10}s"


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    """Wrap ``data`` in one PNG chunk with its length and checksum."""
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def stamp_png(text: str) -> bytes:
    """Draw stamp ``text`` into an RGBA 8-bit PNG, returned as bytes.

    The text may only hold the digits ``0``-``9``, ``"."`` and ``"s"``;
    anything else raises :class:`SheetError`. Every glyph dot becomes a
    4x4 pixel square, characters sit 4 px apart with 6 px of padding on
    every side, the background is black at 60 percent opacity and the
    dots are opaque white. The bytes depend only on the text: fixed PNG
    settings, no timestamp chunks.
    """
    if not isinstance(text, str) or not text:
        raise SheetError(f"bad stamp {text!r}: must be a non-empty string")
    for character in text:
        if character not in _STAMP_CHARS:
            raise SheetError(
                f"bad stamp {text!r}: {character!r} is not a digit, '.' or 's'"
            )
    count = len(text)
    width = _STAMP_PAD * 2 + count * 5 * _STAMP_DOT + (count - 1) * _STAMP_GAP
    height = _STAMP_PAD * 2 + 7 * _STAMP_DOT
    rows = [
        bytearray((0, 0, 0, 153) * width) for _ in range(height)
    ]
    for at, character in enumerate(text):
        left = _STAMP_PAD + at * (5 * _STAMP_DOT + _STAMP_GAP)
        for row, line in enumerate(_GLYPHS[character]):
            for column, dot in enumerate(line):
                if dot != "1":
                    continue
                for dy in range(_STAMP_DOT):
                    pixels = rows[_STAMP_PAD + row * _STAMP_DOT + dy]
                    for dx in range(_STAMP_DOT):
                        base = (left + column * _STAMP_DOT + dx) * 4
                        pixels[base : base + 4] = (255, 255, 255, 255)
    raw = b"".join(b"\x00" + bytes(pixels) for pixels in rows)
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


@dataclass(frozen=True)
class SheetEntry:
    """What the sheets step did with one catalog clip."""

    path: str
    asset_id: str
    status: str
    reason: str | None = None
    sheet: str | None = None
    tile_times: tuple[Fraction, ...] = ()


@dataclass(frozen=True)
class SheetsResult:
    """Every entry of one sheets run, in catalog id order."""

    entries: tuple[SheetEntry, ...] = ()

    @property
    def outcome(self) -> str:
        """The run-record line, like ``"built 3, skipped 1, stale 0, failed 0"``."""
        return (
            f"built {self.built}, skipped {self.skipped}, "
            f"stale {self.stale}, failed {self.failed}"
        )

    @property
    def built(self) -> int:
        """How many sheets were written in this run."""
        return sum(1 for entry in self.entries if entry.status == "built")

    @property
    def skipped(self) -> int:
        """How many clips already had a sheet and were left alone."""
        return sum(1 for entry in self.entries if entry.status == "skipped")

    @property
    def stale(self) -> int:
        """How many rows no longer matched their file."""
        return sum(1 for entry in self.entries if entry.status == "stale")

    @property
    def failed(self) -> int:
        """How many clips ffmpeg could not sheet."""
        return sum(1 for entry in self.entries if entry.status == "failed")


def _clip_key(abs_path: str, size: int, mtime_ns: int) -> str:
    """Return the 8-hex-character state code for a catalog clip row."""
    text = f"{abs_path}\0{size}\0{mtime_ns}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _sheet_stem(path: str) -> str:
    """Return the file stem with unsafe characters replaced by ``_``."""
    stem = Path(path).stem
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in stem
    )


def _sheet_names(row: object) -> tuple[str, str, str]:
    """Return ``(key, sheet name, manifest name)`` for a catalog row."""
    abs_path = os.path.abspath(os.fspath(getattr(row, "path")))
    key = _clip_key(abs_path, getattr(row, "size"), getattr(row, "mtime_ns"))
    stem = _sheet_stem(os.fspath(getattr(row, "path")))
    return key, f"{stem}-{key}.jpg", f"{stem}-{key}.json"


def _sheets_folder(settings: object) -> Path:
    """Return the ``.pipeline/sheets/`` folder for these settings."""
    return Path(os.fspath(getattr(settings, "folder"))) / ".pipeline" / "sheets"


def _stderr_text(completed: object) -> str:
    """Return a completed run's standard error as text."""
    error = getattr(completed, "stderr", b"")
    if isinstance(error, bytes):
        return error.decode("utf-8", "replace")
    return str(error)


def _detect_scenes(
    clip: str, *, runner: object, timeout: int
) -> tuple[list[Fraction], int, str]:
    """Run the scene detector over ``clip``; return ``(times, code, log)``."""
    call = runner  # type: ignore[operator]
    completed = call(
        [
            "ffmpeg",
            "-nostdin",
            "-an",
            "-i",
            clip,
            "-vf",
            SCENE_FILTER,
            "-f",
            "null",
            "-",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    log = _stderr_text(completed)
    if getattr(completed, "returncode", 1) != 0:
        return [], int(getattr(completed, "returncode", 1)), log
    times: list[Fraction] = []
    for match in _PTS_TIME_RE.finditer(log.encode("utf-8")):
        value = float(match.group(1).decode("ascii"))
        times.append(Fraction(round(value * 1000), 1000))
    return sorted(set(times)), 0, log


def _extract_tile(
    clip: str,
    at: Fraction,
    stamp: str,
    tile_path: str,
    *,
    runner: object,
    timeout: int,
) -> object:
    """Pull one stamped 512x288 tile image with a single ffmpeg run."""
    call = runner  # type: ignore[operator]
    return call(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            # Six decimals: exact to the microsecond, far inside any
            # frame, and the same text on every run, so the sheet the
            # seek produces is byte-identical for the same clip.
            "-ss",
            f"{float(at):.6f}",
            "-i",
            clip,
            "-i",
            stamp,
            "-filter_complex",
            "[0:v]scale=512:288:force_original_aspect_ratio=decrease,"
            "pad=512:288:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[bg];"
            "[bg][1:v]overlay=8:8",
            "-frames:v",
            "1",
            tile_path,
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _compose_grid(
    tile_paths: list[str], grid_path: str, *, runner: object, timeout: int
) -> object:
    """Compose the tile images into one 1536x864 JPEG with one ffmpeg run."""
    call = runner  # type: ignore[operator]
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    for tile in tile_paths:
        command += ["-i", tile]
    labels = "".join(f"[{at}:v]" for at in range(len(tile_paths)))
    command += [
        "-filter_complex",
        f"{labels}concat=n={len(tile_paths)}:v=1:a=0,"
        "tile=3x3:padding=0:color=black",
        "-frames:v",
        "1",
        "-q:v",
        "3",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-threads",
        "1",
        grid_path,
    ]
    return call(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _manifest_document(
    asset_id: str,
    sheet_name: str,
    digest: str,
    duration: Fraction,
    scenes: list[Fraction],
    times: list[Fraction],
) -> dict:
    """Build the manifest mapping for one sheet, keys in file order."""
    return {
        "version": 1,
        "asset_id": asset_id,
        "sheet": sheet_name,
        "sheet_sha256": digest,
        "sheet_size": [SHEET_WIDTH, SHEET_HEIGHT],
        "grid": [GRID_COLUMNS, GRID_ROWS],
        "tile_size": [TILE_WIDTH, TILE_HEIGHT],
        "duration": _fraction_text(duration),
        "scene_changes": [_fraction_text(moment) for moment in scenes],
        "tiles": [
            {"index": at, "time": _fraction_text(moment)}
            for at, moment in enumerate(times)
        ],
    }


def _manifest_path(settings: object, manifest_name: str) -> Path:
    """Return the absolute manifest path for a manifest file name."""
    return _sheets_folder(settings) / manifest_name


def read_manifest(settings: object, row: object) -> dict | None:
    """Return the parsed manifest for a catalog row's current key, or None.

    ``None`` means no manifest exists for the row's current state code;
    the row's size or modification time may be unknown (a stale row) or
    the file may simply not have been sheeted yet.
    """
    try:
        size = getattr(row, "size", None)
        mtime_ns = getattr(row, "mtime_ns", None)
        if size is None or mtime_ns is None:
            return None
        _, _, manifest_name = _sheet_names(row)
    except (TypeError, ValueError, OSError):
        return None
    manifest = _manifest_path(settings, manifest_name)
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _manifest_tile_times(document: dict) -> tuple[Fraction, ...]:
    """Read exact tile times back out of a parsed manifest."""
    tiles = document.get("tiles", [])
    return tuple(Fraction(entry["time"]) for entry in tiles)


def build_sheets(
    catalog: Catalog, settings: object, *, runner: object = None
) -> SheetsResult:
    """Build one contact sheet per catalog clip.

    Visits media rows with kind ``"clip"`` in id order. A row with no
    duration, or whose file changed since the scan, is reported
    ``"stale"`` with no tool run. A clip whose manifest already exists
    is ``"skipped"`` with no tool run. Otherwise the sheet is rebuilt
    in a temporary folder: the JPEG is written first, the manifest
    second, both through :class:`SafeWriter`. A JPEG left behind
    without its manifest (an earlier crash) is compared byte for byte:
    identical bytes mean only the manifest is written, different bytes
    make the clip ``"failed"`` and nothing is overwritten. An ffmpeg
    failure on one clip makes that clip ``"failed"`` and the run goes
    on. ``runner`` defaults to :func:`subprocess.run` and is called as
    ``runner(args, stdin=DEVNULL, capture_output=True, timeout, check)``
    for every ffmpeg call.
    """
    call = subprocess.run if runner is None else runner
    if runner is None and shutil.which("ffmpeg") is None:
        raise SheetError(
            "missing required tool: ffmpeg is not on PATH "
            "(install ffmpeg to build contact sheets)"
        )
    writer = SafeWriter(settings)  # type: ignore[arg-type]
    run_id = catalog.start_run("sheets", dry_run=False)
    entries: list[SheetEntry] = []
    try:
        for row in catalog.list_media():
            if getattr(row, "kind", None) != "clip":
                continue
            path = os.fspath(getattr(row, "path"))
            asset_id = getattr(row, "asset_id")
            duration = getattr(row, "duration")
            if duration is None or not catalog.unchanged(path):
                entries.append(
                    SheetEntry(
                        path=path,
                        asset_id=asset_id,
                        status="stale",
                        reason="changed since the scan; run the scan again",
                    )
                )
                continue
            _, sheet_name, manifest_name = _sheet_names(row)
            sheet_path = _sheets_folder(settings) / sheet_name
            manifest_path = _manifest_path(settings, manifest_name)
            if manifest_path.exists():
                try:
                    document = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    tile_times_found = _manifest_tile_times(document)
                except (OSError, ValueError, KeyError, AttributeError):
                    tile_times_found = ()
                entries.append(
                    SheetEntry(
                        path=path,
                        asset_id=asset_id,
                        status="skipped",
                        sheet=sheet_name,
                        tile_times=tile_times_found,
                    )
                )
                continue
            scenes, code, log = _detect_scenes(
                path, runner=call, timeout=_SHEETS_TIMEOUT
            )
            if code != 0:
                tail = log.strip().splitlines()
                detail = tail[-1].strip() if tail else "no error output"
                entries.append(
                    SheetEntry(
                        path=path,
                        asset_id=asset_id,
                        status="failed",
                        reason=(
                            f"ffmpeg scene detection failed for {path} "
                            f"(exit {code}): {detail}"
                        ),
                        tile_times=(),
                    )
                )
                continue
            times = tile_times(duration, scenes)
            with tempfile.TemporaryDirectory() as scratch:
                scratch_dir = Path(scratch)
                tile_paths: list[str] = []
                failed: SheetEntry | None = None
                for at, moment in enumerate(times):
                    stamp_path = os.fspath(
                        scratch_dir / f"stamp-{at}.png"
                    )
                    with open(stamp_path, "wb") as handle:
                        handle.write(stamp_png(stamp_text(moment)))
                    tile_path = os.fspath(scratch_dir / f"tile-{at}.png")
                    completed = _extract_tile(
                        path,
                        moment,
                        stamp_path,
                        tile_path,
                        runner=call,
                        timeout=_SHEETS_TIMEOUT,
                    )
                    if getattr(completed, "returncode", 1) != 0:
                        detail_lines = _stderr_text(completed).strip().splitlines()
                        detail = (
                            detail_lines[-1].strip()
                            if detail_lines
                            else "no error output"
                        )
                        failed = SheetEntry(
                            path=path,
                            asset_id=asset_id,
                            status="failed",
                            reason=(
                                f"ffmpeg tile extraction failed for {path} "
                                f"at {_fraction_text(moment)}s "
                                f"(exit {completed.returncode}): {detail}"
                            ),
                            tile_times=tuple(times),
                        )
                        break
                    tile_paths.append(tile_path)
                if failed is None:
                    grid_path = os.fspath(scratch_dir / "grid.jpg")
                    completed = _compose_grid(
                        tile_paths, grid_path, runner=call, timeout=_SHEETS_TIMEOUT
                    )
                    if getattr(completed, "returncode", 1) != 0:
                        detail_lines = _stderr_text(completed).strip().splitlines()
                        detail = (
                            detail_lines[-1].strip()
                            if detail_lines
                            else "no error output"
                        )
                        failed = SheetEntry(
                            path=path,
                            asset_id=asset_id,
                            status="failed",
                            reason=(
                                f"ffmpeg grid composition failed for {path} "
                                f"(exit {completed.returncode}): {detail}"
                            ),
                            tile_times=tuple(times),
                        )
                if failed is not None:
                    entries.append(failed)
                    continue
                with open(grid_path, "rb") as handle:
                    sheet_bytes = handle.read()
                digest = hashlib.sha256(sheet_bytes).hexdigest()
                document = _manifest_document(
                    asset_id, sheet_name, digest, duration, scenes, times
                )
                manifest_bytes = (
                    json.dumps(document, indent=2) + "\n"
                ).encode("utf-8")
                if sheet_path.exists():
                    if sheet_path.read_bytes() == sheet_bytes:
                        writer.write_new(manifest_path, manifest_bytes)
                        entries.append(
                            SheetEntry(
                                path=path,
                                asset_id=asset_id,
                                status="built",
                                sheet=sheet_name,
                                tile_times=tuple(times),
                            )
                        )
                    else:
                        entries.append(
                            SheetEntry(
                                path=path,
                                asset_id=asset_id,
                                status="failed",
                                reason=(
                                    f"existing sheet {sheet_name} does not "
                                    "match the rebuilt sheet; left untouched"
                                ),
                                tile_times=tuple(times),
                            )
                        )
                    continue
                writer.write_new(sheet_path, sheet_bytes)
                writer.write_new(manifest_path, manifest_bytes)
                entries.append(
                    SheetEntry(
                        path=path,
                        asset_id=asset_id,
                        status="built",
                        sheet=sheet_name,
                        tile_times=tuple(times),
                    )
                )
    except Exception as exc:
        result = SheetsResult(entries=tuple(entries))
        try:
            catalog.finish_run(run_id, f"failed: {exc}")
        except (CatalogError, SafeWriteError, OSError):
            pass
        raise
    result = SheetsResult(entries=tuple(entries))
    catalog.finish_run(run_id, result.outcome)
    return result


def main(argv: list[str] | None = None) -> int:
    """Build contact sheets for one trip folder; return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="fcp_pipeline.sheets",
        description=(
            "Build one 3x3 contact sheet per catalog clip "
            "under .pipeline/sheets/."
        ),
    )
    parser.add_argument(
        "--trip",
        required=True,
        help="Trip folder holding project.toml.",
    )
    args = parser.parse_args(argv)
    try:
        settings = load_trip(args.trip)
    except TripSettingsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if shutil.which("ffmpeg") is None:
        print(
            "error: missing required tool: ffmpeg is not on PATH "
            "(install ffmpeg to build contact sheets)",
            file=sys.stderr,
        )
        return 3
    try:
        catalog = Catalog.open(settings)
    except (CatalogError, SafeWriteError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        result = build_sheets(catalog, settings)
    except (CatalogError, SafeWriteError, OSError, SheetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        catalog.close()
    for entry in result.entries:
        if entry.status in ("failed", "stale") and entry.reason is not None:
            print(f"{entry.status}: {entry.path}: {entry.reason}")
    print(result.outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
