"""Tests for fcp_pipeline.vision.text, the Apple Vision slate reader.

Each tagging test runs in a temporary project folder with a catalog;
clip rows are added with catalog.put_media (kind "clip", a real file in
a temporary footage folder so catalog.unchanged is True, a duration,
camera "F3" on rows whose vocabulary needs it). A shared helper
installs a clip's sheet the way test_sheets.py does (names and manifest
from fcp_pipeline.sheets' _sheet_names, _manifest_document and
_manifest_bytes, files written straight into the temporary
.pipeline/sheets/ folder); the recorded-output tests install
placeholder sheet bytes (the recorded runner never reads them) and the
Vision tests install the drawn JPEG. The recorded runner is a function
that returns subprocess.CompletedProcess with the fixture case as JSON
bytes and counts its calls. Tests that run the real helper are skipped
through a shared decorator that raises unittest.SkipTest with
vision_unavailable_reason(); they share one class-level cache_dir (a
TemporaryDirectory made in setUpClass and removed in tearDownClass) so
only the first run compiles.

Every document, image, name and place here is synthetic and invented.
"""

import contextlib
import functools
import hashlib
import io
import json
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import unittest
import zlib
from fractions import Fraction
from pathlib import Path
from unittest import mock

from fcp_pipeline.catalog import Catalog, MediaRecord, TagRecord
from fcp_pipeline.slates import build_vocabulary, normalize
from fcp_pipeline.trip import load_trip
from fcp_pipeline.vision import text as vision_text
from fcp_pipeline.vision.text import (
    MIN_CONFIDENCE,
    SlateReading,
    TileRef,
    TileSelectionError,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "vision" / "ocr-tiles.json"
)

CASES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]

PROJECT_TOML = """title = "Invented Vision Survey"

[pipeline]
timezone = "UTC"
start_date = 2020-06-15

[[pipeline.places]]
name = "Moonlit Cove"
lat = 12.5
lon = 45.25
radius_m = 500

[[pipeline.places]]
name = "Paper Lantern Bay"
lat = 12.6
lon = 45.3
radius_m = 500
"""

#: The nine even tile times of a 60 s clip, as exact Fractions.
NINE_TILES = [
    Fraction(10, 3),
    Fraction(10, 1),
    Fraction(50, 3),
    Fraction(70, 3),
    Fraction(30, 1),
    Fraction(110, 3),
    Fraction(130, 3),
    Fraction(50, 1),
    Fraction(170, 3),
]

#: 5x7 dot-matrix glyphs for the drawn sheet, one "0"/"1" row each.
GLYPHS = {
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "3": ("01110", "10001", "00001", "00110", "00001", "10001", "01110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}


def make_project(tmp, name="project"):
    """Build a trip folder whose footage folder is a temp folder."""
    tmp = Path(tmp)
    footage = tmp / "footage"
    footage.mkdir(parents=True, exist_ok=True)
    project = tmp / name
    project.mkdir(parents=True, exist_ok=True)
    (project / "project.toml").write_text(PROJECT_TOML, encoding="utf-8")
    return load_trip(project), footage, project


def add_clip(catalog, footage, name, asset_id, camera="F3"):
    """Catalog a small placeholder file as a 60 s clip; return id and path."""
    destination = Path(footage) / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"invented-footage-bytes:" + name.encode("utf-8"))
    info = os.stat(destination)
    media_id = catalog.put_media(
        MediaRecord(
            asset_id=asset_id,
            kind="clip",
            path=os.fspath(destination),
            size=info.st_size,
            mtime_ns=info.st_mtime_ns,
            inode=info.st_ino,
            camera=camera,
            duration=Fraction(60),
        )
    )
    return media_id, os.fspath(destination)


def install_sheet(settings, catalog, media_id, sheet_bytes, times=NINE_TILES):
    """Write a sheet plus manifest for a clip; return the sheet path."""
    from fcp_pipeline import sheets as sheets_module

    row = [item for item in catalog.list_media() if item.id == media_id][0]
    _, sheet_name, manifest_name = sheets_module._sheet_names(row)
    digest = hashlib.sha256(sheet_bytes).hexdigest()
    document = sheets_module._manifest_document(
        row.asset_id, sheet_name, digest, row.duration, 0, list(times)
    )
    payload = sheets_module._manifest_bytes(document)
    folder = Path(os.fspath(settings.folder)) / ".pipeline" / "sheets"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / sheet_name).write_bytes(sheet_bytes)
    (folder / manifest_name).write_bytes(payload)
    return folder / sheet_name


def recorded_runner(case_name, calls):
    """A runner returning one fixture case as JSON bytes, counting calls."""

    def run(args, *, stdin=None, capture_output=False, timeout=None, check=False):
        calls.append(list(args))
        payload = json.dumps(CASES[case_name]).encode("utf-8")
        return subprocess.CompletedProcess(args, 0, stdout=payload, stderr=b"")

    return run


def requires_vision(func):
    """Skip a real-Vision test with the availability reason when missing."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        reason = vision_text.vision_unavailable_reason()
        if reason is not None:
            raise unittest.SkipTest(reason)
        if shutil.which("sips") is None:
            raise unittest.SkipTest(
                "sips is required to convert the drawn sheet "
                "but was not found on PATH"
            )
        return func(*args, **kwargs)

    return wrapper


def draw_sheet(text=None, tile=4):
    """Draw a 1536x864 8-bit RGB PNG sheet with white 5x7 glyph text.

    No font and no ffmpeg drawtext: each glyph dot is a 2x2 pixel
    square, characters advance 6 dot columns (5 plus 1 gap), starting
    at x = 20 and y = 100 pixels inside the requested tile's cell.
    """
    width, height = 1536, 864
    pixels = bytearray(width * height * 3)
    if text is not None:
        column, row = tile % 3, tile // 3
        base_x = 512 * column + 20
        base_y = 288 * row + 100
        for number, character in enumerate(text):
            for line_row, line in enumerate(GLYPHS[character]):
                for column_dot, dot in enumerate(line):
                    if dot != "1":
                        continue
                    for dy in range(2):
                        y = base_y + line_row * 2 + dy
                        for dx in range(2):
                            x = base_x + number * 12 + column_dot * 2 + dx
                            base = (y * width + x) * 3
                            pixels[base : base + 3] = b"\xff\xff\xff"
    raw = b"".join(
        b"\x00" + bytes(pixels[y * width * 3 : (y + 1) * width * 3])
        for y in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def png_to_jpeg(png_path, jpg_path):
    """Convert a drawn PNG sheet to JPEG with sips, or skip the test."""
    completed = subprocess.run(
        [
            "sips",
            "-s",
            "format",
            "jpeg",
            os.fspath(png_path),
            "--out",
            os.fspath(jpg_path),
        ],
        capture_output=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise unittest.SkipTest(
            "sips could not convert the drawn sheet to JPEG "
            "(%s)" % completed.stderr.decode("utf-8", "replace").strip()
        )


def tile_texts(tiles):
    """Map tile index to its TileText for a read_sheet_text result."""
    return {tile.index: tile for tile in tiles}


def manifest_document(times, **overrides):
    """Build a sheets manifest mapping holding the given tile times."""
    document = {
        "version": 1,
        "asset_id": "vision-x",
        "sheet": "clip-12345678.jpg",
        "sheet_sha256": "0" * 64,
        "sheet_size": [1536, 864],
        "grid": [3, 3],
        "tile_size": [512, 288],
        "duration": "60/1",
        "scene_count": 0,
        "truncated": False,
        "tiles": [
            {"index": at, "time": "%s/%s" % (moment.numerator, moment.denominator)}
            for at, moment in enumerate(times)
        ],
    }
    document.update(overrides)
    return document


class TestVision(unittest.TestCase):
    """Real-helper reads and failure wiring (skipped without Vision)."""

    @classmethod
    def setUpClass(cls):
        cls._swift_cache = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls._swift_cache.cleanup()

    def assert_rendered_sheet(self, tiles):
        """Check a "SLATE DAY 3 B-ROLL on tile 4" read end to end."""
        self.assertEqual([tile.index for tile in tiles], list(range(9)))
        by_index = tile_texts(tiles)
        fourth = by_index[4]
        self.assertEqual(len(fourth.lines), 1)
        (line,) = fourth.lines
        self.assertEqual(normalize(line.text), "slate day 3 b roll")
        self.assertGreaterEqual(line.confidence, MIN_CONFIDENCE)
        for value in line.box:
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
        for index, tile in by_index.items():
            if index != 4:
                self.assertEqual(tile.lines, ())

    @requires_vision
    def test_reads_rendered_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            png_path = tmp / "sheet.png"
            png_path.write_bytes(draw_sheet("SLATE DAY 3 B-ROLL", 4))
            jpg_path = tmp / "sheet.jpg"
            png_to_jpeg(png_path, jpg_path)
            settings, footage, _project = make_project(tmp)
            catalog = Catalog.open(settings)
            self.addCleanup(catalog.close)
            media_id, _path = add_clip(catalog, footage, "vision-clip.mov", "vision-1")
            sheet_path = install_sheet(
                settings, catalog, media_id, jpg_path.read_bytes()
            )
            tiles = vision_text.read_sheet_text(
                os.fspath(sheet_path), 9, cache_dir=self._swift_cache.name
            )
            self.assert_rendered_sheet(tiles)
            with self.subTest("png"):
                tiles = vision_text.read_sheet_text(
                    os.fspath(png_path), 9, cache_dir=self._swift_cache.name
                )
                self.assert_rendered_sheet(tiles)

    @requires_vision
    def test_blank_frame_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            sheet_path = Path(tmp) / "blank.png"
            sheet_path.write_bytes(draw_sheet())
            tiles = vision_text.read_sheet_text(
                os.fspath(sheet_path), 9, cache_dir=self._swift_cache.name
            )
            self.assertEqual([tile.index for tile in tiles], list(range(9)))
            for tile in tiles:
                self.assertEqual(tile.lines, ())

    def test_vision_failure_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _project = make_project(tmp)
            catalog = Catalog.open(settings)
            self.addCleanup(catalog.close)
            with mock.patch.object(
                vision_text, "vision_unavailable_reason", return_value=None
            ):
                media_id, _path = add_clip(
                    catalog, footage, "boom-clip.mov", "vision-boom"
                )
                sheet_path = install_sheet(
                    settings, catalog, media_id, b"placeholder-sheet-bytes"
                )
                sheet_arg = os.fspath(sheet_path)

                def failing(args, **kwargs):
                    return subprocess.CompletedProcess(
                        args,
                        1,
                        stdout=b"",
                        stderr=b"read_text: Vision failed on tile 0: boom",
                    )

                with self.assertRaises(vision_text.VisionError) as caught:
                    vision_text.read_sheet_text(sheet_arg, 9, runner=failing)
                self.assertIn("boom", str(caught.exception))
                self.assertNotIsInstance(
                    caught.exception, vision_text.VisionUnavailable
                )

                failures = {
                    "not json": b"not json",
                    "another engine": json.dumps(
                        {
                            "engine": "Something Else",
                            "revision": 3,
                            "tiles": [],
                        }
                    ).encode("utf-8"),
                    "tiles missing index 8": json.dumps(
                        {
                            "engine": vision_text.ENGINE,
                            "revision": 3,
                            "tiles": CASES["full_slate"]["tiles"][:8],
                        }
                    ).encode("utf-8"),
                    "confidence 1.5": self._bad_confidence_case(),
                }
                for label, stdout in failures.items():
                    with self.subTest(label=label):

                        def bad(args, **kwargs):
                            return subprocess.CompletedProcess(
                                args, 0, stdout=stdout, stderr=b""
                            )

                        with self.assertRaises(vision_text.VisionError):
                            vision_text.read_sheet_text(sheet_arg, 9, runner=bad)

                with self.subTest(label="timeout"):

                    def slow(args, **kwargs):
                        raise subprocess.TimeoutExpired(args, 300)

                    with self.assertRaises(vision_text.VisionError):
                        vision_text.read_sheet_text(sheet_arg, 9, runner=slow)

                with self.subTest(label="os error"):

                    def missing(args, **kwargs):
                        raise OSError("swift vanished")

                    with self.assertRaises(vision_text.VisionError):
                        vision_text.read_sheet_text(sheet_arg, 9, runner=missing)

                other_id, _other = add_clip(
                    catalog, footage, "boom-second.mov", "vision-boom-2"
                )
                install_sheet(
                    settings, catalog, other_id, b"placeholder-sheet-bytes"
                )
                calls = []
                fixture = json.dumps(CASES["full_slate"]).encode("utf-8")

                def flaky(args, **kwargs):
                    calls.append(list(args))
                    if len(calls) == 1:
                        return subprocess.CompletedProcess(
                            args, 0, stdout=fixture, stderr=b""
                        )
                    return subprocess.CompletedProcess(
                        args,
                        1,
                        stdout=b"",
                        stderr=b"read_text: Vision failed on tile 0: boom",
                    )

                with self.assertRaises(vision_text.VisionError):
                    vision_text.tag_vision_text(catalog, settings, runner=flaky)
                first_tags = {
                    tag.folder for tag in catalog.list_tags(media_id)
                }
                self.assertEqual(
                    first_tags,
                    {"Day", "Place", "Shot", "Stock", "Camera", "Slate"},
                )
                self.assertEqual(catalog.list_tags(other_id), [])
                runs = catalog.list_runs()
                self.assertTrue(runs[-1].outcome.startswith("failed:"))

    @staticmethod
    def _bad_confidence_case():
        """Full-slate JSON with one confidence of 1.5, as bytes."""
        document = json.loads(json.dumps(CASES["full_slate"]))
        document["tiles"][4]["lines"][0]["confidence"] = 1.5
        return json.dumps(document).encode("utf-8")

    def test_skips_off_macos(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _project = make_project(tmp)
            catalog = Catalog.open(settings)
            self.addCleanup(catalog.close)
            media_id, _path = add_clip(catalog, footage, "skip-clip.mov", "vision-skip")
            calls = []

            def never(args, **kwargs):
                calls.append(list(args))
                raise AssertionError("helper must not run off macOS")

            with mock.patch.object(vision_text.sys, "platform", "linux"):
                reason = vision_text.vision_unavailable_reason()
                self.assertIn("macOS", reason)
                with self.assertRaises(vision_text.VisionUnavailable):
                    vision_text.read_sheet_text("sheet.jpg", 9, runner=never)
                with self.assertRaises(vision_text.VisionUnavailable):
                    vision_text.tag_vision_text(catalog, settings, runner=never)
                self.assertEqual(calls, [])
                self.assertEqual(catalog.list_runs(), [])
                self.assertEqual(catalog.list_tags(media_id), [])
            with (
                mock.patch.object(vision_text.sys, "platform", "darwin"),
                mock.patch.object(
                    vision_text.shutil, "which", return_value=None
                ),
            ):
                reason = vision_text.vision_unavailable_reason()
                self.assertIn("swift", reason)


class TestVisionParse(unittest.TestCase):
    """Tile choice, helper-output checks, stamps and slate readings."""

    def parse_case(self, name, count=9):
        """Parse one recorded fixture case into TileText entries."""
        return vision_text.parse_helper_output(json.dumps(CASES[name]), count)

    def test_select_tiles_positions(self):
        times = [
            Fraction(10, 3),
            Fraction(10, 1),
            Fraction(50, 3),
            Fraction(70, 3),
            Fraction(30, 1),
        ]
        refs = vision_text.select_tiles(manifest_document(times))
        self.assertEqual([ref.index for ref in refs], [0, 1, 2, 3, 4])
        self.assertEqual([ref.column for ref in refs], [0, 1, 2, 0, 1])
        self.assertEqual([ref.row for ref in refs], [0, 0, 0, 1, 1])
        self.assertEqual([ref.time for ref in refs], times)
        (only,) = vision_text.select_tiles(manifest_document([Fraction(30, 1)]))
        self.assertEqual(
            only, TileRef(index=0, time=Fraction(30, 1), column=0, row=0)
        )

    def test_select_tiles_rejects_bad_manifest(self):
        good = [Fraction(10, 3), Fraction(10, 1)]
        bad_manifests = {
            "wrong version": manifest_document(good, version=2),
            "grid [2, 2]": manifest_document(good, grid=[2, 2]),
            "tile_size [256, 144]": manifest_document(
                good, tile_size=[256, 144]
            ),
            "empty tiles": manifest_document(good, tiles=[]),
            "ten tiles": manifest_document([Fraction(at + 1, 3) for at in range(10)]),
            "indexes out of order": manifest_document(
                good,
                tiles=[
                    {"index": 1, "time": "10/3"},
                    {"index": 0, "time": "10/1"},
                ],
            ),
            'time "0.5"': manifest_document(
                good, tiles=[{"index": 0, "time": "0.5"}]
            ),
        }
        for label, document in bad_manifests.items():
            with self.subTest(label=label):
                with self.assertRaises(TileSelectionError):
                    vision_text.select_tiles(document)

    def test_parse_helper_output_checks_shape(self):
        tiles = self.parse_case("full_slate")
        self.assertEqual([tile.index for tile in tiles], list(range(9)))
        fourth = tiles[4]
        self.assertEqual(
            [line.text for line in fourth.lines],
            ["B-ROLL", "SLATE", "DAY 3", "MOONLIT COVE", "PORTRA 400 ON THE F3"],
        )
        self.assertEqual(
            [line.confidence for line in fourth.lines], [1, 1, 1, 1, 0.9]
        )
        self.assertEqual(fourth.lines[0].box, (0.10, 0.45, 0.40, 0.06))
        self.assertEqual(tiles[0].lines[0].box, (0.016, 0.028, 0.09, 0.139))
        for index in (1, 2, 3, 5, 6, 7, 8):
            self.assertEqual(tiles[index].lines, ())
        failures = {
            "not json": b"not json",
            "another engine": json.dumps(
                {"engine": "Something Else", "revision": 3, "tiles": []}
            ),
            "tiles missing index 8": json.dumps(
                {
                    "engine": vision_text.ENGINE,
                    "revision": 3,
                    "tiles": CASES["full_slate"]["tiles"][:8],
                }
            ),
            "confidence 1.5": json.dumps(
                {
                    "engine": vision_text.ENGINE,
                    "revision": 3,
                    "tiles": [
                        {"index": at, "lines": []} for at in range(8)
                    ]
                    + [
                        {
                            "index": 8,
                            "lines": [
                                {
                                    "text": "SLATE",
                                    "confidence": 1.5,
                                    "box": [0.1, 0.2, 0.3, 0.06],
                                }
                            ],
                        }
                    ],
                }
            ),
        }
        for label, stdout in failures.items():
            with self.subTest(label=label):
                with self.assertRaises(vision_text.VisionError):
                    vision_text.parse_helper_output(stdout, 9)

    def test_stamp_lines_dropped(self):
        tiles = self.parse_case("stamp_only")
        by_index = tile_texts(tiles)
        for index in (0, 1, 2):
            self.assertEqual(vision_text.confident_lines(by_index[index]), [])
        (kept,) = vision_text.confident_lines(by_index[3])
        self.assertEqual(kept.text, "3.3s")
        for index in range(4, 9):
            self.assertEqual(vision_text.confident_lines(by_index[index]), [])

    def test_low_confidence_dropped_and_order(self):
        tiles = self.parse_case("full_slate")
        ordered = vision_text.confident_lines(tile_texts(tiles)[4])
        self.assertEqual(
            [line.text for line in ordered],
            ["SLATE", "DAY 3", "MOONLIT COVE", "B-ROLL", "PORTRA 400 ON THE F3"],
        )
        low = self.parse_case("low_confidence")
        self.assertEqual(vision_text.confident_lines(tile_texts(low)[4]), [])

    def test_slate_from_tile_reads_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _project = make_project(tmp)
            catalog = Catalog.in_memory()
            self.addCleanup(catalog.close)
            add_clip(catalog, footage, "vocab-clip.mov", "vision-vocab")
            vocabulary = build_vocabulary(catalog, settings)
            full = self.parse_case("full_slate")
            reading = vision_text.slate_from_tile(
                tile_texts(full)[4], vocabulary
            )
            self.assertIsNotNone(reading)
            assert reading is not None
            self.assertIsInstance(reading, SlateReading)
            self.assertEqual(reading.tile, 4)
            self.assertEqual(
                reading.text, "DAY 3. MOONLIT COVE. B-ROLL. PORTRA 400 ON THE F3"
            )
            self.assertEqual(reading.confidence, 0.9)
            self.assertEqual(
                {part.folder: (part.value, part.status) for part in reading.parts},
                {
                    "Day": ("Day 3", "plain"),
                    "Place": ("Moonlit Cove", "plain"),
                    "Shot": ("B-roll", "plain"),
                    "Stock": ("Portra 400", "plain"),
                    "Camera": ("F3", "plain"),
                },
            )
            one = self.parse_case("one_line")
            single = vision_text.slate_from_tile(tile_texts(one)[2], vocabulary)
            self.assertIsNotNone(single)
            assert single is not None
            self.assertEqual(
                {part.folder: part.value for part in single.parts},
                {"Day": "Day 3", "Shot": "B-roll"},
            )
            for name, index in (("not_a_slate", 1), ("not_a_slate", 5)):
                tiles = self.parse_case(name)
                self.assertIsNone(
                    vision_text.slate_from_tile(tile_texts(tiles)[index], vocabulary)
                )
            low = self.parse_case("low_confidence")
            self.assertIsNone(
                vision_text.slate_from_tile(tile_texts(low)[4], vocabulary)
            )


class TestVisionTags(unittest.TestCase):
    """Tagging from recorded helper output, precedence and the command."""

    def setUp(self):
        unavailable = mock.patch.object(
            vision_text, "vision_unavailable_reason", return_value=None
        )
        unavailable.start()
        self.addCleanup(unavailable.stop)
        offline = mock.patch.object(
            socket,
            "socket",
            side_effect=AssertionError("vision-text step used the network"),
        )
        offline.start()
        self.addCleanup(offline.stop)

    def tags_by_folder(self, catalog, media_id):
        """Map folder name to tag for one media row."""
        return {tag.folder: tag for tag in catalog.list_tags(media_id)}

    def test_full_slate_gives_five_keywords(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _project = make_project(tmp)
            catalog = Catalog.open(settings)
            self.addCleanup(catalog.close)
            media_id, _path = add_clip(catalog, footage, "slate-clip.mov", "vision-1")
            before_rows = catalog.list_media()
            sheet_path = install_sheet(
                settings, catalog, media_id, b"placeholder-sheet-bytes"
            )
            calls = []
            result = vision_text.tag_vision_text(
                catalog, settings, runner=recorded_runner("full_slate", calls)
            )
            (entry,) = result.entries
            self.assertEqual(entry.status, "tagged")
            self.assertEqual(entry.stored, 6)
            self.assertEqual(
                entry.text, "DAY 3. MOONLIT COVE. B-ROLL. PORTRA 400 ON THE F3"
            )
            self.assertEqual(entry.tile_time, Fraction(30, 1))
            tags = self.tags_by_folder(catalog, media_id)
            self.assertEqual(
                set(tags), {"Day", "Place", "Shot", "Stock", "Camera", "Slate"}
            )
            expected = {
                "Day": "Day 3",
                "Place": "Moonlit Cove",
                "Shot": "B-roll",
                "Stock": "Portra 400",
                "Camera": "F3",
                "Slate": "DAY 3. MOONLIT COVE. B-ROLL. PORTRA 400 ON THE F3",
            }
            for folder, value in expected.items():
                with self.subTest(folder=folder):
                    tag = tags[folder]
                    self.assertEqual(tag.value, value)
                    self.assertEqual(tag.source, "vision-text")
                    self.assertEqual(tag.status, "plain")
                    self.assertEqual(tag.confidence, 0.9)
                    self.assertIsNone(tag.start)
                    self.assertIsNone(tag.duration)
            self.assertEqual(len(calls), 1)
            command = calls[0]
            self.assertEqual(command[0], "swift")
            self.assertEqual(command[1], "-module-cache-path")
            self.assertEqual(
                Path(command[3]).resolve(), vision_text.HELPER.resolve()
            )
            self.assertEqual(
                Path(command[4]).resolve(), Path(os.fspath(sheet_path)).resolve()
            )
            self.assertEqual(command[5:], ["3", "3", "9"])
            self.assertEqual(catalog.list_media(), before_rows)
            with self.subTest("symlinked project dir"):
                with tempfile.TemporaryDirectory() as real:
                    base = Path(real)
                    target = base / "actual"
                    target.mkdir()
                    link = base / "link"
                    link.symlink_to(target, target_is_directory=True)
                    linked_settings, linked_footage, _linked = make_project(link)
                    linked = Catalog.open(linked_settings)
                    self.addCleanup(linked.close)
                    linked_id, _linked_path = add_clip(
                        linked, linked_footage, "link-clip.mov", "vision-link"
                    )
                    linked_sheet = install_sheet(
                        linked_settings, linked, linked_id, b"placeholder-sheet-bytes"
                    )
                    linked_calls = []
                    linked_result = vision_text.tag_vision_text(
                        linked,
                        linked_settings,
                        runner=recorded_runner("full_slate", linked_calls),
                    )
                    (linked_entry,) = linked_result.entries
                    self.assertEqual(linked_entry.status, "tagged")
                    self.assertEqual(len(linked_calls), 1)
                    self.assertEqual(
                        Path(linked_calls[0][4]).resolve(),
                        Path(os.fspath(linked_sheet)).resolve(),
                    )

    def test_one_line_slate_tags_day_and_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _project = make_project(tmp)
            catalog = Catalog.open(settings)
            self.addCleanup(catalog.close)
            media_id, _path = add_clip(catalog, footage, "one-line.mov", "vision-2")
            install_sheet(settings, catalog, media_id, b"placeholder-sheet-bytes")
            result = vision_text.tag_vision_text(
                catalog, settings, runner=recorded_runner("one_line", [])
            )
            (entry,) = result.entries
            self.assertEqual(entry.status, "tagged")
            tags = self.tags_by_folder(catalog, media_id)
            self.assertEqual(
                {folder: tag.value for folder, tag in tags.items()},
                {
                    "Day": "Day 3",
                    "Shot": "B-roll",
                    "Slate": "DAY 3. B-ROLL",
                },
            )

    def test_unknown_place_becomes_check_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _project = make_project(tmp)
            catalog = Catalog.open(settings)
            self.addCleanup(catalog.close)
            media_id, _path = add_clip(catalog, footage, "unknown.mov", "vision-3")
            install_sheet(settings, catalog, media_id, b"placeholder-sheet-bytes")
            result = vision_text.tag_vision_text(
                catalog, settings, runner=recorded_runner("unknown_place", [])
            )
            (entry,) = result.entries
            self.assertEqual(entry.status, "tagged")
            tags = self.tags_by_folder(catalog, media_id)
            self.assertEqual(tags["Day"].value, "Day 3")
            self.assertEqual(tags["Day"].status, "plain")
            self.assertEqual(tags["Place"].value, "HARBOR OF NOWHERE")
            self.assertEqual(tags["Place"].status, "check")

    def test_non_slate_text_stores_nothing(self):
        for case_name in ("not_a_slate", "low_confidence", "stamp_only", "blank"):
            with self.subTest(case=case_name):
                with tempfile.TemporaryDirectory() as tmp:
                    settings, footage, _project = make_project(tmp)
                    catalog = Catalog.open(settings)
                    self.addCleanup(catalog.close)
                    media_id, _path = add_clip(
                        catalog, footage, "%s.mov" % case_name, "vision-%s" % case_name
                    )
                    install_sheet(
                        settings, catalog, media_id, b"placeholder-sheet-bytes"
                    )
                    result = vision_text.tag_vision_text(
                        catalog, settings, runner=recorded_runner(case_name, [])
                    )
                    (entry,) = result.entries
                    self.assertEqual(entry.status, "no-slate")
                    self.assertEqual(entry.reason, "no slate text on any tile")
                    self.assertEqual(catalog.list_tags(media_id), [])
                    runs = catalog.list_runs()
                    self.assertEqual(
                        runs[-1].outcome,
                        "tagged 0, no-slate 1, no-sheet 0, stale 0, failed 0",
                    )

    def test_first_slate_tile_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _project = make_project(tmp)
            catalog = Catalog.open(settings)
            self.addCleanup(catalog.close)
            media_id, _path = add_clip(catalog, footage, "two.mov", "vision-4")
            install_sheet(settings, catalog, media_id, b"placeholder-sheet-bytes")
            result = vision_text.tag_vision_text(
                catalog, settings, runner=recorded_runner("two_slates", [])
            )
            (entry,) = result.entries
            self.assertEqual(entry.status, "tagged")
            tags = self.tags_by_folder(catalog, media_id)
            self.assertEqual(tags["Day"].value, "Day 3")
            self.assertEqual(tags["Shot"].value, "B-roll")
            self.assertIn("second slate on tile 6 ignored", entry.notes)

    def test_precedence_follows_the_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _project = make_project(tmp)
            catalog = Catalog.open(settings)
            self.addCleanup(catalog.close)
            media_id, _path = add_clip(catalog, footage, "kept.mov", "vision-5")
            install_sheet(settings, catalog, media_id, b"placeholder-sheet-bytes")
            catalog.put_tag(
                TagRecord(
                    media_id=media_id,
                    folder="Day",
                    value="Day 2",
                    start=None,
                    duration=None,
                    source="slate",
                    confidence=None,
                    status="plain",
                )
            )
            catalog.put_tag(
                TagRecord(
                    media_id=media_id,
                    folder="Place",
                    value="Moonlit Cove",
                    start=None,
                    duration=None,
                    source="ai",
                    confidence=None,
                    status="locked",
                )
            )
            catalog.put_tag(
                TagRecord(
                    media_id=media_id,
                    folder="Camera",
                    value="Invented Cam",
                    start=None,
                    duration=None,
                    source="ai",
                    confidence=None,
                    status="deleted",
                )
            )
            catalog.put_tag(
                TagRecord(
                    media_id=media_id,
                    folder="Shot",
                    value="Insert",
                    start=None,
                    duration=None,
                    source="ai",
                    confidence=None,
                    status="plain",
                )
            )
            calls = []
            result = vision_text.tag_vision_text(
                catalog, settings, runner=recorded_runner("full_slate", calls)
            )
            (entry,) = result.entries
            self.assertEqual(entry.status, "tagged")
            self.assertEqual(entry.stored, 3)
            tags = self.tags_by_folder(catalog, media_id)
            self.assertEqual(tags["Day"].value, "Day 2")
            self.assertEqual(tags["Day"].source, "slate")
            self.assertEqual(tags["Place"].value, "Moonlit Cove")
            self.assertEqual(tags["Place"].status, "locked")
            self.assertEqual(tags["Camera"].value, "Invented Cam")
            self.assertEqual(tags["Camera"].status, "deleted")
            self.assertEqual(tags["Shot"].value, "B-roll")
            self.assertEqual(tags["Shot"].source, "vision-text")
            self.assertEqual(tags["Stock"].value, "Portra 400")
            self.assertIn("Day kept slate tag", entry.notes)
            self.assertIn("Place kept ai tag", entry.notes)
            self.assertIn("Camera kept ai tag", entry.notes)
            again = vision_text.tag_vision_text(
                catalog, settings, runner=recorded_runner("full_slate", [])
            )
            (second,) = again.entries
            self.assertEqual(second.notes, entry.notes)
            self.assertEqual(second.stored, 3)
            rerun = self.tags_by_folder(catalog, media_id)
            self.assertEqual(
                set(rerun), {"Day", "Place", "Shot", "Stock", "Camera", "Slate"}
            )

    def test_stale_no_sheet_and_bad_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, project = make_project(tmp)
            catalog = Catalog.open(settings)
            self.addCleanup(catalog.close)
            stale_id, stale_path = add_clip(catalog, footage, "stale.mov", "vision-6")
            with open(stale_path, "ab") as handle:
                handle.write(b"changed-after-the-scan")
            bare_id, _bare = add_clip(catalog, footage, "bare.mov", "vision-7")
            hashed_id, _hashed = add_clip(
                catalog, footage, "hashed.mov", "vision-8"
            )
            hashed_sheet = install_sheet(
                settings, catalog, hashed_id, b"placeholder-sheet-bytes"
            )
            hashed_sheet.write_bytes(b"different-bytes-after-the-manifest")
            from fcp_pipeline import sheets as sheets_module

            travelled_id, _travelled = add_clip(
                catalog, footage, "travelled.mov", "vision-9"
            )
            travelled_row = [
                item for item in catalog.list_media() if item.id == travelled_id
            ][0]
            _, _, travelled_manifest = sheets_module._sheet_names(travelled_row)
            travelled_folder = (
                Path(os.fspath(settings.folder)) / ".pipeline" / "sheets"
            )
            travelled_folder.mkdir(parents=True, exist_ok=True)
            travelled_manifest_path = travelled_folder / travelled_manifest
            travelled_manifest_path.write_text(
                json.dumps(
                    manifest_document(
                        NINE_TILES, asset_id="vision-9", sheet="../x.jpg"
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            linked_id, _linked = add_clip(
                catalog, footage, "linked.mov", "vision-10"
            )
            linked_row = [
                item for item in catalog.list_media() if item.id == linked_id
            ][0]
            _, linked_name, linked_manifest = sheets_module._sheet_names(linked_row)
            outside = Path(tmp) / "outside.jpg"
            outside.write_bytes(b"outside-sheet-bytes")
            (travelled_folder / linked_name).symlink_to(outside)
            (travelled_folder / linked_manifest).write_text(
                json.dumps(
                    manifest_document(
                        NINE_TILES,
                        asset_id="vision-10",
                        sheet=linked_name,
                        sheet_sha256=hashlib.sha256(
                            b"outside-sheet-bytes"
                        ).hexdigest(),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            calls = []
            result = vision_text.tag_vision_text(
                catalog, settings, runner=recorded_runner("full_slate", calls)
            )
            self.assertEqual(calls, [])
            by_asset = {entry.asset_id: entry for entry in result.entries}
            stale = by_asset["vision-6"]
            self.assertEqual(stale.status, "stale")
            self.assertEqual(
                stale.reason, "changed since the scan; run the scan again"
            )
            bare = by_asset["vision-7"]
            self.assertEqual(bare.status, "no-sheet")
            hashed = by_asset["vision-8"]
            self.assertEqual(hashed.status, "failed")
            self.assertIn(hashed_sheet.name, hashed.reason)
            travelled = by_asset["vision-9"]
            self.assertEqual(travelled.status, "failed")
            self.assertIn("../x.jpg", travelled.reason)
            linked = by_asset["vision-10"]
            self.assertEqual(linked.status, "failed")
            self.assertIn(linked_name, linked.reason)
            self.assertEqual(
                result.outcome,
                "tagged 0, no-slate 0, no-sheet 1, stale 1, failed 3",
            )

    def test_one_cache_dir_per_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _project = make_project(tmp)
            catalog = Catalog.open(settings)
            self.addCleanup(catalog.close)
            for number in range(3):
                media_id, _path = add_clip(
                    catalog,
                    footage,
                    "cache-%d.mov" % number,
                    "vision-cache-%d" % number,
                )
                install_sheet(
                    settings, catalog, media_id, b"placeholder-sheet-bytes"
                )
            calls = []
            vision_text.tag_vision_text(
                catalog, settings, runner=recorded_runner("one_line", calls)
            )
            self.assertEqual(len(calls), 3)
            cache_flags = [command[command.index("-module-cache-path") + 1] for command in calls]
            self.assertEqual(cache_flags[0], cache_flags[1])
            self.assertEqual(cache_flags[1], cache_flags[2])
            self.assertFalse(Path(cache_flags[0]).exists())
            given = Path(tmp) / "given-cache"
            given.mkdir()
            kept_calls = []
            vision_text.tag_vision_text(
                catalog,
                settings,
                runner=recorded_runner("one_line", kept_calls),
                cache_dir=given,
            )
            self.assertEqual(len(kept_calls), 3)
            for command in kept_calls:
                self.assertEqual(
                    command[command.index("-module-cache-path") + 1],
                    os.fspath(given),
                )
            self.assertTrue(given.is_dir())

    @staticmethod
    def snapshot_files(root):
        """Map every relative file path under root to size, mtime and hash."""
        found = {}
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                full = os.path.join(dirpath, name)
                with open(full, "rb") as handle:
                    data = handle.read()
                info = os.lstat(full)
                found[os.path.relpath(full, root)] = (
                    info.st_size,
                    info.st_mtime_ns,
                    hashlib.sha256(data).hexdigest(),
                )
        return found

    def test_offline_read_only_and_catalog_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, project = make_project(tmp)
            catalog = Catalog.open(settings)
            self.addCleanup(catalog.close)
            media_id, _path = add_clip(catalog, footage, "quiet.mov", "vision-11")
            install_sheet(settings, catalog, media_id, b"placeholder-sheet-bytes")
            sheets_folder = (
                Path(os.fspath(settings.folder)) / ".pipeline" / "sheets"
            )
            with mock.patch.object(
                socket,
                "create_connection",
                side_effect=AssertionError("vision-text step used the network"),
            ):
                before_footage = self.snapshot_files(footage)
                before_sheets = self.snapshot_files(sheets_folder)
                before_project = self.snapshot_files(project)
                result = vision_text.tag_vision_text(
                    catalog, settings, runner=recorded_runner("full_slate", [])
                )
                after_footage = self.snapshot_files(footage)
                after_sheets = self.snapshot_files(sheets_folder)
                after_project = self.snapshot_files(project)
            self.assertEqual(result.entries[0].status, "tagged")
            self.assertEqual(after_footage, before_footage)
            self.assertEqual(after_sheets, before_sheets)
            self.assertEqual(set(after_project), set(before_project))
            changed = [
                name
                for name in before_project
                if after_project[name] != before_project[name]
            ]
            self.assertTrue(changed)
            for name in changed:
                self.assertIn("sqlite", Path(name).name)

    def test_command_line_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            settings, footage, project = make_project(tmp)
            catalog = Catalog.open(settings)
            media_id, _path = add_clip(catalog, footage, "cli.mov", "vision-12")
            install_sheet(settings, catalog, media_id, b"placeholder-sheet-bytes")
            catalog.close()
            with mock.patch.object(
                vision_text,
                "vision_unavailable_reason",
                return_value="Apple Vision needs macOS",
            ):
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    code = vision_text.main(["--trip", os.fspath(project)])
                self.assertEqual(code, 3)
                self.assertEqual(errors.getvalue(), "error: Apple Vision needs macOS\n")
            with mock.patch.object(
                vision_text, "vision_unavailable_reason", return_value=None
            ):

                def boom(sheet, tile_count, *, runner=None, cache_dir=None):
                    raise vision_text.VisionError("boom")

                with mock.patch.object(
                    vision_text, "read_sheet_text", side_effect=boom
                ):
                    errors = io.StringIO()
                    with contextlib.redirect_stderr(errors):
                        code = vision_text.main(["--trip", os.fspath(project)])
                    self.assertEqual(code, 1)
                    self.assertIn("error:", errors.getvalue())
                    self.assertIn("boom", errors.getvalue())
                fixture = json.dumps(CASES["two_slates"])

                def recorded(sheet, tile_count, *, runner=None, cache_dir=None):
                    return vision_text.parse_helper_output(fixture, tile_count)

                with mock.patch.object(
                    vision_text, "read_sheet_text", side_effect=recorded
                ):
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        code = vision_text.main(["--trip", os.fspath(project)])
                    self.assertEqual(code, 0)
                    self.assertIn(
                        "tagged 1, no-slate 0, no-sheet 0, stale 0, failed 0",
                        out.getvalue(),
                    )
                    self.assertIn("note:", out.getvalue())
                missing = tmp / "empty-trip"
                missing.mkdir()
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    code = vision_text.main(["--trip", os.fspath(missing)])
                self.assertEqual(code, 1)
            with self.assertRaises(SystemExit) as exited:
                vision_text.main([])
            self.assertEqual(exited.exception.code, 2)

    def test_sources_have_no_network_or_launch(self):
        module_text = (
            Path(vision_text.__file__).read_text(encoding="utf-8")
        )
        helper_text = vision_text.HELPER.read_text(encoding="utf-8")
        for name in (
            "socket",
            "ssl",
            "http",
            "urllib.request",
            "ftplib",
            "fcp_pipeline.net",
        ):
            self.assertNotIn("import %s" % name, module_text)
            self.assertNotIn("from %s" % name, module_text)
        self.assertNotIn("pyobjc", module_text)
        for token in ("URLSession", "NSWorkspace", "Process(", "osascript"):
            self.assertNotIn(token, module_text)
            self.assertNotIn(token, helper_text)
        helper_imports = sorted(
            line.strip()
            for line in helper_text.splitlines()
            if line.strip().startswith("import ")
        )
        self.assertEqual(
            helper_imports,
            [
                "import CoreGraphics",
                "import Foundation",
                "import ImageIO",
                "import Vision",
            ],
        )
