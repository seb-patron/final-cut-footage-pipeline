"""Tests for fcp_pipeline.sheets, the per-clip contact sheet step.

Each sheet test runs in a temporary project folder with a catalog, clips
made by tests.media_fixtures.make_cut_clip (copied into a temporary
footage folder; rows added with catalog.put_media using the copy's real
duration from clip_duration and os.stat size, mtime_ns and inode so
catalog.unchanged is True) and a counting wrapper around subprocess.run
passed as runner. Tests that run ffmpeg skip with a reason naming ffmpeg
(and ffprobe where a test reads a sheet's size) when it is missing, and
the tile and overlay filters are checked the same way by a shared
helper. The tile_times and stamp_text checks need no tools and always
run.
"""

import contextlib
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

from fcp_pipeline.catalog import Catalog, MediaRecord
from fcp_pipeline.sheets import (
    MAX_MANIFEST_BYTES,
    SheetEntry,
    SheetError,
    SheetsResult,
    _manifest_bytes,
    _manifest_document,
    build_sheets,
    main,
    read_manifest,
    stamp_png,
    stamp_text,
    tile_times,
)
from fcp_pipeline.trip import load_trip
from tests.media_fixtures import clip_duration, make_cut_clip

#: The nine even tile times of a 60 s clip, as exact Fractions.
SIXTY_SECOND_TILES = [
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


def _require_tools(*names):
    """Skip with a reason naming each missing tool."""
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise unittest.SkipTest(
            "%s required to build contact sheets but not found on PATH"
            % ", ".join(missing)
        )


def _require_filters(*names):
    """Skip with a reason naming each ffmpeg filter that is missing."""
    _require_tools("ffmpeg")
    ffmpeg = shutil.which("ffmpeg")
    for name in names:
        completed = subprocess.run(
            [ffmpeg, "-hide_banner", "-h", "filter=%s" % name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        if completed.returncode != 0:
            raise unittest.SkipTest(
                "ffmpeg filter %s required to build contact sheets "
                "but not available" % name
            )


def _require_sheet_tools():
    """Skip unless ffmpeg, ffprobe, tile and overlay are all present."""
    _require_tools("ffmpeg", "ffprobe")
    _require_filters("tile", "overlay")


class CountingRunner:
    """A runner wrapper that counts every ffmpeg call it forwards."""

    def __init__(self):
        self.calls = []

    def __call__(self, args, *, stdin=None, capture_output=False,
                 timeout=None, check=False):
        self.calls.append(list(args))
        return subprocess.run(
            args,
            stdin=stdin,
            capture_output=capture_output,
            timeout=timeout,
            check=check,
        )


class TestSheets(unittest.TestCase):
    """Each sheets test runs in a temporary project with invented media."""

    def make_project(self, tmp, name="project"):
        """Build a trip folder whose footage folder is a temp folder."""
        footage = tmp / "footage"
        footage.mkdir(parents=True, exist_ok=True)
        project = tmp / name
        project.mkdir(parents=True, exist_ok=True)
        (project / "project.toml").write_text(
            "title = \"Invented Contact Sheet Survey\"\n"
            "[media]\n"
            f"footage = [\"{footage.as_posix()}\"]\n"
            "[pipeline]\n"
            "timezone = \"UTC\"\n"
            "start_date = 2020-06-15\n",
            encoding="utf-8",
        )
        return load_trip(project), footage, project

    def add_clip(self, catalog, footage, name, asset_id, segments,
                 rate="10/1"):
        """Copy a cached colour-cut clip into footage and catalog it."""
        cached = make_cut_clip(segments, rate=rate)
        destination = footage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, destination)
        info = os.stat(destination)
        total = sum(seconds for _colour, seconds in segments)
        catalog.put_media(
            MediaRecord(
                asset_id=asset_id,
                kind="clip",
                path=os.fspath(destination),
                size=info.st_size,
                mtime_ns=info.st_mtime_ns,
                inode=info.st_ino,
                duration=clip_duration(total, rate),
            )
        )
        return destination

    def sheets_dir(self, project):
        """Return the .pipeline/sheets/ folder under a project."""
        return project.resolve() / ".pipeline" / "sheets"

    def read_manifests(self, project):
        """Map manifest file names to their parsed documents."""
        found = {}
        folder = self.sheets_dir(project)
        if not folder.is_dir():
            return found
        for child in sorted(folder.iterdir()):
            if child.suffix == ".json":
                found[child.name] = json.loads(
                    child.read_text(encoding="utf-8"))
        return found

    def sheet_size(self, sheet):
        """Return the (width, height) of a sheet image via ffprobe."""
        _require_tools("ffprobe")
        ffprobe = shutil.which("ffprobe")
        completed = subprocess.run(
            [ffprobe, "-hide_banner", "-v", "error",
             "-show_entries", "stream=width,height",
             "-of", "json", str(sheet)],
            check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=300)
        stream = json.loads(completed.stdout.decode("utf-8"))["streams"][0]
        return (int(stream["width"]), int(stream["height"]))

    def read_pixel(self, source, x, y, at=None):
        """Return the (red, green, blue) pixel at (x, y) of an image.

        NOTE: the crop runs after an explicit rgb24 conversion: a bare
        1x1 crop on a yuv420p frame is refused (chroma subsampling
        rounds the 1 px width down to 0), so the same rgb24 conversion
        the rawvideo output would do happens first.
        """
        _require_tools("ffmpeg")
        ffmpeg = shutil.which("ffmpeg")
        command = [ffmpeg, "-nostdin", "-hide_banner",
                   "-loglevel", "error"]
        if at is not None:
            command += ["-ss", str(float(at))]
        command += ["-i", str(source), "-vf",
                    "format=rgb24,crop=1:1:%d:%d" % (x, y),
                    "-frames:v", "1", "-f", "rawvideo",
                    "-pix_fmt", "rgb24", "-"]
        completed = subprocess.run(
            command, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=300)
        self.assertEqual(len(completed.stdout), 3)
        red, green, blue = completed.stdout
        return (red, green, blue)

    def cell_centre(self, cell):
        """Return the centre pixel of grid cell ``cell`` (0-8)."""
        column, row = cell % 3, cell // 3
        return (512 * column + 256, 288 * row + 144)

    def assert_white_ish(self, pixel):
        """Every channel of a white tile centre stays bright."""
        for channel in pixel:
            self.assertGreaterEqual(channel, 200, "not white-ish: %r"
                                    % (pixel,))

    def assert_black_ish(self, pixel):
        """Every channel of an empty grid cell stays dark."""
        for channel in pixel:
            self.assertLessEqual(channel, 40, "not black-ish: %r"
                                 % (pixel,))

    def assert_dominant(self, pixel, channel):
        """One channel is the largest by at least 60 over the others."""
        channels = {"red": pixel[0], "green": pixel[1], "blue": pixel[2]}
        wanted = channels[channel]
        for name, value in channels.items():
            if name != channel:
                self.assertGreaterEqual(
                    wanted - value, 60,
                    "%s %r is not dominantly %s" % (channel, pixel, channel))

    def snapshot(self, root):
        """Map every relative path under root to (size, mtime_ns)."""
        found = {}
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                full = os.path.join(dirpath, name)
                info = os.lstat(full)
                found[os.path.relpath(full, root)] = (
                    info.st_size,
                    info.st_mtime_ns,
                )
        return found

    def test_result_types(self):
        empty = SheetsResult(entries=())
        self.assertEqual(empty.built, 0)
        self.assertEqual(empty.skipped, 0)
        self.assertEqual(empty.stale, 0)
        self.assertEqual(empty.failed, 0)
        self.assertEqual(empty.outcome, "built 0, skipped 0, stale 0, failed 0")
        entry = SheetEntry(
            path="clip.mov",
            asset_id="sheet-1",
            status="built",
            sheet="clip-12345678.jpg",
            tile_times=(Fraction(1, 1),),
        )
        self.assertIsNone(entry.reason)
        with self.assertRaises(AttributeError):
            entry.status = "skipped"  # type: ignore[misc]

    def test_sheet_size_and_tiles(self):
        _require_sheet_tools()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, footage, project = self.make_project(tmp)
            runner = CountingRunner()
            with Catalog.open(settings) as catalog:
                clip = self.add_clip(
                    catalog, footage, "white.mov", "sheet-white-60",
                    (("white", 60),))
                self.assertTrue(catalog.unchanged(os.fspath(clip)))
                result = build_sheets(catalog, settings, runner=runner)
            self.assertEqual(result.built, 1)
            self.assertEqual(
                [(entry.status, entry.path) for entry in result.entries],
                [("built", os.fspath(clip))],
            )
            folder = self.sheets_dir(project)
            sheets = sorted(child for child in folder.iterdir()
                            if child.suffix == ".jpg")
            self.assertEqual(len(sheets), 1)
            self.assertEqual(self.sheet_size(sheets[0]), (1536, 864))
            manifests = self.read_manifests(project)
            self.assertEqual(len(manifests), 1)
            document = next(iter(manifests.values()))
            self.assertEqual(document["grid"], [3, 3])
            times = [Fraction(entry["time"])
                     for entry in document["tiles"]]
            self.assertEqual(len(times), 9)
            self.assertEqual(times, sorted(times))
            self.assertEqual(len(set(times)), 9)
            for cell in range(9):
                with self.subTest(cell=cell):
                    x, y = self.cell_centre(cell)
                    self.assert_white_ish(self.read_pixel(sheets[0], x, y))
            self.assertEqual(
                document["sheet_sha256"],
                hashlib.sha256(sheets[0].read_bytes()).hexdigest(),
            )

    def test_even_tile_times(self):
        _require_sheet_tools()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, footage, project = self.make_project(tmp)
            runner = CountingRunner()
            with Catalog.open(settings) as catalog:
                clip = self.add_clip(
                    catalog, footage, "white.mov", "sheet-white-60",
                    (("white", 60),))
                result = build_sheets(catalog, settings, runner=runner)
                row = catalog.get_media(os.fspath(clip))
            self.assertEqual(result.built, 1)
            manifests = self.read_manifests(project)
            self.assertEqual(len(manifests), 1)
            name, document = next(iter(manifests.items()))
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(read_manifest(settings, row), document)
            self.assertEqual(document["sheet"], name.replace(".json", ".jpg"))
            self.assertEqual(
                read_manifest(
                    settings,
                    MediaRecord(
                        asset_id="sheet-missing", kind="clip",
                        path=os.fspath(clip)),
                ),
                None,
            )
            self.assertEqual(
                [Fraction(entry["time"]) for entry in document["tiles"]],
                SIXTY_SECOND_TILES,
            )
            self.assertEqual(document["scene_count"], 0)
            self.assertFalse(document["truncated"])
            self.assertNotIn("scene_changes", document)
            self.assertEqual(
                tile_times(Fraction(60), ()), SIXTY_SECOND_TILES)

    def test_manifest_bytes(self):
        _require_sheet_tools()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, footage, project = self.make_project(tmp)
            runner = CountingRunner()
            with Catalog.open(settings) as catalog:
                clip = self.add_clip(
                    catalog, footage, "white.mov", "sheet-white-60",
                    (("white", 60),))
                result = build_sheets(catalog, settings, runner=runner)
            self.assertEqual(result.built, 1)
            folder = self.sheets_dir(project)
            manifests = sorted(folder.glob("*.json"))
            sheets = sorted(folder.glob("*.jpg"))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(len(sheets), 1)
            info = os.stat(clip)
            key = hashlib.sha256(
                f"{os.path.abspath(os.fspath(clip))}\0"
                f"{info.st_size}\0{info.st_mtime_ns}".encode("utf-8")
            ).hexdigest()[:8]
            sheet_name = "white-%s.jpg" % key
            self.assertEqual(sheets[0].name, sheet_name)
            digest = hashlib.sha256(sheets[0].read_bytes()).hexdigest()
            times = ["10/3", "10/1", "50/3", "70/3", "30/1", "110/3",
                     "130/3", "50/1", "170/3"]
            tiles = "".join(
                '    {\n      "index": %d,\n      "time": "%s"\n    }%s\n'
                % (at, text, "," if at < 8 else "")
                for at, text in enumerate(times)
            )
            expected = (
                '{\n  "version": 1,\n  "asset_id": "sheet-white-60",\n'
                '  "sheet": "%s",\n  "sheet_sha256": "%s",\n'
                '  "sheet_size": [\n    1536,\n    864\n  ],\n'
                '  "grid": [\n    3,\n    3\n  ],\n'
                '  "tile_size": [\n    512,\n    288\n  ],\n'
                '  "duration": "60/1",\n  "scene_count": 0,\n'
                '  "truncated": false,\n'
                '  "tiles": [\n' % (sheet_name, digest)
                + tiles + '  ]\n}\n'
            )
            self.assertEqual(
                manifests[0].read_text(encoding="utf-8"), expected)

    def test_manifest_stays_small_with_many_scenes(self):
        # Serializer-level: 400,000 synthetic scene entries at 60 fps
        # with no video and no ffmpeg. Only the drawn tile times and
        # the scene count reach the manifest, so it stays far under
        # the 5 MB per-file limit.
        scenes = [Fraction(at, 60) for at in range(400_000)]
        times = tile_times(Fraction(60), ())
        document = _manifest_document(
            "sheet-many-scenes",
            "many-12345678.jpg",
            "0" * 64,
            Fraction(60, 1),
            scenes,
            times,
        )
        payload = _manifest_bytes(document)
        self.assertEqual(document["scene_count"], 400_000)
        self.assertEqual(len(document["tiles"]), 9)
        self.assertNotIn("scene_changes", document)
        self.assertLess(len(payload), MAX_MANIFEST_BYTES)
        self.assertLess(len(payload), 10_000)

    def test_manifest_truncation_flag(self):
        times = [Fraction(at, 60) for at in range(12)]
        document = _manifest_document(
            "sheet-truncated",
            "truncated-12345678.jpg",
            "0" * 64,
            Fraction(60, 1),
            [],
            times,
        )
        self.assertTrue(document["truncated"])
        self.assertEqual(len(document["tiles"]), 9)
        payload = _manifest_bytes(document)
        self.assertLess(len(payload), MAX_MANIFEST_BYTES)

    def test_oversize_manifest_raises_and_writes_nothing(self):
        huge = {
            "version": 1,
            "sheet": "huge-12345678.jpg",
            "pad": "x" * (MAX_MANIFEST_BYTES + 1),
        }
        with self.assertRaises(SheetError):
            _manifest_bytes(huge)
        # The size check runs before any write for the clip: force an
        # oversize manifest through build_sheets with synthetic bytes
        # and mocked ffmpeg helpers (no video, no ffmpeg), and prove
        # neither the sheet nor the manifest is left behind.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, footage, _project = self.make_project(tmp)
            clip = footage / "clip.mov"
            clip.write_bytes(b"not a video; ffmpeg is mocked out")
            info = os.stat(clip)
            with Catalog.open(settings) as catalog:
                catalog.put_media(
                    MediaRecord(
                        asset_id="sheet-huge",
                        kind="clip",
                        path=os.fspath(clip),
                        size=info.st_size,
                        mtime_ns=info.st_mtime_ns,
                        inode=info.st_ino,
                        duration=Fraction(60, 1),
                    )
                )
                oversized = {
                    "version": 1,
                    "sheet": "clip-12345678.jpg",
                    "pad": "x" * (MAX_MANIFEST_BYTES + 1),
                }

                def fake_extract(
                    clip_path, at, stamp, tile_path, *,
                    runner, timeout,
                ):
                    Path(tile_path).write_bytes(b"tile")
                    return mock.Mock(returncode=0)

                def fake_compose(
                    tile_paths, grid_path, *, runner, timeout
                ):
                    Path(grid_path).write_bytes(b"grid")
                    return mock.Mock(returncode=0)

                with mock.patch(
                    "fcp_pipeline.sheets._detect_scenes",
                    return_value=([], 0, ""),
                ), mock.patch(
                    "fcp_pipeline.sheets._manifest_document",
                    return_value=oversized,
                ), mock.patch(
                    "fcp_pipeline.sheets._extract_tile",
                    side_effect=fake_extract,
                ), mock.patch(
                    "fcp_pipeline.sheets._compose_grid",
                    side_effect=fake_compose,
                ):
                    with self.assertRaises(SheetError):
                        build_sheets(
                            catalog, settings, runner=CountingRunner())
            folder = self.sheets_dir(_project)
            leftovers = (
                sorted(child.name for child in folder.iterdir())
                if folder.is_dir() else []
            )
            self.assertEqual(leftovers, [])

    def test_scene_change_tile_included(self):
        _require_sheet_tools()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, footage, project = self.make_project(tmp)
            runner = CountingRunner()
            with Catalog.open(settings) as catalog:
                self.add_clip(
                    catalog, footage, "cut.mov", "sheet-cut-60",
                    (("red", 31), ("blue", 29)))
                result = build_sheets(catalog, settings, runner=runner)
            self.assertEqual(result.built, 1)
            manifests = self.read_manifests(project)
            self.assertEqual(len(manifests), 1)
            document = next(iter(manifests.values()))
            times = [Fraction(entry["time"])
                     for entry in document["tiles"]]
            self.assertEqual(len(times), 9)
            self.assertIn(Fraction(31, 1), times)
            self.assertEqual(document["scene_count"], 1)
            self.assertFalse(document["truncated"])
            self.assertNotIn("scene_changes", document)
            sheets = sorted(child for child in self.sheets_dir(project).iterdir()
                            if child.suffix == ".jpg")
            self.assertEqual(len(sheets), 1)
            at = times.index(Fraction(31, 1))
            x, y = self.cell_centre(at)
            self.assert_dominant(self.read_pixel(sheets[0], x, y), "blue")
            before_x, before_y = self.cell_centre(at - 1)
            self.assert_dominant(
                self.read_pixel(sheets[0], before_x, before_y), "red")

    def test_short_clip_one_tile(self):
        _require_sheet_tools()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, footage, project = self.make_project(tmp)
            runner = CountingRunner()
            with Catalog.open(settings) as catalog:
                self.add_clip(
                    catalog, footage, "short.mov", "sheet-short-2",
                    (("white", 2),))
                result = build_sheets(catalog, settings, runner=runner)
            self.assertEqual(result.built, 1)
            manifests = self.read_manifests(project)
            self.assertEqual(len(manifests), 1)
            document = next(iter(manifests.values()))
            self.assertEqual(
                [Fraction(entry["time"]) for entry in document["tiles"]],
                [Fraction(1, 1)],
            )
            sheets = sorted(child for child in self.sheets_dir(project).iterdir()
                            if child.suffix == ".jpg")
            self.assertEqual(len(sheets), 1)
            self.assertEqual(self.sheet_size(sheets[0]), (1536, 864))
            x, y = self.cell_centre(0)
            self.assert_white_ish(self.read_pixel(sheets[0], x, y))
            for cell in range(1, 9):
                with self.subTest(cell=cell):
                    cx, cy = self.cell_centre(cell)
                    self.assert_black_ish(
                        self.read_pixel(sheets[0], cx, cy))

    def test_rerun_skips(self):
        _require_sheet_tools()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, footage, project = self.make_project(tmp)
            with Catalog.open(settings) as catalog:
                clip = self.add_clip(
                    catalog, footage, "white.mov", "sheet-white-60",
                    (("white", 60),))
                first = CountingRunner()
                result = build_sheets(catalog, settings, runner=first)
                self.assertEqual(result.built, 1)
                self.assertGreater(len(first.calls), 0)
            folder = self.sheets_dir(project)
            before = self.snapshot(folder)
            with Catalog.open(settings) as catalog:
                second = CountingRunner()
                rerun = build_sheets(catalog, settings, runner=second)
            self.assertEqual(len(second.calls), 0)
            self.assertTrue(rerun.entries)
            for entry in rerun.entries:
                self.assertEqual(entry.status, "skipped")
            self.assertEqual(self.snapshot(folder), before)
            old_sheets = sorted(child for child in folder.iterdir()
                                if child.suffix == ".jpg")
            self.assertEqual(len(old_sheets), 1)
            old_bytes = old_sheets[0].read_bytes()
            old_mtime = os.stat(old_sheets[0]).st_mtime_ns
            with Catalog.open(settings) as catalog:
                changed = make_cut_clip((("blue", 60),))
                shutil.copyfile(changed, clip)
                info = os.stat(clip)
                catalog.put_media(
                    MediaRecord(
                        asset_id="sheet-white-60",
                        kind="clip",
                        path=os.fspath(clip),
                        size=info.st_size,
                        mtime_ns=info.st_mtime_ns,
                        inode=info.st_ino,
                        duration=clip_duration(60, "10/1"),
                    )
                )
                third = CountingRunner()
                rebuilt = build_sheets(catalog, settings, runner=third)
            self.assertEqual(rebuilt.built, 1)
            self.assertGreater(len(third.calls), 0)
            new_sheets = sorted(child for child in folder.iterdir()
                                if child.suffix == ".jpg")
            self.assertEqual(len(new_sheets), 2)
            fresh = [child for child in new_sheets
                     if child.name != old_sheets[0].name]
            self.assertEqual(len(fresh), 1)
            self.assertEqual(old_sheets[0].read_bytes(), old_bytes)
            self.assertEqual(os.stat(old_sheets[0]).st_mtime_ns, old_mtime)

    def test_stamp_is_drawn(self):
        # Pure checks first: they need no tools and always run.
        pixels = stamp_png("31.0s")
        self.assertTrue(pixels.startswith(b"\x89PNG\r\n\x1a\n"))
        width, height = struct.unpack(">II", pixels[16:24])
        self.assertEqual((width, height), (128, 40))
        self.assertEqual(pixels[24], 8)
        self.assertEqual(pixels[25], 6)
        self.assertNotEqual(pixels, stamp_png("30.0s"))
        self.assertEqual(pixels, stamp_png("31.0s"))
        with self.assertRaises(SheetError):
            stamp_png("3:1")
        position = 8
        raw = b""
        while position < len(pixels):
            length = struct.unpack(">I", pixels[position:position + 4])[0]
            tag = pixels[position + 4:position + 8]
            data = pixels[position + 8:position + 8 + length]
            position += 12 + length
            if tag == b"IDAT":
                raw += data
        decoded = zlib.decompress(raw)
        stride = 128 * 4 + 1
        self.assertEqual(
            list(decoded[1:5]), [0, 0, 0, 153])
        white = sum(
            1
            for row in range(40)
            for column in range(128)
            if decoded[1 + row * stride + column * 4:
                       1 + row * stride + column * 4 + 4]
            == bytes([255, 255, 255, 255])
        )
        self.assertGreater(white, 0)
        _require_sheet_tools()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, footage, project = self.make_project(tmp)
            runner = CountingRunner()
            with Catalog.open(settings) as catalog:
                self.add_clip(
                    catalog, footage, "white.mov", "sheet-white-60",
                    (("white", 60),))
                result = build_sheets(catalog, settings, runner=runner)
            self.assertEqual(result.built, 1)
            sheets = sorted(child for child in self.sheets_dir(project).iterdir()
                            if child.suffix == ".jpg")
            self.assertEqual(len(sheets), 1)
            for cell in range(9):
                column, row = cell % 3, cell // 3
                with self.subTest(cell=cell):
                    stamp_pixel = self.read_pixel(
                        sheets[0], 512 * column + 9, 288 * row + 9)
                    for channel in stamp_pixel:
                        self.assertLess(
                            channel, 180,
                            "stamp box not drawn: %r" % (stamp_pixel,))
                    centre_x, centre_y = self.cell_centre(cell)
                    centre = self.read_pixel(sheets[0], centre_x, centre_y)
                    self.assertFalse(
                        all(channel < 180 for channel in centre),
                        "cell centre stamped over: %r" % (centre,))

    def test_same_clip_same_sheet(self):
        _require_sheet_tools()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            first_settings, first_footage, first_project = self.make_project(
                tmp, name="first")
            second_settings, second_footage, second_project = self.make_project(
                tmp, name="second")
            with Catalog.open(first_settings) as catalog:
                self.add_clip(
                    catalog, first_footage, "clip.mov", "sheet-shared-clip",
                    (("red", 31), ("blue", 29)))
                first_result = build_sheets(
                    catalog, first_settings, runner=CountingRunner())
            self.assertEqual(first_result.built, 1)
            with Catalog.open(second_settings) as catalog:
                self.add_clip(
                    catalog, second_footage, "clip.mov", "sheet-shared-clip",
                    (("red", 31), ("blue", 29)))
                second_result = build_sheets(
                    catalog, second_settings, runner=CountingRunner())
            self.assertEqual(second_result.built, 1)
            first_sheets = sorted(
                child for child in self.sheets_dir(first_project).iterdir()
                if child.suffix == ".jpg")
            second_sheets = sorted(
                child for child in self.sheets_dir(second_project).iterdir()
                if child.suffix == ".jpg")
            self.assertEqual(len(first_sheets), 1)
            self.assertEqual(len(second_sheets), 1)
            # The file names differ: the name's state code covers the
            # path, which differs between the two projects.
            self.assertNotEqual(first_sheets[0].name, second_sheets[0].name)
            self.assertEqual(
                first_sheets[0].read_bytes(), second_sheets[0].read_bytes())
            first_manifests = self.read_manifests(first_project)
            second_manifests = self.read_manifests(second_project)
            self.assertEqual(len(first_manifests), 1)
            self.assertEqual(len(second_manifests), 1)
            first_document = next(iter(first_manifests.values()))
            second_document = next(iter(second_manifests.values()))
            self.assertEqual(
                first_document["tiles"], second_document["tiles"])
            self.assertEqual(
                first_document["sheet_sha256"],
                second_document["sheet_sha256"],
            )

    def test_tile_time_rules(self):
        # (a) A clip under 3 s gets one tile; exactly 3 s uses slices.
        self.assertEqual(tile_times(Fraction(2), ()), [Fraction(1, 1)])
        self.assertEqual(
            tile_times(Fraction(3), ()),
            [Fraction(2 * i + 1, 6) for i in range(9)],
        )
        # (b) Scene times near the ends are dropped; close pairs keep
        # the first.
        even = tile_times(Fraction(60), ())
        self.assertEqual(
            tile_times(
                Fraction(60),
                [Fraction(1, 5), Fraction(597, 10)],
            ),
            even,
        )
        close = tile_times(
            Fraction(60), [Fraction(10, 1), Fraction(103, 10)])
        self.assertIn(Fraction(10, 1), close)
        self.assertNotIn(Fraction(103, 10), close)
        self.assertEqual(len(close), 9)
        # (c) Twelve scene times thin to the nine at (k x 11 + 4) // 8.
        scenes = [Fraction(value) for value in range(2, 14)]
        thinned = tile_times(Fraction(60), scenes)
        self.assertEqual(
            thinned,
            [scenes[(k * 11 + 4) // 8] for k in range(9)],
        )
        # (d) A 31 s cut replaces the 30 s even tile.
        cut = tile_times(Fraction(60), [Fraction(31, 1)])
        self.assertEqual(len(cut), 9)
        self.assertIn(Fraction(31, 1), cut)
        self.assertNotIn(Fraction(30, 1), cut)
        # (e) A zero or negative duration raises.
        with self.assertRaises(SheetError):
            tile_times(Fraction(0), ())
        with self.assertRaises(SheetError):
            tile_times(Fraction(-5, 1), ())
        # (f) Stamp text is seconds rounded half up with an "s".
        self.assertEqual(stamp_text(Fraction(31, 1)), "31.0s")
        self.assertEqual(stamp_text(Fraction(10, 3)), "3.3s")
        self.assertEqual(stamp_text(Fraction(5999, 100)), "60.0s")
        # (g) Thinning ties drop the later non-scene candidate.
        self.assertEqual(
            tile_times(Fraction(60), [Fraction(20, 1), Fraction(40, 1)]),
            [Fraction(10, 3), Fraction(10, 1), Fraction(50, 3),
             Fraction(20, 1), Fraction(70, 3), Fraction(30, 1),
             Fraction(40, 1), Fraction(50, 1), Fraction(170, 3)],
        )
        # (h) Scenes exactly half a second inside the ends are kept, and
        # a pair exactly half a second apart both stay.
        edges = tile_times(
            Fraction(60), [Fraction(1, 2), Fraction(119, 2)])
        self.assertIn(Fraction(1, 2), edges)
        self.assertIn(Fraction(119, 2), edges)
        pair = tile_times(
            Fraction(60), [Fraction(10, 1), Fraction(21, 2)])
        self.assertIn(Fraction(10, 1), pair)
        self.assertIn(Fraction(21, 2), pair)
        # (i) Scene times must be exact: ints are fine, floats refused.
        with self.assertRaises(SheetError):
            tile_times(Fraction(60), [31.0])
        self.assertIn(
            Fraction(31, 1), tile_times(Fraction(60), [31]))

    def test_writes_only_pipeline_sheets(self):
        _require_sheet_tools()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, footage, project = self.make_project(tmp)
            with Catalog.open(settings) as catalog:
                clip = self.add_clip(
                    catalog, footage, "white.mov", "sheet-white-60",
                    (("white", 60),))
                footage_before = self.snapshot(footage)
                project_before = self.snapshot(project)
                with mock.patch.object(
                    socket, "socket",
                    side_effect=AssertionError("no network use"),
                ):
                    result = build_sheets(
                        catalog, settings, runner=CountingRunner())
            self.assertEqual(result.built, 1)
            self.assertEqual(self.snapshot(footage), footage_before)
            after = self.snapshot(project)
            for relative in after:
                if relative not in project_before:
                    self.assertTrue(
                        relative == ".pipeline" or
                        relative.startswith(".pipeline/"),
                        "wrote outside .pipeline/: %s" % relative,
                    )
            self.assertFalse((project / "contact-sheets").exists())
            self.assertFalse((project / "transcripts").exists())
            oversized = [
                relative for relative, (size, _mtime) in after.items()
                if size > 5_000_000
            ]
            self.assertEqual(oversized, [])
            folder = self.sheets_dir(project)
            sheets = sorted(child for child in folder.iterdir()
                            if child.suffix == ".jpg")
            manifests = sorted(child for child in folder.iterdir()
                               if child.suffix == ".json")
            self.assertEqual(len(sheets), 1)
            self.assertEqual(len(manifests), 1)
            sheet_bytes = sheets[0].read_bytes()
            manifests[0].unlink()
            with Catalog.open(settings) as catalog:
                again = build_sheets(
                    catalog, settings, runner=CountingRunner())
            self.assertEqual(again.built, 1)
            self.assertTrue(manifests[0].exists())
            self.assertEqual(sheets[0].read_bytes(), sheet_bytes)
            sheets[0].write_bytes(b"not a contact sheet")
            manifests[0].unlink()
            tampered_bytes = sheets[0].read_bytes()
            tampered_mtime = os.stat(sheets[0]).st_mtime_ns
            with Catalog.open(settings) as catalog:
                conflict = build_sheets(
                    catalog, settings, runner=CountingRunner())
            self.assertEqual(conflict.failed, 1)
            self.assertEqual(
                [entry.status for entry in conflict.entries], ["failed"])
            self.assertIn(sheets[0].name, conflict.entries[0].reason)
            self.assertFalse(manifests[0].exists())
            self.assertEqual(sheets[0].read_bytes(), tampered_bytes)
            self.assertEqual(
                os.stat(sheets[0]).st_mtime_ns, tampered_mtime)

    def test_failures_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            missing = tmp / "missing"
            missing.mkdir()
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                self.assertEqual(main(["--trip", str(missing)]), 1)
            self.assertIn("project.toml", error.getvalue())
            settings, footage, project = self.make_project(tmp)
            error = io.StringIO()
            with mock.patch.dict(os.environ, {"PATH": "/nonexistent"}):
                with contextlib.redirect_stderr(error):
                    self.assertEqual(main(["--trip", str(project)]), 3)
            self.assertIn("ffmpeg", error.getvalue())
            usage = io.StringIO()
            with contextlib.redirect_stderr(usage):
                with self.assertRaises(SystemExit) as exited:
                    main([])
            self.assertEqual(exited.exception.code, 2)
        _require_sheet_tools()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, footage, project = self.make_project(tmp)
            with Catalog.open(settings) as catalog:
                good = self.add_clip(
                    catalog, footage, "good.mov", "sheet-good-60",
                    (("white", 60),))
                broken = footage / "broken.mov"
                broken.write_bytes(b"this is not a video file")
                info = os.stat(broken)
                catalog.put_media(
                    MediaRecord(
                        asset_id="sheet-broken",
                        kind="clip",
                        path=os.fspath(broken),
                        size=info.st_size,
                        mtime_ns=info.st_mtime_ns,
                        inode=info.st_ino,
                        duration=Fraction(5, 1),
                    )
                )
                runner = CountingRunner()
                result = build_sheets(catalog, settings, runner=runner)
            self.assertGreater(len(runner.calls), 0)
            by_path = {entry.path: entry for entry in result.entries}
            self.assertEqual(by_path[os.fspath(good)].status, "built")
            self.assertEqual(by_path[os.fspath(broken)].status, "failed")
            self.assertTrue(by_path[os.fspath(broken)].reason)
            stale_settings, stale_footage, _stale_project = self.make_project(
                tmp, name="stale-only")
            with Catalog.open(stale_settings) as catalog:
                keeper = self.add_clip(
                    catalog, stale_footage, "keeper.mov", "sheet-keeper-60",
                    (("white", 60),))
                keeper_result = build_sheets(
                    catalog, stale_settings, runner=CountingRunner())
                self.assertEqual(keeper_result.built, 1)
                stale_clip = self.add_clip(
                    catalog, stale_footage, "stale.mov", "sheet-stale-60",
                    (("white", 60),))
                # Change the file after its row was stored: the row no
                # longer matches, so the clip is stale.
                with open(stale_clip, "ab") as handle:
                    handle.write(b"\x00" * 64)
                quiet = CountingRunner()
                rerun = build_sheets(
                    catalog, stale_settings, runner=quiet)
            stale_entry = [
                entry for entry in rerun.entries
                if entry.path == os.fspath(stale_clip)
            ]
            self.assertEqual(len(stale_entry), 1)
            self.assertEqual(stale_entry[0].status, "stale")
            self.assertEqual(
                stale_entry[0].reason,
                "changed since the scan; run the scan again")
            keeper_entry = [
                entry for entry in rerun.entries
                if entry.path == os.fspath(keeper)
            ]
            self.assertEqual(len(keeper_entry), 1)
            self.assertEqual(keeper_entry[0].status, "skipped")
            self.assertEqual(len(quiet.calls), 0)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["--trip", str(project)]), 0)
            self.assertIn("built", out.getvalue())


if __name__ == "__main__":
    unittest.main()


