"""Tests for fcp_pipeline.scan, the read-only footage scan step."""

import contextlib
import io
import json
import os
import shutil
import socket
import tempfile
import types
import unittest
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from unittest import mock

import fcp_pipeline.probe
import fcp_pipeline.scan
from fcp_pipeline.catalog import Catalog
from fcp_pipeline.probe import ProbeError, ProbeFacts, parse_timecode, probe_file
from fcp_pipeline.scan import (
    CAPTURE_FILE,
    capture_time_for,
    main,
    scan_catalog,
)
from fcp_pipeline.trip import load_trip
from tests.media_fixtures import make_clip

_UTC = timezone.utc


def require_tools(*tools):
    """Skip with a reason naming each tool that is missing from PATH."""
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        raise unittest.SkipTest(
            "%s is required to run the scan but was not found on PATH"
            % ", ".join(missing)
        )


class TestScan(unittest.TestCase):
    """Each scan runs in a temporary project with invented media only."""

    def make_project(self, tmp):
        """Build a trip folder whose footage folder is a temp folder."""
        footage = tmp / "media" / "footage"
        footage.mkdir(parents=True)
        project = tmp / "project"
        project.mkdir(parents=True, exist_ok=True)
        (project / "project.toml").write_text(
            "title = \"Invented Scan Survey\"\n"
            "[media]\n"
            f"footage = [\"{footage.as_posix()}\"]\n"
            "[pipeline]\n"
            "timezone = \"UTC\"\n"
            "start_date = 2020-06-15\n",
            encoding="utf-8",
        )
        return load_trip(project), {"footage": footage, "project": project}

    def add_clip(self, footage, name, **kwargs):
        """Copy a cached synthetic clip into the footage folder."""
        destination = footage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(make_clip(**kwargs), destination)
        return destination

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

    def test_ntsc_rate_stored_as_fraction(self):
        require_tools("ffmpeg", "ffprobe", "exiftool")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            clip = self.add_clip(
                places["footage"], "clip-ntsc.mov",
                seconds=1, rate="30000/1001",
            )
            with Catalog.open(settings) as catalog:
                result = scan_catalog(catalog, settings)
                self.assertEqual(result.scanned, 1)
                row = catalog.get_media(clip)
            self.assertIsNotNone(row)
            assert row is not None
            self.assertIs(type(row.frame_rate), Fraction)
            self.assertEqual(row.frame_rate, Fraction(30000, 1001))
            self.assertIs(type(row.duration), Fraction)
            self.assertEqual(row.duration, Fraction(1001, 1000))
            matches = [
                entry for entry in result.entries
                if entry.path == os.fspath(clip)
            ]
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].audio_streams, 1)
            self.assertEqual(matches[0].width, 64)
            self.assertEqual(matches[0].height, 36)

    def test_start_timecode_one_hour(self):
        with self.subTest("parse only, no media"):
            self.assertEqual(
                parse_timecode("01:00:00:00", Fraction(25)), 3600
            )
            self.assertEqual(
                parse_timecode("00:01:00;02", Fraction(30000, 1001)),
                Fraction(1800, 1) / Fraction(30000, 1001),
            )
            with self.assertRaises(ProbeError):
                parse_timecode("bad", Fraction(25))
        require_tools("ffmpeg", "ffprobe", "exiftool")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            plain = self.add_clip(
                places["footage"], "clip-hour.mov",
                seconds=1, rate="25/1", start_timecode="01:00:00:00",
            )
            ntsc = self.add_clip(
                places["footage"], "clip-ntsc-tc.mov",
                seconds=1, rate="30000/1001",
                start_timecode="01:00:00:00",
            )
            with Catalog.open(settings) as catalog:
                scan_catalog(catalog, settings)
                plain_row = catalog.get_media(plain)
                ntsc_row = catalog.get_media(ntsc)
            assert plain_row is not None and ntsc_row is not None
            with self.subTest("25 fps hour"):
                self.assertEqual(
                    plain_row.start_timecode, Fraction(3600, 1)
                )
            with self.subTest("ntsc hour"):
                self.assertEqual(
                    ntsc_row.start_timecode, Fraction(18018, 5)
                )

    def ffprobe_payload(
        self, *, format_tags=None, video_tags=None, data_tags=(),
        rate="25/1",
    ):
        """Build a fake ffprobe document with the given tag dicts."""
        if video_tags is None:
            video_tags = ({},)
        streams = []
        for tags in video_tags:
            streams.append(
                {
                    "codec_type": "video",
                    "r_frame_rate": rate,
                    "width": 64,
                    "height": 36,
                    "tags": tags,
                }
            )
        for tags in data_tags:
            streams.append({"codec_type": "data", "tags": tags})
        return {
            "streams": streams,
            "format": {"duration": "1.0", "tags": format_tags or {}},
        }

    def probe_with_payload(self, payload):
        """Run probe_file on a stub serving mocked ffprobe output."""
        def fake_runner(cmd, **kwargs):
            if "-show_format" in cmd:
                return types.SimpleNamespace(
                    returncode=0, stdout=json.dumps(payload)
                )
            return types.SimpleNamespace(returncode=0, stdout="[]")

        with mock.patch("shutil.which", return_value="/fake/tool"):
            return probe_file("clip.mov", runner=fake_runner)

    def test_start_timecode_lookup_order(self):
        cases = [
            (
                "format tags only",
                dict(format_tags={"timecode": "01:00:00:00"}),
                Fraction(3600, 1),
            ),
            (
                "video stream tags only",
                dict(video_tags=({"timecode": "01:00:00:00"},)),
                Fraction(3600, 1),
            ),
            (
                "data stream tags only",
                dict(
                    video_tags=({},),
                    data_tags=({"timecode": "01:00:00:00"},),
                ),
                Fraction(3600, 1),
            ),
            (
                "format beats video and data",
                dict(
                    format_tags={"timecode": "00:00:10:00"},
                    video_tags=({"timecode": "01:00:00:00"},),
                    data_tags=({"timecode": "02:00:00:00"},),
                ),
                Fraction(10, 1),
            ),
            (
                "first video beats second video and data",
                dict(
                    video_tags=(
                        {"timecode": "00:00:10:00"},
                        {"timecode": "01:00:00:00"},
                    ),
                    data_tags=({"timecode": "02:00:00:00"},),
                ),
                Fraction(10, 1),
            ),
            (
                "empty video tag falls through to data",
                dict(
                    video_tags=({"timecode": ""},),
                    data_tags=({"timecode": "01:00:00:00"},),
                ),
                Fraction(3600, 1),
            ),
        ]
        for label, kwargs, expected in cases:
            with self.subTest(label):
                facts = self.probe_with_payload(
                    self.ffprobe_payload(**kwargs)
                )
                self.assertEqual(facts.start_timecode, expected)
        with self.subTest("no timecode anywhere"):
            facts = self.probe_with_payload(
                self.ffprobe_payload(video_tags=({},))
            )
            self.assertIsNone(facts.start_timecode)
        with self.subTest("non-string timecode is rejected"):
            with self.assertRaises(ProbeError):
                self.probe_with_payload(
                    self.ffprobe_payload(
                        data_tags=({"timecode": 42},)
                    )
                )

    def test_capture_time_prefers_quicktime_date(self):
        require_tools("ffmpeg", "ffprobe", "exiftool")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            clip = self.add_clip(
                places["footage"], "DJI_20190101120000_0001_D.MP4",
                seconds=1, rate="25/1", container="mp4",
                creation_time=datetime(2020, 6, 15, 8, 30, tzinfo=_UTC),
                make="DJI", model="SyntheticDrone S1",
                location="+00.0000-030.0000/",
            )
            with Catalog.open(settings) as catalog:
                result = scan_catalog(catalog, settings)
                row = catalog.get_media(clip)
            assert row is not None
            self.assertEqual(row.capture_time, "2020-06-15T08:30:00Z")
            self.assertEqual(row.capture_time_source, "quicktime")
            self.assertEqual(row.camera, "DJI SyntheticDrone S1")
            self.assertEqual(row.gps_lat, 0.0)
            self.assertEqual(row.gps_lon, -30.0)
            matches = [
                entry for entry in result.entries
                if entry.path == os.fspath(clip)
            ]
            self.assertEqual(len(matches), 1)
            self.assertTrue(matches[0].capture_verified)

    def test_dji_name_fallback(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            probe = tmp / "probe.txt"
            probe.write_bytes(b"probe")
            facts = ProbeFacts(
                duration=None,
                frame_rate=None,
                start_timecode=None,
                audio_streams=0,
                width=None,
                height=None,
                make=None,
                model=None,
                gps_lat=None,
                gps_lon=None,
                quicktime_created=None,
            )
            with self.subTest("invalid calendar values fall through"):
                text, source, verified = capture_time_for(
                    facts, "DJI_20201340250000_0001_D.MP4", os.stat(probe)
                )
                self.assertEqual(source, CAPTURE_FILE)
                self.assertFalse(verified)
                self.assertRegex(text, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        require_tools("ffmpeg", "ffprobe", "exiftool")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            clip = self.add_clip(
                places["footage"], "DJI_20200102100000_0001_D.MP4",
                seconds=1, rate="25/1", container="mp4",
                creation_time=None,
            )
            with Catalog.open(settings) as catalog:
                scan_catalog(catalog, settings)
                row = catalog.get_media(clip)
            assert row is not None
            self.assertEqual(row.capture_time, "2020-01-02T10:00:00Z")
            self.assertEqual(row.capture_time_source, "filename")

    def test_file_time_only_flagged_unverified(self):
        require_tools("ffmpeg", "ffprobe", "exiftool")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            clip = self.add_clip(
                places["footage"], "clip-plain.mov",
                seconds=1, rate="25/1", creation_time=None,
            )
            known = datetime(2021, 3, 4, 5, 6, 7, tzinfo=_UTC)
            stamp = known.timestamp()
            os.utime(clip, (stamp, stamp))
            with Catalog.open(settings) as catalog:
                result = scan_catalog(catalog, settings)
                row = catalog.get_media(clip)
            assert row is not None
            self.assertEqual(row.capture_time, "2021-03-04T05:06:07Z")
            self.assertEqual(row.capture_time_source, "file")
            matches = [
                entry for entry in result.entries
                if entry.path == os.fspath(clip)
            ]
            self.assertEqual(len(matches), 1)
            self.assertFalse(matches[0].capture_verified)

    def test_second_run_skips_unchanged(self):
        require_tools("ffmpeg", "ffprobe", "exiftool")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            first_clip = self.add_clip(
                places["footage"], "clip-a.mov",
                seconds=1, rate="25/1",
            )
            second_clip = self.add_clip(
                places["footage"], "clip-b.mov",
                seconds=1, rate="30000/1001",
            )
            calls = []

            def counting(path, *, runner=None):
                calls.append(os.fspath(path))
                return probe_file(path, runner=runner)

            with Catalog.open(settings) as catalog:
                first = scan_catalog(catalog, settings)
                self.assertEqual(first.scanned, 2)
                second = scan_catalog(catalog, settings, probe=counting)
                self.assertEqual(calls, [])
                self.assertEqual(second.skipped, 2)
                self.assertEqual(second.scanned, 0)
                replacement = make_clip(seconds=2, rate="25/1")
                shutil.copyfile(replacement, second_clip)
                third = scan_catalog(catalog, settings, probe=counting)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0], os.fspath(second_clip))
                self.assertEqual(third.scanned, 1)
                self.assertEqual(third.skipped, 1)
                row = catalog.get_media(first_clip)
            assert row is not None
            self.assertIsNotNone(row.duration)

    def test_unreadable_file_reported(self):
        require_tools("ffmpeg", "ffprobe", "exiftool")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            good = self.add_clip(
                places["footage"], "clip-good.mov",
                seconds=1, rate="25/1",
            )
            broken = places["footage"] / "broken.mov"
            broken.write_bytes(b"not a video file at all")
            before_bytes = broken.read_bytes()
            before_mtime = os.stat(broken).st_mtime_ns
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = main(["--trip", os.fspath(places["project"])])
            out = buffer.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("unreadable:", out)
            self.assertIn(os.fspath(broken), out)
            with Catalog.open(settings) as catalog:
                good_row = catalog.get_media(good)
                broken_row = catalog.get_media(broken)
                result = scan_catalog(catalog, settings)
            assert good_row is not None and broken_row is not None
            self.assertIsNotNone(good_row.duration)
            self.assertIsNone(broken_row.duration)
            self.assertIn(
                os.fspath(broken), result.unreadable_paths
            )
            self.assertEqual(broken.read_bytes(), before_bytes)
            self.assertEqual(os.stat(broken).st_mtime_ns, before_mtime)

    def test_failed_probe_after_change_stops_run(self):
        require_tools("ffprobe", "exiftool")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            target = places["footage"] / "clip-changing.mov"
            target.write_bytes(b"stub-bytes")

            def mutating_probe(path, *, runner=None):
                info = os.stat(os.fspath(path))
                os.utime(
                    os.fspath(path),
                    (info.st_atime, info.st_mtime + 2),
                )
                raise ProbeError(
                    "cannot read %s: boom" % os.fspath(path)
                )

            with Catalog.open(settings) as catalog:
                with self.assertRaisesRegex(
                    ProbeError, "file changed during scan"
                ):
                    scan_catalog(
                        catalog, settings, probe=mutating_probe
                    )
                self.assertEqual(catalog.list_media(), [])
            with contextlib.redirect_stderr(io.StringIO()):
                with mock.patch.object(
                    fcp_pipeline.scan.scan_catalog,
                    "__kwdefaults__",
                    {"probe": mutating_probe},
                ):
                    code = main(
                        ["--trip", os.fspath(places["project"])]
                    )
            self.assertEqual(code, 1)
            with Catalog.open(settings) as catalog:
                self.assertEqual(catalog.list_media(), [])

    def test_failed_probe_unchanged_is_unreadable(self):
        require_tools("ffprobe", "exiftool")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            target = places["footage"] / "clip-broken.mov"
            target.write_bytes(b"stub-bytes")
            calls = []

            def failing_probe(path, *, runner=None):
                calls.append(os.fspath(path))
                raise ProbeError(
                    "cannot read %s: boom" % os.fspath(path)
                )

            with Catalog.open(settings) as catalog:
                result = scan_catalog(
                    catalog, settings, probe=failing_probe
                )
                self.assertEqual(result.unreadable, 1)
                self.assertEqual(
                    result.unreadable_paths, (os.fspath(target),)
                )
                row = catalog.get_media(target)
                self.assertIsNotNone(row)
                assert row is not None
                self.assertIsNone(row.duration)
                again = scan_catalog(
                    catalog, settings, probe=failing_probe
                )
                self.assertEqual(again.unreadable, 1)
                kept = catalog.get_media(target)
                self.assertIsNotNone(kept)
                assert kept is not None
                self.assertIsNone(kept.duration)
            self.assertEqual(calls, [os.fspath(target)] * 2)

    def test_media_mtime_unchanged(self):
        require_tools("ffmpeg", "ffprobe", "exiftool")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            self.add_clip(
                places["footage"], "clip-a.mov",
                seconds=1, rate="25/1",
            )
            self.add_clip(
                places["footage"], "clip-b.mov",
                seconds=1, rate="30000/1001",
            )
            footage = places["footage"]
            past = datetime(2020, 1, 1, tzinfo=_UTC).timestamp()
            for dirpath, _dirnames, filenames in os.walk(footage):
                for name in filenames:
                    full = os.path.join(dirpath, name)
                    os.utime(full, (past, past))
            before = self.snapshot(footage)

            real_utime = os.utime
            real_rename = os.rename
            real_replace = os.replace
            real_remove = os.remove
            real_move = shutil.move

            def guard(label, original):
                def checked(path, *args, **kwargs):
                    if os.fspath(path).startswith(os.fspath(footage)):
                        self.fail(
                            f"scan must not touch footage: {label} {path}"
                        )
                    return original(path, *args, **kwargs)

                return checked

            patches = [
                mock.patch(
                    "os.utime", side_effect=guard("utime", real_utime)
                ),
                mock.patch(
                    "os.rename", side_effect=guard("rename", real_rename)
                ),
                mock.patch(
                    "os.replace", side_effect=guard("replace", real_replace)
                ),
                mock.patch(
                    "os.remove", side_effect=guard("remove", real_remove)
                ),
                mock.patch(
                    "shutil.move", side_effect=guard("move", real_move)
                ),
            ]
            with Catalog.open(settings) as catalog:
                with contextlib.ExitStack() as stack:
                    for patch in patches:
                        stack.enter_context(patch)
                    scan_catalog(catalog, settings)
                    scan_catalog(catalog, settings)
            self.assertEqual(self.snapshot(footage), before)

    def test_reads_media_inside_fcpbundle(self):
        require_tools("ffmpeg", "ffprobe", "exiftool")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            originals = (
                places["footage"] / "Trip.fcpbundle"
                / "Invented Event" / "Original Media"
            )
            originals.mkdir(parents=True)
            clip = self.add_clip(
                originals, "clip-bundle.mov",
                seconds=1, rate="25/1",
            )
            before_bytes = clip.read_bytes()
            before_mtime = os.stat(clip).st_mtime_ns
            bundle = places["footage"] / "Trip.fcpbundle"
            before = self.snapshot(bundle)
            with Catalog.open(settings) as catalog:
                result = scan_catalog(catalog, settings)
                row = catalog.get_media(clip)
            assert row is not None
            self.assertIsNotNone(row.duration)
            matches = [
                entry for entry in result.entries
                if entry.path == os.fspath(clip)
            ]
            self.assertEqual(len(matches), 1)
            self.assertEqual(clip.read_bytes(), before_bytes)
            self.assertEqual(os.stat(clip).st_mtime_ns, before_mtime)
            self.assertEqual(self.snapshot(bundle), before)

    def test_missing_tool_is_not_unreadable(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            stub = places["footage"] / "stub.mov"
            stub.write_bytes(b"stub-bytes")
            real_which = shutil.which

            def fake_which(command, *args, **kwargs):
                if command == "ffprobe":
                    return None
                return real_which(command, *args, **kwargs)

            target = "fcp_pipeline.probe.shutil.which"
            with mock.patch(target, side_effect=fake_which):
                with Catalog.open(settings) as catalog:
                    with self.assertRaises(ProbeError) as caught:
                        scan_catalog(catalog, settings)
                    self.assertIn("ffprobe", str(caught.exception))
                    self.assertEqual(catalog.list_media(), [])
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(
                        main(["--trip", os.fspath(places["project"])]), 3
                    )

    def test_scan_writes_only_pipeline_folder(self):
        require_tools("ffmpeg", "ffprobe", "exiftool")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            self.add_clip(
                places["footage"], "clip-a.mov",
                seconds=1, rate="25/1",
            )
            before = self.snapshot(tmp)
            with Catalog.open(settings) as catalog:
                scan_catalog(catalog, settings)
            after = self.snapshot(tmp)
            pipeline = os.fspath(places["project"] / ".pipeline")
            for relative in after:
                if relative not in before:
                    full = os.path.join(os.fspath(tmp), relative)
                    self.assertTrue(
                        os.fspath(full).startswith(pipeline + os.sep),
                        f"new file outside .pipeline: {relative}",
                    )
            for relative, values in before.items():
                if relative in after:
                    self.assertEqual(
                        after[relative], values,
                        f"changed file outside scan writes: {relative}",
                    )

    def test_no_network(self):
        for module in (fcp_pipeline.scan, fcp_pipeline.probe):
            source = Path(module.__file__).read_text(encoding="utf-8")
            self.assertNotRegex(
                source,
                r"(?m)^\s*(import|from)\s+"
                r"(socket|ssl|http|urllib\.request|ftplib)\b",
            )
            self.assertNotIn("osascript", source)
        require_tools("ffmpeg", "ffprobe", "exiftool")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            self.add_clip(
                places["footage"], "clip-a.mov",
                seconds=1, rate="25/1",
            )
            with Catalog.open(settings) as catalog:
                with mock.patch(
                    "socket.socket",
                    side_effect=AssertionError("scan must not use sockets"),
                ):
                    scan_catalog(catalog, settings)


if __name__ == "__main__":
    unittest.main()
