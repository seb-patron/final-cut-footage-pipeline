"""Tests for fcp_pipeline.transcribe_import, the existing-results importer.

Every test runs in a temporary project folder with an invented
project.toml, a catalog, and placeholder clip files of a few bytes under
a temporary footage folder; rows are added with catalog.put_media using
the file's real size, modification time and inode so catalog.unchanged
is True. Results are small invented JSON files written into a separate
temporary folder, built from tests/fixtures/whisper-json/ files (copied,
never read in place by the module). No speech model, no ffprobe and no
network is ever used.
"""

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

from fcp_pipeline import transcribe as transcribe_module
from fcp_pipeline import transcribe_import as import_module
from fcp_pipeline.backends import whisper_local
from fcp_pipeline.catalog import Catalog, MediaRecord
from fcp_pipeline.transcribe import main
from fcp_pipeline.transcribe_import import (
    ResultFormatError,
    import_results,
    parse_result,
)
from fcp_pipeline.trip import load_trip

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "whisper-json"

CREATED = "2020-06-15T08:30:00Z"


def fail_if_called(name):
    """A callable failing the test when the forbidden call happens."""

    def fail(*args, **kwargs):
        raise AssertionError(f"forbidden call during import: {name}")

    return fail


class TestImport(unittest.TestCase):
    """Each run uses a temporary project with invented media only."""

    def setUp(self):
        self._socket_patches = [
            mock.patch(
                "socket.socket", side_effect=AssertionError("no socket")
            ),
            mock.patch(
                "socket.create_connection",
                side_effect=AssertionError("no connection"),
            ),
        ]
        for patcher in self._socket_patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_project(self, tmp):
        """Build a trip folder whose footage and results folders are temp."""
        footage = tmp / "media" / "footage"
        footage.mkdir(parents=True)
        results = tmp / "results"
        results.mkdir(parents=True)
        project = tmp / "project"
        project.mkdir(parents=True, exist_ok=True)
        (project / "project.toml").write_text(
            'title = "Invented Import Survey"\n'
            "[media]\n"
            f'footage = ["{footage.as_posix()}"]\n'
            "[pipeline]\n"
            'timezone = "UTC"\n'
            "start_date = 2020-06-15\n",
            encoding="utf-8",
        )
        return (
            load_trip(project),
            {"footage": footage, "project": project, "results": results},
        )

    def add_clip(
        self,
        catalog,
        footage,
        name,
        asset_id,
        duration=Fraction(5, 2),
        frame_rate=Fraction(30, 1),
    ):
        """Write invented bytes as a clip and record them with a real stat."""
        destination = footage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"invented-clip-bytes-" + name.encode())
        info = os.stat(destination)
        media_id = catalog.put_media(
            MediaRecord(
                asset_id=asset_id,
                kind="clip",
                path=os.fspath(destination),
                size=info.st_size,
                mtime_ns=info.st_mtime_ns,
                inode=info.st_ino,
                duration=duration,
                frame_rate=frame_rate,
            )
        )
        return destination, media_id

    def fixture_data(self, name):
        """Load one invented fixture result as decoded JSON."""
        return json.loads((FIXTURES / name).read_text(encoding="utf-8"))

    def add_result(self, results, name, data):
        """Write one invented result file into the results folder."""
        (results / name).write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
        return results / name

    def add_fixture_result(self, results, name, fixture, **overrides):
        """Copy one fixture into the results folder with tweaks applied."""
        data = self.fixture_data(fixture)
        data.update(overrides)
        return self.add_result(results, name, data)

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

    def file_snapshot(self, root):
        """Map every relative path under root to (bytes, mtime_ns)."""
        found = {}
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                full = os.path.join(dirpath, name)
                found[os.path.relpath(full, root)] = (
                    Path(full).read_bytes(),
                    os.stat(full).st_mtime_ns,
                )
        return found

    def test_imports_matching_json(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage, project, results = (
                places["footage"],
                places["project"],
                places["results"],
            )
            with Catalog.open(settings) as catalog:
                clip, media_id = self.add_clip(
                    catalog, footage, "harbor-a.mov", "asset-a"
                )
                self.add_fixture_result(
                    results, "harbor-a.json", "harbor-a.json"
                )
                result = import_results(
                    catalog, settings, results, created=CREATED
                )
                (entry,) = result.entries
                self.assertEqual(entry.status, "imported")
                self.assertIsNone(entry.reason)
                self.assertEqual(entry.path, os.path.abspath(clip))
                self.assertEqual(result.imported, 1)
                self.assertEqual(
                    result.outcome,
                    "imported 1, skipped 0, unmatched 0, stale 0, failed 0",
                )
                transcripts = project / "transcripts"
                names = sorted(os.listdir(transcripts))
                self.assertEqual(len(names), 3)
                keys = set()
                for name in names:
                    stem, dot, extension = name.rpartition(".")
                    self.assertTrue(dot)
                    self.assertIn(extension, ("json", "srt", "txt"))
                    self.assertTrue(stem.startswith("harbor-a-"))
                    keys.add(stem[len("harbor-a-"):])
                    self.assertRegex(
                        stem[len("harbor-a-"):], r"^[0-9a-f]{8}$"
                    )
                self.assertEqual(len(keys), 1)
                rows = catalog.list_transcripts(media_id)
                self.assertEqual(len(rows), 3)
                for row in rows:
                    with self.subTest(path=row.path):
                        self.assertEqual(
                            row.engine, "mlx-whisper:imported"
                        )
                        self.assertEqual(row.language, "en")
                        self.assertEqual(row.created, CREATED)
                        self.assertEqual(
                            row.sha256,
                            hashlib.sha256(
                                Path(row.path).read_bytes()
                            ).hexdigest(),
                        )
                json_rows = [
                    row for row in rows if row.path.endswith(".json")
                ]
                (json_row,) = json_rows
                (segment,) = catalog.list_transcript_segments(json_row.id)
                self.assertEqual(segment.idx, 0)
                self.assertEqual(segment.start, Fraction(0, 1))
                self.assertEqual(segment.stop, Fraction(5, 2))
                self.assertEqual(segment.text, "invented harbor dawn")
                first, second = segment.words
                self.assertEqual(first.idx, 0)
                self.assertEqual(first.word, "invented")
                self.assertEqual(first.start, Fraction(0, 1))
                self.assertEqual(first.stop, Fraction(2, 5))
                self.assertIsNone(first.probability)
                self.assertEqual(second.idx, 1)
                self.assertEqual(second.word, "harbor")
                self.assertEqual(second.start, Fraction(1, 2))
                self.assertEqual(second.stop, Fraction(9, 10))
                self.assertEqual(second.probability, 0.85)
                written = json.loads(
                    Path(json_row.path).read_text(encoding="utf-8")
                )
                self.assertEqual(written["model"], "imported")
                runs = catalog.list_runs()
                self.assertEqual(len(runs), 1)
                self.assertEqual(runs[0].step, "transcribe-import")
                self.assertEqual(runs[0].outcome, result.outcome)
            with self.subTest("a second import skips without touching"):
                before = self.file_snapshot(transcripts)
                with Catalog.open(settings) as catalog:
                    rerun = import_results(catalog, settings, results)
                    self.assertEqual(rerun.imported, 0)
                    self.assertEqual(rerun.skipped, 1)
                    self.assertEqual(
                        rerun.outcome,
                        "imported 0, skipped 1, unmatched 0, "
                        "stale 0, failed 0",
                    )
                    self.assertEqual(
                        len(catalog.list_transcripts(media_id)), 3
                    )
                self.assertEqual(self.file_snapshot(transcripts), before)

    def test_duration_mismatch_rejected(self):
        cases = [
            (2.6, "unmatched"),
            (2.533, "imported"),
            (2.54, "unmatched"),
        ]
        for frame_rate in (Fraction(30, 1), Fraction(30000, 1001)):
            for duration, expected in cases:
                with self.subTest(
                    frame_rate=str(frame_rate), duration=duration
                ):
                    with tempfile.TemporaryDirectory() as raw:
                        tmp = Path(raw)
                        settings, places = self.make_project(tmp)
                        footage, project, results = (
                            places["footage"],
                            places["project"],
                            places["results"],
                        )
                        with Catalog.open(settings) as catalog:
                            _, media_id = self.add_clip(
                                catalog,
                                footage,
                                "harbor-a.mov",
                                "asset-a",
                                frame_rate=frame_rate,
                            )
                            self.add_fixture_result(
                                results,
                                "harbor-a.json",
                                "harbor-a.json",
                                duration=duration,
                            )
                            result = import_results(catalog, settings, results)
                            (entry,) = result.entries
                            self.assertEqual(entry.status, expected)
                            if expected == "unmatched":
                                self.assertIn("duration", entry.reason or "")
                                self.assertIn(
                                    "one frame", entry.reason or ""
                                )
                                self.assertEqual(
                                    catalog.list_transcripts(media_id), []
                                )
                                self.assertFalse(
                                    (project / "transcripts").exists()
                                )
                            else:
                                self.assertEqual(
                                    len(catalog.list_transcripts(media_id)),
                                    3,
                                )

    def test_unmatched_listed(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage, project, results = (
                places["footage"],
                places["project"],
                places["results"],
            )
            with Catalog.open(settings) as catalog:
                self.add_clip(
                    catalog, footage, "no-duration.mov", "asset-n"
                )
            self.add_fixture_result(
                results, "no-such-clip.json", "harbor-a.json"
            )
            (results / "broken.json").write_bytes(
                b"{{{ not json at all {{{ invented"
            )
            self.add_result(
                results,
                "no-duration.json",
                {
                    "language": "en",
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 1.5,
                            "text": "invented quiet cove",
                        }
                    ],
                },
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(
                    ["--trip", os.fspath(project), "--import", str(results)]
                )
            self.assertEqual(code, 0)
            lines = out.getvalue().splitlines()
            self.assertEqual(
                lines,
                [
                    "unmatched: broken.json: result is not valid JSON",
                    "unmatched: no-duration.json: "
                    "no duration in the result",
                    "unmatched: no-such-clip.json: "
                    "no clip named no-such-clip in the catalog",
                    "imported 0, skipped 0, unmatched 3, stale 0, failed 0",
                ],
            )
            self.assertNotIn("{{{ not json", out.getvalue())
            with Catalog.open(settings) as catalog:
                self.assertEqual(catalog.list_transcripts(1), [])

    def test_name_matching_rules(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage, _project, results = (
                places["footage"],
                places["project"],
                places["results"],
            )
            with Catalog.open(settings) as catalog:
                harbor_a, harbor_a_id = self.add_clip(
                    catalog, footage, "harbor-a.mov", "asset-a"
                )
                harbor_b, harbor_b_id = self.add_clip(
                    catalog,
                    footage,
                    "harbor-b.mov",
                    "asset-b",
                    duration=Fraction(3, 1),
                )
                twin_a, twin_a_id = self.add_clip(
                    catalog, footage, "a/twin.mov", "asset-twin-a"
                )
                twin_b, _twin_b_id = self.add_clip(
                    catalog,
                    footage,
                    "b/twin.mov",
                    "asset-twin-b",
                    duration=Fraction(4, 1),
                )
                self.add_fixture_result(
                    results, "HARBOR-A.json", "harbor-a.json"
                )
                data_b = self.fixture_data("harbor-a.json")
                data_b["duration"] = 3.0
                data_b["segments"][0]["end"] = 3.0
                self.add_result(results, "harbor-b.mov.json", data_b)
                data_twin = self.fixture_data("harbor-a.json")
                self.add_result(results, "twin.json", data_twin)
                result = import_results(catalog, settings, results)
                by_name = {entry.name: entry for entry in result.entries}
                self.assertEqual(result.imported, 3)
                self.assertEqual(
                    by_name["HARBOR-A.json"].path,
                    os.path.abspath(harbor_a),
                )
                self.assertEqual(
                    by_name["harbor-b.mov.json"].path,
                    os.path.abspath(harbor_b),
                )
                self.assertEqual(
                    by_name["twin.json"].path, os.path.abspath(twin_a)
                )
                self.assertEqual(len(catalog.list_transcripts(harbor_a_id)), 3)
                self.assertEqual(len(catalog.list_transcripts(harbor_b_id)), 3)
                self.assertEqual(len(catalog.list_transcripts(twin_a_id)), 3)
        with self.subTest("two clips with one duration fit"):
            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                settings, places = self.make_project(tmp)
                footage, _project, results = (
                    places["footage"],
                    places["project"],
                    places["results"],
                )
                with Catalog.open(settings) as catalog:
                    self.add_clip(
                        catalog, footage, "c1/twin.mov", "asset-twin-1"
                    )
                    self.add_clip(
                        catalog, footage, "c2/twin.mov", "asset-twin-2"
                    )
                    self.add_fixture_result(
                        results, "twin.json", "harbor-a.json"
                    )
                    result = import_results(catalog, settings, results)
                    (entry,) = result.entries
                    self.assertEqual(entry.status, "unmatched")
                    self.assertIn("more than one clip matches", entry.reason)
        with self.subTest("two results for one clip"):
            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                settings, places = self.make_project(tmp)
                footage, _project, results = (
                    places["footage"],
                    places["project"],
                    places["results"],
                )
                with Catalog.open(settings) as catalog:
                    self.add_clip(
                        catalog, footage, "clip-a.mov", "asset-a"
                    )
                    self.add_fixture_result(
                        results, "clip-a.json", "harbor-a.json"
                    )
                    data = self.fixture_data("harbor-a.json")
                    self.add_result(results, "clip-a.mov.json", data)
                    result = import_results(catalog, settings, results)
                    by_name = {entry.name: entry for entry in result.entries}
                    self.assertEqual(result.imported, 1)
                    self.assertEqual(result.unmatched, 1)
                    self.assertEqual(
                        by_name["clip-a.mov.json"].status, "unmatched"
                    )
                    self.assertIn(
                        "clip already matched by clip-a.json",
                        by_name["clip-a.mov.json"].reason or "",
                    )

    def test_stale_clip_not_imported(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage, project, results = (
                places["footage"],
                places["project"],
                places["results"],
            )
            with Catalog.open(settings) as catalog:
                clip, media_id = self.add_clip(
                    catalog, footage, "harbor-a.mov", "asset-a"
                )
                clip.write_bytes(b"invented-clip-bytes-changed-unrecorded")
                self.add_fixture_result(
                    results, "harbor-a.json", "harbor-a.json"
                )
                result = import_results(catalog, settings, results)
                (entry,) = result.entries
                self.assertEqual(entry.status, "stale")
                self.assertEqual(
                    entry.reason,
                    "changed since the scan; run the scan again",
                )
                self.assertEqual(result.stale, 1)
                self.assertEqual(catalog.list_transcripts(media_id), [])
                self.assertFalse((project / "transcripts").exists())

    def test_parse_rejects_bad_shapes(self):
        good_segments = [
            {"start": 0.0, "end": 2.5, "text": "invented harbor dawn"}
        ]
        cases = [
            ("result", ["not", "an", "object"], "segments"),
            ("segments", {"language": "en"}, "segments"),
            (
                "start",
                {
                    "segments": [
                        {
                            "start": 2.0,
                            "end": 1.0,
                            "text": "invented harbor dawn",
                        }
                    ]
                },
                "start",
            ),
            (
                "start",
                {"segments": [{"start": -1.0, "end": 1.0, "text": "x"}]},
                "start",
            ),
            (
                "start",
                {"segments": [{"start": True, "end": 1.0, "text": "x"}]},
                "start",
            ),
            (
                "start",
                json.loads(
                    '{"segments": [{"start": NaN, "end": 1.0, '
                    '"text": "invented"}], "duration": 1.0}'
                ),
                "start",
            ),
            (
                "end",
                json.loads(
                    '{"segments": [{"start": 0.0, "end": Infinity, '
                    '"text": "invented"}], "duration": 1.0}'
                ),
                "end",
            ),
            (
                "text",
                {"segments": [{"start": 0.0, "end": 1.0, "text": 5}]},
                "text",
            ),
            (
                "probability",
                {
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "text": "invented",
                            "words": [
                                {
                                    "word": "invented",
                                    "start": 0.0,
                                    "end": 0.4,
                                    "probability": 2,
                                }
                            ],
                        }
                    ]
                },
                "probability",
            ),
            (
                "duration",
                {"segments": good_segments, "duration": 0},
                "duration",
            ),
        ]
        for label, data, key in cases:
            with self.subTest(label=label):
                with self.assertRaises(ResultFormatError) as caught:
                    parse_result(data)
                self.assertIn(key, str(caught.exception))
        with self.subTest("extra keys parse with an empty word list"):
            transcription, duration = parse_result(
                self.fixture_data("extra-keys.json")
            )
            self.assertEqual(duration, Fraction(5, 2))
            self.assertEqual(transcription.language, "en")
            self.assertEqual(len(transcription.segments), 1)
            self.assertEqual(len(transcription.segments[0].words), 2)
            plain, plain_duration = parse_result(
                self.fixture_data("no-words.json")
            )
            self.assertEqual(plain_duration, Fraction(3, 2))
            (segment,) = plain.segments
            self.assertEqual(segment.words, ())
        with self.subTest("wrapped shape reads both duration sources"):
            wrapped = self.fixture_data("wrapped.json")
            transcription, duration = parse_result(wrapped)
            self.assertEqual(duration, Fraction(4, 1))
            self.assertEqual(transcription.language, "en")
            self.assertEqual(len(transcription.segments), 1)
            fallback = dict(wrapped)
            del fallback["asset"]
            _, fallback_duration = parse_result(fallback)
            self.assertEqual(fallback_duration, Fraction(4, 1))
            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                settings, places = self.make_project(tmp)
                footage, _project, results = (
                    places["footage"],
                    places["project"],
                    places["results"],
                )
                with Catalog.open(settings) as catalog:
                    self.add_clip(
                        catalog,
                        footage,
                        "invented-harbor-c.mov",
                        "asset-c",
                        duration=Fraction(4, 1),
                    )
                    self.add_result(results, "invented-harbor-c.json", wrapped)
                    failing = dict(wrapped)
                    failing["status"] = "failed"
                    self.add_result(results, "invented-harbor-d.json", failing)
                    no_asset = dict(wrapped)
                    del no_asset["asset"]
                    self.add_clip(
                        catalog,
                        footage,
                        "invented-harbor-e.mov",
                        "asset-e",
                        duration=Fraction(4, 1),
                    )
                    self.add_result(
                        results, "invented-harbor-e.json", no_asset
                    )
                    result = import_results(catalog, settings, results)
                    by_name = {
                        entry.name: entry for entry in result.entries
                    }
                    self.assertEqual(
                        by_name["invented-harbor-c.json"].status, "imported"
                    )
                    self.assertEqual(
                        by_name["invented-harbor-e.json"].status, "imported"
                    )
                    failed_entry = by_name["invented-harbor-d.json"]
                    self.assertEqual(failed_entry.status, "unmatched")
                    self.assertIn("failed", failed_entry.reason or "")

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage, project, results = (
                places["footage"],
                places["project"],
                places["results"],
            )
            with Catalog.open(settings) as catalog:
                _, media_id = self.add_clip(
                    catalog, footage, "harbor-a.mov", "asset-a"
                )
                self.add_fixture_result(
                    results, "harbor-a.json", "harbor-a.json"
                )
                before = self.snapshot(project)
                result = import_results(
                    catalog, settings, results, dry_run=True
                )
                (entry,) = result.entries
                self.assertEqual(entry.status, "would-import")
                self.assertEqual(result.would_import, 1)
                self.assertEqual(
                    result.outcome,
                    "would-import 1, imported 0, skipped 0, "
                    "unmatched 0, stale 0, failed 0",
                )
                self.assertEqual(self.snapshot(project), before)
                self.assertEqual(catalog.list_runs(), [])
                self.assertEqual(
                    catalog.list_transcripts(media_id), []
                )

    def test_no_speech_backend_no_network(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage, project, results = (
                places["footage"],
                places["project"],
                places["results"],
            )
            with Catalog.open(settings) as catalog:
                self.add_clip(
                    catalog, footage, "harbor-a.mov", "asset-a"
                )
            self.add_fixture_result(
                results, "harbor-a.json", "harbor-a.json"
            )
            out = io.StringIO()
            with mock.patch.object(
                whisper_local,
                "transcribe_file",
                side_effect=fail_if_called("transcribe_file"),
            ), mock.patch.object(
                whisper_local, "available", return_value=False
            ), mock.patch.object(
                transcribe_module,
                "clip_has_audio",
                side_effect=fail_if_called("clip_has_audio"),
            ), contextlib.redirect_stdout(out):
                code = main(
                    ["--trip", os.fspath(project), "--import", str(results)]
                )
            self.assertEqual(code, 0)
            self.assertEqual(
                out.getvalue().strip(),
                "imported 1, skipped 0, unmatched 0, stale 0, failed 0",
            )

    def test_inputs_read_only(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage, _project, results = (
                places["footage"],
                places["project"],
                places["results"],
            )
            with Catalog.open(settings) as catalog:
                self.add_clip(
                    catalog, footage, "harbor-a.mov", "asset-a"
                )
                self.add_clip(
                    catalog, footage, "harbor-b.mov", "asset-b"
                )
            self.add_fixture_result(
                results, "harbor-a.json", "harbor-a.json"
            )
            footage_before = self.snapshot(footage)
            results_before = self.snapshot(results)
            with Catalog.open(settings) as catalog:
                import_results(catalog, settings, results)
            self.assertEqual(self.snapshot(footage), footage_before)
            self.assertEqual(self.snapshot(results), results_before)
            with Catalog.open(settings) as catalog:
                import_results(catalog, settings, results, dry_run=True)
            self.assertEqual(self.snapshot(footage), footage_before)
            self.assertEqual(self.snapshot(results), results_before)

    def test_oversize_result_unmatched(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage, project, results = (
                places["footage"],
                places["project"],
                places["results"],
            )
            with Catalog.open(settings) as catalog:
                _, media_id = self.add_clip(
                    catalog, footage, "harbor-a.mov", "asset-a"
                )
                data = self.fixture_data("harbor-a.json")
                data["segments"][0]["text"] = "invented word " * 40
                self.add_result(results, "harbor-a.json", data)
                self.assertGreater(
                    (results / "harbor-a.json").stat().st_size, 200
                )
                with mock.patch.object(
                    import_module, "MAX_RESULT_BYTES", 100
                ):
                    result = import_results(catalog, settings, results)
                (entry,) = result.entries
                self.assertEqual(entry.status, "unmatched")
                self.assertIn("100", entry.reason or "")
                self.assertIn("limit", entry.reason or "")
                self.assertEqual(catalog.list_transcripts(media_id), [])
                self.assertFalse((project / "transcripts").exists())

    def test_main_exit_codes(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            with self.subTest("a missing project.toml returns 1"):
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    code = main(["--trip", os.fspath(tmp / "nope")])
                self.assertEqual(code, 1)
                self.assertTrue(errors.getvalue().startswith("error: "))
            settings, places = self.make_project(tmp)
            project, results = places["project"], places["results"]
            with self.subTest("a missing import folder returns 1"):
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    code = main(
                        [
                            "--trip",
                            os.fspath(project),
                            "--import",
                            str(results / "missing"),
                        ]
                    )
                self.assertEqual(code, 1)
                self.assertTrue(errors.getvalue().startswith("error: "))
            with self.subTest("an import file returns 1"):
                marker = results / "marker.txt"
                marker.write_text("invented", encoding="utf-8")
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    code = main(
                        [
                            "--trip",
                            os.fspath(project),
                            "--import",
                            str(marker),
                        ]
                    )
                self.assertEqual(code, 1)
                self.assertTrue(errors.getvalue().startswith("error: "))
            with self.subTest("the old missing-backend code is 2"):
                errors = io.StringIO()
                with mock.patch.object(
                    whisper_local, "available", return_value=False
                ), contextlib.redirect_stderr(errors):
                    code = main(["--trip", os.fspath(project)])
                self.assertEqual(code, 2)
                self.assertIn("pip install mlx-whisper", errors.getvalue())
            with self.subTest("a usage error raises SystemExit(2)"):
                with self.assertRaises(SystemExit) as caught:
                    main([])
                self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
