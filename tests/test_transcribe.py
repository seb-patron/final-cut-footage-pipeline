"""Tests for fcp_pipeline.transcribe, the local footage transcription step.

Every test runs in a temporary project folder with a catalog and
invented media. A fake backend (a plain function returning invented
transcription data and recording its calls) stands in for mlx-whisper;
the real package is never imported. Clips are small placeholder files
or synthetic clips from tests.media_fixtures.make_clip copied into a
temporary footage folder; rows are added with catalog.put_media using
the file's real size, modification time and inode so catalog.unchanged
is True. Tests that need a real ffprobe skip with a reason naming
ffprobe; the others pass an audio_probe and run everywhere.
"""

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

from fcp_pipeline import transcribe as transcribe_module
from fcp_pipeline.backends import whisper_local
from fcp_pipeline.backends.whisper_local import (
    DEFAULT_MODEL,
    ENGINE,
    INSTALL_HINT,
    BackendMissing,
    Segment,
    Transcription,
    Word,
)
from fcp_pipeline.catalog import Catalog, CatalogError, MediaRecord
from fcp_pipeline.transcribe import (
    TranscribeError,
    build_initial_prompt,
    clip_has_audio,
    main,
    transcribe_catalog,
)
from fcp_pipeline.trip import load_trip
from tests.media_fixtures import make_clip

REPO_ROOT = Path(__file__).resolve().parents[1]

SMALL_MODEL = "mlx-community/whisper-small-mlx"


def require_tools(*tools):
    """Skip with a reason naming each tool that is missing from PATH."""
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        raise unittest.SkipTest(
            "%s is required for this test but was not found on PATH"
            % ", ".join(missing)
        )


def invented_answer(language="en"):
    """One invented backend answer: 0.0-2.5 s with two words."""
    return Transcription(
        language=language,
        segments=(
            Segment(
                start=Fraction(0, 1),
                stop=Fraction(5, 2),
                text="invented harbor dawn",
                words=(
                    Word(
                        start=Fraction(0, 1),
                        stop=Fraction(2, 5),
                        word="invented",
                        probability=None,
                    ),
                    Word(
                        start=Fraction(1, 2),
                        stop=Fraction(9, 10),
                        word="harbor",
                        probability=0.9,
                    ),
                ),
            ),
        ),
    )


class FakeBackend:
    """A plain callable standing in for mlx-whisper."""

    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer if answer is not None else invented_answer()

    def __call__(self, path, *, model, initial_prompt):
        self.calls.append(
            {"path": path, "model": model, "initial_prompt": initial_prompt}
        )
        return self.answer


class TestTranscribe(unittest.TestCase):
    """Each run uses a temporary project with invented media only."""

    def make_project(self, tmp, extra_toml="", places=()):
        """Build a trip folder whose footage folder is a temp folder."""
        footage = tmp / "media" / "footage"
        footage.mkdir(parents=True)
        project = tmp / "project"
        project.mkdir(parents=True, exist_ok=True)
        place_tables = "".join(
            f'[[pipeline.places]]\nname = "{name}"\n' for name in places
        )
        (project / "project.toml").write_text(
            "title = \"Invented Transcribe Survey\"\n"
            "[media]\n"
            f"footage = [\"{footage.as_posix()}\"]\n"
            "[pipeline]\n"
            "timezone = \"UTC\"\n"
            "start_date = 2020-06-15\n"
            f"{extra_toml}"
            f"{place_tables}",
            encoding="utf-8",
        )
        return load_trip(project), {"footage": footage, "project": project}

    def add_clip(self, catalog, footage, name, asset_id, camera=None, **kwargs):
        """Copy a cached synthetic clip into the footage folder and record it."""
        destination = footage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(make_clip(**kwargs), destination)
        return destination, self.record_clip(
            catalog, destination, asset_id, camera=camera
        )

    def add_placeholder(self, catalog, footage, name, asset_id, camera=None):
        """Write invented bytes as a clip and record them with a real stat."""
        destination = footage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"invented-clip-bytes-" + name.encode())
        return destination, self.record_clip(
            catalog, destination, asset_id, camera=camera
        )

    def record_clip(self, catalog, destination, asset_id, camera=None):
        """Record a footage file with its real stat so unchanged is True."""
        info = os.stat(destination)
        return catalog.put_media(
            MediaRecord(
                asset_id=asset_id,
                kind="clip",
                path=os.fspath(destination),
                size=info.st_size,
                mtime_ns=info.st_mtime_ns,
                inode=info.st_ino,
                camera=camera,
            )
        )

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

    def test_skips_no_audio(self):
        require_tools("ffmpeg", "ffprobe")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage = places["footage"]
            with Catalog.open(settings) as catalog:
                silent, _ = self.add_clip(
                    catalog, footage, "silent.mov", "asset-silent",
                    audio=None,
                )
                tone, _ = self.add_clip(
                    catalog, footage, "tone.mov", "asset-tone",
                    audio="tone",
                )
                fake = FakeBackend()
                result = transcribe_catalog(
                    catalog, settings, backend=fake
                )
                self.assertEqual(len(fake.calls), 1)
                self.assertEqual(
                    fake.calls[0]["path"], os.path.abspath(tone)
                )
                by_path = {entry.path: entry for entry in result.entries}
                self.assertEqual(
                    by_path[os.path.abspath(silent)].status, "no-audio"
                )
                self.assertEqual(
                    by_path[os.path.abspath(silent)].files, ()
                )
                self.assertEqual(
                    by_path[os.path.abspath(tone)].status, "transcribed"
                )
                self.assertEqual(result.no_audio, 1)
                self.assertEqual(result.transcribed, 1)
        with self.subTest("a stub probe denying everything calls nothing"):
            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                settings, places = self.make_project(tmp)
                footage = places["footage"]
                with Catalog.open(settings) as catalog:
                    self.add_placeholder(
                        catalog, footage, "clip-a.mov", "asset-a"
                    )
                    self.add_placeholder(
                        catalog, footage, "clip-b.mov", "asset-b"
                    )
                    fake = FakeBackend()
                    result = transcribe_catalog(
                        catalog,
                        settings,
                        backend=fake,
                        audio_probe=lambda path: False,
                    )
                    self.assertEqual(fake.calls, [])
                    self.assertEqual(result.no_audio, 2)
                    self.assertEqual(result.transcribed, 0)

    def test_stores_word_times(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage = places["footage"]
            created = "2020-06-15T08:30:00Z"
            answer = Transcription(
                language="en",
                segments=(
                    Segment(
                        start=Fraction(0, 1),
                        stop=Fraction(5, 2),
                        text="invented hello there",
                        words=(
                            Word(
                                start=Fraction(0, 1),
                                stop=Fraction(2, 5),
                                word="hello",
                                probability=None,
                            ),
                            Word(
                                start=Fraction(1, 2),
                                stop=Fraction(9, 10),
                                word="there",
                                probability=0.8,
                            ),
                        ),
                    ),
                ),
            )
            with Catalog.open(settings) as catalog:
                clip, media_id = self.add_placeholder(
                    catalog, footage, "clip-a.mov", "asset-a"
                )
                fake = FakeBackend(answer=answer)
                result = transcribe_catalog(
                    catalog,
                    settings,
                    backend=fake,
                    audio_probe=lambda path: True,
                    created=created,
                )
                self.assertEqual(result.transcribed, 1)
                rows = catalog.list_transcripts(media_id)
                self.assertEqual(len(rows), 3)
                json_rows = [row for row in rows if row.path.endswith(".json")]
                (json_row,) = json_rows
                for row in rows:
                    with self.subTest(path=row.path):
                        self.assertEqual(
                            row.engine, f"{ENGINE}:{DEFAULT_MODEL}"
                        )
                        self.assertEqual(row.language, "en")
                        self.assertEqual(row.created, created)
                        self.assertRegex(row.sha256, r"^[0-9a-f]{64}$")
                        self.assertEqual(
                            row.sha256,
                            hashlib.sha256(
                                Path(row.path).read_bytes()
                            ).hexdigest(),
                        )
                (segment,) = catalog.list_transcript_segments(json_row.id)
                self.assertEqual(segment.start, Fraction(0, 1))
                self.assertEqual(segment.stop, Fraction(5, 2))
                first, second = segment.words
                self.assertEqual(first.word, "hello")
                self.assertEqual(first.start, Fraction(0, 1))
                self.assertEqual(first.stop, Fraction(2, 5))
                self.assertIsNone(first.probability)
                self.assertEqual(second.word, "there")
                self.assertEqual(second.start, Fraction(1, 2))
                self.assertEqual(second.stop, Fraction(9, 10))
                self.assertEqual(second.probability, 0.8)
            with self.subTest("a word starting after its stop fails the clip"):
                with tempfile.TemporaryDirectory() as raw2:
                    tmp2 = Path(raw2)
                    settings2, places2 = self.make_project(tmp2)
                    footage2 = places2["footage"]
                    with Catalog.open(settings2) as catalog2:
                        bad_answer = Transcription(
                            language="en",
                            segments=(
                                Segment(
                                    start=Fraction(0, 1),
                                    stop=Fraction(5, 2),
                                    text="invented bad order",
                                    words=(
                                        Word(
                                            start=Fraction(9, 10),
                                            stop=Fraction(1, 2),
                                            word="bad",
                                            probability=None,
                                        ),
                                    ),
                                ),
                            ),
                        )
                        clip2, media_id2 = self.add_placeholder(
                            catalog2, footage2, "clip-a.mov", "asset-a"
                        )
                        bad = FakeBackend(answer=bad_answer)
                        result2 = transcribe_catalog(
                            catalog2,
                            settings2,
                            backend=bad,
                            audio_probe=lambda path: True,
                            created=created,
                        )
                        self.assertEqual(result2.failed, 1)
                        (entry,) = result2.entries
                        self.assertEqual(entry.status, "failed")
                        self.assertEqual(entry.files, ())
                        self.assertEqual(
                            catalog2.list_transcripts(media_id2), []
                        )

    def test_writes_transcript_files(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage = places["footage"]
            project = places["project"]
            with Catalog.open(settings) as catalog:
                self.add_placeholder(
                    catalog, footage, "clip-a.mov", "asset-a"
                )
                self.add_placeholder(
                    catalog, footage, "clip-b.mov", "asset-b"
                )
                footage_before = self.snapshot(footage)
                fake = FakeBackend()
                result = transcribe_catalog(
                    catalog,
                    settings,
                    backend=fake,
                    audio_probe=lambda path: True,
                )
                self.assertEqual(result.transcribed, 2)
                transcripts = project / "transcripts"
                names = sorted(os.listdir(transcripts))
                self.assertEqual(len(names), 6)
                for clip_stem in ("clip-a", "clip-b"):
                    with self.subTest(clip=clip_stem):
                        owned = sorted(
                            name
                            for name in names
                            if name.startswith(clip_stem + "-")
                        )
                        self.assertEqual(len(owned), 3)
                        keys = {
                            re.fullmatch(
                                re.escape(clip_stem)
                                + r"-([0-9a-f]{8})\.(json|srt|txt)",
                                name,
                            )
                            for name in owned
                        }
                        self.assertTrue(all(match is not None for match in keys))
                        self.assertEqual(
                            {match.group(1) for match in keys},  # type: ignore[union-attr]
                            {next(iter(keys)).group(1)},  # type: ignore[union-attr]
                        )
                        suffixes = sorted(
                            match.group(2)  # type: ignore[union-attr]
                            for match in keys
                        )
                        self.assertEqual(suffixes, ["json", "srt", "txt"])
                expected_srt = (
                    "1\n00:00:00,000 --> 00:00:02,500\n"
                    "invented harbor dawn\n"
                )
                expected_txt = "invented harbor dawn\n"
                expected_json = {
                    "version": 1,
                    "engine": ENGINE,
                    "model": DEFAULT_MODEL,
                    "language": "en",
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 2.5,
                            "text": "invented harbor dawn",
                            "words": [
                                {
                                    "word": "invented",
                                    "start": 0.0,
                                    "end": 0.4,
                                    "probability": None,
                                },
                                {
                                    "word": "harbor",
                                    "start": 0.5,
                                    "end": 0.9,
                                    "probability": 0.9,
                                },
                            ],
                        }
                    ],
                }
                for name in names:
                    with self.subTest(file=name):
                        blob = (transcripts / name).read_bytes()
                        self.assertLessEqual(len(blob), 5_000_000)
                        if name.endswith(".srt"):
                            self.assertEqual(blob.decode("utf-8"), expected_srt)
                        elif name.endswith(".txt"):
                            self.assertEqual(blob.decode("utf-8"), expected_txt)
                        else:
                            self.assertEqual(
                                json.loads(blob.decode("utf-8")), expected_json
                            )
                with self.subTest("nothing else was created"):
                    for dirpath, _dirnames, filenames in os.walk(project):
                        for name in filenames:
                            relative = os.path.relpath(
                                os.path.join(dirpath, name), project
                            )
                            self.assertTrue(
                                relative == "project.toml"
                                or relative.startswith("transcripts/")
                                or relative.startswith(".pipeline/"),
                                f"unexpected file {relative}",
                            )
                self.assertEqual(self.snapshot(footage), footage_before)
                with self.subTest("a rerun writes nothing and calls nothing"):
                    again = FakeBackend()
                    rerun = transcribe_catalog(
                        catalog,
                        settings,
                        backend=again,
                        audio_probe=lambda path: True,
                    )
                    self.assertEqual(again.calls, [])
                    self.assertEqual(rerun.skipped, 2)
                    self.assertEqual(
                        sorted(os.listdir(transcripts)), names
                    )

    def test_vocab_in_prompt(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp, places=("Invented Cove",))
            footage = places["footage"]
            with Catalog.open(settings) as catalog:
                self.add_placeholder(
                    catalog,
                    footage,
                    "clip-a.mov",
                    "asset-a",
                    camera="Invented Maker Cam A",
                )
                self.add_placeholder(
                    catalog,
                    footage,
                    "clip-b.mov",
                    "asset-b",
                    camera="Invented Maker Cam B",
                )
                self.add_placeholder(
                    catalog,
                    footage,
                    "clip-c.mov",
                    "asset-c",
                    camera="Invented Maker Cam A",
                )
                fake = FakeBackend()
                transcribe_catalog(
                    catalog,
                    settings,
                    backend=fake,
                    audio_probe=lambda path: True,
                )
                self.assertTrue(fake.calls)
                for call in fake.calls:
                    with self.subTest(call=call["path"]):
                        self.assertEqual(
                            call["initial_prompt"],
                            "Gear: Invented Maker Cam A, "
                            "Invented Maker Cam B. "
                            "Places: Invented Cove.",
                        )
                        self.assertEqual(call["model"], DEFAULT_MODEL)
        with self.subTest("no places leaves the gear part alone"):
            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                settings, places = self.make_project(tmp)
                footage = places["footage"]
                with Catalog.open(settings) as catalog:
                    self.add_placeholder(
                        catalog,
                        footage,
                        "clip-a.mov",
                        "asset-a",
                        camera="Invented Maker Cam A",
                    )
                    fake = FakeBackend()
                    transcribe_catalog(
                        catalog,
                        settings,
                        backend=fake,
                        audio_probe=lambda path: True,
                    )
                    (call,) = fake.calls
                    self.assertEqual(
                        call["initial_prompt"], "Gear: Invented Maker Cam A."
                    )
        with self.subTest("a long catalog is cut at a whole name"):
            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                settings, places = self.make_project(tmp)
                with Catalog.open(settings) as catalog:
                    cameras = [
                        f"Invented Maker Cam {number:02d} Harbor Unit"
                        for number in range(40)
                    ]
                    for number, camera in enumerate(cameras):
                        self.add_placeholder(
                            catalog,
                            places["footage"],
                            f"clip-{number:02d}.mov",
                            f"asset-{number:02d}",
                            camera=camera,
                        )
                    text = build_initial_prompt(catalog, settings)
                    self.assertLessEqual(len(text), 600)
                    self.assertTrue(text.endswith("."))
                    last = text[:-1].split(", ")[-1]
                    self.assertIn(last, cameras)
        with self.subTest("the model follows project.toml"):
            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                settings, places = self.make_project(
                    tmp,
                    extra_toml="[pipeline.transcribe]\n"
                    f'model = "{SMALL_MODEL}"\n',
                )
                footage = places["footage"]
                with Catalog.open(settings) as catalog:
                    self.add_placeholder(
                        catalog, footage, "clip-a.mov", "asset-a"
                    )
                    fake = FakeBackend()
                    transcribe_catalog(
                        catalog,
                        settings,
                        backend=fake,
                        audio_probe=lambda path: True,
                    )
                    (call,) = fake.calls
                    self.assertEqual(call["model"], SMALL_MODEL)

    def test_resume_after_crash(self):
        class CrashError(BaseException):
            """A test-only crash the step must never swallow."""

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage = places["footage"]
            project = places["project"]
            with Catalog.open(settings) as catalog:
                clips = []
                for number in range(4):
                    clip, _ = self.add_placeholder(
                        catalog,
                        footage,
                        f"clip-{number}.mov",
                        f"asset-{number}",
                    )
                    clips.append(os.path.abspath(clip))

                class CrashingBackend(FakeBackend):
                    def __call__(self, path, *, model, initial_prompt):
                        self.calls.append(
                            {
                                "path": path,
                                "model": model,
                                "initial_prompt": initial_prompt,
                            }
                        )
                        if len(self.calls) == 3:
                            raise CrashError("invented crash on clip 3")
                        return self.answer

                crashing = CrashingBackend()
                with self.assertRaises(CrashError):
                    transcribe_catalog(
                        catalog,
                        settings,
                        backend=crashing,
                        audio_probe=lambda path: True,
                    )
                self.assertEqual(
                    [call["path"] for call in crashing.calls], clips[:3]
                )
                first_files = {}
                for media_id in (1, 2):
                    rows = catalog.list_transcripts(media_id)
                    self.assertEqual(len(rows), 3)
                    for row in rows:
                        first_files[row.path] = (
                            Path(row.path).read_bytes(),
                            os.stat(row.path).st_mtime_ns,
                        )
                self.assertEqual(catalog.list_transcripts(3), [])
                self.assertEqual(catalog.list_transcripts(4), [])

                second = FakeBackend()
                result = transcribe_catalog(
                    catalog,
                    settings,
                    backend=second,
                    audio_probe=lambda path: True,
                )
                self.assertEqual(
                    [call["path"] for call in second.calls], clips[2:]
                )
                self.assertEqual(result.transcribed, 2)
                self.assertEqual(result.skipped, 2)
                for path, (data, mtime_ns) in first_files.items():
                    with self.subTest(path=path):
                        self.assertEqual(Path(path).read_bytes(), data)
                        self.assertEqual(
                            os.stat(path).st_mtime_ns, mtime_ns
                        )

                final = FakeBackend()
                last = transcribe_catalog(
                    catalog,
                    settings,
                    backend=final,
                    audio_probe=lambda path: True,
                )
                self.assertEqual(final.calls, [])
                self.assertEqual(last.skipped, 4)
                self.assertEqual(last.transcribed, 0)
        with self.subTest("a leftover file moves the next run to -2"):
            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                settings, places = self.make_project(tmp)
                footage = places["footage"]
                project = places["project"]
                with Catalog.open(settings) as catalog:
                    clip, _ = self.add_placeholder(
                        catalog, footage, "clip-a.mov", "asset-a"
                    )
                    info = os.stat(clip)
                    key = transcribe_module._clip_key(
                        os.path.abspath(clip),
                        info.st_size,
                        info.st_mtime_ns,
                        DEFAULT_MODEL,
                    )
                    stem = transcribe_module._safe_stem(os.fspath(clip))
                    transcripts = project / "transcripts"
                    transcripts.mkdir(parents=True, exist_ok=True)
                    leftover = transcripts / f"{stem}-{key}.srt"
                    leftover.write_bytes(b"leftover invented bytes")
                    fake = FakeBackend()
                    result = transcribe_catalog(
                        catalog,
                        settings,
                        backend=fake,
                        audio_probe=lambda path: True,
                    )
                    self.assertEqual(result.transcribed, 1)
                    (entry,) = result.entries
                    self.assertTrue(
                        all(f"-{key}-2." in path for path in entry.files)
                    )
                    self.assertEqual(
                        leftover.read_bytes(), b"leftover invented bytes"
                    )
                    with self.subTest("the suffixed clip is not retried"):
                        names_before = sorted(
                            os.listdir(project / "transcripts")
                        )
                        quiet = FakeBackend()
                        rerun = transcribe_catalog(
                            catalog,
                            settings,
                            backend=quiet,
                            audio_probe=lambda path: True,
                        )
                        self.assertEqual(quiet.calls, [])
                        self.assertEqual(rerun.skipped, 1)
                        self.assertEqual(rerun.transcribed, 0)
                        self.assertEqual(
                            sorted(os.listdir(project / "transcripts")),
                            names_before,
                        )
        with self.subTest("a changed clip is transcribed again"):
            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                settings, places = self.make_project(tmp)
                footage = places["footage"]
                project = places["project"]
                with Catalog.open(settings) as catalog:
                    clip, _ = self.add_placeholder(
                        catalog, footage, "clip-a.mov", "asset-a"
                    )
                    fake = FakeBackend()
                    first = transcribe_catalog(
                        catalog,
                        settings,
                        backend=fake,
                        audio_probe=lambda path: True,
                    )
                    self.assertEqual(first.transcribed, 1)
                    old_names = sorted(os.listdir(project / "transcripts"))
                    self.assertEqual(len(old_names), 3)
                    clip.write_bytes(b"invented-clip-bytes-rewritten-longer")
                    self.record_clip(catalog, clip, "asset-a")
                    second = FakeBackend()
                    rerun = transcribe_catalog(
                        catalog,
                        settings,
                        backend=second,
                        audio_probe=lambda path: True,
                    )
                    self.assertEqual(len(second.calls), 1)
                    self.assertEqual(rerun.transcribed, 1)
                    names = sorted(os.listdir(project / "transcripts"))
                    self.assertEqual(len(names), 6)
                    for name in old_names:
                        with self.subTest(old=name):
                            self.assertIn(name, names)

    def test_missing_backend_gives_install_hint(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            project = places["project"]
            saved = sys.modules.get("mlx_whisper")
            sys.modules["mlx_whisper"] = None
            try:
                self.assertFalse(whisper_local.available())
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    code = main(
                        ["--trip", os.fspath(project)]
                    )
                self.assertEqual(code, 2)
                self.assertIn(INSTALL_HINT, errors.getvalue())
                self.assertFalse((project / ".pipeline").exists())
                self.assertFalse((project / "transcripts").exists())
                with self.subTest("transcribe_file names the install"):
                    with self.assertRaises(BackendMissing) as caught:
                        whisper_local.transcribe_file(
                            project / "clip.mov",
                            model=DEFAULT_MODEL,
                            initial_prompt="",
                        )
                    self.assertIn(INSTALL_HINT, str(caught.exception))
            finally:
                if saved is None:
                    sys.modules.pop("mlx_whisper", None)
                else:
                    sys.modules["mlx_whisper"] = saved
        with self.subTest("a missing ffprobe exits 3 naming ffprobe"):
            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                settings, places = self.make_project(tmp)
                footage = places["footage"]
                project = places["project"]
                with Catalog.open(settings) as catalog:
                    self.add_placeholder(
                        catalog, footage, "clip-a.mov", "asset-a"
                    )
                errors = io.StringIO()
                with mock.patch.object(
                    whisper_local, "available", return_value=True
                ), mock.patch.dict(
                    os.environ, {"PATH": "/nonexistent-test-dir"}
                ), contextlib.redirect_stderr(errors):
                    code = main(["--trip", os.fspath(project)])
                self.assertEqual(code, 3)
                self.assertIn("ffprobe", errors.getvalue())
        with self.subTest("imports stay free of mlx_whisper and socket"):
            code = (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "import fcp_pipeline.transcribe, "
                "fcp_pipeline.backends.whisper_local; "
                "sys.exit(0 if 'mlx_whisper' not in sys.modules "
                "and 'socket' not in sys.modules else 3)"
            )
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code, str(REPO_ROOT)],
                capture_output=True,
                text=True,
                timeout=120,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_no_network(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage = places["footage"]
            with Catalog.open(settings) as catalog:
                self.add_placeholder(
                    catalog, footage, "clip-a.mov", "asset-a"
                )
                self.add_placeholder(
                    catalog, footage, "clip-b.mov", "asset-b"
                )
                fake = FakeBackend()
                with mock.patch(
                    "socket.socket",
                    side_effect=AssertionError("real network use: socket"),
                ), mock.patch(
                    "socket.create_connection",
                    side_effect=AssertionError(
                        "real network use: create_connection"
                    ),
                ):
                    result = transcribe_catalog(
                        catalog,
                        settings,
                        backend=fake,
                        audio_probe=lambda path: True,
                    )
                self.assertEqual(result.transcribed, 2)
                self.assertEqual(len(fake.calls), 2)

    def test_dry_run_changes_nothing(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage = places["footage"]
            project = places["project"]
            with Catalog.open(settings) as catalog:
                self.add_placeholder(
                    catalog, footage, "clip-a.mov", "asset-a"
                )
                self.add_placeholder(
                    catalog, footage, "clip-b.mov", "asset-b"
                )
                fake = FakeBackend()
                result = transcribe_catalog(
                    catalog,
                    settings,
                    backend=fake,
                    audio_probe=lambda path: True,
                    dry_run=True,
                )
                self.assertEqual(fake.calls, [])
                self.assertFalse((project / "transcripts").exists())
                self.assertEqual(catalog.list_transcripts(1), [])
                self.assertEqual(catalog.list_transcripts(2), [])
                self.assertEqual(catalog.list_runs(), [])
                self.assertEqual(
                    [entry.status for entry in result.entries],
                    ["would-transcribe", "would-transcribe"],
                )
                self.assertEqual(result.would_transcribe, 2)

    def test_failed_clip_does_not_stop_run(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage = places["footage"]
            project = places["project"]
            with Catalog.open(settings) as catalog:
                first, first_id = self.add_placeholder(
                    catalog, footage, "clip-a.mov", "asset-a"
                )
                second, second_id = self.add_placeholder(
                    catalog, footage, "clip-b.mov", "asset-b"
                )

                class FlakyBackend(FakeBackend):
                    def __call__(self, path, *, model, initial_prompt):
                        self.calls.append(
                            {
                                "path": path,
                                "model": model,
                                "initial_prompt": initial_prompt,
                            }
                        )
                        if len(self.calls) == 1:
                            raise ValueError("invented backend failure")
                        return self.answer

                flaky = FlakyBackend()
                result = transcribe_catalog(
                    catalog,
                    settings,
                    backend=flaky,
                    audio_probe=lambda path: True,
                )
                self.assertEqual(len(flaky.calls), 2)
                by_path = {entry.path: entry for entry in result.entries}
                failed = by_path[os.path.abspath(first)]
                self.assertEqual(failed.status, "failed")
                self.assertIn("invented backend failure", failed.reason or "")
                self.assertEqual(failed.files, ())
                self.assertEqual(catalog.list_transcripts(first_id), [])
                good = by_path[os.path.abspath(second)]
                self.assertEqual(good.status, "transcribed")
                self.assertEqual(len(good.files), 3)
                self.assertEqual(len(catalog.list_transcripts(second_id)), 3)
                self.assertEqual(
                    result.outcome,
                    "transcribed 1, skipped 0, no-audio 0, "
                    "stale 0, failed 1",
                )
                names = sorted(os.listdir(project / "transcripts"))
                self.assertEqual(len(names), 3)

    def test_backend_transcribe_error_fails_the_clip(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage = places["footage"]
            project = places["project"]
            with Catalog.open(settings) as catalog:
                first, first_id = self.add_placeholder(
                    catalog, footage, "clip-a.mov", "asset-a"
                )
                second, second_id = self.add_placeholder(
                    catalog, footage, "clip-b.mov", "asset-b"
                )

                class UnluckyBackend(FakeBackend):
                    def __call__(self, path, *, model, initial_prompt):
                        self.calls.append(
                            {
                                "path": path,
                                "model": model,
                                "initial_prompt": initial_prompt,
                            }
                        )
                        if len(self.calls) == 1:
                            raise TranscribeError(
                                "invented transcribe failure"
                            )
                        return self.answer

                unlucky = UnluckyBackend()
                result = transcribe_catalog(
                    catalog,
                    settings,
                    backend=unlucky,
                    audio_probe=lambda path: True,
                )
                self.assertEqual(len(unlucky.calls), 2)
                by_path = {entry.path: entry for entry in result.entries}
                failed = by_path[os.path.abspath(first)]
                self.assertEqual(failed.status, "failed")
                self.assertIn(
                    "invented transcribe failure", failed.reason or ""
                )
                self.assertEqual(failed.files, ())
                self.assertEqual(catalog.list_transcripts(first_id), [])
                good = by_path[os.path.abspath(second)]
                self.assertEqual(good.status, "transcribed")
                self.assertEqual(len(good.files), 3)
                self.assertEqual(len(catalog.list_transcripts(second_id)), 3)
                self.assertEqual(
                    result.outcome,
                    "transcribed 1, skipped 0, no-audio 0, "
                    "stale 0, failed 1",
                )
                names = sorted(os.listdir(project / "transcripts"))
                self.assertEqual(len(names), 3)
        with self.subTest("main reports failed clips with exit 0"):
            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                settings, places = self.make_project(tmp)
                footage = places["footage"]
                project = places["project"]
                with Catalog.open(settings) as catalog:
                    self.add_placeholder(
                        catalog, footage, "clip-a.mov", "asset-a"
                    )
                    self.add_placeholder(
                        catalog, footage, "clip-b.mov", "asset-b"
                    )

                class UnluckyBackend(FakeBackend):
                    def __call__(self, path, *, model, initial_prompt):
                        self.calls.append(
                            {
                                "path": path,
                                "model": model,
                                "initial_prompt": initial_prompt,
                            }
                        )
                        if len(self.calls) == 1:
                            raise TranscribeError(
                                "invented transcribe failure"
                            )
                        return self.answer

                unlucky = UnluckyBackend()
                out, errors = io.StringIO(), io.StringIO()
                with mock.patch.object(
                    whisper_local, "available", return_value=True
                ), mock.patch.object(
                    transcribe_module, "clip_has_audio", return_value=True
                ), mock.patch.object(
                    whisper_local, "transcribe_file", unlucky
                ), contextlib.redirect_stdout(
                    out
                ), contextlib.redirect_stderr(errors):
                    code = main(["--trip", os.fspath(project)])
                self.assertEqual(code, 0)
                lines = out.getvalue().splitlines()
                self.assertTrue(
                    any(
                        line.startswith("failed: ")
                        and "invented transcribe failure" in line
                        for line in lines
                    ),
                    out.getvalue(),
                )
                self.assertEqual(
                    lines[-1],
                    "transcribed 1, skipped 0, no-audio 0, "
                    "stale 0, failed 1",
                )

    def test_stale_clip_is_left_alone(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage = places["footage"]
            project = places["project"]
            with Catalog.open(settings) as catalog:
                clip, media_id = self.add_placeholder(
                    catalog, footage, "clip-a.mov", "asset-a"
                )
                clip.write_bytes(b"invented-clip-bytes-changed-unrecorded")
                fake = FakeBackend()
                result = transcribe_catalog(
                    catalog,
                    settings,
                    backend=fake,
                    audio_probe=lambda path: True,
                )
                self.assertEqual(fake.calls, [])
                (entry,) = result.entries
                self.assertEqual(entry.status, "stale")
                self.assertEqual(
                    entry.reason,
                    "changed since the scan; run the scan again",
                )
                self.assertEqual(entry.files, ())
                self.assertEqual(result.stale, 1)
                self.assertEqual(catalog.list_transcripts(media_id), [])
                self.assertFalse((project / "transcripts").exists())
        with self.subTest("a missing file is stale too"):
            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                settings, places = self.make_project(tmp)
                footage = places["footage"]
                with Catalog.open(settings) as catalog:
                    clip, _ = self.add_placeholder(
                        catalog, footage, "clip-a.mov", "asset-a"
                    )
                    clip.unlink()
                    fake = FakeBackend()
                    result = transcribe_catalog(
                        catalog,
                        settings,
                        backend=fake,
                        audio_probe=lambda path: True,
                    )
                    self.assertEqual(fake.calls, [])
                    (entry,) = result.entries
                    self.assertEqual(entry.status, "stale")

    def test_empty_answer_is_recorded_not_retried(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage = places["footage"]
            project = places["project"]
            with Catalog.open(settings) as catalog:
                _, media_id = self.add_placeholder(
                    catalog, footage, "clip-a.mov", "asset-a"
                )
                silent = FakeBackend(
                    answer=Transcription(language=None, segments=())
                )
                result = transcribe_catalog(
                    catalog,
                    settings,
                    backend=silent,
                    audio_probe=lambda path: True,
                )
                self.assertEqual(result.transcribed, 1)
                (entry,) = result.entries
                self.assertEqual(len(entry.files), 3)
                rows = catalog.list_transcripts(media_id)
                self.assertEqual(len(rows), 3)
                by_suffix = {
                    Path(row.path).suffix: Path(row.path).read_bytes()
                    for row in rows
                }
                self.assertEqual(by_suffix[".srt"], b"")
                self.assertEqual(by_suffix[".txt"], b"")
                self.assertEqual(
                    json.loads(by_suffix[".json"].decode("utf-8")),
                    {
                        "version": 1,
                        "engine": ENGINE,
                        "model": DEFAULT_MODEL,
                        "language": None,
                        "segments": [],
                    },
                )
                names_before = sorted(os.listdir(project / "transcripts"))
                quiet = FakeBackend()
                rerun = transcribe_catalog(
                    catalog,
                    settings,
                    backend=quiet,
                    audio_probe=lambda path: True,
                )
                self.assertEqual(quiet.calls, [])
                self.assertEqual(rerun.skipped, 1)
                self.assertEqual(
                    sorted(os.listdir(project / "transcripts")), names_before
                )

    def test_oversize_write_fails_the_clip(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage = places["footage"]
            project = places["project"]
            with Catalog.open(settings) as catalog:
                _, media_id = self.add_placeholder(
                    catalog, footage, "clip-a.mov", "asset-a"
                )
                big = FakeBackend(
                    answer=Transcription(
                        language="en",
                        segments=(
                            Segment(
                                start=Fraction(0, 1),
                                stop=Fraction(1, 1),
                                text="invented-word " * 400_000,
                                words=(),
                            ),
                        ),
                    )
                )
                result = transcribe_catalog(
                    catalog,
                    settings,
                    backend=big,
                    audio_probe=lambda path: True,
                )
                self.assertEqual(result.failed, 1)
                (entry,) = result.entries
                self.assertEqual(entry.status, "failed")
                self.assertEqual(entry.files, ())
                self.assertEqual(catalog.list_transcripts(media_id), [])
                self.assertFalse((project / "transcripts").exists())

    def test_main_reports_problems_and_exit_codes(self):
        require_tools("ffmpeg", "ffprobe")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage = places["footage"]
            project = places["project"]
            first = footage / "clip-a.mov"
            second = footage / "clip-b.mov"
            shutil.copyfile(make_clip(audio="tone"), first)
            shutil.copyfile(make_clip(audio="tone"), second)
            with Catalog.open(settings) as catalog:
                self.record_clip(catalog, first, "asset-a")
                self.record_clip(catalog, second, "asset-b")

            class FlakyBackend(FakeBackend):
                def __call__(self, path, *, model, initial_prompt):
                    self.calls.append(
                        {
                            "path": path,
                            "model": model,
                            "initial_prompt": initial_prompt,
                        }
                    )
                    if len(self.calls) == 1:
                        raise ValueError("invented backend failure")
                    return self.answer

            flaky = FlakyBackend()
            out, errors = io.StringIO(), io.StringIO()
            with mock.patch.object(
                whisper_local, "available", return_value=True
            ), mock.patch.object(
                whisper_local, "transcribe_file", flaky
            ), contextlib.redirect_stdout(
                out
            ), contextlib.redirect_stderr(errors):
                code = main(["--trip", os.fspath(project)])
            self.assertEqual(code, 0)
            lines = out.getvalue().splitlines()
            self.assertTrue(
                any(
                    line.startswith("failed: ")
                    and "invented backend failure" in line
                    for line in lines
                ),
                out.getvalue(),
            )
            self.assertEqual(
                lines[-1],
                "transcribed 1, skipped 0, no-audio 0, stale 0, failed 1",
            )
        with self.subTest("an unreadable trip exits 1"):
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                code = main(["--trip", "/nonexistent-test-trip"])
            self.assertEqual(code, 1)
            self.assertTrue(errors.getvalue().startswith("error: "))

    def test_backend_call_shape(self):
        recorded = {}

        class FakeWhisper:
            @staticmethod
            def transcribe(
                path,
                *,
                path_or_hf_repo,
                word_timestamps,
                initial_prompt,
                verbose,
            ):
                recorded.update(
                    path=path,
                    path_or_hf_repo=path_or_hf_repo,
                    word_timestamps=word_timestamps,
                    initial_prompt=initial_prompt,
                    verbose=verbose,
                )
                return {
                    "language": "en",
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 2.5,
                            "text": "invented hello",
                            "words": [
                                {
                                    "word": "invented",
                                    "start": 0.0,
                                    "end": 0.4,
                                    "probability": None,
                                },
                                {
                                    "word": "hello",
                                    "start": 0.5,
                                    "end": 0.9,
                                    "probability": 0.8,
                                },
                            ],
                        },
                        {
                            "start": 3.0,
                            "end": 4.0,
                            "text": "invented quiet",
                        },
                    ],
                }

        saved = sys.modules.get("mlx_whisper")
        fake_module = type(sys)("mlx_whisper")
        fake_module.transcribe = FakeWhisper.transcribe
        sys.modules["mlx_whisper"] = fake_module
        try:
            answer = whisper_local.transcribe_file(
                Path("/invented/clip-a.mov"),
                model=DEFAULT_MODEL,
                initial_prompt="Gear: Invented Cam.",
            )
        finally:
            if saved is None:
                sys.modules.pop("mlx_whisper", None)
            else:
                sys.modules["mlx_whisper"] = saved
        self.assertEqual(recorded["path"], "/invented/clip-a.mov")
        self.assertEqual(recorded["path_or_hf_repo"], DEFAULT_MODEL)
        self.assertTrue(recorded["word_timestamps"])
        self.assertEqual(recorded["initial_prompt"], "Gear: Invented Cam.")
        self.assertIsNone(recorded["verbose"])
        self.assertEqual(answer.language, "en")
        first, second = answer.segments
        self.assertEqual(first.start, Fraction(0, 1))
        self.assertEqual(first.stop, Fraction(5, 2))
        self.assertEqual(first.words[1].start, Fraction(1, 2))
        self.assertEqual(first.words[1].probability, 0.8)
        self.assertEqual(second.words, ())


if __name__ == "__main__":
    unittest.main()
