"""Tests for fcp_pipeline.vo, the voiceover transcription step.

Every test runs in a temporary project folder with an invented
project.toml (title "Invented Voiceover Survey", ``[pipeline]``
timezone "UTC" and start_date 2020-06-15, ``[media]`` voiceover
pointing at a synthetic recording in a separate temporary folder
standing in for Voice Memos, and where a test needs the hosted path
``[pipeline.ai.vo_audio]`` provider "openai" and model
"fake-audio-model"). The recording is a small synthetic WAV built with
the standard wave module (a tone), not real speech. The fake hosted
client is a tiny class whose audio.transcriptions.create records the
keyword arguments and returns a deep copy of the recorded invented
response (or raises, or returns a given dict); it is passed as
api_client. The fake local backend is a function that records its
calls and returns an invented whisper_local.Transcription with word
times. No test makes a live API call and no test uses the network:
socket is patched to fail and the offline switch is managed per test
as in tests/test_memo_transcribe.py.
"""

import array
import copy
import hashlib
import io
import json
import math
import os
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path
from unittest import mock

from fcp_pipeline import vo as vo_module
from fcp_pipeline.ai import AiConfigError
from fcp_pipeline.ai.audio_openai import (
    AudioFormatError,
    build_request,
    parse_transcription,
)
from fcp_pipeline.backends import whisper_local
from fcp_pipeline.catalog import Catalog
from fcp_pipeline.vo import (
    VoError,
    main,
    recordings,
    transcribe_voiceover,
)
from fcp_pipeline import net as net_module
from fcp_pipeline.net import set_offline
from fcp_pipeline.trip import load_trip

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "openai" / "vo-words-response.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

FAKE_MODEL = "fake-audio-model"
RATE = 16000


def write_wav(path, samples, rate=RATE):
    """Write mono 16-bit samples as a WAV file."""
    with wave.open(os.fspath(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(array.array("h", samples).tobytes())


def tone_samples(seconds, rate=RATE, freq=440.0, amp=10000):
    """Synthetic tone samples (never real speech)."""
    count = int(seconds * rate)
    return [
        int(amp * math.sin(2.0 * math.pi * freq * index / rate))
        for index in range(count)
    ]


def write_project(root, voiceover=None, ai_extra="", offline=None):
    """Write a trip folder with an invented project.toml; return its path."""
    trip = Path(root) / "project"
    trip.mkdir(parents=True, exist_ok=True)
    lines = ['title = "Invented Voiceover Survey"', "[media]"]
    if voiceover is not None:
        if isinstance(voiceover, list):
            lines.append(f"voiceover = {json.dumps([str(item) for item in voiceover])}")
        else:
            lines.append(f"voiceover = {json.dumps(str(voiceover))}")
    lines.extend(
        ["[pipeline]", 'timezone = "UTC"', "start_date = 2020-06-15"]
    )
    if offline is not None:
        lines.append(f"offline = {'true' if offline else 'false'}")
    (trip / "project.toml").write_text(
        "\n".join(lines) + "\n" + ai_extra, encoding="utf-8"
    )
    return trip


OPENAI_TOML = (
    "[pipeline.ai.vo_audio]\n"
    'provider = "openai"\n'
    f'model = "{FAKE_MODEL}"\n'
)

LOCAL_TOML = '[pipeline.ai.vo_audio]\nprovider = "local"\n'

MEMO_OPENAI_TOML = (
    "[pipeline.ai.memo_audio]\n"
    'provider = "openai"\n'
    f'model = "{FAKE_MODEL}"\n'
)


class FakeAudioClient:
    """Tiny fake hosted client; audio.transcriptions.create records kwargs."""

    def __init__(self, answer=None, error=None):
        self.calls = []
        self._answer = answer
        self._error = error
        self.audio = self._Audio(self)

    class _Audio:
        def __init__(self, client):
            self.transcriptions = client._Transcriptions(client)

    class _Transcriptions:
        def __init__(self, client):
            self._client = client

        def create(self, **kwargs):
            client = self._client
            handle = kwargs.get("file")
            record = dict(kwargs)
            if handle is not None:
                name = getattr(handle, "name", "")
                record["file"] = os.path.basename(name)
                record["file_path"] = os.fspath(name)
                try:
                    position = handle.tell()
                    handle.seek(0, 2)
                    record["size"] = handle.tell()
                    handle.seek(position)
                except OSError:
                    record["size"] = None
            client.calls.append(record)
            if client._error is not None:
                raise client._error
            if client._answer is not None:
                return copy.deepcopy(client._answer)
            return copy.deepcopy(FIXTURE)


def invented_local(language="en", text="invented local voiceover line"):
    """One invented local backend answer: one segment with two words."""
    return whisper_local.Transcription(
        language=language,
        segments=(
            whisper_local.Segment(
                start=Fraction(0, 1),
                stop=Fraction(5, 2),
                text=text,
                words=(
                    whisper_local.Word(
                        start=Fraction(0, 1),
                        stop=Fraction(1, 2),
                        word="invented",
                        probability=None,
                    ),
                    whisper_local.Word(
                        start=Fraction(1, 2),
                        stop=Fraction(1, 1),
                        word="local",
                        probability=None,
                    ),
                ),
            ),
        ),
    )


def make_local(answer=None, error=None):
    """A fake local backend function recording its calls."""

    def fake(path, *, model, initial_prompt):
        fake.calls.append(
            {"path": path, "model": model, "initial_prompt": initial_prompt}
        )
        if fake.error is not None:
            raise fake.error
        return copy.deepcopy(fake.answer)

    fake.calls = []
    fake.answer = answer if answer is not None else invented_local()
    fake.error = error
    return fake


class TestVo(unittest.TestCase):
    def setUp(self):
        # String targets so mock.patch imports socket itself; this module
        # never imports it, and any real use fails the test.
        self._offline = net_module._PROCESS_OFFLINE
        set_offline(False)
        self._patchers = []
        for target in ("socket.socket", "socket.create_connection"):
            patcher = mock.patch(
                target, side_effect=AssertionError(f"real network use: {target}")
            )
            patcher.start()
            self._patchers.append(patcher)

    def tearDown(self):
        for patcher in reversed(self._patchers):
            patcher.stop()
        self._patchers = []
        set_offline(self._offline)

    def make_recording(self, media, name="vo-a.wav", seconds=0.5):
        """Write a small synthetic tone WAV; return its path."""
        media.mkdir(parents=True, exist_ok=True)
        recording = media / name
        write_wav(recording, tone_samples(seconds))
        return recording

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

    def test_word_times_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "media"
            recording = self.make_recording(media)
            trip = write_project(tmp, voiceover=recording, ai_extra=OPENAI_TOML)
            settings = load_trip(trip)
            client = FakeAudioClient()
            local = make_local()
            with Catalog.open(settings) as catalog:
                result = transcribe_voiceover(
                    catalog, settings, api_client=client, local_backend=local
                )
            self.assertEqual(
                result.outcome, "transcribed 1, skipped 0, failed 0"
            )
            self.assertEqual(result.transcribed, 1)
            self.assertEqual(len(client.calls), 1)
            call = client.calls[0]
            self.assertEqual(call["model"], FAKE_MODEL)
            self.assertEqual(call["response_format"], "verbose_json")
            self.assertIn("word", call["timestamp_granularities"])
            self.assertIn("segment", call["timestamp_granularities"])
            self.assertEqual(call["file"], "vo-a.wav")
            self.assertEqual(call["size"], os.path.getsize(recording))
            self.assertEqual(local.calls, [])
            saved = sorted((trip / "transcripts").glob("*.json"))
            self.assertEqual(len(saved), 1)
            blob = saved[0].read_bytes()
            document = json.loads(blob.decode("utf-8"))
            self.assertEqual(document["version"], 1)
            self.assertEqual(document["role"], "voiceover")
            self.assertEqual(
                document["recording"]["path"], os.path.abspath(recording)
            )
            self.assertEqual(
                document["recording"]["sha256"],
                hashlib.sha256(recording.read_bytes()).hexdigest(),
            )
            self.assertEqual(document["engine"], "openai")
            self.assertEqual(document["model"], FAKE_MODEL)
            self.assertEqual(document["language"], "english")
            words = [
                word
                for segment in document["segments"]
                for word in segment["words"]
            ]
            self.assertEqual(len(words), 6)
            for word in words:
                self.assertIsInstance(word["start"], (int, float))
                self.assertIsInstance(word["end"], (int, float))
                self.assertLessEqual(word["start"], word["end"])
            self.assertTrue(blob.endswith(b"\n"))
            with Catalog.open(settings) as catalog:
                rows = [
                    row
                    for row in catalog.list_media()
                    if row.kind == "voiceover"
                ]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0].path, os.path.abspath(recording))
                transcripts = catalog.list_transcripts(rows[0].id)
                self.assertEqual(len(transcripts), 1)
                segments = catalog.list_transcript_segments(transcripts[0].id)
                catalog_words = [
                    word for segment in segments for word in segment.words
                ]
                self.assertEqual(len(catalog_words), 6)
                expected = [
                    (item["word"], item["start"], item["end"])
                    for item in FIXTURE["words"]
                ]
                for record, (text, start, end) in zip(catalog_words, expected):
                    self.assertEqual(record.word, text)
                    self.assertEqual(
                        record.start, Fraction(int(round(start * 1000)), 1000)
                    )
                    self.assertEqual(
                        record.stop, Fraction(int(round(end * 1000)), 1000)
                    )
            log_lines = (
                (trip / ".pipeline" / "network-log.tsv")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual(len(log_lines), 2)
            fields = log_lines[1].split("\t")
            self.assertEqual(
                fields[1:4], ["vo_audio", "openai", FAKE_MODEL]
            )
            self.assertRegex(fields[4], r"^1 audio [\d,]+ bytes$")
            self.assertEqual(fields[5], "sent")
            self.assertNotIn("invented", log_lines[1])
            self.assertNotIn(os.fspath(recording), log_lines[1])

    def test_no_word_times_falls_back_to_local(self):
        answers = {
            "missing words": {
                "language": "english",
                "segments": [
                    {"start": 0.0, "end": 2.5, "text": "no times here"}
                ],
            },
            "empty words": {
                "language": "english",
                "segments": [
                    {"start": 0.0, "end": 2.5, "text": "no times here"}
                ],
                "words": [],
            },
        }
        for label, answer in answers.items():
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as tmp:
                    media = Path(tmp) / "media"
                    recording = self.make_recording(media)
                    trip = write_project(
                        tmp, voiceover=recording, ai_extra=OPENAI_TOML
                    )
                    settings = load_trip(trip)
                    client = FakeAudioClient(answer=answer)
                    local = make_local()
                    with Catalog.open(settings) as catalog:
                        result = transcribe_voiceover(
                            catalog,
                            settings,
                            api_client=client,
                            local_backend=local,
                        )
                    (entry,) = result.entries
                    self.assertEqual(entry.status, "transcribed")
                    self.assertEqual(entry.engine, "mlx-whisper")
                    self.assertEqual(len(client.calls), 1)
                    self.assertEqual(len(local.calls), 1)
                    self.assertEqual(len(entry.notes), 1)
                    self.assertIn("AudioFormatError", entry.notes[0])
                    saved = sorted((trip / "transcripts").glob("*.json"))
                    self.assertEqual(len(saved), 1)
                    document = json.loads(saved[0].read_text(encoding="utf-8"))
                    self.assertEqual(document["engine"], "mlx-whisper")
                    file_words = [
                        word["word"]
                        for segment in document["segments"]
                        for word in segment["words"]
                    ]
                    self.assertEqual(file_words, ["invented", "local"])
                    for segment in document["segments"]:
                        for word in segment["words"]:
                            self.assertLessEqual(word["start"], word["end"])

    def test_offline_local_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "media"
            recording = self.make_recording(media)
            trip = write_project(
                tmp, voiceover=recording, ai_extra=OPENAI_TOML, offline=True
            )
            settings = load_trip(trip)
            client = FakeAudioClient()
            local = make_local()
            with Catalog.open(settings) as catalog:
                result = transcribe_voiceover(
                    catalog, settings, api_client=client, local_backend=local
                )
            self.assertEqual(len(local.calls), 1)
            self.assertEqual(client.calls, [])
            (entry,) = result.entries
            self.assertEqual(entry.status, "transcribed")
            self.assertEqual(entry.notes, ("offline: transcribed locally",))
            log = trip / ".pipeline" / "network-log.tsv"
            if log.exists():
                for line in log.read_text(encoding="utf-8").splitlines()[1:]:
                    self.assertNotEqual(line.split("\t")[-1], "sent")
        with self.subTest("process switch on an online project"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(
                    tmp, voiceover=recording, ai_extra=OPENAI_TOML
                )
                settings = load_trip(trip)
                client = FakeAudioClient()
                local = make_local()
                set_offline(True)
                try:
                    with Catalog.open(settings) as catalog:
                        result = transcribe_voiceover(
                            catalog,
                            settings,
                            api_client=client,
                            local_backend=local,
                        )
                finally:
                    set_offline(False)
                self.assertEqual(len(local.calls), 1)
                self.assertEqual(client.calls, [])
                (entry,) = result.entries
                self.assertEqual(entry.notes, ("offline: transcribed locally",))
                log = trip / ".pipeline" / "network-log.tsv"
                if log.exists():
                    for line in log.read_text(encoding="utf-8").splitlines()[1:]:
                        self.assertNotEqual(line.split("\t")[-1], "sent")

    def test_skips_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "media"
            recording = self.make_recording(media)
            trip = write_project(tmp, voiceover=recording, ai_extra=LOCAL_TOML)
            settings = load_trip(trip)
            with Catalog.open(settings) as catalog:
                first = transcribe_voiceover(
                    catalog, settings, local_backend=make_local()
                )
                self.assertEqual(first.transcribed, 1)
                saved = sorted((trip / "transcripts").glob("*.json"))
                self.assertEqual(len(saved), 1)
                saved_path = saved[0]
                before_bytes = saved_path.read_bytes()
                before_mtime = os.stat(saved_path).st_mtime_ns
                idle_client = FakeAudioClient()
                idle_local = make_local()
                again = transcribe_voiceover(
                    catalog,
                    settings,
                    api_client=idle_client,
                    local_backend=idle_local,
                )
                (entry,) = again.entries
                self.assertEqual(entry.status, "skipped")
                self.assertEqual(idle_client.calls, [])
                self.assertEqual(idle_local.calls, [])
                self.assertEqual(again.skipped, 1)
                self.assertEqual(
                    again.outcome, "transcribed 0, skipped 1, failed 0"
                )
                self.assertEqual(saved_path.read_bytes(), before_bytes)
                self.assertEqual(
                    os.stat(saved_path).st_mtime_ns, before_mtime
                )
                self.assertEqual(
                    sorted((trip / "transcripts").glob("*.json")), [saved_path]
                )
                with open(recording, "ab") as handle:
                    handle.write(b"\x00")
                digest = hashlib.sha256(recording.read_bytes()).hexdigest()
                third = transcribe_voiceover(
                    catalog, settings, local_backend=make_local()
                )
                self.assertEqual(third.transcribed, 1)
                second_path = (
                    trip / "transcripts" / f"vo-a-{digest[:8]}.json"
                )
                self.assertTrue(second_path.exists())
                self.assertEqual(saved_path.read_bytes(), before_bytes)

    def resolved_snapshot(self, root):
        """Map every resolve()d path under root to (size, mtime_ns)."""
        found = {}
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                full = os.path.join(dirpath, name)
                info = os.lstat(full)
                found[str(Path(full).resolve())] = (
                    info.st_size,
                    info.st_mtime_ns,
                )
        return found

    def still_snapshot(self, root):
        """Snapshot minus the catalog database the run itself records in."""
        return {
            path: value
            for path, value in self.resolved_snapshot(root).items()
            if "/.pipeline/catalog.sqlite" not in path
        }

    def test_vo_audio_stays_outside(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "media"
            recording = self.make_recording(media, seconds=0.5)
            trip = write_project(tmp, voiceover=recording, ai_extra=OPENAI_TOML)
            raw = recording.read_bytes()
            before_stat = os.stat(recording)
            before_media = self.snapshot(media)
            settings = load_trip(trip)
            before_trip = self.resolved_snapshot(trip)
            transcripts_dir = str(Path(settings.transcripts).resolve())
            pipeline_dir = str(Path(settings.pipeline).resolve())
            client = FakeAudioClient()
            with Catalog.open(settings) as catalog:
                result = transcribe_voiceover(
                    catalog, settings, api_client=client, local_backend=make_local()
                )
            self.assertEqual(result.transcribed, 1)
            self.assertTrue(recording.exists())
            self.assertEqual(recording.read_bytes(), raw)
            after_stat = os.stat(recording)
            self.assertEqual(after_stat.st_size, before_stat.st_size)
            self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)
            self.assertEqual(self.snapshot(media), before_media)
            after_trip = self.resolved_snapshot(trip)
            new_paths = set(after_trip) - set(before_trip)
            self.assertTrue(new_paths)
            for path in new_paths:
                self.assertTrue(
                    path == transcripts_dir
                    or path.startswith(transcripts_dir + os.sep)
                    or path == pipeline_dir
                    or path.startswith(pipeline_dir + os.sep),
                    f"new file outside transcripts/ and .pipeline/: {path}",
                )
            for path, (size, _mtime) in after_trip.items():
                self.assertLessEqual(size, 5_000_000)
            self.assertFalse((trip / "outline.md").exists())
        with self.subTest("a recording symlinked into transcripts fails"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(tmp, voiceover=recording, ai_extra=LOCAL_TOML)
                (trip / "transcripts").mkdir(parents=True, exist_ok=True)
                link = trip / "transcripts" / "vo-link.wav"
                os.symlink(recording, link)
                (trip / "project.toml").write_text(
                    (trip / "project.toml").read_text(encoding="utf-8").replace(
                        f"voiceover = {json.dumps(str(recording))}",
                        f"voiceover = {json.dumps(str(link))}",
                    ),
                    encoding="utf-8",
                )
                settings = load_trip(trip)
                local = make_local()
                with Catalog.open(settings) as catalog:
                    before = self.still_snapshot(trip)
                    result = transcribe_voiceover(
                        catalog, settings, local_backend=local
                    )
                (entry,) = result.entries
                self.assertEqual(entry.status, "failed")
                self.assertEqual(
                    entry.reason,
                    "recording is inside the project's output folders",
                )
                self.assertEqual(local.calls, [])
                self.assertEqual(self.still_snapshot(trip), before)
        with self.subTest("a symlink outside pointing into transcripts fails"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                media.mkdir(parents=True, exist_ok=True)
                trip = write_project(tmp, voiceover="placeholder", ai_extra=LOCAL_TOML)
                (trip / "transcripts").mkdir(parents=True, exist_ok=True)
                target = trip / "transcripts" / "vo-target.wav"
                write_wav(target, tone_samples(0.5))
                link = media / "vo-link.wav"
                os.symlink(target, link)
                (trip / "project.toml").write_text(
                    (trip / "project.toml").read_text(encoding="utf-8").replace(
                        'voiceover = "placeholder"',
                        f"voiceover = {json.dumps(str(link))}",
                    ),
                    encoding="utf-8",
                )
                settings = load_trip(trip)
                local = make_local()
                with Catalog.open(settings) as catalog:
                    before = self.still_snapshot(trip)
                    result = transcribe_voiceover(
                        catalog, settings, local_backend=local
                    )
                (entry,) = result.entries
                self.assertEqual(entry.status, "failed")
                self.assertEqual(
                    entry.reason,
                    "recording is inside the project's output folders",
                )
                self.assertEqual(local.calls, [])
                self.assertEqual(self.still_snapshot(trip), before)
        with self.subTest("a trip reached through a symlink still refuses"):
            with tempfile.TemporaryDirectory() as tmp:
                real = Path(tmp) / "real"
                real.mkdir()
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(real, voiceover=recording, ai_extra=LOCAL_TOML)
                link_dir = Path(tmp) / "linked"
                os.symlink(real, link_dir)
                linked_trip = link_dir / "project"
                (linked_trip / "transcripts").mkdir(parents=True, exist_ok=True)
                link = linked_trip / "transcripts" / "vo-link.wav"
                os.symlink(recording, link)
                (linked_trip / "project.toml").write_text(
                    (linked_trip / "project.toml").read_text(encoding="utf-8").replace(
                        f"voiceover = {json.dumps(str(recording))}",
                        f"voiceover = {json.dumps(str(link))}",
                    ),
                    encoding="utf-8",
                )
                settings = load_trip(linked_trip)
                local = make_local()
                with Catalog.open(settings) as catalog:
                    before = self.still_snapshot(linked_trip)
                    result = transcribe_voiceover(
                        catalog, settings, local_backend=local
                    )
                (entry,) = result.entries
                self.assertEqual(entry.status, "failed")
                self.assertEqual(
                    entry.reason,
                    "recording is inside the project's output folders",
                )
                self.assertEqual(local.calls, [])
                self.assertEqual(self.still_snapshot(linked_trip), before)

    def test_default_is_local(self):
        for label, ai_extra in (
            ("absent", ""),
            ("local", LOCAL_TOML),
        ):
            with self.subTest(table=label):
                with tempfile.TemporaryDirectory() as tmp:
                    media = Path(tmp) / "media"
                    recording = self.make_recording(media)
                    trip = write_project(
                        tmp, voiceover=recording, ai_extra=ai_extra
                    )
                    settings = load_trip(trip)
                    client = FakeAudioClient()
                    local = make_local()
                    with Catalog.open(settings) as catalog:
                        result = transcribe_voiceover(
                            catalog,
                            settings,
                            api_client=client,
                            local_backend=local,
                        )
                    self.assertEqual(result.transcribed, 1)
                    self.assertEqual(len(local.calls), 1)
                    self.assertEqual(
                        local.calls[0]["model"], whisper_local.DEFAULT_MODEL
                    )
                    self.assertEqual(client.calls, [])
                    document = json.loads(
                        next((trip / "transcripts").glob("*.json")).read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(document["engine"], "mlx-whisper")
        with self.subTest("memo_audio openai leaves vo local"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(
                    tmp, voiceover=recording, ai_extra=MEMO_OPENAI_TOML
                )
                settings = load_trip(trip)
                client = FakeAudioClient()
                local = make_local()
                with Catalog.open(settings) as catalog:
                    result = transcribe_voiceover(
                        catalog,
                        settings,
                        api_client=client,
                        local_backend=local,
                    )
                self.assertEqual(result.transcribed, 1)
                self.assertEqual(len(local.calls), 1)
                self.assertEqual(client.calls, [])

    def test_local_without_words_fails(self):
        answer = whisper_local.Transcription(
            language="en",
            segments=(
                whisper_local.Segment(
                    start=Fraction(0, 1),
                    stop=Fraction(5, 2),
                    text="invented wordless line",
                    words=(),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "media"
            recording = self.make_recording(media)
            trip = write_project(tmp, voiceover=recording, ai_extra=LOCAL_TOML)
            settings = load_trip(trip)
            local = make_local(answer=answer)
            with Catalog.open(settings) as catalog:
                result = transcribe_voiceover(
                    catalog, settings, local_backend=local
                )
            (entry,) = result.entries
            self.assertEqual(entry.status, "failed")
            self.assertEqual(entry.reason, "no word times in the answer")
            self.assertEqual(len(local.calls), 1)
            self.assertFalse((trip / "transcripts").exists())
            with Catalog.open(settings) as catalog:
                self.assertEqual(catalog.list_media(), [])
                for row in catalog.list_media():
                    self.assertEqual(catalog.list_transcripts(row.id), [])

    def test_huge_numbers_refused(self):
        bad_values = [1e308, 10**400, float("inf"), float("nan"), -1.0, True, "x"]
        for value in bad_values:
            label = repr(value)[:40]
            with self.subTest(kind="word start", value=label):
                document = copy.deepcopy(FIXTURE)
                document["words"][2]["start"] = value
                with self.assertRaises(AudioFormatError):
                    parse_transcription(document, require_words=True)
            with self.subTest(kind="word end", value=label):
                document = copy.deepcopy(FIXTURE)
                document["words"][2]["end"] = value
                with self.assertRaises(AudioFormatError):
                    parse_transcription(document, require_words=True)
            with self.subTest(kind="segment end", value=label):
                document = copy.deepcopy(FIXTURE)
                document["segments"][1]["end"] = value
                with self.assertRaises(AudioFormatError):
                    parse_transcription(document, require_words=True)
        for label, words in (
            ("missing words", None),
            ("empty words", []),
            ("non-list words", "nope"),
        ):
            with self.subTest(case=label):
                document = copy.deepcopy(FIXTURE)
                if words is None:
                    document.pop("words", None)
                else:
                    document["words"] = words
                with self.assertRaises(AudioFormatError):
                    parse_transcription(document, require_words=True)
        with self.subTest("huge word end falls back to local"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(
                    tmp, voiceover=recording, ai_extra=OPENAI_TOML
                )
                settings = load_trip(trip)
                document = copy.deepcopy(FIXTURE)
                document["words"][0]["end"] = 1e308
                client = FakeAudioClient(answer=document)
                local = make_local()
                with Catalog.open(settings) as catalog:
                    result = transcribe_voiceover(
                        catalog,
                        settings,
                        api_client=client,
                        local_backend=local,
                    )
                (entry,) = result.entries
                self.assertEqual(entry.status, "transcribed")
                self.assertEqual(len(local.calls), 1)
                self.assertIn("AudioFormatError", entry.notes[0])
        with self.subTest("pinned request bodies"):
            self.assertEqual(
                build_request(FAKE_MODEL),
                {
                    "model": FAKE_MODEL,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment"],
                },
            )
            self.assertEqual(
                build_request(FAKE_MODEL, words=True),
                {
                    "model": FAKE_MODEL,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment", "word"],
                },
            )

    def test_hosted_failure_and_size_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "media"
            recording = self.make_recording(media)
            trip = write_project(tmp, voiceover=recording, ai_extra=OPENAI_TOML)
            settings = load_trip(trip)
            client = FakeAudioClient(error=RuntimeError("invented secret text"))
            local = make_local()
            with Catalog.open(settings) as catalog:
                result = transcribe_voiceover(
                    catalog, settings, api_client=client, local_backend=local
                )
            (entry,) = result.entries
            self.assertEqual(entry.status, "transcribed")
            self.assertEqual(entry.engine, "mlx-whisper")
            self.assertEqual(
                entry.notes,
                (
                    "vo-a.wav: the hosted transcription failed "
                    "(RuntimeError); transcribed locally",
                ),
            )
            document = json.loads(
                next((trip / "transcripts").glob("*.json")).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(document["engine"], "mlx-whisper")
        with self.subTest("an oversize recording never uploads"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(
                    tmp, voiceover=recording, ai_extra=OPENAI_TOML
                )
                settings = load_trip(trip)
                client = FakeAudioClient()
                local = make_local()
                with mock.patch.object(vo_module, "MAX_UPLOAD_BYTES", 10):
                    with Catalog.open(settings) as catalog:
                        result = transcribe_voiceover(
                            catalog,
                            settings,
                            api_client=client,
                            local_backend=local,
                        )
                (entry,) = result.entries
                self.assertEqual(entry.status, "transcribed")
                self.assertEqual(
                    entry.notes,
                    ("vo-a.wav: over the hosted size limit; transcribed locally",),
                )
                self.assertEqual(client.calls, [])
                self.assertEqual(len(local.calls), 1)
        with self.subTest("a local failure after a fallback keeps notes"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(
                    tmp, voiceover=recording, ai_extra=OPENAI_TOML
                )
                settings = load_trip(trip)
                client = FakeAudioClient(
                    error=RuntimeError("invented hosted boom")
                )
                local = make_local(error=RuntimeError("invented local boom"))
                with Catalog.open(settings) as catalog:
                    result = transcribe_voiceover(
                        catalog,
                        settings,
                        api_client=client,
                        local_backend=local,
                    )
                (entry,) = result.entries
                self.assertEqual(entry.status, "failed")
                self.assertEqual(entry.reason, "invented local boom")
                self.assertEqual(len(entry.notes), 1)
                self.assertIn("RuntimeError", entry.notes[0])
                self.assertEqual(
                    sorted((trip / "transcripts").glob("*.json")), []
                )

    def test_settings_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "media"
            first = self.make_recording(media, name="vo-one.wav", seconds=0.5)
            second = self.make_recording(media, name="vo-two.wav", seconds=0.6)
            trip = write_project(
                tmp, voiceover=[first, second], ai_extra=LOCAL_TOML
            )
            settings = load_trip(trip)
            self.assertEqual(recordings(settings), [first, second])
            local = make_local()
            with Catalog.open(settings) as catalog:
                result = transcribe_voiceover(
                    catalog, settings, local_backend=local
                )
            self.assertEqual(
                result.outcome, "transcribed 2, skipped 0, failed 0"
            )
            self.assertEqual(
                [entry.name for entry in result.entries],
                ["vo-one.wav", "vo-two.wav"],
            )
            self.assertEqual(len(local.calls), 2)
            self.assertEqual(
                local.calls[0]["initial_prompt"],
                local.calls[1]["initial_prompt"],
            )
            digest = hashlib.sha256(first.read_bytes()).hexdigest()
            self.assertTrue(
                (trip / "transcripts" / f"vo-one-{digest[:8]}.json").exists()
            )
        with self.subTest("absent voiceover transcribes nothing"):
            with tempfile.TemporaryDirectory() as tmp:
                trip = write_project(tmp, ai_extra=LOCAL_TOML)
                settings = load_trip(trip)
                self.assertEqual(recordings(settings), [])
                with Catalog.open(settings) as catalog:
                    result = transcribe_voiceover(
                        catalog, settings, local_backend=make_local()
                    )
                self.assertEqual(
                    result.outcome, "transcribed 0, skipped 0, failed 0"
                )
                self.assertEqual(result.entries, ())
        with self.subTest("wrong type names the key"):
            with tempfile.TemporaryDirectory() as tmp:
                trip = write_project(tmp, ai_extra=LOCAL_TOML)
                (trip / "project.toml").write_text(
                    'title = "Invented Voiceover Survey"\n'
                    "[media]\n"
                    "voiceover = 5\n"
                    "[pipeline]\n"
                    'timezone = "UTC"\n'
                    "start_date = 2020-06-15\n",
                    encoding="utf-8",
                )
                settings = load_trip(trip)
                with self.assertRaises(VoError) as raised:
                    recordings(settings)
                self.assertIn("[media].voiceover", str(raised.exception))
                with Catalog.open(settings) as catalog:
                    with self.assertRaises(VoError):
                        transcribe_voiceover(
                            catalog, settings, local_backend=make_local()
                        )
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = main(["transcribe", "--trip", os.fspath(trip)])
                self.assertEqual(code, 1)
                self.assertIn("[media].voiceover", err.getvalue())
        with self.subTest("missing recording fails the entry"):
            with tempfile.TemporaryDirectory() as tmp:
                trip = write_project(
                    tmp,
                    voiceover=str(Path(tmp) / "media" / "gone.wav"),
                    ai_extra=LOCAL_TOML,
                )
                settings = load_trip(trip)
                local = make_local()
                with Catalog.open(settings) as catalog:
                    result = transcribe_voiceover(
                        catalog, settings, local_backend=local
                    )
                (entry,) = result.entries
                self.assertEqual(entry.status, "failed")
                self.assertEqual(entry.reason, "recording not found")
                self.assertEqual(local.calls, [])
        with self.subTest("openai without a model raises AiConfigError"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(
                    tmp,
                    voiceover=recording,
                    ai_extra="[pipeline.ai.vo_audio]\n"
                    'provider = "openai"\n',
                )
                settings = load_trip(trip)
                with Catalog.open(settings) as catalog:
                    with self.assertRaises(AiConfigError):
                        transcribe_voiceover(
                            catalog, settings, local_backend=make_local()
                        )
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = main(["transcribe", "--trip", os.fspath(trip)])
                self.assertEqual(code, 2)
                self.assertIn("[pipeline.ai.vo_audio].model", err.getvalue())

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "media"
            recording = self.make_recording(media)
            trip = write_project(tmp, voiceover=recording, ai_extra=LOCAL_TOML)
            settings = load_trip(trip)
            client = FakeAudioClient()
            local = make_local()
            with Catalog.open(settings) as catalog:
                self.assertEqual(catalog.list_runs(), [])
                before = self.snapshot(trip)
                result = transcribe_voiceover(
                    catalog,
                    settings,
                    dry_run=True,
                    api_client=client,
                    local_backend=local,
                )
                (entry,) = result.entries
                self.assertEqual(entry.status, "would-transcribe")
                self.assertEqual(
                    result.outcome,
                    "would-transcribe 1, transcribed 0, skipped 0, failed 0",
                )
                self.assertEqual(client.calls, [])
                self.assertEqual(local.calls, [])
                self.assertEqual(catalog.list_runs(), [])
                self.assertEqual(self.snapshot(trip), before)
            with Catalog.open(settings) as catalog:
                real = transcribe_voiceover(
                    catalog, settings, local_backend=make_local()
                )
                self.assertEqual(real.transcribed, 1)
                runs = catalog.list_runs()
                self.assertEqual(len(runs), 1)
                self.assertEqual(runs[0].step, "vo-transcribe")
                self.assertEqual(runs[0].outcome, real.outcome)

    def test_main_exit_codes(self):
        with self.subTest("missing project.toml returns 1"):
            with tempfile.TemporaryDirectory() as tmp:
                trip = Path(tmp) / "project"
                trip.mkdir()
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = main(["transcribe", "--trip", os.fspath(trip)])
                self.assertEqual(code, 1)
                self.assertIn("error:", err.getvalue())
        with self.subTest("a good local run returns 0"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(tmp, voiceover=recording)
                fake = make_local()
                out, err = io.StringIO(), io.StringIO()
                with mock.patch.object(
                    whisper_local, "transcribe_file", fake
                ):
                    with redirect_stdout(out), redirect_stderr(err):
                        code = main(
                            ["transcribe", "--trip", os.fspath(trip)]
                        )
                self.assertEqual(code, 0)
                self.assertEqual(
                    out.getvalue().strip(), "transcribed 1, skipped 0, failed 0"
                )
                self.assertEqual(len(fake.calls), 1)
        with self.subTest("a failed recording returns 1"):
            with tempfile.TemporaryDirectory() as tmp:
                trip = write_project(
                    tmp,
                    voiceover=str(Path(tmp) / "media" / "gone.wav"),
                    ai_extra=LOCAL_TOML,
                )
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = main(["transcribe", "--trip", os.fspath(trip)])
                self.assertEqual(code, 1)
                self.assertIn("failed: gone.wav: recording not found", out.getvalue())
        with self.subTest("hosted without a model returns 2"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(
                    tmp,
                    voiceover=recording,
                    ai_extra="[pipeline.ai.vo_audio]\n"
                    'provider = "openai"\n',
                )
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = main(["transcribe", "--trip", os.fspath(trip)])
                self.assertEqual(code, 2)
                self.assertIn("[pipeline.ai.vo_audio].model", err.getvalue())
        with self.subTest("--dry-run returns 0 and writes nothing"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(tmp, voiceover=recording, ai_extra=LOCAL_TOML)
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = main(
                        ["transcribe", "--trip", os.fspath(trip), "--dry-run"]
                    )
                self.assertEqual(code, 0)
                self.assertIn("would-transcribe 1", out.getvalue())
                self.assertFalse((trip / "transcripts").exists())
                self.assertFalse(
                    (trip / ".pipeline" / "network-log.tsv").exists()
                )


if __name__ == "__main__":
    unittest.main()
