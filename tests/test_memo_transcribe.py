"""Tests for fcp_pipeline.memo, the trip memo transcription step.

Every test runs in a temporary project folder with an invented
project.toml (title "Invented Memo Survey", ``[pipeline]`` timezone
"UTC" and start_date 2020-06-15, ``[media]`` memo pointing at a
synthetic recording in a separate temporary folder standing in for
Voice Memos, and where a test needs the hosted path
``[pipeline.ai.memo_audio]`` provider "openai" and model
"fake-audio-model"). The recording is a small synthetic WAV built with
the standard wave module (a tone), not real speech and not an ffmpeg
output, so most tests run everywhere. The fake hosted client is a tiny
class whose audio.transcriptions.create records the keyword arguments
and returns a deep copy of the recorded invented response (or raises,
or returns a given dict); it is passed as api_client. The fake local
backend is a function that records its calls and returns an invented
whisper_local.Transcription. No test makes a live API call and no test
uses the network: socket is patched to fail and the offline switch is
managed per test as in tests/test_ai.py.
"""

import array
import copy
import hashlib
import io
import json
import math
import os
import shutil
import socket
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path
from unittest import mock

from fcp_pipeline import memo as memo_module
from fcp_pipeline.ai import audio_openai
from fcp_pipeline.ai.audio_openai import (
    AudioFormatError,
    AudioSegment,
    AudioTranscript,
    build_request,
    parse_transcription,
    transcribe_audio,
)
from fcp_pipeline.backends import whisper_local
from fcp_pipeline.backends.whisper_local import BackendMissing
from fcp_pipeline.catalog import Catalog
from fcp_pipeline.memo import (
    MemoError,
    main,
    plan_chunks,
    recordings,
    transcribe_memos,
)
from fcp_pipeline import net as net_module
from fcp_pipeline.net import NetworkGate, OfflineError, set_offline
from fcp_pipeline.trip import load_trip

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "openai" / "transcription-response.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

FAKE_MODEL = "fake-audio-model"
RATE = 16000


def require_tools(*tools):
    """Skip with a reason naming each tool that is missing from PATH."""
    absent = [tool for tool in tools if shutil.which(tool) is None]
    if absent:
        raise unittest.SkipTest(
            "%s is required for this test but was not found on PATH"
            % ", ".join(absent)
        )


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


def write_project(root, memo=None, ai_extra="", offline=None):
    """Write a trip folder with an invented project.toml; return its path."""
    trip = Path(root) / "project"
    trip.mkdir(parents=True, exist_ok=True)
    lines = ['title = "Invented Memo Survey"', "[media]"]
    if memo is not None:
        if isinstance(memo, list):
            lines.append(f"memo = {json.dumps([str(item) for item in memo])}")
        else:
            lines.append(f"memo = {json.dumps(str(memo))}")
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
    "[pipeline.ai.memo_audio]\n"
    'provider = "openai"\n'
    f'model = "{FAKE_MODEL}"\n'
)

LOCAL_TOML = '[pipeline.ai.memo_audio]\nprovider = "local"\n'


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


def invented_local(language="en", text="invented local memo line"):
    """One invented local backend answer: a single 0.0-2.5 s segment."""
    return whisper_local.Transcription(
        language=language,
        segments=(
            whisper_local.Segment(
                start=Fraction(0, 1),
                stop=Fraction(5, 2),
                text=text,
                words=(),
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


class TestMemo(unittest.TestCase):
    def setUp(self):
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

    def make_recording(self, media, name="memo-a.wav", seconds=0.5):
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

    def test_api_response_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "media"
            recording = self.make_recording(media)
            trip = write_project(tmp, memo=recording, ai_extra=OPENAI_TOML)
            settings = load_trip(trip)
            client = FakeAudioClient()
            local = make_local()
            with Catalog.open(settings) as catalog:
                result = transcribe_memos(
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
            self.assertEqual(call["timestamp_granularities"], ["segment"])
            self.assertEqual(call["file"], "memo-a.wav")
            self.assertEqual(call["size"], os.path.getsize(recording))
            self.assertEqual(local.calls, [])
            blob = (trip / "memo" / "memo-a.json").read_bytes()
            document = json.loads(blob.decode("utf-8"))
            self.assertEqual(document["version"], 1)
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
            self.assertEqual(document["chunks"], 1)
            self.assertEqual(
                document["segments"],
                [
                    {
                        "start": 0.0,
                        "end": 3.2,
                        "text": "invented memo opening line",
                    },
                    {
                        "start": 3.2,
                        "end": 6.4,
                        "text": "invented memo second line",
                    },
                ],
            )
            self.assertTrue(blob.endswith(b"\n"))
            log_lines = (
                (trip / ".pipeline" / "network-log.tsv")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual(len(log_lines), 2)
            fields = log_lines[1].split("\t")
            self.assertEqual(
                fields[1:4], ["memo_audio", "openai", FAKE_MODEL]
            )
            self.assertRegex(fields[4], r"^1 audio [\d,]+ bytes$")
            self.assertEqual(fields[5], "sent")
            self.assertNotIn("invented", log_lines[1])
            self.assertNotIn(os.fspath(recording), log_lines[1])

    def test_offline_uses_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "media"
            recording = self.make_recording(media)
            trip = write_project(
                tmp, memo=recording, ai_extra=OPENAI_TOML, offline=True
            )
            settings = load_trip(trip)
            client = FakeAudioClient()
            local = make_local()
            with Catalog.open(settings) as catalog:
                result = transcribe_memos(
                    catalog, settings, api_client=client, local_backend=local
                )
            self.assertEqual(len(local.calls), 1)
            self.assertEqual(client.calls, [])
            document = json.loads(
                (trip / "memo" / "memo-a.json").read_text(encoding="utf-8")
            )
            self.assertEqual(document["engine"], "mlx-whisper")
            (entry,) = result.entries
            self.assertEqual(entry.notes, ("offline: transcribed locally",))
        with self.subTest("process switch on an online project"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(tmp, memo=recording, ai_extra=OPENAI_TOML)
                settings = load_trip(trip)
                client = FakeAudioClient()
                local = make_local()
                set_offline(True)
                try:
                    with Catalog.open(settings) as catalog:
                        result = transcribe_memos(
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
        with self.subTest("direct hosted call refuses before the client"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(tmp, memo=recording, ai_extra=OPENAI_TOML)
                settings = load_trip(trip)
                client = FakeAudioClient()
                gate = NetworkGate(settings, "memo_audio")
                set_offline(True)
                try:
                    with self.assertRaises(OfflineError):
                        transcribe_audio(
                            gate, FAKE_MODEL, recording, client=client
                        )
                finally:
                    set_offline(False)
                self.assertEqual(client.calls, [])

    def test_local_setting_uses_whisper(self):
        for ai_extra in (LOCAL_TOML, ""):
            with self.subTest(table="local" if ai_extra else "absent"):
                with tempfile.TemporaryDirectory() as tmp:
                    media = Path(tmp) / "media"
                    recording = self.make_recording(media)
                    trip = write_project(tmp, memo=recording, ai_extra=ai_extra)
                    settings = load_trip(trip)
                    client = FakeAudioClient()
                    local = make_local()
                    with Catalog.open(settings) as catalog:
                        result = transcribe_memos(
                            catalog,
                            settings,
                            api_client=client,
                            local_backend=local,
                        )
                    self.assertEqual(len(local.calls), 1)
                    self.assertEqual(
                        local.calls[0]["model"], whisper_local.DEFAULT_MODEL
                    )
                    self.assertEqual(client.calls, [])
                    document = json.loads(
                        (trip / "memo" / "memo-a.json")
                        .read_text(encoding="utf-8")
                    )
                    self.assertEqual(document["engine"], "mlx-whisper")
                    self.assertFalse(
                        (trip / ".pipeline" / "network-log.tsv").exists()
                    )
                    self.assertEqual(result.transcribed, 1)

    def test_api_error_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "media"
            recording = self.make_recording(media)
            trip = write_project(tmp, memo=recording, ai_extra=OPENAI_TOML)
            settings = load_trip(trip)
            client = FakeAudioClient(
                error=RuntimeError("invented secret text")
            )
            local = make_local()
            with Catalog.open(settings) as catalog:
                result = transcribe_memos(
                    catalog, settings, api_client=client, local_backend=local
                )
            (entry,) = result.entries
            self.assertEqual(entry.status, "transcribed")
            self.assertEqual(entry.engine, "mlx-whisper")
            self.assertEqual(len(entry.notes), 1)
            (note,) = entry.notes
            self.assertIn("RuntimeError", note)
            self.assertIn("memo-a.wav", note)
            self.assertNotIn("invented secret text", note)
            self.assertEqual(
                note,
                "memo-a.wav: the hosted transcription failed "
                "(RuntimeError); transcribed locally",
            )
            document = json.loads(
                (trip / "memo" / "memo-a.json").read_text(encoding="utf-8")
            )
            self.assertEqual(document["engine"], "mlx-whisper")
            log_lines = (
                (trip / ".pipeline" / "network-log.tsv")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual(len(log_lines), 2)
            self.assertEqual(log_lines[1].split("\t")[5], "failed")
        with self.subTest("main shows the fallback note as a warning"):
            with tempfile.TemporaryDirectory() as tmp:
                trip = write_project(tmp)
                settings = load_trip(trip)
                fallback = memo_module.MemoEntry(
                    name="memo-a.wav",
                    status="transcribed",
                    engine="mlx-whisper",
                    notes=(
                        "memo-a.wav: the hosted transcription failed "
                        "(RuntimeError); transcribed locally",
                    ),
                )
                with Catalog.open(settings):
                    pass
                with mock.patch.object(
                    memo_module,
                    "transcribe_memos",
                    return_value=memo_module.MemoResult(
                        entries=(fallback,)
                    ),
                ):
                    out, err = io.StringIO(), io.StringIO()
                    with redirect_stdout(out), redirect_stderr(err):
                        code = main(
                            ["transcribe", "--trip", os.fspath(trip)]
                        )
                self.assertEqual(code, 0)
                self.assertIn(
                    "warning: memo-a.wav: the hosted transcription failed "
                    "(RuntimeError); transcribed locally",
                    err.getvalue(),
                )
        with self.subTest("answer without segment times falls back"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(tmp, memo=recording, ai_extra=OPENAI_TOML)
                settings = load_trip(trip)
                client = FakeAudioClient(answer={"language": "english"})
                local = make_local()
                with Catalog.open(settings) as catalog:
                    result = transcribe_memos(
                        catalog,
                        settings,
                        api_client=client,
                        local_backend=local,
                    )
                (entry,) = result.entries
                self.assertEqual(entry.status, "transcribed")
                self.assertEqual(entry.engine, "mlx-whisper")
                self.assertIn("AudioFormatError", entry.notes[0])
        with self.subTest("missing key falls back"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(tmp, memo=recording, ai_extra=OPENAI_TOML)
                settings = load_trip(trip)
                local = make_local()
                with mock.patch.dict(os.environ, clear=False):
                    os.environ.pop("OPENAI_API_KEY", None)
                    with mock.patch(
                        "fcp_pipeline.net._read_key",
                        side_effect=RuntimeError("no key in this test"),
                    ):
                        with Catalog.open(settings) as catalog:
                            result = transcribe_memos(
                                catalog, settings, local_backend=local
                            )
                (entry,) = result.entries
                self.assertEqual(entry.status, "transcribed")
                self.assertEqual(entry.engine, "mlx-whisper")
                self.assertIn("RuntimeError", entry.notes[0])
        with self.subTest("missing local package fails the entry"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(tmp, memo=recording, ai_extra=LOCAL_TOML)
                settings = load_trip(trip)
                local = make_local(
                    error=BackendMissing(whisper_local.INSTALL_HINT)
                )
                with Catalog.open(settings) as catalog:
                    result = transcribe_memos(
                        catalog, settings, local_backend=local
                    )
                (entry,) = result.entries
                self.assertEqual(entry.status, "failed")
                self.assertEqual(entry.reason, whisper_local.INSTALL_HINT)

    def test_skips_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "media"
            recording = self.make_recording(media)
            trip = write_project(tmp, memo=recording, ai_extra=LOCAL_TOML)
            settings = load_trip(trip)
            with Catalog.open(settings) as catalog:
                first = transcribe_memos(
                    catalog, settings, local_backend=make_local()
                )
                self.assertEqual(first.transcribed, 1)
                saved_path = trip / "memo" / "memo-a.json"
                before_bytes = saved_path.read_bytes()
                before_mtime = os.stat(saved_path).st_mtime_ns
                again = transcribe_memos(
                    catalog,
                    settings,
                    api_client=FakeAudioClient(),
                    local_backend=make_local(),
                )
                (entry,) = again.entries
                self.assertEqual(entry.status, "skipped")
                self.assertEqual(again.skipped, 1)
                self.assertEqual(saved_path.read_bytes(), before_bytes)
                self.assertEqual(
                    os.stat(saved_path).st_mtime_ns, before_mtime
                )
                with open(recording, "ab") as handle:
                    handle.write(b"\x00")
                digest = hashlib.sha256(recording.read_bytes()).hexdigest()
                third = transcribe_memos(
                    catalog, settings, local_backend=make_local()
                )
                self.assertEqual(third.transcribed, 1)
                second_path = trip / "memo" / f"memo-a-{digest[:8]}.json"
                self.assertTrue(second_path.exists())
                self.assertEqual(saved_path.read_bytes(), before_bytes)
                renamed = trip / "memo" / "renamed-memo.json"
                os.rename(saved_path, renamed)
                os.rename(second_path, saved_path)
                fourth = transcribe_memos(
                    catalog,
                    settings,
                    api_client=FakeAudioClient(),
                    local_backend=make_local(),
                )
                self.assertEqual(fourth.skipped, 1)
                self.assertEqual(
                    sorted(os.listdir(trip / "memo")),
                    sorted(["renamed-memo.json", "memo-a.json"]),
                )

    def test_long_memo_split_at_silence(self):
        rate = RATE
        tone = [10000] * rate
        silence = [0] * rate
        samples = array.array("h", tone + silence + tone + silence + tone)
        pieces = plan_chunks(samples, rate, 80044)
        self.assertEqual(
            pieces, [(0, 24000), (24000, 56000), (56000, 80000)]
        )
        limit = (80044 - 44) // 2
        for first, last in pieces:
            self.assertLessEqual(last - first, limit)
        with self.subTest("a recording that fits gives one piece"):
            self.assertEqual(
                plan_chunks(array.array("h", [7] * 100), rate, 80044),
                [(0, 100)],
            )
        with self.subTest("audio with no silence is hard-cut at the limit"):
            self.assertEqual(
                plan_chunks(array.array("h", [10000] * 80000), rate, 80044),
                [(0, 40000), (40000, 80000)],
            )
        require_tools("ffmpeg")
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "media"
            media.mkdir(parents=True, exist_ok=True)
            recording = media / "memo-long.wav"
            write_wav(recording, list(samples))
            trip = write_project(tmp, memo=recording, ai_extra=OPENAI_TOML)
            settings = load_trip(trip)
            client = FakeAudioClient()
            local = make_local()
            with mock.patch.object(
                memo_module, "MAX_UPLOAD_BYTES", 80044
            ):
                with Catalog.open(settings) as catalog:
                    result = transcribe_memos(
                        catalog,
                        settings,
                        api_client=client,
                        local_backend=local,
                    )
            self.assertEqual(result.transcribed, 1)
            self.assertEqual(len(client.calls), 3)
            names = [call["file"] for call in client.calls]
            self.assertEqual(
                names, ["chunk-001.wav", "chunk-002.wav", "chunk-003.wav"]
            )
            for call in client.calls:
                self.assertLessEqual(call["size"], 80044)
            self.assertEqual(local.calls, [])
            document = json.loads(
                (trip / "memo" / "memo-long.json").read_text(encoding="utf-8")
            )
            self.assertEqual(document["chunks"], 3)
            segments = document["segments"]
            self.assertEqual(len(segments), 6)
            self.assertEqual(
                [(item["start"], item["end"]) for item in segments],
                [
                    (0.0, 3.2),
                    (3.2, 6.4),
                    (1.5, 4.7),
                    (4.7, 7.9),
                    (3.5, 6.7),
                    (6.7, 9.9),
                ],
            )
            scratch = os.path.dirname(client.calls[0]["file_path"])
            self.assertEqual(
                scratch, os.path.dirname(client.calls[1]["file_path"])
            )
            self.assertFalse(os.path.exists(scratch))
            for root in (trip, media, media.parent):
                with self.subTest(root=os.fspath(root)):
                    self.assertFalse(
                        os.path.abspath(scratch).startswith(
                            os.path.abspath(root) + os.sep
                        )
                    )
        with self.subTest("a silent gap cut is used, not a hard cut"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                media.mkdir(parents=True, exist_ok=True)
                recording = media / "memo-quiet.wav"
                write_wav(recording, [0] * 80000)
                trip = write_project(tmp, memo=recording, ai_extra=OPENAI_TOML)
                settings = load_trip(trip)
                client = FakeAudioClient()
                with mock.patch.object(
                    memo_module, "MAX_UPLOAD_BYTES", 80044
                ):
                    with Catalog.open(settings) as catalog:
                        result = transcribe_memos(
                            catalog,
                            settings,
                            api_client=client,
                            local_backend=make_local(),
                        )
                (entry,) = result.entries
                self.assertEqual(entry.status, "transcribed")
                self.assertEqual(entry.notes, ())

    def test_audio_never_copied(self):
        scenarios = []
        with tempfile.TemporaryDirectory() as tmp:
            audio_suffixes = (
                ".wav", ".m4a", ".mp3", ".mp4", ".caf", ".aac", ".flac",
                ".ogg",
            )
            # One project per run kind, each with its own recording.
            for kind, ai_extra, name, seconds in [
                ("api", OPENAI_TOML, "memo-api.wav", 0.5),
                ("local", LOCAL_TOML, "memo-local.wav", 0.5),
            ]:
                base = Path(tmp) / kind
                project_media = base / "voice-memos"
                recording = self.make_recording(
                    project_media, name=name, seconds=seconds
                )
                trip = write_project(base, memo=recording, ai_extra=ai_extra)
                raw = recording.read_bytes()
                before = self.snapshot(project_media)
                settings = load_trip(trip)
                with Catalog.open(settings) as catalog:
                    result = transcribe_memos(
                        catalog,
                        settings,
                        api_client=FakeAudioClient(),
                        local_backend=make_local(),
                    )
                self.assertEqual(result.transcribed, 1)
                scenarios.append((trip, project_media, recording, raw, before))
            # A split run uploads pieces but keeps nothing in the project.
            base = Path(tmp) / "split"
            project_media = base / "voice-memos"
            recording = self.make_recording(
                project_media, name="memo-split.wav", seconds=5.0
            )
            trip = write_project(base, memo=recording, ai_extra=OPENAI_TOML)
            raw = recording.read_bytes()
            before = self.snapshot(project_media)
            settings = load_trip(trip)
            require_tools("ffmpeg")
            with mock.patch.object(memo_module, "MAX_UPLOAD_BYTES", 80044):
                with Catalog.open(settings) as catalog:
                    result = transcribe_memos(
                        catalog,
                        settings,
                        api_client=FakeAudioClient(),
                        local_backend=make_local(),
                    )
            self.assertEqual(result.transcribed, 1)
            scenarios.append((trip, project_media, recording, raw, before))
            for trip, project_media, recording, raw, before in scenarios:
                with self.subTest(trip=trip.name):
                    self.assertEqual(
                        self.snapshot(project_media), before
                    )
                    memo_files = sorted(os.listdir(trip / "memo"))
                    self.assertTrue(memo_files)
                    for name in memo_files:
                        self.assertTrue(name.endswith(".json"))
                    for dirpath, _dirnames, filenames in os.walk(trip):
                        for name in filenames:
                            full = os.path.join(dirpath, name)
                            self.assertNotIn(
                                os.path.splitext(name)[1].lower(),
                                audio_suffixes,
                                f"audio file under the project: {full}",
                            )
                            with open(full, "rb") as handle:
                                content = handle.read()
                            if full.endswith(".json") or full.endswith(".tsv"):
                                self.assertNotIn(raw[:64], content)
                    document = json.loads(
                        (trip / "memo" / memo_files[0]).read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(
                        set(document),
                        {
                            "version", "recording", "engine", "model",
                            "language", "chunks", "segments",
                        },
                    )
                    self.assertEqual(
                        set(document["recording"]), {"path", "sha256"}
                    )
                    self.assertEqual(
                        document["recording"]["path"],
                        os.path.abspath(recording),
                    )
                    self.assertEqual(
                        document["recording"]["sha256"],
                        hashlib.sha256(raw).hexdigest(),
                    )

    def test_settings_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "media"
            first = self.make_recording(media, name="memo-one.wav", seconds=0.5)
            second = self.make_recording(
                media, name="memo-two.wav", seconds=0.6
            )
            trip = write_project(
                tmp, memo=[first, second], ai_extra=LOCAL_TOML
            )
            settings = load_trip(trip)
            self.assertEqual(recordings(settings), [first, second])
            with Catalog.open(settings) as catalog:
                result = transcribe_memos(
                    catalog, settings, local_backend=make_local()
                )
            self.assertEqual(
                result.outcome, "transcribed 2, skipped 0, failed 0"
            )
            self.assertEqual(
                [entry.name for entry in result.entries],
                ["memo-one.wav", "memo-two.wav"],
            )
            self.assertTrue((trip / "memo" / "memo-one.json").exists())
            self.assertTrue((trip / "memo" / "memo-two.json").exists())
        with self.subTest("absent memo transcribes nothing"):
            with tempfile.TemporaryDirectory() as tmp:
                trip = write_project(tmp, ai_extra=LOCAL_TOML)
                settings = load_trip(trip)
                self.assertEqual(recordings(settings), [])
                with Catalog.open(settings) as catalog:
                    result = transcribe_memos(
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
                    'title = "Invented Memo Survey"\n'
                    "[media]\n"
                    "memo = 5\n"
                    "[pipeline]\n"
                    'timezone = "UTC"\n'
                    "start_date = 2020-06-15\n",
                    encoding="utf-8",
                )
                settings = load_trip(trip)
                with self.assertRaises(MemoError) as raised:
                    recordings(settings)
                self.assertIn("[media].memo", str(raised.exception))
                with Catalog.open(settings) as catalog:
                    with self.assertRaises(MemoError):
                        transcribe_memos(
                            catalog, settings, local_backend=make_local()
                        )
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = main(["transcribe", "--trip", os.fspath(trip)])
                self.assertEqual(code, 1)
                self.assertIn("[media].memo", err.getvalue())
        with self.subTest("missing recording fails the entry"):
            with tempfile.TemporaryDirectory() as tmp:
                trip = write_project(
                    tmp,
                    memo=str(Path(tmp) / "media" / "gone.wav"),
                    ai_extra=LOCAL_TOML,
                )
                settings = load_trip(trip)
                local = make_local()
                with Catalog.open(settings) as catalog:
                    result = transcribe_memos(
                        catalog, settings, local_backend=local
                    )
                (entry,) = result.entries
                self.assertEqual(entry.status, "failed")
                self.assertEqual(entry.reason, "recording not found")
                self.assertEqual(local.calls, [])
        with self.subTest("openai without a model stops with exit 2"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media)
                trip = write_project(
                    tmp,
                    memo=recording,
                    ai_extra="[pipeline.ai.memo_audio]\n"
                    'provider = "openai"\n',
                )
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = main(["transcribe", "--trip", os.fspath(trip)])
                self.assertEqual(code, 2)
                self.assertIn(
                    "[pipeline.ai.memo_audio].model", err.getvalue()
                )
        with self.subTest("a colliding name uses the hash suffix"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media, name="memo-a.wav")
                trip = write_project(tmp, memo=recording, ai_extra=LOCAL_TOML)
                settings = load_trip(trip)
                (trip / "memo").mkdir(parents=True, exist_ok=True)
                blocker = trip / "memo" / "memo-a.json"
                blocker.write_bytes(b'{"version": 1}\n')
                with Catalog.open(settings) as catalog:
                    result = transcribe_memos(
                        catalog, settings, local_backend=make_local()
                    )
                (entry,) = result.entries
                self.assertEqual(entry.status, "transcribed")
                digest = hashlib.sha256(recording.read_bytes()).hexdigest()
                self.assertTrue(
                    (trip / "memo" / f"memo-a-{digest[:8]}.json").exists()
                )
                self.assertEqual(blocker.read_bytes(), b'{"version": 1}\n')
        with self.subTest("a second collision fails without overwriting"):
            with tempfile.TemporaryDirectory() as tmp:
                media = Path(tmp) / "media"
                recording = self.make_recording(media, name="memo-a.wav")
                trip = write_project(tmp, memo=recording, ai_extra=LOCAL_TOML)
                settings = load_trip(trip)
                digest = hashlib.sha256(recording.read_bytes()).hexdigest()
                (trip / "memo").mkdir(parents=True, exist_ok=True)
                first = trip / "memo" / "memo-a.json"
                second = trip / "memo" / f"memo-a-{digest[:8]}.json"
                first.write_bytes(b'{"version": 1}\n')
                second.write_bytes(b'{"version": 2}\n')
                with Catalog.open(settings) as catalog:
                    result = transcribe_memos(
                        catalog, settings, local_backend=make_local()
                    )
                (entry,) = result.entries
                self.assertEqual(entry.status, "failed")
                self.assertEqual(entry.reason, "target already exists")
                self.assertEqual(first.read_bytes(), b'{"version": 1}\n')
                self.assertEqual(second.read_bytes(), b'{"version": 2}\n')

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "media"
            recording = self.make_recording(media)
            trip = write_project(tmp, memo=recording, ai_extra=LOCAL_TOML)
            settings = load_trip(trip)
            client = FakeAudioClient()
            local = make_local()
            with Catalog.open(settings) as catalog:
                self.assertEqual(catalog.list_runs(), [])
                before = self.snapshot(trip)
                result = transcribe_memos(
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
                real = transcribe_memos(
                    catalog, settings, local_backend=make_local()
                )
                self.assertEqual(real.transcribed, 1)
                runs = catalog.list_runs()
                self.assertEqual(len(runs), 1)
                self.assertEqual(runs[0].step, "memo-transcribe")
                self.assertEqual(runs[0].outcome, real.outcome)

    def test_audio_module_shapes(self):
        document = copy.deepcopy(FIXTURE)
        parsed = parse_transcription(document)
        self.assertIsInstance(parsed, AudioTranscript)
        self.assertEqual(parsed.language, "english")
        self.assertEqual(
            parsed.segments,
            (
                AudioSegment(
                    start=Fraction(0, 1),
                    end=Fraction(16, 5),
                    text="invented memo opening line",
                ),
                AudioSegment(
                    start=Fraction(16, 5),
                    end=Fraction(32, 5),
                    text="invented memo second line",
                ),
            ),
        )
        with self.subTest("an object with model_dump parses the same"):
            class Wrapped:
                def __init__(self, payload):
                    self._payload = payload

                def model_dump(self):
                    return copy.deepcopy(self._payload)

            self.assertEqual(parse_transcription(Wrapped(document)), parsed)
        cases = {
            "missing segments": {},
            "empty segments": {"segments": []},
            "segments not a list": {"segments": "nope"},
            "start after end": {
                "segments": [{"start": 3.2, "end": 1.0, "text": "late"}]
            },
            "boolean time": {
                "segments": [{"start": True, "end": 1.0, "text": "yes"}]
            },
            "NaN time": {
                "segments": [
                    {"start": float("nan"), "end": 1.0, "text": "vague"}
                ]
            },
            "infinite time": {
                "segments": [
                    {"start": 0.0, "end": float("inf"), "text": "far"}
                ]
            },
            "non-string text": {
                "segments": [{"start": 0.0, "end": 1.0, "text": 7}]
            },
            "non-object segment": {"segments": ["nope"]},
            "missing text": {"segments": [{"start": 0.0, "end": 1.0}]},
        }
        for name, payload in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(AudioFormatError):
                    parse_transcription(copy.deepcopy(payload))
        with self.subTest("pinned request body"):
            self.assertEqual(
                build_request(FAKE_MODEL),
                {
                    "model": FAKE_MODEL,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment"],
                },
            )

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
        with self.subTest("no memo table transcribes nothing"):
            with tempfile.TemporaryDirectory() as tmp:
                trip = write_project(tmp)
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = main(
                        ["transcribe", "--trip", os.fspath(trip), "--offline"]
                    )
                self.assertEqual(code, 0)
                self.assertEqual(
                    out.getvalue().strip(), "transcribed 0, skipped 0, failed 0"
                )
        with self.subTest("usage errors raise SystemExit(2)"):
            with self.assertRaises(SystemExit) as raised:
                main([])
            self.assertEqual(raised.exception.code, 2)
            with self.assertRaises(SystemExit) as raised:
                main(["transcribe"])
            self.assertEqual(raised.exception.code, 2)
        with self.subTest("--offline sets the process switch"):
            with tempfile.TemporaryDirectory() as tmp:
                trip = write_project(tmp)
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = main(
                        ["transcribe", "--trip", os.fspath(trip), "--offline"]
                    )
                self.assertEqual(code, 0)
                self.assertTrue(net_module._PROCESS_OFFLINE)


if __name__ == "__main__":
    unittest.main()
