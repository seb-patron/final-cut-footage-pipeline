"""Tests for fcp_pipeline.tag_batch, the batch-mode tag step.

No test here makes a real network call, reads a real key, or needs an
installed SDK: socket functions fail when called, sys.modules entries
for openai and anthropic are restored in tearDown, every project is a
temporary folder with an invented project.toml, and sheets are
synthetic header-only JPEGs. Fake backends subclass the real ones and
override only the SDK-touching methods (``create_batch``,
``retrieve_batch``, ``fetch_batch_results``), recording every call;
building and parsing stay real. Clips are catalogued in order so their
media ids are 1, 2, 3, and every name, id and model string is invented.
"""

import base64
import hashlib
import io
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fcp_pipeline import ai, net, tag, tag_batch
from fcp_pipeline.ai import AnthropicBackend, OpenAIBackend
from fcp_pipeline.ai import batch as ai_batch
from fcp_pipeline.catalog import (
    Catalog,
    MediaRecord,
    TagRecord,
    TranscriptRecord,
)
from fcp_pipeline.net import NetworkGate, OfflineError, set_offline
from fcp_pipeline.trip import load_trip

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ai" / "batch"
TAG_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ai" / "tag-responses"

FAKE_OPENAI_MODEL = "fake-openai-batch-model"
FAKE_ANTHROPIC_MODEL = "fake-anthropic-batch-model"

#: Tile times of a 60 s clip: centres of nine equal slices.
TILE_TIMES = (
    Fraction(10, 3),
    Fraction(10, 1),
    Fraction(50, 3),
    Fraction(70, 3),
    Fraction(30, 1),
    Fraction(110, 3),
    Fraction(130, 3),
    Fraction(50, 1),
    Fraction(170, 3),
)

#: Vocabulary used across these tests.
VOCABULARY = {
    "place": ("Invented Cove",),
    "gear": ("Invented Maker Cam A",),
    "subject": ("invented-boat", "invented-gull"),
}

#: Three distinct recorded answers, keyed so a mix-up is visible.
ANSWER_ONE = {
    "description": "Invented dawn watch one",
    "shot": "wide",
    "ranges": [
        {
            "confidence": 0.86,
            "end": 9.0,
            "start": 3.0,
            "tags": [{"folder": "place", "value": "Invented Cove"}],
        }
    ],
    "problems": [],
    "outline_hint": None,
}
ANSWER_TWO = {
    "description": "Invented pier survey two",
    "shot": "medium",
    "ranges": [
        {
            "confidence": 0.62,
            "end": 33.0,
            "start": 25.0,
            "tags": [{"folder": "subject", "value": "invented-boat"}],
        }
    ],
    "problems": [],
    "outline_hint": None,
}
ANSWER_THREE = {
    "description": "Invented gull count three",
    "shot": "close-up",
    "ranges": [
        {
            "confidence": 0.9,
            "end": 12.0,
            "start": 11.0,
            "tags": [{"folder": "subject", "value": "invented-gull"}],
        }
    ],
    "problems": [],
    "outline_hint": None,
}


def load_fixture(name):
    """Return the parsed JSON batch fixture ``name``."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def load_tag_fixture(name):
    """Return the parsed JSON one-clip tag fixture ``name``."""
    return json.loads((TAG_FIXTURES / name).read_text(encoding="utf-8"))


def fake_jpeg(width, height):
    """Return a header-only JPEG ``width`` x ``height`` readable by jpeg_size."""
    return (
        b"\xff\xd8"
        + b"\xff\xc0"
        + struct.pack(">HBHHB", 11, 8, height, width, 1)
        + bytes([1, 0x11, 0])
        + b"\xff\xd9"
    )


def openai_raw(answer):
    """Recorded-style OpenAI Responses result carrying ``answer``."""
    text = answer if isinstance(answer, str) else json.dumps(answer)
    return {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {"input_tokens": 2140, "output_tokens": 210},
    }


def anthropic_raw(answer):
    """Recorded-style Anthropic Messages result carrying ``answer``."""
    text = answer if isinstance(answer, str) else json.dumps(answer)
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 2210, "output_tokens": 205},
    }


def openai_line_succeeded(custom_id, answer):
    """One OpenAI batch output line carrying a 200 answer."""
    return json.dumps(
        {
            "custom_id": custom_id,
            "response": {"status_code": 200, "body": openai_raw(answer)},
            "error": None,
        }
    )


def openai_line_errored(custom_id, code, message, status_code=400):
    """One OpenAI batch output line carrying a provider error."""
    return json.dumps(
        {
            "custom_id": custom_id,
            "response": {"status_code": status_code, "body": None},
            "error": {"code": code, "message": message},
        }
    )


def anthropic_entry_succeeded(custom_id, answer):
    """One Anthropic batch result entry carrying an answer."""
    return {
        "custom_id": custom_id,
        "result": {"type": "succeeded", "message": anthropic_raw(answer)},
    }


def anthropic_entry_errored(custom_id, error_type, message):
    """One Anthropic batch result entry carrying a provider error."""
    return {
        "custom_id": custom_id,
        "result": {
            "type": "errored",
            "error": {"type": error_type, "message": message},
        },
    }


def anthropic_entry_status(custom_id, status):
    """One Anthropic batch result entry with an ``expired``/``canceled`` type."""
    return {"custom_id": custom_id, "result": {"type": status}}


class FakeOpenAI(OpenAIBackend):
    """Real OpenAI building and parsing; only ``create`` is faked."""

    def __init__(self, gate, raws):
        super().__init__(gate)
        self.requests = []
        self.raws = list(raws)

    def create(self, request):
        self.requests.append(request)
        return self.raws.pop(0)


class FakeAnthropic(AnthropicBackend):
    """Real Anthropic building and parsing; only ``create`` is faked."""

    def __init__(self, gate, raws):
        super().__init__(gate)
        self.requests = []
        self.raws = list(raws)

    def create(self, request):
        self.requests.append(request)
        return self.raws.pop(0)


class FakeBatchOpenAI(OpenAIBackend):
    """Real OpenAI batch building and parsing; only SDK calls are faked."""

    def __init__(self, gate):
        super().__init__(gate)
        self.batch_inputs = []
        self.retrieve_calls = []
        self.fetch_calls = []
        self.status_raw = {
            "status": "completed",
            "request_counts": {"total": 0, "completed": 0, "failed": 0},
        }
        self.output_text = ""
        self.error_text = ""
        self.fail_on = set()
        self._created = 0

    @property
    def batch_id(self):
        """The invented batch id the most recent submit handed out."""
        return f"batch-invented-{self._created:03d}"

    def create_batch(self, batch_input):
        self.batch_inputs.append(batch_input)
        self._created += 1
        if self._created in self.fail_on:
            raise RuntimeError("invented batch failure")
        return self.batch_id

    def retrieve_batch(self, batch_id):
        self.retrieve_calls.append(batch_id)
        return dict(self.status_raw)

    def fetch_batch_results(self, batch_id, raw_status):
        self.fetch_calls.append(batch_id)
        return {
            "output_text": self.output_text,
            "error_text": self.error_text,
        }


class FakeBatchAnthropic(AnthropicBackend):
    """Real Anthropic batch building and parsing; only SDK calls are faked."""

    def __init__(self, gate):
        super().__init__(gate)
        self.batch_inputs = []
        self.retrieve_calls = []
        self.fetch_calls = []
        self.status_raw = {
            "processing_status": "ended",
            "request_counts": {
                "processing": 0,
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
        }
        self.result_entries: list = []
        self.fail_on = set()
        self._created = 0

    @property
    def batch_id(self):
        """The invented batch id the most recent submit handed out."""
        return f"msgbatch-invented-{self._created:03d}"

    def create_batch(self, batch_input):
        self.batch_inputs.append(batch_input)
        self._created += 1
        if self._created in self.fail_on:
            raise RuntimeError("invented batch failure")
        return self.batch_id

    def retrieve_batch(self, batch_id):
        self.retrieve_calls.append(batch_id)
        return dict(self.status_raw)

    def fetch_batch_results(self, batch_id, raw_status):
        self.fetch_calls.append(batch_id)
        return [dict(entry) for entry in self.result_entries]


class TestBatch(unittest.TestCase):
    def setUp(self):
        self._offline = net._PROCESS_OFFLINE
        set_offline(False)
        self._saved_modules = {
            name: sys.modules.get(name) for name in ("openai", "anthropic")
        }
        self._patchers = []
        for target in ("socket.socket", "socket.create_connection"):
            patcher = mock.patch(
                target, side_effect=AssertionError(f"real network use: {target}")
            )
            patcher.start()
            self._patchers.append(patcher)
        self._catalogs = []

    def tearDown(self):
        for catalog in self._catalogs:
            try:
                catalog.close()
            except Exception:
                pass
        self._catalogs = []
        for patcher in reversed(self._patchers):
            patcher.stop()
        self._patchers = []
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        set_offline(self._offline)

    # -- project scaffolding (copied style, not imported) -------------------

    def make_project(self, tmp, provider="openai", model=None, offline=None):
        """Build a trip folder with one invented place and a tag model."""
        tmp = Path(tmp)
        footage = tmp / "footage"
        footage.mkdir(parents=True, exist_ok=True)
        project = tmp / "project"
        project.mkdir(parents=True, exist_ok=True)
        model = model if model is not None else (
            FAKE_OPENAI_MODEL if provider == "openai" else FAKE_ANTHROPIC_MODEL
        )
        toml = (
            'title = "Invented Tag Survey"\n'
            "[pipeline]\n"
            'timezone = "UTC"\n'
            "start_date = 2020-06-15\n"
            + (f"offline = {'true' if offline else 'false'}\n" if offline is not None else "")
            + '[[pipeline.places]]\nname = "Invented Cove"\n'
            + "[pipeline.ai.tag]\n"
            + f'provider = "{provider}"\nmodel = "{model}"\n'
        )
        (project / "project.toml").write_text(toml, encoding="utf-8")
        return load_trip(project), footage, project

    def open_catalog(self, settings):
        """Open the trip catalog, tracking it for tearDown."""
        catalog = Catalog.open(settings)
        self._catalogs.append(catalog)
        return catalog

    def add_clip(self, catalog, footage, name, asset_id, camera="Invented Maker Cam A"):
        """Catalog a small placeholder file as a 60 s clip; return its id."""
        destination = Path(footage) / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"invented-footage-bytes:" + name.encode("utf-8"))
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
                duration=Fraction(60),
            )
        ), os.fspath(destination)

    def write_sheet(self, settings, catalog, media_id, data=None, tiles=TILE_TIMES):
        """Write a synthetic sheet plus manifest for a clip; return bytes."""
        from fcp_pipeline import sheets as sheets_module

        row = [r for r in catalog.list_media() if r.id == media_id][0]
        _, sheet_name, manifest_name = sheets_module._sheet_names(row)
        sheet_bytes = data if data is not None else fake_jpeg(1536, 864)
        folder = Path(os.fspath(settings.pipeline)) / "sheets"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / sheet_name).write_bytes(sheet_bytes)
        manifest = {
            "version": 1,
            "asset_id": row.asset_id,
            "sheet": sheet_name,
            "sheet_sha256": hashlib.sha256(sheet_bytes).hexdigest(),
            "sheet_size": [1536, 864],
            "grid": [3, 3],
            "tile_size": [512, 288],
            "duration": "60/1",
            "scene_count": 0,
            "truncated": False,
            "tiles": [
                {"index": at, "time": f"{moment.numerator}/{moment.denominator}"}
                for at, moment in enumerate(tiles)
            ],
        }
        (folder / manifest_name).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return sheet_bytes

    def add_transcript(self, catalog, media_id, tmp, text):
        """Record a .txt transcript file holding ``text``."""
        folder = Path(tmp) / "transcripts"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"clip-{media_id}.txt"
        path.write_text(text, encoding="utf-8")
        catalog.add_transcript(
            TranscriptRecord(media_id=media_id, path=os.fspath(path))
        )
        return os.fspath(path)

    def tag_key_list(self, catalog, media_id):
        """Return a clip's tags as sorted comparable tuples."""
        return sorted(
            (
                found.folder,
                found.value,
                str(found.start),
                str(found.duration),
                found.source,
                found.status,
            )
            for found in catalog.list_tags(media_id)
        )

    def network_log_text(self, settings):
        """Return the network log text, or "" when it does not exist."""
        path = Path(os.fspath(settings.folder)) / ".pipeline" / "network-log.tsv"
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def walk_request(self, request):
        """Yield every key of ``request`` for the no-video check."""
        if isinstance(request, dict):
            for key, value in request.items():
                yield key
                yield from self.walk_request(value)
        elif isinstance(request, list):
            for value in request:
                yield from self.walk_request(value)

    # -- submit -------------------------------------------------------------

    def test_submit_saves_batch_id(self):
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                with tempfile.TemporaryDirectory() as tmp:
                    settings, footage, _ = self.make_project(tmp, provider=provider)
                    catalog = self.open_catalog(settings)
                    media_ids = []
                    sheets = []
                    for at, asset in enumerate(("clip-a", "clip-b", "clip-c")):
                        media_id, _ = self.add_clip(
                            catalog, footage, f"cove-{asset}.mov", asset
                        )
                        media_ids.append(media_id)
                        sheets.append(
                            self.write_sheet(settings, catalog, media_id)
                        )
                    trip = Path(os.fspath(settings.folder))
                    before = {
                        path.relative_to(trip)
                        for path in (trip / ".pipeline").rglob("*")
                        if path.is_file()
                    }
                    gate = net.NetworkGate(settings, "tag")
                    if provider == "openai":
                        fake = FakeBatchOpenAI(gate)
                        model = FAKE_OPENAI_MODEL
                    else:
                        fake = FakeBatchAnthropic(gate)
                        model = FAKE_ANTHROPIC_MODEL
                    result = tag_batch.submit_batch(
                        catalog,
                        settings,
                        vocabulary=dict(VOCABULARY),
                        backends={provider: fake},
                    )
                    batch_id = fake.batch_id
                    self.assertEqual(result.submitted, 3)
                    job_path = (
                        Path(os.fspath(settings.folder))
                        / ".pipeline"
                        / "tag-batch"
                        / "001.job.json"
                    )
                    self.assertTrue(job_path.is_file())
                    job = json.loads(job_path.read_text(encoding="utf-8"))
                    self.assertEqual(job["version"], 1)
                    self.assertEqual(job["provider"], provider)
                    self.assertEqual(job["model"], model)
                    self.assertEqual(job["batch_id"], batch_id)
                    self.assertEqual(
                        [clip["custom_id"] for clip in job["clips"]],
                        ["clip-1", "clip-2", "clip-3"],
                    )
                    self.assertEqual(
                        [clip["media_id"] for clip in job["clips"]],
                        media_ids,
                    )
                    self.assertEqual(
                        [clip["asset_id"] for clip in job["clips"]],
                        ["clip-a", "clip-b", "clip-c"],
                    )
                    self.assertEqual(
                        [clip["sheet_sha256"] for clip in job["clips"]],
                        [hashlib.sha256(data).hexdigest() for data in sheets],
                    )
                    self.assertEqual(
                        job["vocabulary"],
                        {
                            folder: list(values)
                            for folder, values in VOCABULARY.items()
                        },
                    )
                    runs = catalog.list_runs()
                    self.assertEqual(runs[-1].step, "tag-batch")
                    self.assertIn(batch_id, runs[-1].outcome)
                    self.assertIn(provider, runs[-1].outcome)
                    self.assertEqual(len(fake.batch_inputs), 1)
                    log_lines = self.network_log_text(settings).splitlines()
                    self.assertEqual(len(log_lines), 2)
                    self.assertIn(provider, log_lines[1])
                    self.assertIn("sent", log_lines[1])
                    after = {
                        path.relative_to(trip)
                        for path in (trip / ".pipeline").rglob("*")
                        if path.is_file()
                    }
                    added = {str(path) for path in after - before}
                    self.assertEqual(
                        added,
                        {
                            str(Path(".pipeline") / "tag-batch" / "001.job.json"),
                            str(Path(".pipeline") / "network-log.tsv"),
                        },
                    )
                    for media_id in media_ids:
                        self.assertEqual(catalog.list_tags(media_id), [])

    def test_batch_request_matches_single_request(self):
        good = load_tag_fixture("good-answer.json")
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                with tempfile.TemporaryDirectory() as tmp:
                    settings, footage, _ = self.make_project(tmp, provider=provider)
                    catalog = self.open_catalog(settings)
                    media_id, clip_path = self.add_clip(
                        catalog, footage, "cove-a.mov", "clip-a"
                    )
                    sheet_bytes = self.write_sheet(settings, catalog, media_id)
                    gate = net.NetworkGate(settings, "tag")
                    if provider == "openai":
                        batch_fake = FakeBatchOpenAI(gate)
                        single_fake = FakeOpenAI(gate, [openai_raw(good)])
                    else:
                        batch_fake = FakeBatchAnthropic(gate)
                        single_fake = FakeAnthropic(
                            gate, [anthropic_raw(good)]
                        )
                    tag_batch.submit_batch(
                        catalog,
                        settings,
                        vocabulary=dict(VOCABULARY),
                        backends={provider: batch_fake},
                    )
                    tag.tag_catalog(
                        catalog,
                        settings,
                        vocabulary=dict(VOCABULARY),
                        retag=True,
                        backends={provider: single_fake},
                    )
                    self.assertEqual(len(batch_fake.batch_inputs), 1)
                    self.assertEqual(len(single_fake.requests), 1)
                    if provider == "openai":
                        lines = batch_fake.batch_inputs[0].split(b"\n")
                        lines = [line for line in lines if line.strip()]
                        self.assertEqual(len(lines), 1)
                        bundle = json.loads(lines[0].decode("utf-8"))
                        self.assertEqual(
                            set(bundle),
                            {"custom_id", "method", "url", "body"},
                        )
                        self.assertEqual(bundle["method"], "POST")
                        self.assertEqual(bundle["url"], "/v1/responses")
                        batch_request = bundle["body"]
                    else:
                        bundle = batch_fake.batch_inputs[0]
                        self.assertEqual(len(bundle), 1)
                        self.assertEqual(set(bundle[0]), {"custom_id", "params"})
                        self.assertEqual(bundle[0]["custom_id"], "clip-1")
                        batch_request = bundle[0]["params"]
                    single_request = single_fake.requests[0]
                    self.assertEqual(set(batch_request), set(single_request))
                    self.assertEqual(batch_request, single_request)
                    text = json.dumps(batch_request)
                    for key in self.walk_request(batch_request):
                        self.assertNotIn("video", key)
                    self.assertNotIn("cove-a.mov", text)
                    self.assertNotIn(clip_path, text)
                    if provider == "openai":
                        parts = batch_request["input"][0]["content"]
                        images = [
                            part
                            for part in parts
                            if part.get("type") == "input_image"
                        ]
                        self.assertEqual(len(images), 1)
                        payload = base64.b64decode(
                            images[0]["image_url"].split(",", 1)[1]
                        )
                        self.assertEqual(payload, sheet_bytes)
                    else:
                        blocks = batch_request["messages"][0]["content"]
                        pictures = [
                            block
                            for block in blocks
                            if block.get("type") == "image"
                        ]
                        self.assertEqual(len(pictures), 1)
                        source = pictures[0]["source"]
                        self.assertEqual(source["media_type"], "image/jpeg")
                        self.assertEqual(
                            base64.b64decode(source["data"]), sheet_bytes
                        )

    # -- collect ------------------------------------------------------------

    def test_collect_out_of_order(self):
        answers = [ANSWER_ONE, ANSWER_TWO, ANSWER_THREE]
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                with tempfile.TemporaryDirectory() as tmp:
                    settings, footage, _ = self.make_project(tmp, provider=provider)
                    catalog = self.open_catalog(settings)
                    media_ids = []
                    for asset in ("clip-a", "clip-b", "clip-c"):
                        media_id, _ = self.add_clip(
                            catalog, footage, f"cove-{asset}.mov", asset
                        )
                        self.write_sheet(settings, catalog, media_id)
                        media_ids.append(media_id)
                    gate = net.NetworkGate(settings, "tag")
                    if provider == "openai":
                        fake = FakeBatchOpenAI(gate)
                    else:
                        fake = FakeBatchAnthropic(gate)
                    submit_result = tag_batch.submit_batch(
                        catalog,
                        settings,
                        vocabulary=dict(VOCABULARY),
                        backends={provider: fake},
                    )
                    self.assertEqual(submit_result.submitted, 3)
                    if provider == "openai":
                        fake.status_raw = {
                            "status": "completed",
                            "request_counts": {
                                "total": 3,
                                "completed": 3,
                                "failed": 0,
                            },
                        }
                        fake.output_text = "\n".join(
                            openai_line_succeeded(f"clip-{at}", answer)
                            for at, answer in (
                                (3, answers[2]),
                                (2, answers[1]),
                                (1, answers[0]),
                            )
                        )
                        fake.error_text = ""
                    else:
                        fake.status_raw = {
                            "processing_status": "ended",
                            "request_counts": {
                                "processing": 0,
                                "succeeded": 3,
                                "errored": 0,
                                "canceled": 0,
                                "expired": 0,
                            },
                        }
                        fake.result_entries = [
                            anthropic_entry_succeeded("clip-2", answers[1]),
                            anthropic_entry_succeeded("clip-3", answers[2]),
                            anthropic_entry_succeeded("clip-1", answers[0]),
                        ]
                    result = tag_batch.collect_batch(
                        catalog, settings, backends={provider: fake}
                    )
                    self.assertEqual(
                        result.outcome,
                        "tagged 3, failed 0, expired 0, stale 0, running 0",
                    )
                    shots = {}
                    for media_id in media_ids:
                        for found in catalog.list_tags(media_id):
                            if found.folder == "shot":
                                shots[media_id] = found.value
                    self.assertEqual(
                        shots,
                        {
                            media_ids[0]: "wide",
                            media_ids[1]: "medium",
                            media_ids[2]: "close-up",
                        },
                    )
                    done_path = (
                        Path(os.fspath(settings.folder))
                        / ".pipeline"
                        / "tag-batch"
                        / "001.done.json"
                    )
                    self.assertTrue(done_path.is_file())
                    usage = (
                        Path(os.fspath(settings.folder))
                        / ".pipeline"
                        / "ai-usage.tsv"
                    ).read_text(encoding="utf-8").splitlines()
                    self.assertEqual(len(usage), 4)
                    for row_line in usage[1:]:
                        self.assertIn(provider, row_line)
                        self.assertIn(
                            FAKE_OPENAI_MODEL
                            if provider == "openai"
                            else FAKE_ANTHROPIC_MODEL,
                            row_line,
                        )

    def test_collect_twice_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_ids = []
            for asset in ("clip-a", "clip-b", "clip-c"):
                media_id, _ = self.add_clip(
                    catalog, footage, f"cove-{asset}.mov", asset
                )
                self.write_sheet(settings, catalog, media_id)
                media_ids.append(media_id)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeBatchOpenAI(gate)
            tag_batch.submit_batch(
                catalog, settings,
                vocabulary=dict(VOCABULARY), backends={"openai": fake},
            )
            fake.status_raw = {
                "status": "completed",
                "request_counts": {"total": 3, "completed": 3, "failed": 0},
            }
            fake.output_text = "\n".join(
                openai_line_succeeded(f"clip-{at}", answer)
                for at, answer in (
                    (1, ANSWER_ONE),
                    (2, ANSWER_TWO),
                    (3, ANSWER_THREE),
                )
            )
            first = tag_batch.collect_batch(
                catalog, settings, backends={"openai": fake}
            )
            self.assertEqual(first.tagged, 3)
            before_tags = {
                media_id: self.tag_key_list(catalog, media_id)
                for media_id in media_ids
            }
            before_log = self.network_log_text(settings)
            retrieve_calls = len(fake.retrieve_calls)
            fetch_calls = len(fake.fetch_calls)
            second = tag_batch.collect_batch(
                catalog, settings, backends={"openai": fake}
            )
            self.assertEqual(second.outcome, "no open batch")
            self.assertEqual(len(fake.retrieve_calls), retrieve_calls)
            self.assertEqual(len(fake.fetch_calls), fetch_calls)
            self.assertEqual(self.network_log_text(settings), before_log)
            for media_id in media_ids:
                self.assertEqual(
                    self.tag_key_list(catalog, media_id), before_tags[media_id]
                )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = tag.main(
                    ["--trip", os.fspath(Path(tmp) / "project"), "--collect"]
                )
            self.assertEqual(code, 0)
            self.assertIn("no open batch", stdout.getvalue().splitlines())
            for media_id in media_ids:
                self.assertEqual(
                    self.tag_key_list(catalog, media_id), before_tags[media_id]
                )

    def test_errored_items_listed(self):
        message = "invented provider message that must never surface"
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                with tempfile.TemporaryDirectory() as tmp:
                    settings, footage, project = self.make_project(
                        tmp, provider=provider
                    )
                    catalog = self.open_catalog(settings)
                    media_ids = []
                    for asset in ("clip-a", "clip-b", "clip-c", "clip-d"):
                        media_id, _ = self.add_clip(
                            catalog, footage, f"cove-{asset}.mov", asset
                        )
                        self.write_sheet(settings, catalog, media_id)
                        media_ids.append(media_id)
                    gate = net.NetworkGate(settings, "tag")
                    if provider == "openai":
                        fake = FakeBatchOpenAI(gate)
                    else:
                        fake = FakeBatchAnthropic(gate)
                    tag_batch.submit_batch(
                        catalog,
                        settings,
                        vocabulary=dict(VOCABULARY),
                        backends={provider: fake},
                    )
                    if provider == "openai":
                        status_raw = {
                            "status": "completed",
                            "request_counts": {
                                "total": 4,
                                "completed": 1,
                                "failed": 3,
                            },
                            "output_file_id": "file-invented-out-001",
                            "error_file_id": "file-invented-err-001",
                        }
                        fetch_raw = {
                            "output_text": openai_line_succeeded(
                                "clip-1", ANSWER_ONE
                            ),
                            "error_text": "\n".join(
                                [
                                    openai_line_errored(
                                        "clip-2", "invalid_request", message
                                    ),
                                    json.dumps(
                                        {
                                            "custom_id": "clip-3",
                                            "response": {
                                                "status_code": 500,
                                                "body": None,
                                            },
                                            "error": {
                                                "code": "expired",
                                                "message": message,
                                            },
                                        }
                                    ),
                                ]
                            ),
                        }
                    else:
                        status_raw = {
                            "processing_status": "ended",
                            "request_counts": {
                                "processing": 0,
                                "succeeded": 1,
                                "errored": 1,
                                "canceled": 0,
                                "expired": 1,
                            },
                        }
                        fetch_raw = [
                            anthropic_entry_succeeded("clip-1", ANSWER_ONE),
                            anthropic_entry_errored(
                                "clip-2", "invalid_request_error", message
                            ),
                            anthropic_entry_status("clip-3", "expired"),
                        ]
                    backend = OpenAIBackend if provider == "openai" else AnthropicBackend
                    stdout = io.StringIO()
                    with mock.patch.object(
                        backend, "retrieve_batch", return_value=status_raw
                    ), mock.patch.object(
                        backend, "fetch_batch_results", return_value=fetch_raw
                    ), redirect_stdout(stdout):
                        code = tag.main(
                            ["--trip", os.fspath(project), "--collect"]
                        )
                    self.assertEqual(code, 0)
                    lines = stdout.getvalue().splitlines()
                    failed_lines = [
                        line for line in lines if line.startswith("failed: ")
                    ]
                    expired_lines = [
                        line for line in lines if line.startswith("expired: ")
                    ]
                    self.assertEqual(len(failed_lines), 1)
                    self.assertEqual(len(expired_lines), 2)
                    self.assertNotIn(message, stdout.getvalue())
                    self.assertNotIn(
                        ANSWER_ONE["description"], stdout.getvalue()
                    )
                    self.assertTrue(catalog.list_tags(media_ids[0]))
                    for media_id in media_ids[1:]:
                        self.assertEqual(catalog.list_tags(media_id), [])
                    gate_b = net.NetworkGate(settings, "tag")
                    if provider == "openai":
                        fake_b = FakeBatchOpenAI(gate_b)
                    else:
                        fake_b = FakeBatchAnthropic(gate_b)
                    again = tag_batch.submit_batch(
                        catalog,
                        settings,
                        vocabulary=dict(VOCABULARY),
                        backends={provider: fake_b},
                    )
                    self.assertEqual(again.submitted, 3)
                    second_job = (
                        Path(os.fspath(settings.folder))
                        / ".pipeline"
                        / "tag-batch"
                        / "002.job.json"
                    )
                    self.assertTrue(second_job.is_file())
                    clips = json.loads(
                        second_job.read_text(encoding="utf-8")
                    )["clips"]
                    self.assertEqual(
                        [clip["custom_id"] for clip in clips],
                        ["clip-2", "clip-3", "clip-4"],
                    )

    def test_duplicate_and_unknown_items_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_ids = []
            for asset in ("clip-a", "clip-b", "clip-c"):
                media_id, _ = self.add_clip(
                    catalog, footage, f"cove-{asset}.mov", asset
                )
                self.write_sheet(settings, catalog, media_id)
                media_ids.append(media_id)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeBatchOpenAI(gate)
            tag_batch.submit_batch(
                catalog,
                settings,
                vocabulary=dict(VOCABULARY),
                backends={"openai": fake},
            )
            fake.status_raw = {
                "status": "completed",
                "request_counts": {"total": 3, "completed": 3, "failed": 0},
            }
            fake.output_text = "\n".join(
                [
                    openai_line_succeeded("clip-1", ANSWER_ONE),
                    openai_line_succeeded("clip-2", ANSWER_TWO),
                    openai_line_succeeded("clip-2", ANSWER_TWO),
                    openai_line_succeeded("clip-99", ANSWER_THREE),
                ]
            )
            result = tag_batch.collect_batch(
                catalog, settings, backends={"openai": fake}
            )
            self.assertEqual(
                result.outcome,
                "tagged 1, failed 0, expired 2, stale 0, running 0",
            )
            self.assertEqual(len(result.notes), 2)
            self.assertTrue(
                any("clip-99" in note for note in result.notes)
            )
            self.assertTrue(
                any("clip-2" in note for note in result.notes)
            )
            self.assertTrue(catalog.list_tags(media_ids[0]))
            self.assertEqual(catalog.list_tags(media_ids[1]), [])
            self.assertEqual(catalog.list_tags(media_ids[2]), [])
            by_id = {}
            for entry in result.entries:
                found = catalog.get_media(entry.path)
                by_id[found.id] = entry
            self.assertEqual(by_id[media_ids[0]].status, "tagged")
            self.assertEqual(by_id[media_ids[1]].status, "expired")
            self.assertEqual(by_id[media_ids[2]].status, "expired")
            done = json.loads(
                (
                    Path(os.fspath(settings.folder))
                    / ".pipeline"
                    / "tag-batch"
                    / "001.done.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                [(clip["asset_id"], clip["status"]) for clip in done["clips"]],
                [
                    ("clip-a", "tagged"),
                    ("clip-b", "expired"),
                    ("clip-c", "expired"),
                ],
            )

    def test_finished_failed_batch_expires_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_ids = []
            for asset in ("clip-a", "clip-b", "clip-c"):
                media_id, _ = self.add_clip(
                    catalog, footage, f"cove-{asset}.mov", asset
                )
                self.write_sheet(settings, catalog, media_id)
                media_ids.append(media_id)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeBatchOpenAI(gate)
            tag_batch.submit_batch(
                catalog,
                settings,
                vocabulary=dict(VOCABULARY),
                backends={"openai": fake},
            )
            fake.status_raw = {
                "status": "failed",
                "request_counts": {"total": 3, "completed": 0, "failed": 3},
            }
            fake.output_text = ""
            fake.error_text = ""
            result = tag_batch.collect_batch(
                catalog, settings, backends={"openai": fake}
            )
            self.assertEqual(
                result.outcome,
                "tagged 0, failed 0, expired 3, stale 0, running 0",
            )
            self.assertEqual(len(fake.fetch_calls), 1)
            for media_id in media_ids:
                self.assertEqual(catalog.list_tags(media_id), [])
            done = json.loads(
                (
                    Path(os.fspath(settings.folder))
                    / ".pipeline"
                    / "tag-batch"
                    / "001.done.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(done["state"], "failed")
            self.assertEqual(
                [clip["status"] for clip in done["clips"]],
                ["expired", "expired", "expired"],
            )
            self.assertFalse(
                (
                    Path(os.fspath(settings.folder))
                    / ".pipeline"
                    / "ai-usage.tsv"
                ).exists()
            )

    def test_corrupt_job_values_rejected(self):
        def base_document():
            return {
                "version": 1,
                "provider": "openai",
                "model": FAKE_OPENAI_MODEL,
                "batch_id": "batch-invented-000",
                "submitted": tag_batch._utc_now(),
                "cutoff": 0.75,
                "retag": False,
                "vocabulary": {
                    folder: list(values)
                    for folder, values in VOCABULARY.items()
                },
                "clips": [],
            }

        def good_clip():
            return {
                "custom_id": "clip-9",
                "media_id": 9,
                "asset_id": "clip-i",
                "sheet_sha256": "0" * 64,
            }

        mutations = {
            "provider": {"provider": "openaI"},
            "model": {"model": ""},
            "batch_id": {"batch_id": ""},
            "submitted": {"submitted": ""},
            "cutoff": {"cutoff": "high"},
            "retag": {"retag": "yes"},
            "clips-not-list": {"clips": "nope"},
            "clip-custom-id": {"clips": [dict(good_clip(), custom_id="")]},
            "clip-media-id": {"clips": [dict(good_clip(), media_id=True)]},
            "vocabulary-values": {"vocabulary": {"place": "Invented Cove"}},
        }
        for bad_kind, patch in mutations.items():
            with self.subTest(value=bad_kind):
                with tempfile.TemporaryDirectory() as tmp:
                    settings, footage, _ = self.make_project(tmp)
                    catalog = self.open_catalog(settings)
                    media_id, _ = self.add_clip(
                        catalog, footage, "cove-a.mov", "clip-a"
                    )
                    self.write_sheet(settings, catalog, media_id)
                    state_dir = (
                        Path(os.fspath(settings.folder))
                        / ".pipeline"
                        / "tag-batch"
                    )
                    state_dir.mkdir(parents=True, exist_ok=True)
                    document = base_document()
                    document.update(patch)
                    (state_dir / "001.job.json").write_text(
                        json.dumps(document) + "\n", encoding="utf-8"
                    )
                    gate = net.NetworkGate(settings, "tag")
                    fake = FakeBatchOpenAI(gate)
                    with self.assertRaises(tag_batch.TagBatchError) as caught:
                        tag_batch.submit_batch(
                            catalog,
                            settings,
                            vocabulary=dict(VOCABULARY),
                            backends={"openai": fake},
                        )
                    self.assertIn("001.job.json", str(caught.exception))
                    self.assertEqual(fake.batch_inputs, [])
                    self.assertEqual(fake.retrieve_calls, [])

    def legacy_vocabulary(self):
        """Pin the lowercase folders the recorded fixtures use."""
        return {
            "place": ("Invented Cove",),
            "gear": ("Invented Maker Cam A",),
            "subject": ("invented-boat", "invented-gull"),
        }

    def test_answers_checked_like_single_tagging(self):
        good = load_tag_fixture("good-answer.json")
        edge = load_tag_fixture("edge-answer.json")
        malformed = load_tag_fixture("malformed-answer.json")
        huge = {
            "description": "Invented huge-number check",
            "shot": "wide",
            "ranges": [
                {
                    "confidence": 0.9,
                    "end": 2000001,
                    "start": 2000000,
                    "tags": [{"folder": "place", "value": "Invented Cove"}],
                }
            ],
            "problems": [],
            "outline_hint": None,
        }
        answers = [good, edge, malformed, huge]
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_ids = []
            for asset in ("clip-a", "clip-b", "clip-c", "clip-d"):
                media_id, _ = self.add_clip(
                    catalog, footage, f"cove-{asset}.mov", asset
                )
                self.write_sheet(settings, catalog, media_id)
                media_ids.append(media_id)
            catalog.put_tag(
                TagRecord(
                    media_id=media_ids[0],
                    folder="place",
                    value="Invented Locked Cove",
                    start=Fraction(0),
                    duration=Fraction(10),
                    source="editor",
                    confidence=None,
                    status="locked",
                )
            )
            gate = net.NetworkGate(settings, "tag")
            fake = FakeBatchOpenAI(gate)
            tag_batch.submit_batch(
                catalog,
                settings,
                vocabulary=dict(self.legacy_vocabulary()),
                backends={"openai": fake},
            )
            fake.status_raw = {
                "status": "completed",
                "request_counts": {"total": 4, "completed": 4, "failed": 0},
            }
            fake.output_text = "\n".join(
                openai_line_succeeded(f"clip-{at}", answer)
                for at, answer in enumerate(answers, start=1)
            )
            result = tag_batch.collect_batch(
                catalog, settings, backends={"openai": fake}
            )
            self.assertEqual(len(fake.batch_inputs), 1)
            malformed_entry = [
                entry
                for entry in result.entries
                if catalog.get_media(entry.path).id == media_ids[2]
            ][0]
            self.assertEqual(malformed_entry.status, "failed")
            self.assertTrue(
                malformed_entry.reason.startswith("answer failed checks:")
            )
            huge_entry = [
                entry
                for entry in result.entries
                if catalog.get_media(entry.path).id == media_ids[3]
            ][0]
            self.assertEqual(huge_entry.status, "failed")
            self.assertIn("out of range", huge_entry.reason)
            locked = [
                found
                for found in catalog.list_tags(media_ids[0])
                if found.folder == "place"
                and found.start == Fraction(0)
                and found.duration == Fraction(10)
            ]
            self.assertEqual(len(locked), 1)
            self.assertEqual(locked[0].value, "Invented Locked Cove")
            void = catalog.list_tags(media_ids[2])
            self.assertEqual(void, [])
            void_huge = catalog.list_tags(media_ids[3])
            self.assertEqual(void_huge, [])

            with self.subTest("twin one-clip run stores the same tags"):
                settings_b, footage_b, _ = self.make_project(Path(tmp) / "twin")
                catalog_b = self.open_catalog(settings_b)
                twin_ids = []
                for asset in ("clip-a", "clip-b", "clip-c", "clip-d"):
                    twin_id, _ = self.add_clip(
                        catalog_b, footage_b, f"cove-{asset}.mov", asset
                    )
                    self.write_sheet(settings_b, catalog_b, twin_id)
                    twin_ids.append(twin_id)
                catalog_b.put_tag(
                    TagRecord(
                        media_id=twin_ids[0],
                        folder="place",
                        value="Invented Locked Cove",
                        start=Fraction(0),
                        duration=Fraction(10),
                        source="editor",
                        confidence=None,
                        status="locked",
                    )
                )
                gate_b = net.NetworkGate(settings_b, "tag")
                twin_fake = FakeOpenAI(
                    gate_b,
                    [
                        openai_raw(good),
                        openai_raw(edge),
                        openai_raw(malformed),
                        openai_raw(malformed),
                        openai_raw(huge),
                    ],
                )
                twin_result = tag.tag_catalog(
                    catalog_b,
                    settings_b,
                    vocabulary=dict(self.legacy_vocabulary()),
                    backends={"openai": twin_fake},
                )
                self.assertEqual(twin_result.tagged, 2)
                for media_id, twin_id in zip(media_ids, twin_ids):
                    self.assertEqual(
                        self.tag_key_list(catalog, media_id),
                        self.tag_key_list(catalog_b, twin_id),
                    )

    def test_resume_without_double_tagging(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_ids = []
            for asset in ("clip-a", "clip-b", "clip-c"):
                media_id, _ = self.add_clip(
                    catalog, footage, f"cove-{asset}.mov", asset
                )
                self.write_sheet(settings, catalog, media_id)
                media_ids.append(media_id)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeBatchOpenAI(gate)
            first = tag_batch.submit_batch(
                catalog,
                settings,
                vocabulary=dict(VOCABULARY),
                backends={"openai": fake},
            )
            self.assertEqual(first.submitted, 3)
            batch_id = fake.batch_id
            again = tag_batch.submit_batch(
                catalog,
                settings,
                vocabulary=dict(VOCABULARY),
                backends={"openai": fake},
            )
            self.assertEqual(len(fake.batch_inputs), 1)
            self.assertEqual(again.submitted, 0)
            self.assertEqual(again.pending, 3)
            for entry in again.entries:
                self.assertEqual(entry.status, "pending")
                self.assertIn(batch_id, entry.reason or "")

        with self.subTest("failed job leaves the first two clips pending"):
            with tempfile.TemporaryDirectory() as tmp:
                settings, footage, _ = self.make_project(tmp)
                catalog = self.open_catalog(settings)
                for asset in ("clip-a", "clip-b", "clip-c", "clip-d", "clip-e"):
                    media_id, _ = self.add_clip(
                        catalog, footage, f"cove-{asset}.mov", asset
                    )
                    self.write_sheet(settings, catalog, media_id)
                gate = net.NetworkGate(settings, "tag")
                fake = FakeBatchOpenAI(gate)
                fake.fail_on = {2}
                with self.assertRaises(RuntimeError):
                    with mock.patch.object(
                        tag_batch, "MAX_JOB_REQUESTS", 2
                    ):
                        tag_batch.submit_batch(
                            catalog,
                            settings,
                            vocabulary=dict(VOCABULARY),
                            backends={"openai": fake},
                        )
                first_job = (
                    Path(os.fspath(settings.folder))
                    / ".pipeline"
                    / "tag-batch"
                    / "001.job.json"
                )
                self.assertTrue(first_job.is_file())
                first_clips = json.loads(
                    first_job.read_text(encoding="utf-8")
                )["clips"]
                self.assertEqual(
                    [clip["custom_id"] for clip in first_clips],
                    ["clip-1", "clip-2"],
                )
                fake.fail_on = set()
                with mock.patch.object(tag_batch, "MAX_JOB_REQUESTS", 2):
                    rerun = tag_batch.submit_batch(
                        catalog,
                        settings,
                        vocabulary=dict(VOCABULARY),
                        backends={"openai": fake},
                    )
                self.assertEqual(rerun.submitted, 3)
                self.assertEqual(rerun.pending, 2)
                for name in ("002.job.json", "003.job.json"):
                    self.assertTrue(
                        (
                            Path(os.fspath(settings.folder))
                            / ".pipeline"
                            / "tag-batch"
                            / name
                        ).is_file()
                    )
                second_clips = json.loads(
                    (
                        Path(os.fspath(settings.folder))
                        / ".pipeline"
                        / "tag-batch"
                        / "002.job.json"
                    ).read_text(encoding="utf-8")
                )["clips"]
                self.assertEqual(
                    [clip["custom_id"] for clip in second_clips],
                    ["clip-3", "clip-4"],
                )

        with self.subTest("unwritable job file names the batch"):
            with tempfile.TemporaryDirectory() as tmp:
                settings, footage, _ = self.make_project(tmp)
                catalog = self.open_catalog(settings)
                media_id, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
                self.write_sheet(settings, catalog, media_id)
                gate = net.NetworkGate(settings, "tag")
                fake = FakeBatchOpenAI(gate)
                planted = tag_batch._job_document(
                    "openai",
                    FAKE_OPENAI_MODEL,
                    "batch-invented-000",
                    tag_batch._utc_now(),
                    0.75,
                    False,
                    dict(VOCABULARY),
                    [],
                )
                tag_batch._write_state_file(settings, "001.job.json", planted)
                planted_path = (
                    Path(os.fspath(settings.folder))
                    / ".pipeline"
                    / "tag-batch"
                    / "001.job.json"
                )
                before_bytes = planted_path.read_bytes()
                before_mtime = os.stat(planted_path).st_mtime_ns
                with mock.patch.object(
                    tag_batch, "_next_job_number", return_value=1
                ):
                    with self.assertRaises(tag_batch.TagBatchError) as caught:
                        tag_batch.submit_batch(
                            catalog,
                            settings,
                            vocabulary=dict(VOCABULARY),
                            backends={"openai": fake},
                        )
                self.assertIn(fake.batch_id, str(caught.exception))
                self.assertIn("openai", str(caught.exception))
                self.assertIn(fake.batch_id, catalog.list_runs()[-1].outcome)
                self.assertEqual(planted_path.read_bytes(), before_bytes)
                self.assertEqual(
                    os.stat(planted_path).st_mtime_ns, before_mtime
                )

        with self.subTest("interrupted collect finishes on rerun"):
            answers = [ANSWER_ONE, ANSWER_TWO, ANSWER_THREE]

            def build(parent):
                inner_settings, inner_footage, _ = self.make_project(parent)
                inner_catalog = self.open_catalog(inner_settings)
                inner_ids = []
                for asset in ("clip-a", "clip-b", "clip-c"):
                    inner_id, _ = self.add_clip(
                        inner_catalog, inner_footage, f"cove-{asset}.mov", asset
                    )
                    self.write_sheet(inner_settings, inner_catalog, inner_id)
                    inner_ids.append(inner_id)
                inner_gate = net.NetworkGate(inner_settings, "tag")
                inner_fake = FakeBatchOpenAI(inner_gate)
                tag_batch.submit_batch(
                    inner_catalog,
                    inner_settings,
                    vocabulary=dict(VOCABULARY),
                    backends={"openai": inner_fake},
                )
                inner_fake.status_raw = {
                    "status": "completed",
                    "request_counts": {"total": 3, "completed": 3, "failed": 0},
                }
                inner_fake.output_text = "\n".join(
                    openai_line_succeeded(f"clip-{at}", answer)
                    for at, answer in enumerate(answers, start=1)
                )
                return inner_settings, inner_catalog, inner_ids, inner_fake

            with tempfile.TemporaryDirectory() as twin_tmp:
                twin_settings, twin_catalog, twin_ids, twin_fake = build(
                    Path(twin_tmp) / "twin"
                )
                twin_result = tag_batch.collect_batch(
                    twin_catalog, twin_settings, backends={"openai": twin_fake}
                )
                self.assertEqual(twin_result.tagged, 3)
            with tempfile.TemporaryDirectory() as tmp_b:
                settings_b, footage_b, _ = self.make_project(tmp_b)
                catalog_b = self.open_catalog(settings_b)
                ids_b = []
                for asset in ("clip-a", "clip-b", "clip-c"):
                    media_id, _ = self.add_clip(
                        catalog_b, footage_b, f"cove-{asset}.mov", asset
                    )
                    self.write_sheet(settings_b, catalog_b, media_id)
                    ids_b.append(media_id)
                gate_b = net.NetworkGate(settings_b, "tag")
                fake_b = FakeBatchOpenAI(gate_b)
                tag_batch.submit_batch(
                    catalog_b,
                    settings_b,
                    vocabulary=dict(VOCABULARY),
                    backends={"openai": fake_b},
                )
                fake_b.status_raw = {
                    "status": "completed",
                    "request_counts": {"total": 3, "completed": 3, "failed": 0},
                }
                fake_b.output_text = "\n".join(
                    openai_line_succeeded(f"clip-{at}", answer)
                    for at, answer in enumerate(answers, start=1)
                )
                original_put_tag = catalog_b.put_tag
                calls = []

                def flaky(record):
                    calls.append(record)
                    if len(calls) == 2:
                        raise RuntimeError("invented interrupt")
                    return original_put_tag(record)

                with mock.patch.object(
                    catalog_b, "put_tag", side_effect=flaky
                ):
                    with self.assertRaises(RuntimeError):
                        tag_batch.collect_batch(
                            catalog_b, settings_b, backends={"openai": fake_b}
                        )
                self.assertFalse(
                    (
                        Path(os.fspath(settings_b.folder))
                        / ".pipeline"
                        / "tag-batch"
                        / "001.done.json"
                    ).exists()
                )
                rerun = tag_batch.collect_batch(
                    catalog_b, settings_b, backends={"openai": fake_b}
                )
                self.assertEqual(rerun.tagged, 3)
                for media_id, twin_id in zip(ids_b, twin_ids):
                    self.assertEqual(
                        self.tag_key_list(catalog_b, media_id),
                        self.tag_key_list(twin_catalog, twin_id),
                    )
                    slots = [
                        (found[0], found[2], found[3])
                        for found in self.tag_key_list(catalog_b, media_id)
                    ]
                    self.assertEqual(len(slots), len(set(slots)))

    def test_job_split_by_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            for asset in ("clip-a", "clip-b", "clip-c", "clip-d", "clip-e"):
                media_id, _ = self.add_clip(
                    catalog, footage, f"cove-{asset}.mov", asset
                )
                self.write_sheet(settings, catalog, media_id)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeBatchOpenAI(gate)
            with mock.patch.object(tag_batch, "MAX_JOB_REQUESTS", 2):
                result = tag_batch.submit_batch(
                    catalog,
                    settings,
                    vocabulary=dict(VOCABULARY),
                    backends={"openai": fake},
                )
            self.assertEqual(result.submitted, 5)
            self.assertEqual(len(fake.batch_inputs), 3)
            self.assertEqual(len(result.jobs), 3)
            for name, count in (
                ("001.job.json", 2),
                ("002.job.json", 2),
                ("003.job.json", 1),
            ):
                clips = json.loads(
                    (
                        Path(os.fspath(settings.folder))
                        / ".pipeline"
                        / "tag-batch"
                        / name
                    ).read_text(encoding="utf-8")
                )["clips"]
                self.assertEqual(len(clips), count)
            for job in result.jobs:
                self.assertIn(job.batch_id, result.outcome)
            self.assertIn("submitted 5 in 3 batches", result.outcome)

        with self.subTest("byte cap splits one request per job"):
            with tempfile.TemporaryDirectory() as tmp:
                settings, footage, _ = self.make_project(tmp)
                catalog = self.open_catalog(settings)
                for asset in ("clip-a", "clip-b", "clip-c", "clip-d", "clip-e"):
                    media_id, _ = self.add_clip(
                        catalog, footage, f"cove-{asset}.mov", asset
                    )
                    self.write_sheet(settings, catalog, media_id)
                row = [
                    row
                    for row in catalog.list_media()
                    if row.asset_id == "clip-a"
                ][0]
                from fcp_pipeline.sheets import read_manifest

                manifest = read_manifest(settings, row)
                data, failure = tag._load_sheet_bytes(settings, manifest)
                self.assertIsNone(failure)
                instructions = tag.build_instructions(
                    tag.metadata_line(row),
                    tag.slate_text(catalog, row.id),
                    tag.transcript_text(catalog, row.id)[0],
                    dict(VOCABULARY),
                    tag._manifest_tile_times(manifest),
                )
                _, request = ai_batch.prepare(
                    settings,
                    "tag",
                    instructions,
                    tag.answer_schema(),
                    [ai.Image(data=data, media_type="image/jpeg")],
                )
                size = len(json.dumps(request).encode("utf-8"))
                gate = net.NetworkGate(settings, "tag")
                fake = FakeBatchOpenAI(gate)
                with mock.patch.object(tag_batch, "MAX_JOB_BYTES", size):
                    result = tag_batch.submit_batch(
                        catalog,
                        settings,
                        vocabulary=dict(VOCABULARY),
                        backends={"openai": fake},
                    )
                self.assertEqual(result.submitted, 5)
                self.assertEqual(len(fake.batch_inputs), 5)
                self.assertIn("submitted 5 in 5 batches", result.outcome)

    def test_stale_clip_not_tagged_on_collect(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_ids = []
            for asset in ("clip-a", "clip-b", "clip-c"):
                media_id, _ = self.add_clip(
                    catalog, footage, f"cove-{asset}.mov", asset
                )
                self.write_sheet(settings, catalog, media_id)
                media_ids.append(media_id)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeBatchOpenAI(gate)
            tag_batch.submit_batch(
                catalog,
                settings,
                vocabulary=dict(VOCABULARY),
                backends={"openai": fake},
            )
            with open(os.fspath(Path(footage) / "cove-clip-a.mov"), "ab") as handle:
                handle.write(b"changed-bytes")
            from fcp_pipeline import sheets as sheets_module

            row_b = [r for r in catalog.list_media() if r.id == media_ids[1]][0]
            _, _, manifest_name = sheets_module._sheet_names(row_b)
            manifest_path = (
                Path(os.fspath(settings.pipeline)) / "sheets" / manifest_name
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sheet_sha256"] = "0" * 64
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            fake.status_raw = {
                "status": "completed",
                "request_counts": {"total": 3, "completed": 3, "failed": 0},
            }
            fake.output_text = "\n".join(
                openai_line_succeeded(f"clip-{at}", answer)
                for at, answer in (
                    (1, ANSWER_ONE),
                    (2, ANSWER_TWO),
                    (3, ANSWER_THREE),
                )
            )
            result = tag_batch.collect_batch(
                catalog, settings, backends={"openai": fake}
            )
            self.assertEqual(
                result.outcome,
                "tagged 1, failed 0, expired 0, stale 2, running 0",
            )
            by_id = {}
            for entry in result.entries:
                found = catalog.get_media(entry.path)
                by_id[found.id] = entry
            self.assertEqual(by_id[media_ids[0]].status, "stale")
            self.assertEqual(by_id[media_ids[1]].status, "stale")
            self.assertEqual(by_id[media_ids[2]].status, "tagged")
            self.assertEqual(catalog.list_tags(media_ids[0]), [])
            self.assertEqual(catalog.list_tags(media_ids[1]), [])
            self.assertTrue(catalog.list_tags(media_ids[2]))

    def test_still_running_stays_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, project = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            for asset in ("clip-a", "clip-b"):
                media_id, _ = self.add_clip(
                    catalog, footage, f"cove-{asset}.mov", asset
                )
                self.write_sheet(settings, catalog, media_id)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeBatchOpenAI(gate)
            tag_batch.submit_batch(
                catalog,
                settings,
                vocabulary=dict(VOCABULARY),
                backends={"openai": fake},
            )
            running_raw = {
                "status": "in_progress",
                "request_counts": {"total": 2, "completed": 0, "failed": 0},
            }
            done_raw = {
                "status": "completed",
                "request_counts": {"total": 2, "completed": 2, "failed": 0},
            }
            fetch_raw = {
                "output_text": "\n".join(
                    [
                        openai_line_succeeded("clip-1", ANSWER_ONE),
                        openai_line_succeeded("clip-2", ANSWER_TWO),
                    ]
                ),
                "error_text": "",
            }
            with mock.patch.object(
                OpenAIBackend, "retrieve_batch", return_value=running_raw
            ), mock.patch.object(
                OpenAIBackend,
                "fetch_batch_results",
                side_effect=AssertionError("no fetch while running"),
            ):
                result = tag_batch.collect_batch(catalog, settings)
                self.assertEqual(result.running, 2)
                self.assertEqual(
                    result.outcome,
                    "tagged 0, failed 0, expired 0, stale 0, running 2",
                )
                done_path = (
                    Path(os.fspath(settings.folder))
                    / ".pipeline"
                    / "tag-batch"
                    / "001.done.json"
                )
                self.assertFalse(done_path.exists())
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    code = tag.main(
                        ["--trip", os.fspath(project), "--collect"]
                    )
                self.assertEqual(code, 0)
                self.assertEqual(
                    len(
                        [
                            line
                            for line in stdout.getvalue().splitlines()
                            if line.startswith("running: ")
                        ]
                    ),
                    2,
                )
            with mock.patch.object(
                OpenAIBackend, "retrieve_batch", return_value=done_raw
            ), mock.patch.object(
                OpenAIBackend, "fetch_batch_results", return_value=fetch_raw
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    code = tag.main(
                        ["--trip", os.fspath(project), "--collect"]
                    )
                self.assertEqual(code, 0)
                self.assertIn(
                    "tagged 2, failed 0, expired 0, stale 0, running 0",
                    stdout.getvalue().splitlines(),
                )
                media_one = catalog.get_media(
                    os.fspath(Path(footage) / "cove-clip-a.mov")
                )
                self.assertTrue(
                    [
                        found
                        for found in catalog.list_tags(media_one.id)
                        if found.folder == "shot" and found.value == "wide"
                    ]
                )

    # -- gates and errors ---------------------------------------------------

    def check_offline_batch_refused(self, settings, project, catalog):
        """Shared assertions for a refused offline ``--batch``."""
        state_dir = (
            Path(os.fspath(settings.folder)) / ".pipeline" / "tag-batch"
        )
        with mock.patch.object(
            OpenAIBackend,
            "create_batch",
            side_effect=AssertionError("must not upload while offline"),
        ):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = tag.main(["--trip", os.fspath(project), "--batch"])
        self.assertEqual(code, 3)
        lines = stderr.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("error:"))
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertFalse(state_dir.exists())
        self.assertTrue(
            catalog.list_runs()[-1].outcome.startswith("stopped: OfflineError")
        )
        log_lines = self.network_log_text(settings).splitlines()
        self.assertTrue(
            any("refused offline" in line for line in log_lines)
        )

    def test_offline_refuses_before_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, project = self.make_project(tmp, offline=True)
            catalog = self.open_catalog(settings)
            media_id, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            self.write_sheet(settings, catalog, media_id)
            self.check_offline_batch_refused(settings, project, catalog)

        with self.subTest("process switch"):
            with tempfile.TemporaryDirectory() as tmp:
                settings, footage, project = self.make_project(tmp)
                catalog = self.open_catalog(settings)
                media_id, _ = self.add_clip(
                    catalog, footage, "cove-a.mov", "clip-a"
                )
                self.write_sheet(settings, catalog, media_id)
                stderr = io.StringIO()
                with mock.patch.object(
                    OpenAIBackend,
                    "create_batch",
                    side_effect=AssertionError("must not upload while offline"),
                ), redirect_stderr(stderr):
                    code = tag.main(
                        ["--trip", os.fspath(project), "--batch", "--offline"]
                    )
                self.assertEqual(code, 3)
                self.assertEqual(len(stderr.getvalue().splitlines()), 1)

        with self.subTest("collect while offline"):
            set_offline(False)
            with tempfile.TemporaryDirectory() as tmp:
                settings, footage, project = self.make_project(tmp)
                catalog = self.open_catalog(settings)
                media_id, _ = self.add_clip(
                    catalog, footage, "cove-a.mov", "clip-a"
                )
                self.write_sheet(settings, catalog, media_id)
                gate = net.NetworkGate(settings, "tag")
                fake = FakeBatchOpenAI(gate)
                submitted = tag_batch.submit_batch(
                    catalog,
                    settings,
                    vocabulary=dict(VOCABULARY),
                    backends={"openai": fake},
                )
                self.assertEqual(submitted.submitted, 1)
                job_path = (
                    Path(os.fspath(settings.folder))
                    / ".pipeline"
                    / "tag-batch"
                    / "001.job.json"
                )
                before_bytes = job_path.read_bytes()
                before_mtime = os.stat(job_path).st_mtime_ns
                with mock.patch.object(
                    OpenAIBackend,
                    "retrieve_batch",
                    side_effect=AssertionError("must not download offline"),
                ), mock.patch.object(
                    OpenAIBackend,
                    "fetch_batch_results",
                    side_effect=AssertionError("must not download offline"),
                ):
                    stderr = io.StringIO()
                    with redirect_stderr(stderr):
                        code = tag.main(
                            [
                                "--trip",
                                os.fspath(project),
                                "--collect",
                                "--offline",
                            ]
                        )
                self.assertEqual(code, 3)
                self.assertEqual(len(stderr.getvalue().splitlines()), 1)
                self.assertEqual(job_path.read_bytes(), before_bytes)
                self.assertEqual(
                    os.stat(job_path).st_mtime_ns, before_mtime
                )
                self.assertFalse(
                    (
                        Path(os.fspath(settings.folder))
                        / ".pipeline"
                        / "tag-batch"
                        / "001.done.json"
                    ).exists()
                )

    def test_missing_model_stops_before_anything_is_sent(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, project = self.make_project(tmp, model="<model-id>")
            catalog = self.open_catalog(settings)
            media_id, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            self.write_sheet(settings, catalog, media_id)
            with mock.patch.object(
                tag,
                "_load_sheet_bytes",
                side_effect=AssertionError("must not read sheets"),
            ), mock.patch.object(
                OpenAIBackend,
                "create_batch",
                side_effect=AssertionError("must not upload"),
            ):
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    code = tag.main(["--trip", os.fspath(project), "--batch"])
            self.assertEqual(code, 2)
            self.assertIn("[pipeline.ai.tag].model", stderr.getvalue())
            self.assertFalse(
                (
                    Path(os.fspath(settings.folder))
                    / ".pipeline"
                    / "tag-batch"
                ).exists()
            )

    def test_state_files_stay_small_and_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_ids = []
            for asset in ("clip-a", "clip-b", "clip-c"):
                media_id, _ = self.add_clip(
                    catalog, footage, f"cove-{asset}.mov", asset
                )
                self.write_sheet(settings, catalog, media_id)
                media_ids.append(media_id)
            trip = Path(os.fspath(settings.folder))

            def snapshot():
                found = {}
                for root, _, files in os.walk(trip):
                    for name in files:
                        full = Path(root) / name
                        found[str(full.relative_to(trip))] = full
                return found

            before = snapshot()
            gate = net.NetworkGate(settings, "tag")
            fake = FakeBatchOpenAI(gate)
            tag_batch.submit_batch(
                catalog,
                settings,
                vocabulary=dict(VOCABULARY),
                backends={"openai": fake},
            )
            fake.status_raw = {
                "status": "completed",
                "request_counts": {"total": 3, "completed": 3, "failed": 0},
            }
            fake.output_text = "\n".join(
                openai_line_succeeded(f"clip-{at}", answer)
                for at, answer in (
                    (1, ANSWER_ONE),
                    (2, ANSWER_TWO),
                    (3, ANSWER_THREE),
                )
            )
            tag_batch.collect_batch(
                catalog, settings, backends={"openai": fake}
            )
            after = snapshot()
            added = set(after) - set(before)
            self.assertIn(
                str(Path(".pipeline") / "tag-batch" / "001.job.json"), added
            )
            self.assertIn(
                str(Path(".pipeline") / "tag-batch" / "001.done.json"), added
            )
            self.assertTrue(
                added
                <= {
                    str(Path(".pipeline") / "tag-batch" / "001.job.json"),
                    str(Path(".pipeline") / "tag-batch" / "001.done.json"),
                    str(Path(".pipeline") / "network-log.tsv"),
                    str(Path(".pipeline") / "ai-usage.tsv"),
                }
            )
            state_dir = trip / ".pipeline" / "tag-batch"
            for name in ("001.job.json", "001.done.json"):
                size = os.stat(state_dir / name).st_size
                self.assertLessEqual(size, tag_batch.MAX_STATE_BYTES)

        with self.subTest("oversized state raises before any call"):
            with tempfile.TemporaryDirectory() as tmp:
                settings, footage, _ = self.make_project(tmp)
                catalog = self.open_catalog(settings)
                media_id, _ = self.add_clip(
                    catalog, footage, "cove-a.mov", "clip-a"
                )
                self.write_sheet(settings, catalog, media_id)
                gate = net.NetworkGate(settings, "tag")
                fake = FakeBatchOpenAI(gate)
                with mock.patch.object(tag_batch, "MAX_STATE_BYTES", 100):
                    with self.assertRaises(tag_batch.TagBatchError):
                        tag_batch.submit_batch(
                            catalog,
                            settings,
                            vocabulary=dict(VOCABULARY),
                            backends={"openai": fake},
                        )
                self.assertEqual(fake.batch_inputs, [])

        with self.subTest("existing job file is never overwritten"):
            with tempfile.TemporaryDirectory() as tmp:
                settings, footage, _ = self.make_project(tmp)
                catalog = self.open_catalog(settings)
                media_id, _ = self.add_clip(
                    catalog, footage, "cove-a.mov", "clip-a"
                )
                self.write_sheet(settings, catalog, media_id)
                planted = tag_batch._job_document(
                    "openai",
                    FAKE_OPENAI_MODEL,
                    "batch-invented-000",
                    tag_batch._utc_now(),
                    0.75,
                    False,
                    dict(VOCABULARY),
                    [],
                )
                tag_batch._write_state_file(settings, "001.job.json", planted)
                planted_path = (
                    Path(os.fspath(settings.folder))
                    / ".pipeline"
                    / "tag-batch"
                    / "001.job.json"
                )
                before_bytes = planted_path.read_bytes()
                before_mtime = os.stat(planted_path).st_mtime_ns
                gate = net.NetworkGate(settings, "tag")
                fake = FakeBatchOpenAI(gate)
                with mock.patch.object(
                    tag_batch, "_next_job_number", return_value=1
                ):
                    with self.assertRaises(tag_batch.TagBatchError) as caught:
                        tag_batch.submit_batch(
                            catalog,
                            settings,
                            vocabulary=dict(VOCABULARY),
                            backends={"openai": fake},
                        )
                self.assertIn("001.job.json", str(caught.exception))
                self.assertEqual(planted_path.read_bytes(), before_bytes)
                self.assertEqual(
                    os.stat(planted_path).st_mtime_ns, before_mtime
                )

        for bad_kind in ("symlink", "not-json", "version-2"):
            with self.subTest(job=bad_kind):
                with tempfile.TemporaryDirectory() as tmp:
                    settings, footage, _ = self.make_project(tmp)
                    catalog = self.open_catalog(settings)
                    media_id, _ = self.add_clip(
                        catalog, footage, "cove-a.mov", "clip-a"
                    )
                    self.write_sheet(settings, catalog, media_id)
                    state_dir = (
                        Path(os.fspath(settings.folder))
                        / ".pipeline"
                        / "tag-batch"
                    )
                    state_dir.mkdir(parents=True, exist_ok=True)
                    target = state_dir / "001.job.json"
                    if bad_kind == "symlink":
                        other = state_dir / "other.json"
                        other.write_text("{}\n", encoding="utf-8")
                        target.symlink_to(other)
                    elif bad_kind == "not-json":
                        target.write_text("not json\n", encoding="utf-8")
                    else:
                        target.write_text(
                            json.dumps({"version": 2}) + "\n",
                            encoding="utf-8",
                        )
                    gate = net.NetworkGate(settings, "tag")
                    fake = FakeBatchOpenAI(gate)
                    with self.assertRaises(tag_batch.TagBatchError) as caught:
                        tag_batch.submit_batch(
                            catalog,
                            settings,
                            vocabulary=dict(VOCABULARY),
                            backends={"openai": fake},
                        )
                    self.assertIn("001.job.json", str(caught.exception))
                    self.assertEqual(fake.batch_inputs, [])

        with self.subTest("state files hold no answer, transcript or key text"):
            with tempfile.TemporaryDirectory() as tmp:
                settings, footage, _ = self.make_project(tmp)
                catalog = self.open_catalog(settings)
                media_id, _ = self.add_clip(
                    catalog, footage, "cove-a.mov", "clip-a"
                )
                self.write_sheet(settings, catalog, media_id)
                transcript_words = "invented transcript words xyzzy"
                self.add_transcript(
                    catalog, media_id, tmp, transcript_words * 10
                )
                gate = net.NetworkGate(settings, "tag")
                fake = FakeBatchOpenAI(gate)
                with mock.patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "invented-key-xyzzy-12345"},
                ):
                    tag_batch.submit_batch(
                        catalog,
                        settings,
                        vocabulary=dict(VOCABULARY),
                        backends={"openai": fake},
                    )
                    fake.status_raw = {
                        "status": "completed",
                        "request_counts": {
                            "total": 1,
                            "completed": 1,
                            "failed": 0,
                        },
                    }
                    fake.output_text = openai_line_succeeded(
                        "clip-1", ANSWER_ONE
                    )
                    tag_batch.collect_batch(
                        catalog, settings, backends={"openai": fake}
                    )
                state_dir = (
                    Path(os.fspath(settings.folder))
                    / ".pipeline"
                    / "tag-batch"
                )
                state_text = "\n".join(
                    path.read_text(encoding="utf-8")
                    for path in sorted(state_dir.iterdir())
                )
                self.assertNotIn(ANSWER_ONE["description"], state_text)
                self.assertNotIn("xyzzy", state_text)
                self.assertNotIn("invented-key-xyzzy-12345", state_text)

    def test_footage_and_other_folders_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, project = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_id, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            self.write_sheet(settings, catalog, media_id)
            self.add_transcript(catalog, media_id, tmp, "invented words here")

            def snapshot(folder):
                found = {}
                for root, _, files in os.walk(folder):
                    for name in files:
                        full = Path(root) / name
                        info = os.stat(full)
                        found[os.fspath(full.relative_to(tmp))] = (
                            info.st_size,
                            info.st_mtime_ns,
                            hashlib.sha256(
                                full.read_bytes()
                            ).hexdigest(),
                        )
                return found

            def snapshot_project():
                found = {}
                for root, dirs, files in os.walk(project):
                    dirs[:] = [
                        d for d in dirs if d not in (".pipeline", "__pycache__")
                    ]
                    for name in files:
                        full = Path(root) / name
                        info = os.stat(full)
                        found[os.fspath(full.relative_to(project))] = (
                            info.st_size,
                            info.st_mtime_ns,
                            hashlib.sha256(
                                full.read_bytes()
                            ).hexdigest(),
                        )
                return found

            transcripts = Path(tmp) / "transcripts"
            sheets = (
                Path(os.fspath(settings.folder)) / ".pipeline" / "sheets"
            )
            before_footage = snapshot(footage)
            before_transcripts = snapshot(transcripts)
            before_sheets = snapshot(sheets)
            before_project = snapshot_project()
            self.assertTrue(before_footage)
            self.assertTrue(before_sheets)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeBatchOpenAI(gate)
            tag_batch.submit_batch(
                catalog,
                settings,
                vocabulary=dict(VOCABULARY),
                backends={"openai": fake},
            )
            fake.status_raw = {
                "status": "completed",
                "request_counts": {"total": 1, "completed": 1, "failed": 0},
            }
            fake.output_text = openai_line_succeeded("clip-1", ANSWER_ONE)
            tag_batch.collect_batch(
                catalog, settings, backends={"openai": fake}
            )
            self.assertEqual(snapshot(footage), before_footage)
            self.assertEqual(snapshot(transcripts), before_transcripts)
            self.assertEqual(snapshot(sheets), before_sheets)
            self.assertEqual(snapshot_project(), before_project)
            for root, _, files in os.walk(project):
                for name in files:
                    full = Path(root) / name
                    self.assertLessEqual(
                        os.stat(full).st_size, 5_000_000, f"{full} is over 5 MB"
                    )

    def test_dry_run_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, project = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            runs_before = catalog.list_runs()
            for asset in ("clip-a", "clip-b"):
                media_id, _ = self.add_clip(
                    catalog, footage, f"cove-{asset}.mov", asset
                )
                self.write_sheet(settings, catalog, media_id)
            with mock.patch.object(
                OpenAIBackend,
                "create_batch",
                side_effect=AssertionError("dry run must not call"),
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    code = tag.main(
                        ["--trip", os.fspath(project), "--batch", "--dry-run"]
                    )
            self.assertEqual(code, 0)
            lines = stdout.getvalue().splitlines()
            self.assertEqual(
                len(
                    [
                        line
                        for line in lines
                        if line.startswith("would-submit: ")
                    ]
                ),
                2,
            )
            self.assertIn(
                "would-submit 2, skipped 0, pending 0, stale 0, "
                "no-sheet 0, failed 0",
                lines,
            )
            self.assertFalse(
                (
                    Path(os.fspath(settings.folder))
                    / ".pipeline"
                    / "tag-batch"
                ).exists()
            )
            self.assertEqual(catalog.list_runs(), runs_before)
            self.assertFalse(
                (
                    Path(os.fspath(settings.folder))
                    / ".pipeline"
                    / "network-log.tsv"
                ).exists()
            )
        with self.subTest("dry run with collect exits 2"):
            with tempfile.TemporaryDirectory() as tmp:
                _, _, project = self.make_project(tmp)
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    code = tag.main(
                        [
                            "--trip",
                            os.fspath(project),
                            "--collect",
                            "--dry-run",
                        ]
                    )
                self.assertEqual(code, 2)
                self.assertEqual(len(stderr.getvalue().splitlines()), 1)

    def test_flags_exclusive_and_default_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, project = self.make_project(tmp)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = tag.main(
                    ["--trip", os.fspath(project), "--batch", "--collect"]
                )
            self.assertEqual(code, 2)
            self.assertEqual(len(stderr.getvalue().splitlines()), 1)

        with self.subTest("default path still tags one clip"):
            with tempfile.TemporaryDirectory() as tmp:
                settings, footage, project = self.make_project(tmp)
                catalog = self.open_catalog(settings)
                media_id, _ = self.add_clip(
                    catalog, footage, "cove-a.mov", "clip-a"
                )
                self.write_sheet(settings, catalog, media_id)
                good = load_tag_fixture("good-answer.json")
                with mock.patch.object(
                    OpenAIBackend, "create", return_value=openai_raw(good)
                ):
                    stdout = io.StringIO()
                    with redirect_stdout(stdout):
                        code = tag.main(["--trip", os.fspath(project)])
                self.assertEqual(code, 0)
                self.assertIn(
                    "tagged 1, skipped 0, stale 0, no-sheet 0, failed 0",
                    stdout.getvalue().splitlines(),
                )
                self.assertTrue(
                    [
                        found
                        for found in catalog.list_tags(media_id)
                        if found.source == "ai"
                    ]
                )

    # -- backend and gate boundary ------------------------------------------

    def test_real_backends_call_sdk_shapes(self):
        recorded: dict = {
            "openai_files_create": [],
            "openai_batches_create": [],
            "openai_batches_retrieve": [],
            "openai_files_content": [],
            "anthropic_batches_create": [],
            "anthropic_batches_retrieve": [],
            "anthropic_batches_results": [],
        }

        class Dumpable(dict):
            def model_dump(self):
                return dict(self)

        class StubFiles:
            def create(self, *, file, purpose):
                recorded["openai_files_create"].append((file, purpose))
                return Dumpable({"id": "file-invented-001"})

            def content(self, file_id):
                recorded["openai_files_content"].append(file_id)
                return SimpleNamespace(text="invented-result-text")

        class StubOpenAIBatches:
            def create(self, *, input_file_id, endpoint, completion_window):
                recorded["openai_batches_create"].append(
                    (input_file_id, endpoint, completion_window)
                )
                return Dumpable({"id": "batch-invented-001"})

            def retrieve(self, batch_id):
                recorded["openai_batches_retrieve"].append(batch_id)
                return Dumpable({"id": batch_id, "status": "completed"})

        class StubOpenAIClient:
            def __init__(self):
                self.files = StubFiles()
                self.batches = StubOpenAIBatches()

        class StubAnthropicBatches:
            def create(self, *, requests):
                recorded["anthropic_batches_create"].append(requests)
                return Dumpable({"id": "msgbatch-invented-001"})

            def retrieve(self, batch_id):
                recorded["anthropic_batches_retrieve"].append(batch_id)
                return Dumpable(
                    {"id": batch_id, "processing_status": "ended"}
                )

            def results(self, batch_id):
                recorded["anthropic_batches_results"].append(batch_id)
                return [Dumpable({"custom_id": "clip-1"})]

        class StubAnthropicMessages:
            def __init__(self):
                self.batches = StubAnthropicBatches()

        class StubAnthropicClient:
            def __init__(self):
                self.messages = StubAnthropicMessages()

        with tempfile.TemporaryDirectory() as tmp:
            settings, _, _ = self.make_project(tmp)
            gate = net.NetworkGate(settings, "tag")
            stub_openai = StubOpenAIClient()
            stub_anthropic = StubAnthropicClient()
            with mock.patch.object(
                NetworkGate, "openai_client", return_value=stub_openai
            ):
                backend = OpenAIBackend(gate)
                batch_id = backend.create_batch(b'{"custom_id": "clip-1"}')
                self.assertEqual(batch_id, "batch-invented-001")
                self.assertEqual(len(recorded["openai_files_create"]), 1)
                name, payload = recorded["openai_files_create"][0][0]
                self.assertEqual(recorded["openai_files_create"][0][1], "batch")
                self.assertTrue(name.endswith(".jsonl"))
                self.assertIsInstance(payload, bytes)
                self.assertEqual(len(recorded["openai_batches_create"]), 1)
                file_id, endpoint, window = recorded["openai_batches_create"][0]
                self.assertEqual(file_id, "file-invented-001")
                self.assertEqual(endpoint, "/v1/responses")
                self.assertEqual(window, "24h")
                status = backend.retrieve_batch("batch-invented-001")
                self.assertEqual(status, {"id": "batch-invented-001", "status": "completed"})
                self.assertEqual(
                    recorded["openai_batches_retrieve"], ["batch-invented-001"]
                )
                fetched = backend.fetch_batch_results(
                    "batch-invented-001",
                    {
                        "output_file_id": "file-invented-out-001",
                        "error_file_id": None,
                    },
                )
                self.assertEqual(
                    fetched,
                    {
                        "output_text": "invented-result-text",
                        "error_text": "",
                    },
                )
                self.assertEqual(
                    recorded["openai_files_content"], ["file-invented-out-001"]
                )
            with mock.patch.object(
                NetworkGate, "anthropic_client", return_value=stub_anthropic
            ):
                backend = AnthropicBackend(gate)
                batch_id = backend.create_batch([{"custom_id": "clip-1"}])
                self.assertEqual(batch_id, "msgbatch-invented-001")
                self.assertEqual(len(recorded["anthropic_batches_create"]), 1)
                self.assertEqual(
                    recorded["anthropic_batches_create"][0],
                    [{"custom_id": "clip-1"}],
                )
                status = backend.retrieve_batch("msgbatch-invented-001")
                self.assertEqual(
                    status,
                    {"id": "msgbatch-invented-001", "processing_status": "ended"},
                )
                self.assertEqual(
                    recorded["anthropic_batches_retrieve"],
                    ["msgbatch-invented-001"],
                )
                fetched = backend.fetch_batch_results(
                    "msgbatch-invented-001", status
                )
                self.assertEqual(fetched, [{"custom_id": "clip-1"}])
                self.assertEqual(
                    recorded["anthropic_batches_results"],
                    ["msgbatch-invented-001"],
                )

        with self.subTest("offline builds no stub"):
            with tempfile.TemporaryDirectory() as tmp:
                settings, _, _ = self.make_project(tmp, offline=True)
                gate = net.NetworkGate(settings, "tag")
                # The real gate refuses before importing any SDK, reading
                # any key or building any client, so these raise OfflineError
                # instead of the missing-SDK or missing-key error.
                openai_backend = OpenAIBackend(gate)
                anthropic_backend = AnthropicBackend(gate)
                with self.assertRaises(OfflineError):
                    openai_backend.create_batch(b"{}")
                with self.assertRaises(OfflineError):
                    openai_backend.retrieve_batch("batch-invented-001")
                with self.assertRaises(OfflineError):
                    openai_backend.fetch_batch_results(
                        "batch-invented-001", {}
                    )
                with self.assertRaises(OfflineError):
                    anthropic_backend.create_batch([])
                with self.assertRaises(OfflineError):
                    anthropic_backend.retrieve_batch("msgbatch-invented-001")
                with self.assertRaises(OfflineError):
                    anthropic_backend.fetch_batch_results(
                        "msgbatch-invented-001", {}
                    )
                self.assertNotIn("openai", sys.modules)
                self.assertNotIn("anthropic", sys.modules)

    def test_recorded_batch_shapes_parse(self):
        openai_backend = OpenAIBackend.__new__(OpenAIBackend)
        anthropic_backend = AnthropicBackend.__new__(AnthropicBackend)
        with self.subTest(backend="OpenAIBackend"):
            status = openai_backend.parse_batch_status(
                load_fixture("openai-status-completed.json")
            )
            self.assertEqual(status.state, "done")
            self.assertEqual(status.total, 4)
            output = (FIXTURES / "openai-output.jsonl").read_text(
                encoding="utf-8"
            )
            errors = (FIXTURES / "openai-errors.jsonl").read_text(
                encoding="utf-8"
            )
            items = openai_backend.parse_batch_results(
                {"output_text": output, "error_text": errors}
            )
            self.assertEqual(
                [(item.custom_id, item.status) for item in items],
                [
                    ("clip-1", "succeeded"),
                    ("clip-2", "errored"),
                    ("clip-3", "expired"),
                    ("clip-4", "canceled"),
                ],
            )
            self.assertEqual(
                items[0].reply.data["description"],
                "Invented lighthouse watch at dawn",
            )
            self.assertEqual(items[1].error, "status 400: invalid_request")
            self.assertNotIn("must never surface", items[1].error or "")
            self.assertEqual(items[2].error, "expired")
            self.assertEqual(items[3].error, "canceled")
        with self.subTest(backend="AnthropicBackend"):
            status = anthropic_backend.parse_batch_status(
                load_fixture("anthropic-status-ended.json")
            )
            self.assertEqual(status.state, "done")
            self.assertEqual(status.total, 4)
            entries = load_fixture("anthropic-results.json")
            items = anthropic_backend.parse_batch_results(entries)
            self.assertEqual(
                [(item.custom_id, item.status) for item in items],
                [
                    ("clip-1", "succeeded"),
                    ("clip-2", "errored"),
                    ("clip-3", "expired"),
                    ("clip-4", "canceled"),
                ],
            )
            self.assertEqual(
                items[0].reply.data["description"],
                "Invented lighthouse watch at dawn",
            )
            self.assertEqual(items[1].error, "invalid_request_error")
            self.assertNotIn("must never surface", items[1].error or "")
        for backend in (openai_backend, anthropic_backend):
            for raw, want in [
                ({"status": "validating"}, "running"),
                ({"status": "in_progress"}, "running"),
                ({"status": "finalizing"}, "running"),
                ({"status": "cancelling"}, "running"),
                ({"status": "completed"}, "done"),
                ({"status": "failed"}, "failed"),
                ({"status": "expired"}, "expired"),
                ({"status": "cancelled"}, "cancelled"),
                ({"status": "mystery"}, "failed"),
                ({"processing_status": "in_progress"}, "running"),
                ({"processing_status": "canceling"}, "running"),
                ({"processing_status": "ended"}, "done"),
                ({"processing_status": "mystery"}, "failed"),
            ]:
                with self.subTest(
                    backend=type(backend).__name__, raw=raw
                ):
                    self.assertEqual(
                        backend.parse_batch_status(raw).state, want
                    )

    def test_import_is_quiet(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = (
                "import sys; "
                f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
                "import fcp_pipeline.tag_batch; "
                "import fcp_pipeline.ai.batch; "
                "assert 'openai' not in sys.modules, 'openai imported'; "
                "assert 'anthropic' not in sys.modules, 'anthropic imported'; "
                "print('quiet-ok')"
            )
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            completed = subprocess.run(
                [sys.executable, "-I", "-c", script],
                capture_output=True,
                text=True,
                cwd=tmp,
                timeout=120,
                env=env,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("quiet-ok", completed.stdout)
            leftovers = [
                child
                for child in Path(tmp).iterdir()
                if child.name != "__pycache__"
            ]
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
