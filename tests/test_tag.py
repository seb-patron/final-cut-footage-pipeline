"""Tests for fcp_pipeline.tag, the single-clip tagging step.

No test here makes a real network call, reads a real key, or needs an
installed SDK: socket functions fail when called, fake backends override
``create`` only (building and parsing stay real), and every project is
a temporary folder with an invented project.toml. Sheets are synthetic
header-only JPEGs, except one test that builds a real sheet with ffmpeg
(skipped where ffmpeg is missing). Every name, place, model id and path
here is invented.
"""

import base64
import hashlib
import io
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path
from unittest import mock

from fcp_pipeline import net, tag
from fcp_pipeline.ai import (
    AiConfigError,
    AnthropicBackend,
    OpenAIBackend,
)
from fcp_pipeline.catalog import (
    Catalog,
    MediaRecord,
    TagRecord,
    TranscriptRecord,
)
from fcp_pipeline.net import OfflineError, set_offline
from fcp_pipeline.trip import load_trip

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ai" / "tag-responses"

FAKE_OPENAI_MODEL = "fake-openai-tag-model"
FAKE_ANTHROPIC_MODEL = "fake-anthropic-tag-model"

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


def load_fixture(name):
    """Return the parsed JSON fixture ``name``."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


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


class TestTag(unittest.TestCase):
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

    # -- project scaffolding -------------------------------------------

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

    def vocabulary(self, catalog, settings):
        """The trip vocabulary used across these tests."""
        return tag.build_vocabulary(
            catalog,
            settings,
            extra={"subject": ["invented-boat", "invented-gull"]},
        )

    def legacy_vocabulary(self):
        """Pin the lowercase folders the recorded fixtures were written against."""
        return {
            "place": ("Invented Cove",),
            "gear": ("Invented Maker Cam A",),
            "subject": ("invented-boat", "invented-gull"),
        }

    def openai_text(self, request):
        """Return the instructions text of a recorded OpenAI request."""
        parts = request["input"][0]["content"]
        texts = [part["text"] for part in parts if part.get("type") == "input_text"]
        self.assertEqual(len(texts), 1)
        return texts[0]

    def transcript_section(self, instructions):
        """Return the words between the transcript marker lines."""
        start_marker = "\nBEGIN TRANSCRIPT\n"
        end_marker = "\nEND TRANSCRIPT"
        start = instructions.index(start_marker) + len(start_marker)
        end = instructions.index(end_marker, start)
        return instructions[start:end].split()

    def assert_no_video_or_file_keys(self, request):
        """Walk every key of ``request``; none names video or file media."""

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    self.assertNotIn("video", key)
                    self.assertNotIn("file", key)
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(request)

    # -- the request ----------------------------------------------------

    def test_request_has_one_image_no_video(self):
        good = load_fixture("good-answer.json")
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_id, clip_path = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            sheet_bytes = self.write_sheet(settings, catalog, media_id)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeOpenAI(gate, [load_fixture("openai-good.json")])
            result = tag.tag_catalog(
                catalog,
                settings,
                vocabulary=self.vocabulary(catalog, settings),
                backends={"openai": fake},
            )
            self.assertEqual(result.tagged, 1)
            self.assertEqual(len(fake.requests), 1)
            request = fake.requests[0]
            parts = request["input"][0]["content"]
            images = [p for p in parts if p.get("type") == "input_image"]
            self.assertEqual(len(images), 1)
            self.assertEqual(len(parts), 2)
            uri = images[0]["image_url"]
            self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
            payload = base64.b64decode(uri.split(",", 1)[1])
            self.assertEqual(payload, sheet_bytes)
            self.assertLessEqual(tag.jpeg_size(payload)[0], 1536)
            self.assertLessEqual(tag.jpeg_size(payload)[1], 864)
            self.assert_no_video_or_file_keys(request)
            text = self.openai_text(request)
            self.assertIn("camera=Invented Maker Cam A; duration=60.000s", text)
            self.assertNotIn(clip_path, text)
            self.assertNotIn("cove-a.mov", text)

            with self.subTest("anthropic"):
                settings_b, footage_b, _ = self.make_project(
                    Path(tmp) / "b", provider="anthropic"
                )
                catalog_b = self.open_catalog(settings_b)
                media_b, _ = self.add_clip(catalog_b, footage_b, "cove-b.mov", "clip-b")
                sheet_b = self.write_sheet(settings_b, catalog_b, media_b)
                gate_b = net.NetworkGate(settings_b, "tag")
                fake_b = FakeAnthropic(
                    gate_b, [load_fixture("anthropic-good.json")]
                )
                result_b = tag.tag_catalog(
                    catalog_b,
                    settings_b,
                    vocabulary=self.vocabulary(catalog_b, settings_b),
                    backends={"anthropic": fake_b},
                )
                self.assertEqual(result_b.tagged, 1)
                self.assertEqual(len(fake_b.requests), 1)
                blocks = fake_b.requests[0]["messages"][0]["content"]
                pictures = [b for b in blocks if b.get("type") == "image"]
                self.assertEqual(len(pictures), 1)
                source = pictures[0]["source"]
                self.assertEqual(source["type"], "base64")
                self.assertEqual(source["media_type"], "image/jpeg")
                self.assertEqual(base64.b64decode(source["data"]), sheet_b)
                self.assert_no_video_or_file_keys(fake_b.requests[0])

            with self.subTest("oversize sheet is failed with no call"):
                settings_c, footage_c, _ = self.make_project(Path(tmp) / "c")
                catalog_c = self.open_catalog(settings_c)
                media_c, _ = self.add_clip(catalog_c, footage_c, "cove-c.mov", "clip-c")
                self.write_sheet(
                    settings_c, catalog_c, media_c, data=fake_jpeg(1537, 864)
                )
                gate_c = net.NetworkGate(settings_c, "tag")
                fake_c = FakeOpenAI(gate_c, [openai_raw(good)])
                result_c = tag.tag_catalog(
                    catalog_c,
                    settings_c,
                    vocabulary=self.vocabulary(catalog_c, settings_c),
                    backends={"openai": fake_c},
                )
                self.assertEqual(result_c.failed, 1)
                self.assertEqual(fake_c.requests, [])

            with self.subTest("checksum mismatch is failed with no call"):
                settings_d, footage_d, _ = self.make_project(Path(tmp) / "d")
                catalog_d = self.open_catalog(settings_d)
                media_d, _ = self.add_clip(catalog_d, footage_d, "cove-d.mov", "clip-d")
                sheet_d = self.write_sheet(settings_d, catalog_d, media_d)
                from fcp_pipeline import sheets as sheets_module

                row = [r for r in catalog_d.list_media() if r.id == media_d][0]
                _, sheet_name, _ = sheets_module._sheet_names(row)
                sheet_path = (
                    Path(os.fspath(settings_d.pipeline)) / "sheets" / sheet_name
                )
                sheet_path.write_bytes(sheet_d + b"\x00")
                gate_d = net.NetworkGate(settings_d, "tag")
                fake_d = FakeOpenAI(gate_d, [openai_raw(good)])
                result_d = tag.tag_catalog(
                    catalog_d,
                    settings_d,
                    vocabulary=self.vocabulary(catalog_d, settings_d),
                    backends={"openai": fake_d},
                )
                self.assertEqual(result_d.failed, 1)
                self.assertEqual(fake_d.requests, [])

    def test_transcript_capped_at_300_words(self):
        lines = "\n".join(
            " ".join(f"w{n:03d}" for n in range(start, start + 45))
            for start in range(1, 451, 45)
        )
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_id, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            self.write_sheet(settings, catalog, media_id)
            self.add_transcript(catalog, media_id, tmp, lines)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeOpenAI(gate, [openai_raw(load_fixture("good-answer.json"))])
            result = tag.tag_catalog(
                catalog,
                settings,
                vocabulary=self.vocabulary(catalog, settings),
                backends={"openai": fake},
            )
            self.assertEqual(result.tagged, 1)
            section = self.transcript_section(self.openai_text(fake.requests[0]))
            self.assertEqual(len(section), 300)
            self.assertEqual(section[-1], "w300")
            self.assertNotIn("w301", section)
            self.assertEqual(tag.first_words("a  b\nc", 2), "a b")

            with self.subTest("no transcript record is a note, not a failure"):
                settings_b, footage_b, _ = self.make_project(Path(tmp) / "b")
                catalog_b = self.open_catalog(settings_b)
                media_b, _ = self.add_clip(catalog_b, footage_b, "cove-b.mov", "clip-b")
                self.write_sheet(settings_b, catalog_b, media_b)
                gate_b = net.NetworkGate(settings_b, "tag")
                fake_b = FakeOpenAI(
                    gate_b, [openai_raw(load_fixture("good-answer.json"))]
                )
                result_b = tag.tag_catalog(
                    catalog_b,
                    settings_b,
                    vocabulary=self.vocabulary(catalog_b, settings_b),
                    backends={"openai": fake_b},
                )
                self.assertEqual(result_b.tagged, 1)
                section_b = self.transcript_section(
                    self.openai_text(fake_b.requests[0])
                )
                self.assertEqual(section_b, [])
                self.assertTrue(result_b.entries[0].notes)

            with self.subTest("missing transcript file is a note, not a failure"):
                settings_c, footage_c, _ = self.make_project(Path(tmp) / "c")
                catalog_c = self.open_catalog(settings_c)
                media_c, _ = self.add_clip(catalog_c, footage_c, "cove-c.mov", "clip-c")
                self.write_sheet(settings_c, catalog_c, media_c)
                catalog_c.add_transcript(
                    TranscriptRecord(
                        media_id=media_c,
                        path=os.fspath(Path(tmp) / "gone" / "missing.txt"),
                    )
                )
                gate_c = net.NetworkGate(settings_c, "tag")
                fake_c = FakeOpenAI(
                    gate_c, [openai_raw(load_fixture("good-answer.json"))]
                )
                result_c = tag.tag_catalog(
                    catalog_c,
                    settings_c,
                    vocabulary=self.vocabulary(catalog_c, settings_c),
                    backends={"openai": fake_c},
                )
                self.assertEqual(result_c.tagged, 1)
                section_c = self.transcript_section(
                    self.openai_text(fake_c.requests[0])
                )
                self.assertEqual(section_c, [])
                self.assertTrue(result_c.entries[0].notes)

    def test_cutoff_splits_plain_and_check(self):
        edge = load_fixture("edge-answer.json")
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_id, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            self.write_sheet(settings, catalog, media_id)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeOpenAI(gate, [openai_raw(edge)])
            result = tag.tag_catalog(
                catalog,
                settings,
                vocabulary=self.legacy_vocabulary(),
                backends={"openai": fake},
            )
            self.assertEqual(result.tagged, 1)

            def slot(folder, start, duration):
                return [
                    t
                    for t in catalog.list_tags(media_id)
                    if t.folder == folder
                    and t.start == start
                    and t.duration == duration
                ]

            boat = slot("subject", Fraction(0), Fraction(10, 3))
            self.assertEqual(len(boat), 1)
            self.assertEqual(boat[0].status, "plain")
            self.assertEqual(boat[0].source, "ai")
            self.assertAlmostEqual(boat[0].confidence, 0.86)
            gull = slot("subject", Fraction(10), Fraction(20, 3))
            self.assertEqual(len(gull), 1)
            self.assertEqual(gull[0].status, "check")
            cove = slot("place", Fraction(110, 3), Fraction(20, 3))
            self.assertEqual(len(cove), 1)
            self.assertEqual(cove[0].status, "plain")
            self.assertAlmostEqual(cove[0].confidence, 0.75)

            with self.subTest("cutoff 0.9 demotes the 0.86 tag"):
                settings_b, footage_b, _ = self.make_project(Path(tmp) / "b")
                catalog_b = self.open_catalog(settings_b)
                media_b, _ = self.add_clip(
                    catalog_b, footage_b, "cove-b.mov", "clip-b"
                )
                self.write_sheet(settings_b, catalog_b, media_b)
                gate_b = net.NetworkGate(settings_b, "tag")
                fake_b = FakeOpenAI(gate_b, [openai_raw(edge)])
                tag.tag_catalog(
                    catalog_b,
                    settings_b,
                    vocabulary=self.vocabulary(catalog_b, settings_b),
                    cutoff=0.9,
                    backends={"openai": fake_b},
                )
                boat_b = [
                    t
                    for t in catalog_b.list_tags(media_b)
                    if t.folder == "subject"
                    and t.start == Fraction(0)
                    and t.duration == Fraction(10, 3)
                ]
                self.assertEqual(len(boat_b), 1)
                self.assertEqual(boat_b[0].status, "check")

            with self.subTest("bad cutoff raises before any call"):
                settings_c, footage_c, _ = self.make_project(Path(tmp) / "c")
                catalog_c = self.open_catalog(settings_c)
                media_c, _ = self.add_clip(
                    catalog_c, footage_c, "cove-c.mov", "clip-c"
                )
                self.write_sheet(settings_c, catalog_c, media_c)
                gate_c = net.NetworkGate(settings_c, "tag")
                fake_c = FakeOpenAI(gate_c, [openai_raw(edge)])
                with self.assertRaises(tag.TagError):
                    tag.tag_catalog(
                        catalog_c,
                        settings_c,
                        vocabulary=self.vocabulary(catalog_c, settings_c),
                        cutoff=1.5,
                        backends={"openai": fake_c},
                    )
                self.assertEqual(fake_c.requests, [])

    def test_range_outside_clip_dropped(self):
        edge = load_fixture("edge-answer.json")
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_id, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            self.write_sheet(settings, catalog, media_id)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeOpenAI(gate, [openai_raw(edge)])
            result = tag.tag_catalog(
                catalog,
                settings,
                vocabulary=self.vocabulary(catalog, settings),
                backends={"openai": fake},
            )
            self.assertEqual(result.tagged, 1)
            entry = result.entries[0]
            outside = [n for n in entry.notes if "outside the clip" in n]
            self.assertGreaterEqual(len(outside), 3)
            ranges = [
                t for t in catalog.list_tags(media_id) if t.duration is not None
            ]
            self.assertTrue(ranges)

            with self.subTest("bad confidence names its own rule"):
                settings_b, footage_b, _ = self.make_project(Path(tmp) / "b")
                catalog_b = self.open_catalog(settings_b)
                media_b, _ = self.add_clip(
                    catalog_b, footage_b, "cove-b.mov", "clip-b"
                )
                self.write_sheet(settings_b, catalog_b, media_b)
                answer = {
                    "description": "Invented cove survey",
                    "shot": "wide",
                    "ranges": [
                        {
                            "start": 1.0,
                            "end": 2.0,
                            "tags": [
                                {"folder": "subject", "value": "invented-boat"}
                            ],
                            "confidence": 1.5,
                        },
                        {
                            "start": 11.0,
                            "end": 12.0,
                            "tags": [
                                {"folder": "subject", "value": "invented-gull"}
                            ],
                            "confidence": 0.8,
                        },
                    ],
                    "problems": [],
                    "outline_hint": None,
                }
                gate_b = net.NetworkGate(settings_b, "tag")
                fake_b = FakeOpenAI(gate_b, [openai_raw(answer)])
                result_b = tag.tag_catalog(
                    catalog_b,
                    settings_b,
                    vocabulary=self.vocabulary(catalog_b, settings_b),
                    backends={"openai": fake_b},
                )
                self.assertEqual(result_b.tagged, 1)
                self.assertTrue(
                    any(
                        "confidence out of range" in n
                        for n in result_b.entries[0].notes
                    )
                )
                kept = [
                    t
                    for t in catalog_b.list_tags(media_b)
                    if t.folder == "subject" and t.value == "invented-gull"
                ]
                self.assertEqual(len(kept), 1)

    def test_unknown_vocabulary_dropped(self):
        edge = load_fixture("edge-answer.json")
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_id, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            self.write_sheet(settings, catalog, media_id)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeOpenAI(gate, [openai_raw(edge)])
            result = tag.tag_catalog(
                catalog,
                settings,
                vocabulary=self.legacy_vocabulary(),
                backends={"openai": fake},
            )
            self.assertEqual(result.tagged, 1)
            stored = catalog.list_tags(media_id)
            self.assertFalse(
                [t for t in stored if t.value == "invented-unknown"]
            )
            self.assertFalse(
                [t for t in stored if t.folder == "invented-nofolder"]
            )
            vocab_notes = [
                n for n in result.entries[0].notes if "not in the trip vocabulary" in n
            ]
            self.assertGreaterEqual(len(vocab_notes), 2)
            gear = [
                t
                for t in stored
                if t.folder == "gear" and t.value == "Invented Maker Cam A"
            ]
            self.assertEqual(len(gear), 1)
            dup_slot = [
                t
                for t in stored
                if t.folder == "subject"
                and t.start == Fraction(50)
                and t.duration == Fraction(20, 3)
            ]
            self.assertEqual(len(dup_slot), 1)
            self.assertEqual(dup_slot[0].value, "invented-gull")
            self.assertAlmostEqual(dup_slot[0].confidence, 0.8)
            self.assertTrue(
                any(
                    "duplicate in the same range" in n
                    for n in result.entries[0].notes
                )
            )

    def test_huge_answer_numbers_rejected_as_invalid(self):
        cases = {
            "huge int start": (10**4000, 2.0, 0.8),
            "huge float start": (1e308, 2.0, 0.8),
            "huge int confidence": (1.0, 2.0, 10**4000),
        }
        for label, (start, end, confidence) in cases.items():
            with self.subTest(label):
                with tempfile.TemporaryDirectory() as tmp:
                    settings, footage, _ = self.make_project(tmp)
                    catalog = self.open_catalog(settings)
                    media_id, _ = self.add_clip(
                        catalog, footage, "cove-a.mov", "clip-a"
                    )
                    self.write_sheet(settings, catalog, media_id)
                    answer = {
                        "description": "Invented cove survey",
                        "shot": "wide",
                        "ranges": [
                            {
                                "start": start,
                                "end": end,
                                "tags": [
                                    {
                                        "folder": "subject",
                                        "value": "invented-boat",
                                    }
                                ],
                                "confidence": confidence,
                            }
                        ],
                        "problems": [],
                        "outline_hint": None,
                    }
                    gate = net.NetworkGate(settings, "tag")
                    fake = FakeOpenAI(gate, [openai_raw(answer)])
                    result = tag.tag_catalog(
                        catalog,
                        settings,
                        vocabulary=self.vocabulary(catalog, settings),
                        backends={"openai": fake},
                    )
                    self.assertEqual(result.tagged, 0)
                    self.assertEqual(result.failed, 1)
                    entry = result.entries[0]
                    self.assertEqual(entry.status, "failed")
                    self.assertIn("invalid answer", entry.reason)
                    self.assertEqual(catalog.list_tags(media_id), [])

    def test_vocabulary_uses_day_place_gear_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_a, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            media_b, _ = self.add_clip(catalog, footage, "cove-b.mov", "clip-b")
            long_slate = "slate line " * 30
            self.assertGreater(len(long_slate), tag._MAX_SLATE_VOCAB_CHARS)
            self.assertTrue(
                catalog.put_tag(
                    TagRecord(
                        media_id=media_a,
                        folder="Slate",
                        value=long_slate,
                        start=None,
                        duration=None,
                        source="slate",
                        confidence=None,
                        status="plain",
                    )
                )
            )
            self.assertTrue(
                catalog.put_tag(
                    TagRecord(
                        media_id=media_a,
                        folder="Day",
                        value="Day 1",
                        start=None,
                        duration=None,
                        source="metadata",
                        confidence=None,
                        status="plain",
                    )
                )
            )
            self.assertTrue(
                catalog.put_tag(
                    TagRecord(
                        media_id=media_b,
                        folder="Slate",
                        value="Harbor sign",
                        start=None,
                        duration=None,
                        source="slate",
                        confidence=None,
                        status="plain",
                    )
                )
            )
            vocabulary = tag.build_vocabulary(catalog, settings)
            self.assertEqual(vocabulary["Place"], ("Invented Cove",))
            self.assertEqual(vocabulary["Gear"], ("Invented Maker Cam A",))
            self.assertEqual(vocabulary["Day"], ("Day 1",))
            self.assertNotIn("place", vocabulary)
            self.assertNotIn("gear", vocabulary)
            self.assertIn("Harbor sign", vocabulary["Slate"])
            for values in vocabulary.values():
                self.assertNotIn(long_slate, values)

    def test_ranges_not_a_list_is_a_note(self):
        kept, notes = tag.check_answer(
            {"ranges": "not-a-list"},
            {"subject": ("invented-boat",)},
            Fraction(60),
            [Fraction(0), Fraction(60)],
        )
        self.assertEqual(kept, [])
        self.assertEqual(notes, ["ranges: dropped (not a list of ranges)"])

    def test_manifest_edges_clamp_out_of_range_tile_times(self):
        manifest = {
            "tiles": [
                {"index": 0, "time": "10/1"},
                {"index": 1, "time": "-5/1"},
                {"index": 2, "time": "61/1"},
                {"index": 3, "time": "not-a-number"},
                {"index": 4, "time": "70/1"},
            ]
        }
        self.assertEqual(
            tag._manifest_edges(manifest, Fraction(60)),
            [Fraction(0), Fraction(10), Fraction(60)],
        )

    def test_times_snap_to_tile_edges(self):
        good = load_fixture("good-answer.json")
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_id, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            self.write_sheet(settings, catalog, media_id)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeOpenAI(gate, [openai_raw(good)])
            result = tag.tag_catalog(
                catalog,
                settings,
                vocabulary=self.legacy_vocabulary(),
                backends={"openai": fake},
            )
            self.assertEqual(result.tagged, 1)

            def slot(folder, start, duration):
                return [
                    t
                    for t in catalog.list_tags(media_id)
                    if t.folder == folder
                    and t.start == start
                    and t.duration == duration
                ]

            first = slot("place", Fraction(0), Fraction(10))
            self.assertEqual(len(first), 1)
            second = slot("subject", Fraction(70, 3), Fraction(40, 3))
            self.assertEqual(len(second), 1)

            with self.subTest("a range ending at the duration keeps end 60"):
                settings_b, footage_b, _ = self.make_project(Path(tmp) / "b")
                catalog_b = self.open_catalog(settings_b)
                media_b, _ = self.add_clip(
                    catalog_b, footage_b, "cove-b.mov", "clip-b"
                )
                self.write_sheet(settings_b, catalog_b, media_b)
                answer = {
                    "description": "Invented cove survey",
                    "shot": "wide",
                    "ranges": [
                        {
                            "start": 55.0,
                            "end": 60.0,
                            "tags": [
                                {"folder": "subject", "value": "invented-boat"}
                            ],
                            "confidence": 0.9,
                        }
                    ],
                    "problems": [],
                    "outline_hint": None,
                }
                gate_b = net.NetworkGate(settings_b, "tag")
                fake_b = FakeOpenAI(gate_b, [openai_raw(answer)])
                result_b = tag.tag_catalog(
                    catalog_b,
                    settings_b,
                    vocabulary=self.vocabulary(catalog_b, settings_b),
                    backends={"openai": fake_b},
                )
                self.assertEqual(result_b.tagged, 1)
                tail = [
                    t
                    for t in catalog_b.list_tags(media_b)
                    if t.folder == "subject"
                ]
                self.assertEqual(len(tail), 1)
                self.assertEqual(tail[0].start, Fraction(50))
                self.assertEqual(tail[0].start + tail[0].duration, Fraction(60))

            for owner, media in ((catalog, media_id), (catalog_b, media_b)):
                for record in owner.list_tags(media):
                    if record.start is not None:
                        self.assertIsInstance(record.start, Fraction)
                    if record.duration is not None:
                        self.assertIsInstance(record.duration, Fraction)

    def test_stores_note_todo_shot_and_hint(self):
        good = load_fixture("good-answer.json")
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_id, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            self.write_sheet(settings, catalog, media_id)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeOpenAI(gate, [openai_raw(good)])
            result = tag.tag_catalog(
                catalog,
                settings,
                vocabulary=self.vocabulary(catalog, settings),
                backends={"openai": fake},
            )
            self.assertEqual(result.tagged, 1)

            def whole(folder):
                return [
                    t
                    for t in catalog.list_tags(media_id)
                    if t.folder == folder
                    and t.start is None
                    and t.duration is None
                ]

            shot = whole("shot")
            self.assertEqual(len(shot), 1)
            self.assertEqual(shot[0].value, "wide")
            self.assertEqual(shot[0].status, "plain")
            self.assertEqual(shot[0].source, "ai")
            self.assertIsNone(shot[0].confidence)
            note = whole("note")
            self.assertEqual(len(note), 1)
            self.assertEqual(note[0].value, "Invented harbor walk at dusk")
            self.assertEqual(note[0].source, "ai")
            todo = whole("todo")
            self.assertEqual(len(todo), 1)
            self.assertEqual(todo[0].value, "lens smudge in the last tile")
            hint = whole("outline-hint")
            self.assertEqual(len(hint), 1)
            self.assertEqual(hint[0].value, "Arrival at the cove")
            self.assertEqual(hint[0].status, "check")
            self.assertEqual(hint[0].source, "ai")

            with self.subTest("an empty description stores no note"):
                settings_b, footage_b, _ = self.make_project(Path(tmp) / "b")
                catalog_b = self.open_catalog(settings_b)
                media_b, _ = self.add_clip(
                    catalog_b, footage_b, "cove-b.mov", "clip-b"
                )
                self.write_sheet(settings_b, catalog_b, media_b)
                answer = dict(good, description="   ")
                gate_b = net.NetworkGate(settings_b, "tag")
                fake_b = FakeOpenAI(gate_b, [openai_raw(answer)])
                tag.tag_catalog(
                    catalog_b,
                    settings_b,
                    vocabulary=self.vocabulary(catalog_b, settings_b),
                    backends={"openai": fake_b},
                )
                notes_b = [
                    t
                    for t in catalog_b.list_tags(media_b)
                    if t.folder == "note"
                ]
                self.assertEqual(notes_b, [])

    def test_malformed_retried_then_failed(self):
        malformed = load_fixture("malformed-answer.json")
        good = load_fixture("good-answer.json")
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_a, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            media_b, _ = self.add_clip(catalog, footage, "cove-b.mov", "clip-b")
            self.write_sheet(settings, catalog, media_a)
            self.write_sheet(settings, catalog, media_b)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeOpenAI(
                gate,
                [
                    openai_raw(malformed),
                    openai_raw(malformed),
                    openai_raw(good),
                ],
            )
            result = tag.tag_catalog(
                catalog,
                settings,
                vocabulary=self.vocabulary(catalog, settings),
                backends={"openai": fake},
            )
            self.assertEqual(len(fake.requests), 3)
            first, second = result.entries
            self.assertEqual(first.status, "failed")
            self.assertIn("tag", first.reason)
            self.assertIn("openai", first.reason)
            self.assertNotIn("Invented harbor", first.reason)
            self.assertEqual(catalog.list_tags(media_a), [])
            self.assertEqual(second.status, "tagged")
            self.assertEqual(
                result.outcome, "tagged 1, skipped 0, stale 0, no-sheet 0, failed 1"
            )
            self.assertEqual(result.failed, 1)

            with self.subTest("malformed then good is tagged after two calls"):
                settings_b, footage_b, _ = self.make_project(Path(tmp) / "b")
                catalog_b = self.open_catalog(settings_b)
                media_c, _ = self.add_clip(
                    catalog_b, footage_b, "cove-c.mov", "clip-c"
                )
                self.write_sheet(settings_b, catalog_b, media_c)
                gate_b = net.NetworkGate(settings_b, "tag")
                fake_b = FakeOpenAI(
                    gate_b, [openai_raw(malformed), openai_raw(good)]
                )
                result_b = tag.tag_catalog(
                    catalog_b,
                    settings_b,
                    vocabulary=self.vocabulary(catalog_b, settings_b),
                    backends={"openai": fake_b},
                )
                self.assertEqual(result_b.tagged, 1)
                self.assertEqual(len(fake_b.requests), 2)

    def test_locked_tag_untouched(self):
        good = load_fixture("good-answer.json")
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_id, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            self.write_sheet(settings, catalog, media_id)
            seeded = [
                TagRecord(
                    media_id=media_id,
                    folder="place",
                    value="Locked Cove",
                    start=Fraction(0),
                    duration=Fraction(10),
                    source="editor",
                    confidence=None,
                    status="locked",
                ),
                TagRecord(
                    media_id=media_id,
                    folder="subject",
                    value="Locked Boat",
                    start=Fraction(70, 3),
                    duration=Fraction(40, 3),
                    source="ai",
                    confidence=0.5,
                    status="locked",
                ),
                TagRecord(
                    media_id=media_id,
                    folder="todo",
                    value="Old todo",
                    start=None,
                    duration=None,
                    source="ai",
                    confidence=None,
                    status="deleted",
                ),
                TagRecord(
                    media_id=media_id,
                    folder="shot",
                    value="medium",
                    start=None,
                    duration=None,
                    source="slate",
                    confidence=None,
                    status="plain",
                ),
                TagRecord(
                    media_id=media_id,
                    folder="note",
                    value="Old note",
                    start=None,
                    duration=None,
                    source="ai",
                    confidence=None,
                    status="plain",
                ),
            ]
            for record in seeded:
                self.assertTrue(catalog.put_tag(record))
            before = {
                (t.folder, t.start, t.duration): t for t in catalog.list_tags(media_id)
            }
            gate = net.NetworkGate(settings, "tag")
            fake = FakeOpenAI(gate, [openai_raw(good)])
            result = tag.tag_catalog(
                catalog,
                settings,
                vocabulary=self.vocabulary(catalog, settings),
                retag=True,
                backends={"openai": fake},
            )
            self.assertEqual(result.tagged, 1)
            after = {
                (t.folder, t.start, t.duration): t for t in catalog.list_tags(media_id)
            }
            for record in seeded[:4]:
                key = (record.folder, record.start, record.duration)
                kept = after[key]
                self.assertEqual(kept.id, before[key].id)
                self.assertEqual(kept.value, record.value)
                self.assertEqual(kept.status, record.status)
                self.assertEqual(kept.source, record.source)
                self.assertEqual(kept.confidence, record.confidence)
            self.assertGreaterEqual(len(result.entries[0].notes), 4)
            note = after[("note", None, None)]
            self.assertEqual(note.value, "Invented harbor walk at dusk")
            self.assertEqual(note.source, "ai")

    def test_same_answer_either_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            projects = {}
            for provider in ("openai", "anthropic"):
                settings, footage, _ = self.make_project(
                    Path(tmp) / provider, provider=provider
                )
                catalog = self.open_catalog(settings)
                media_id, _ = self.add_clip(
                    catalog, footage, "cove-a.mov", "clip-a"
                )
                self.write_sheet(settings, catalog, media_id)
                gate = net.NetworkGate(settings, "tag")
                if provider == "openai":
                    fake_o = FakeOpenAI(
                        gate, [load_fixture("openai-good.json")]
                    )
                    fake_a = FakeAnthropic(gate, [])
                    backends = {"openai": fake_o, "anthropic": fake_a}
                else:
                    fake_a = FakeAnthropic(
                        gate, [load_fixture("anthropic-good.json")]
                    )
                    fake_o = FakeOpenAI(gate, [])
                    backends = {"openai": fake_o, "anthropic": fake_a}
                result = tag.tag_catalog(
                    catalog,
                    settings,
                    vocabulary=self.vocabulary(catalog, settings),
                    backends=backends,
                )
                self.assertEqual(result.tagged, 1)
                projects[provider] = (
                    catalog,
                    media_id,
                    fake_o,
                    fake_a,
                )
            catalog_o, media_o, fake_o, idle_a = projects["openai"]
            catalog_a, media_a, idle_o, fake_a = projects["anthropic"]
            self.assertEqual(len(fake_o.requests), 1)
            self.assertEqual(idle_a.requests, [])
            self.assertEqual(len(fake_a.requests), 1)
            self.assertEqual(idle_o.requests, [])

            def signature(catalog, media_id):
                return sorted(
                    (
                        t.folder,
                        t.value,
                        str(t.start),
                        str(t.duration),
                        t.source,
                        t.confidence,
                        t.status,
                    )
                    for t in catalog.list_tags(media_id)
                )

            self.assertEqual(
                signature(catalog_o, media_o), signature(catalog_a, media_a)
            )

    def test_offline_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp, offline=True)
            catalog = self.open_catalog(settings)
            media_id, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            self.write_sheet(settings, catalog, media_id)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeOpenAI(
                gate, [openai_raw(load_fixture("good-answer.json"))]
            )
            with self.assertRaises(OfflineError) as caught:
                tag.tag_catalog(
                    catalog,
                    settings,
                    vocabulary=self.vocabulary(catalog, settings),
                    backends={"openai": fake},
                )
            self.assertIn("offline", str(caught.exception))
            self.assertEqual(fake.requests, [])
            self.assertEqual(catalog.list_tags(media_id), [])
            log_path = Path(os.fspath(settings.folder)) / ".pipeline" / "network-log.tsv"
            lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("refused offline", lines[1])
            runs = catalog.list_runs()
            self.assertEqual(len(runs), 1)
            self.assertTrue(runs[0].outcome.startswith("stopped"))
            usage_path = (
                Path(os.fspath(settings.folder)) / ".pipeline" / "ai-usage.tsv"
            )
            self.assertFalse(usage_path.exists())

            with self.subTest("main --offline returns 3 on stderr"):
                settings_b, footage_b, _ = self.make_project(Path(tmp) / "b")
                catalog_b = self.open_catalog(settings_b)
                media_b, _ = self.add_clip(
                    catalog_b, footage_b, "cove-b.mov", "clip-b"
                )
                self.write_sheet(settings_b, catalog_b, media_b)
                catalog_b.close()
                self._catalogs.remove(catalog_b)
                stderr = io.StringIO()
                try:
                    with redirect_stderr(stderr):
                        code = tag.main(
                            ["--trip", os.fspath(settings_b.folder), "--offline"]
                        )
                finally:
                    set_offline(False)
                self.assertEqual(code, 3)
                error_lines = stderr.getvalue().splitlines()
                self.assertEqual(len(error_lines), 1)
                self.assertTrue(error_lines[0].startswith("error:"))
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_missing_model_stops_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir(parents=True, exist_ok=True)
            (project / "project.toml").write_text(
                'title = "Invented Tag Survey"\n'
                "[pipeline]\n"
                'timezone = "UTC"\n'
                "start_date = 2020-06-15\n"
                '[[pipeline.places]]\nname = "Invented Cove"\n',
                encoding="utf-8",
            )
            settings = load_trip(project)
            footage = Path(tmp) / "footage"
            footage.mkdir(parents=True, exist_ok=True)
            catalog = self.open_catalog(settings)
            media_id, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            self.write_sheet(settings, catalog, media_id)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeOpenAI(
                gate, [openai_raw(load_fixture("good-answer.json"))]
            )
            with self.assertRaises(AiConfigError) as caught:
                tag.tag_catalog(
                    catalog,
                    settings,
                    vocabulary=self.vocabulary(catalog, settings),
                    backends={"openai": fake},
                )
            self.assertIn("[pipeline.ai.tag].model", str(caught.exception))
            self.assertEqual(fake.requests, [])
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = tag.main(["--trip", os.fspath(project)])
            self.assertEqual(code, 2)
            self.assertIn("[pipeline.ai.tag].model", stderr.getvalue())

    def test_skips_stale_no_sheet_and_tagged(self):
        good = load_fixture("good-answer.json")
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_stale, path_stale = self.add_clip(
                catalog, footage, "stale.mov", "clip-stale"
            )
            media_bare, _ = self.add_clip(catalog, footage, "bare.mov", "clip-bare")
            media_done, _ = self.add_clip(catalog, footage, "done.mov", "clip-done")
            self.write_sheet(settings, catalog, media_stale)
            self.write_sheet(settings, catalog, media_done)
            catalog.put_tag(
                TagRecord(
                    media_id=media_done,
                    folder="note",
                    value="Already there",
                    start=None,
                    duration=None,
                    source="ai",
                    confidence=None,
                    status="plain",
                )
            )
            with open(path_stale, "ab") as handle:
                handle.write(b"changed")
            gate = net.NetworkGate(settings, "tag")
            fake = FakeOpenAI(gate, [openai_raw(good)])
            result = tag.tag_catalog(
                catalog,
                settings,
                vocabulary=self.vocabulary(catalog, settings),
                backends={"openai": fake},
            )
            by_status = {entry.path: entry for entry in result.entries}
            self.assertEqual(by_status[path_stale].status, "stale")
            bare = [e for e in result.entries if e.status == "no-sheet"][0]
            self.assertEqual(catalog.get_media(bare.path).id, media_bare)
            done = [
                e
                for e in result.entries
                if catalog.get_media(e.path).id == media_done
            ][0]
            self.assertEqual(done.status, "skipped")
            self.assertEqual(fake.requests, [])
            for entry in result.entries:
                self.assertTrue(entry.reason)
            self.assertEqual(
                result.outcome, "tagged 0, skipped 1, stale 1, no-sheet 1, failed 0"
            )

            with self.subTest("retag tags the skipped clip again"):
                gate_b = net.NetworkGate(settings, "tag")
                fake_b = FakeOpenAI(gate_b, [openai_raw(good)])
                result_b = tag.tag_catalog(
                    catalog,
                    settings,
                    vocabulary=self.vocabulary(catalog, settings),
                    retag=True,
                    backends={"openai": fake_b},
                )
                done_b = [
                    e
                    for e in result_b.entries
                    if catalog.get_media(e.path).id == media_done
                ][0]
                self.assertEqual(done_b.status, "tagged")
                self.assertEqual(len(fake_b.requests), 1)

    def test_dry_run_changes_nothing(self):
        good = load_fixture("good-answer.json")
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_a, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            media_b, _ = self.add_clip(catalog, footage, "cove-b.mov", "clip-b")
            self.write_sheet(settings, catalog, media_a)
            self.write_sheet(settings, catalog, media_b)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeOpenAI(gate, [openai_raw(good), openai_raw(good)])
            result = tag.tag_catalog(
                catalog,
                settings,
                vocabulary=self.vocabulary(catalog, settings),
                dry_run=True,
                backends={"openai": fake},
            )
            self.assertEqual(
                result.outcome,
                "would-tag 2, skipped 0, stale 0, no-sheet 0, failed 0",
            )
            self.assertEqual(fake.requests, [])
            self.assertEqual(catalog.list_tags(media_a), [])
            self.assertEqual(catalog.list_tags(media_b), [])
            self.assertEqual(catalog.list_runs(), [])
            pipeline = Path(os.fspath(settings.folder)) / ".pipeline"
            self.assertFalse((pipeline / "network-log.tsv").exists())
            self.assertFalse((pipeline / "ai-usage.tsv").exists())

    def test_footage_and_other_folders_untouched(self):
        good = load_fixture("good-answer.json")
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
                        )
                return found

            transcripts = Path(tmp) / "transcripts"
            before_footage = snapshot(footage)
            before_transcripts = snapshot(transcripts)
            before_project = snapshot_project()
            self.assertTrue(before_footage)
            self.assertTrue(before_transcripts)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeOpenAI(gate, [openai_raw(good)])
            result = tag.tag_catalog(
                catalog,
                settings,
                vocabulary=self.vocabulary(catalog, settings),
                backends={"openai": fake},
            )
            self.assertEqual(result.tagged, 1)
            self.assertEqual(snapshot(footage), before_footage)
            self.assertEqual(snapshot(transcripts), before_transcripts)
            self.assertEqual(snapshot_project(), before_project)
            for root, _, files in os.walk(project):
                for name in files:
                    full = Path(root) / name
                    self.assertLessEqual(
                        os.stat(full).st_size, 5_000_000, f"{full} is over 5 MB"
                    )

    def test_usage_row_per_call(self):
        good = load_fixture("good-answer.json")
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, _ = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            media_id, _ = self.add_clip(catalog, footage, "cove-a.mov", "clip-a")
            sheet_bytes = self.write_sheet(settings, catalog, media_id)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeOpenAI(gate, [openai_raw(good)])
            result = tag.tag_catalog(
                catalog,
                settings,
                vocabulary=self.vocabulary(catalog, settings),
                backends={"openai": fake},
            )
            self.assertEqual(result.tagged, 1)
            pipeline = Path(os.fspath(settings.folder)) / ".pipeline"
            usage = (pipeline / "ai-usage.tsv").read_text(encoding="utf-8")
            rows = usage.splitlines()
            self.assertEqual(len(rows), 2)
            self.assertIn("tag", rows[1])
            self.assertIn("openai", rows[1])
            self.assertIn(FAKE_OPENAI_MODEL, rows[1])
            self.assertNotIn(base64.b64encode(sheet_bytes).decode("ascii"), usage)
            self.assertNotIn("Invented harbor walk at dusk", usage)
            network = (pipeline / "network-log.tsv").read_text(encoding="utf-8")
            log_lines = network.splitlines()
            self.assertEqual(len(log_lines), 2)
            self.assertIn("1 image", log_lines[1])
            self.assertIn("characters", log_lines[1])
            self.assertNotIn("Invented harbor walk at dusk", network)

    def test_schema_is_strict_and_small(self):
        from fcp_pipeline.ai import check_schema, validate

        schema = tag.answer_schema()
        check_schema(schema)
        self.assertEqual(validate(load_fixture("good-answer.json"), schema), [])
        self.assertTrue(validate(load_fixture("malformed-answer.json"), schema))
        self.assertEqual(validate(load_fixture("edge-answer.json"), schema), [])
        text = (REPO_ROOT / "fcp_pipeline" / "tag_schema.json").read_text(
            encoding="utf-8"
        )
        for keyword in (
            "minimum",
            "maximum",
            "minLength",
            "maxLength",
            "minItems",
            "maxItems",
        ):
            self.assertNotIn(f'"{keyword}"', text)

    def test_snap_range_and_jpeg_size(self):
        edges = [Fraction(0), Fraction(10, 3), Fraction(10), Fraction(60)]
        self.assertEqual(
            tag.snap_range(Fraction(3), Fraction(9), edges),
            (Fraction(0), Fraction(10)),
        )
        self.assertEqual(
            tag.snap_range(Fraction(10), Fraction(12), edges),
            (Fraction(10), Fraction(60)),
        )
        self.assertEqual(tag.jpeg_size(fake_jpeg(1536, 864)), (1536, 864))
        with self.assertRaises(tag.TagError):
            tag.jpeg_size(b"not a jpeg")

    def test_import_is_quiet(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = (
                "import sys; "
                f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
                "import fcp_pipeline.tag; "
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

    def test_main_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = tag.main(["--trip", os.fspath(Path(tmp) / "missing")])
            self.assertEqual(code, 1)
            settings, footage, project = self.make_project(Path(tmp) / "bad")
            bad_vocab = Path(tmp) / "vocab.json"
            bad_vocab.write_text('["not", "an", "object"]', encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = tag.main(
                    [
                        "--trip",
                        os.fspath(project),
                        "--vocabulary",
                        os.fspath(bad_vocab),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn(os.fspath(bad_vocab), stderr.getvalue())
            settings_e, _, project_e = self.make_project(Path(tmp) / "empty")
            catalog_e = self.open_catalog(settings_e)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = tag.main(["--trip", os.fspath(project_e)])
            catalog_e.close()
            self._catalogs.remove(catalog_e)
            self.assertEqual(code, 0)
            self.assertIn(
                "tagged 0, skipped 0, stale 0, no-sheet 0, failed 0",
                stdout.getvalue().splitlines(),
            )

    def test_tags_a_real_sheet(self):
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg and ffprobe are required to build a real sheet")
        from tests.media_fixtures import clip_duration, make_cut_clip

        from fcp_pipeline import sheets as sheets_module

        good = load_fixture("good-answer.json")
        with tempfile.TemporaryDirectory() as tmp:
            settings, footage, project = self.make_project(tmp)
            catalog = self.open_catalog(settings)
            cached = make_cut_clip((("red", 2), ("blue", 2)), rate="10/1")
            destination = Path(footage) / "cut.mov"
            shutil.copyfile(cached, destination)
            info = os.stat(destination)
            media_id = catalog.put_media(
                MediaRecord(
                    asset_id="clip-cut",
                    kind="clip",
                    path=os.fspath(destination),
                    size=info.st_size,
                    mtime_ns=info.st_mtime_ns,
                    inode=info.st_ino,
                    camera="Invented Maker Cam A",
                    duration=clip_duration(4, "10/1"),
                )
            )
            sheets_module.build_sheets(catalog, settings)
            row = [r for r in catalog.list_media() if r.id == media_id][0]
            _, sheet_name, _ = sheets_module._sheet_names(row)
            sheet_path = (
                Path(os.fspath(settings.pipeline)) / "sheets" / sheet_name
            )
            sheet_bytes = sheet_path.read_bytes()
            width, height = tag.jpeg_size(sheet_bytes)
            self.assertLessEqual(width, 1536)
            self.assertLessEqual(height, 864)
            gate = net.NetworkGate(settings, "tag")
            fake = FakeOpenAI(gate, [openai_raw(good)])
            result = tag.tag_catalog(
                catalog,
                settings,
                vocabulary=self.vocabulary(catalog, settings),
                backends={"openai": fake},
            )
            self.assertEqual(result.tagged, 1)
            uri = fake.requests[0]["input"][0]["content"][1]["image_url"]
            self.assertEqual(
                base64.b64decode(uri.split(",", 1)[1]), sheet_bytes
            )

            def whole(folder):
                return [
                    t
                    for t in catalog.list_tags(media_id)
                    if t.folder == folder
                    and t.start is None
                    and t.duration is None
                ]

            self.assertEqual(whole("shot")[0].value, "wide")
            self.assertEqual(
                whole("note")[0].value, "Invented harbor walk at dusk"
            )
            self.assertEqual(
                whole("todo")[0].value, "lens smudge in the last tile"
            )
            self.assertEqual(whole("outline-hint")[0].value, "Arrival at the cove")


if __name__ == "__main__":
    unittest.main()
