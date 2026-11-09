"""Tests for the photo-card hand-check sample."""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.parse
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path


def _run_main(args):
    from tests import fcp_samples

    with contextlib.redirect_stdout(io.StringIO()):
        return fcp_samples.main(args)


def _run_main_capture(args):
    from tests import fcp_samples

    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fcp_samples.main(args)
    return code, out.getvalue(), err.getvalue()


def _parse_time(text):
    body = text.strip()
    assert body.endswith("s"), text
    body = body[:-1]
    if "/" in body:
        num, den = body.split("/", 1)
        return Fraction(int(num), int(den))
    return Fraction(int(body), 1)


def _uri_to_path(uri):
    split = urllib.parse.urlsplit(uri)
    assert split.scheme == "file", uri
    return Path(urllib.parse.unquote(split.path))


def _have_ffmpeg():
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


class TestPhotoCardSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests import media_fixtures

        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.have_ffmpeg = _have_ffmpeg()
        cls.ffmpeg_reason = (media_fixtures.FFMPEG_SKIP_REASON
                             + ": photo-card sample not built")
        cls.tmp = Path(tempfile.mkdtemp(prefix="photo-card-sample-test-"))
        cls.out = cls.tmp / "photo-run"
        if cls.have_ffmpeg:
            code = _run_main(["--output", str(cls.out), "--photo-card"])
            assert code == 0, code
            cls.xml_path = cls.out / "photo-card.fcpxml"
            cls.manifest_path = cls.out / "photo-card-manifest.json"
            cls.manifest = json.loads(cls.manifest_path.read_text(
                encoding="utf-8"))
            cls.root = ET.parse(str(cls.xml_path)).getroot()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(str(cls.tmp), ignore_errors=True)

    def _require_media(self):
        if not self.have_ffmpeg:
            raise unittest.SkipTest(self.ffmpeg_reason)

    def _format_frame_duration(self):
        resources = self.root.find("resources")
        formats = {}
        for fmt in resources.findall("format"):
            if fmt.get("id") is not None:
                formats[fmt.get("id")] = fmt
        self.assertEqual(len(formats), 1)
        fmt = next(iter(formats.values()))
        return _parse_time(fmt.get("frameDuration"))

    def _spine_videos(self):
        event = self.root.find("library/event")
        project = event.find("project")
        spine = project.find("sequence").find("spine")
        return list(spine)

    def _event_stills(self):
        event = self.root.find("library/event")
        return list(event.findall("asset-clip"))

    def test_sample_passes_dtd(self):
        self._require_media()
        from tests import golden

        golden.assert_dtd_valid(str(self.xml_path))
        from fcp_pipeline.bridge.check import check_fcpxml

        result = check_fcpxml(str(self.xml_path))
        self.assertEqual(result.missing_media, [])
        self.assertEqual(result.status, "valid")

    def test_cards_and_stills_pair_one_to_one(self):
        self._require_media()
        videos = self._spine_videos()
        stills = self._event_stills()
        self.assertEqual(len(videos), 4)
        self.assertEqual(len(stills), 4)
        for video in videos:
            self.assertEqual(video.tag, "video")
        cards = {}
        for video in videos:
            name = video.get("name") or ""
            self.assertTrue(name.startswith("P"), name)
            number = name[1:4]
            self.assertTrue(number.isdigit(), name)
            stem = name.split("|", 1)[1].strip()
            cards[number] = stem
        found = {}
        for still in stills:
            name = still.get("name") or ""
            self.assertTrue(name.startswith("S"), name)
            number = name[1:4]
            self.assertTrue(number.isdigit(), name)
            stem = name.split("|", 1)[1].strip()
            found[number] = stem
        self.assertEqual(set(cards), {"001", "002", "003", "004"})
        self.assertEqual(set(found), {"001", "002", "003", "004"})
        for number in ("001", "002", "003", "004"):
            self.assertEqual(cards[number], found[number])
            self.assertEqual(cards[number], "check-photo-%s" % number)
        event = self.root.find("library/event")
        project = event.find("project")
        spine = project.find("sequence").find("spine")
        self.assertEqual([c.tag for c in spine], ["video"] * 4)
        for video in videos:
            self.assertEqual(list(video.findall("asset-clip")), [])
        for still in stills:
            self.assertIsNone(still.get("offset"))
            self.assertIsNone(still.get("lane"))
            self.assertEqual(still.get("start"), "0s")

    def test_names_and_timings_match_manifest(self):
        self._require_media()
        cards = self.manifest["cards"]
        self.assertEqual(self.manifest["version"], 1)
        self.assertEqual(len(cards), 4)
        resources = self.root.find("resources")
        effects = {}
        for effect in resources.findall("effect"):
            effects[effect.get("id")] = effect
        videos = self._spine_videos()
        stills = self._event_stills()
        by_card = {v.get("name"): v for v in videos}
        by_still = {s.get("name"): s for s in stills}
        frame_dur = self._format_frame_duration()
        offset = Fraction(0, 1)
        for pos, entry in enumerate(cards):
            with self.subTest(card=entry["number"]):
                number = entry["number"]
                self.assertEqual(number, pos + 1)
                variant = entry["variant"]
                video = by_card[entry["card_name"]]
                still = by_still[entry["still_name"]]
                effect = effects[video.get("ref")]
                self.assertEqual(effect.get("name"), variant)
                self.assertEqual(
                    effect.get("uid"),
                    "~/Generators.localized/Film Photo Cards/"
                    "%s/%s.motn" % (variant, variant))
                self.assertEqual(video.get("offset"), entry["offset"])
                self.assertEqual(video.get("duration"), entry["duration"])
                self.assertEqual(still.get("duration"), entry["duration"])
                off = _parse_time(entry["offset"])
                dur = _parse_time(entry["duration"])
                self.assertEqual(off, offset)
                self.assertEqual(off % frame_dur, 0)
                self.assertEqual(dur % frame_dur, 0)
                offset += dur
                params = {p.get("name"): p for p in video.findall("param")}
                self.assertEqual(params["Text"].get("value"), entry["text"])
                self.assertEqual(params["Show camera/film"].get("value"), "1")
                still_file = self.out / "media" / entry["still_file"]
                self.assertTrue(still_file.is_file(), str(still_file))
        sequence = (self.root.find("library/event").find("project")
                    .find("sequence"))
        self.assertEqual(_parse_time(sequence.get("duration")), offset)

    def test_no_private_paths(self):
        self._require_media()
        user_part = "/" + "Users" + "/"
        home_part = "/" + "home" + "/"
        xml_text = self.xml_path.read_text(encoding="utf-8")
        manifest_text = self.manifest_path.read_text(encoding="utf-8")
        doc_path = (self.repo_root / "docs" / "photo-card-encoding.md")
        doc_text = doc_path.read_text(encoding="utf-8")
        out_uri = self.out.resolve().as_uri()
        scrubbed = xml_text.replace(out_uri, "placeholder")
        self.assertNotIn(user_part, scrubbed)
        self.assertNotIn(home_part, scrubbed)
        self.assertNotIn("file://", scrubbed)
        self.assertNotIn(user_part, manifest_text)
        self.assertNotIn(home_part, manifest_text)
        self.assertNotIn("file://", manifest_text)
        self.assertNotIn(user_part, doc_text)
        self.assertNotIn(home_part, doc_text)
        self.assertNotIn("file://", doc_text)
        out_root = self.out.resolve()
        for rep in self.root.iter("media-rep"):
            src = rep.get("src")
            self.assertTrue(src.startswith("file://"), src)
            target = _uri_to_path(src).resolve()
            self.assertTrue(target.is_file(), str(target))
            self.assertTrue(target.is_relative_to(out_root), str(target))

    def test_no_drop_zone_param(self):
        self._require_media()
        needle = "100" + "62"
        xml_text = self.xml_path.read_text(encoding="utf-8")
        self.assertNotIn(needle, xml_text)
        self.assertNotIn("Photo Drop Zone", xml_text)
        for param in self.root.iter("param"):
            key = param.get("key") or ""
            self.assertNotIn(needle, key)

    def test_refuses_from_export_combination(self):
        fixture = (self.repo_root / "tests" / "fixtures"
                   / "metadata-preservation" / "native" / "baseline.fcpxml")
        target = self.tmp / "combo-run"
        code, _, err = _run_main_capture(
            ["--output", str(target), "--from-export", str(fixture),
             "--photo-card"])
        self.assertEqual(code, 2)
        self.assertIn("--photo-card", err)
        self.assertIn("--from-export", err)
        self.assertFalse(target.exists()
                         and any(target.iterdir()))

    def test_existing_media_sample_unchanged(self):
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "tests.test_fcp_samples"],
            cwd=str(self.repo_root), capture_output=True, text=True,
            timeout=600,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])


if __name__ == "__main__":
    unittest.main()
