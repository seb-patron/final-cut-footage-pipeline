"""Tests for the Final Cut hand-check sample generator."""

import contextlib
import hashlib
import io
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


def _norm_text(text):
    return None if text is None or text.strip() == "" else text


def _assert_trees_equal(case, first, second, path="root"):
    case.assertEqual(first.tag, second.tag, "tag at %s" % path)
    case.assertEqual(dict(first.attrib), dict(second.attrib),
                     "attributes at %s" % path)
    case.assertEqual(_norm_text(first.text), _norm_text(second.text),
                     "text at %s" % path)
    case.assertEqual(_norm_text(first.tail), _norm_text(second.tail),
                     "tail at %s" % path)
    first_kids = list(first)
    second_kids = list(second)
    case.assertEqual(len(first_kids), len(second_kids),
                     "child count at %s" % path)
    for pos, (left, right) in enumerate(zip(first_kids, second_kids)):
        _assert_trees_equal(case, left, right, "%s/%s[%d]" % (path, right.tag, pos))


def _format_map(root):
    found = {}
    resources = root.find("resources")
    if resources is not None:
        for fmt in resources.findall("format"):
            if fmt.get("id") is not None:
                found[fmt.get("id")] = fmt
    return found


def _asset_map(root):
    found = {}
    resources = root.find("resources")
    if resources is not None:
        for asset in resources.findall("asset"):
            if asset.get("id") is not None:
                found[asset.get("id")] = asset
    return found


def _browser_clips(root):
    """(event, clip, frame_duration, start, duration) in document order."""
    assets = _asset_map(root)
    formats = _format_map(root)
    events = root.findall("library/event") + root.findall("event")
    clips = []
    for event in events:
        for clip in event.findall("asset-clip"):
            ref = clip.get("ref")
            asset = assets.get(ref)
            fmt_id = clip.get("format")
            if fmt_id is None and asset is not None:
                fmt_id = asset.get("format")
            fmt = formats.get(fmt_id) if fmt_id is not None else None
            if fmt is None or fmt.get("frameDuration") is None:
                continue
            frame_dur = _parse_time(fmt.get("frameDuration"))
            start_text = clip.get("start")
            if start_text is None and asset is not None:
                start_text = asset.get("start")
            dur_text = clip.get("duration")
            if dur_text is None and asset is not None:
                dur_text = asset.get("duration")
            if dur_text is None:
                continue
            clips.append((event, clip, frame_dur,
                          _parse_time(start_text or "0s"), _parse_time(dur_text)))
    return clips


class TestSamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests import media_fixtures

        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.fixture = Path(__file__).resolve().parent / "fixtures" / \
            "metadata-preservation" / "native" / "baseline.fcpxml"
        cls.baseline_bytes = cls.fixture.read_bytes()
        cls.have_ffmpeg = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
        cls.ffmpeg_reason = media_fixtures.FFMPEG_SKIP_REASON + \
            ": media run not built"
        cls.tmp = Path(tempfile.mkdtemp(prefix="fcp-samples-test-"))
        cls.media_out = cls.tmp / "media-run"
        cls.export_out = cls.tmp / "export-run"
        if cls.have_ffmpeg:
            code = _run_main(["--output", str(cls.media_out)])
            assert code == 0, code
        code = _run_main(
            ["--output", str(cls.export_out), "--from-export", str(cls.fixture)])
        assert code == 0, code

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(str(cls.tmp), ignore_errors=True)

    def _require_media(self):
        if not self.have_ffmpeg:
            raise unittest.SkipTest(self.ffmpeg_reason)

    def test_all_samples_pass_dtd(self):
        from tests import golden

        names = ["full.fcpxml", "vo-spine.fcpxml",
                 "replace-noop.fcpxml", "replace-annotations.fcpxml",
                 "project-only.fcpxml"]
        for name in names:
            with self.subTest(sample=name):
                if name in ("full.fcpxml", "vo-spine.fcpxml"):
                    if not self.have_ffmpeg:
                        raise unittest.SkipTest(self.ffmpeg_reason)
                    path = self.media_out / name
                else:
                    path = self.export_out / name
                golden.assert_dtd_valid(str(path))

    def test_full_sample_has_every_item(self):
        self._require_media()
        root = ET.parse(str(self.media_out / "full.fcpxml")).getroot()
        self.assertIsNone(root.find("library"))
        events = root.findall("event")
        self.assertEqual(len(events), 3)
        self.assertFalse(root.get("uid"))
        for event in events:
            self.assertFalse(event.get("uid"))
        smarts = list(root.iter("smart-collection"))
        self.assertEqual(len(smarts), 2)
        self.assertTrue(any(
            s.find("match-ratings") is not None and
            s.find("match-ratings").get("value") == "favorites" for s in smarts))
        self.assertTrue(any(
            s.find("match-media") is not None and
            s.find("match-media").get("type") == "stills" for s in smarts))
        self.assertTrue(any(True for _ in root.iter("keyword-collection")))
        self.assertTrue(any(True for _ in root.iter("collection-folder")))
        clips = list(root.iter("asset-clip"))
        ranged = False
        for clip in clips:
            if clip.get("duration") is None:
                continue
            span = _parse_time(clip.get("duration"))
            for keyword in clip.findall("keyword"):
                if keyword.get("duration") and _parse_time(keyword.get("duration")) < span:
                    ranged = True
        self.assertTrue(ranged)
        self.assertTrue(any(
            m.get("completed") is None for m in root.iter("marker")))
        self.assertTrue(any(
            m.get("completed") == "0" for m in root.iter("marker")))
        self.assertTrue(any(True for _ in root.iter("chapter-marker")))
        self.assertTrue(any(
            r.get("value") == "favorite" for r in root.iter("rating")))
        self.assertTrue(any(
            r.get("value") == "reject" for r in root.iter("rating")))
        self.assertTrue(any(True for _ in root.iter("note")))
        projects = list(root.iter("project"))
        self.assertTrue(projects)
        assets = _asset_map(root)
        formats = _format_map(root)
        seen = False
        for project in projects:
            spine = project.find("sequence").find("spine")
            kinds = [child.tag for child in spine]
            if "gap" not in kinds:
                continue
            refs = [c.get("ref") for c in spine if c.tag == "asset-clip"]
            still_here = any(
                assets.get(r) is not None and
                (assets[r].find("media-rep") is not None) and
                (assets[r].find("media-rep").get("src") or "").endswith(".jpg")
                for r in refs)
            fmt_ids = {assets[r].get("format") for r in refs
                       if assets.get(r) is not None and assets[r].get("format")}
            if still_here and len(fmt_ids) >= 2:
                seen = True
        self.assertTrue(seen, "project spine needs a gap, a still, two cameras")

    def test_media_is_distinct_and_referenced(self):
        self._require_media()
        from tests.media_fixtures import clip_duration

        media_dir = self.media_out / "media"
        movs = sorted(media_dir.glob("*.mov"))
        jpgs = sorted(media_dir.glob("*.jpg"))
        wavs = sorted(media_dir.glob("*.wav"))
        self.assertEqual(len(movs), 6)
        self.assertEqual(len(jpgs), 3)
        self.assertEqual(len(wavs), 1)
        seen = {}
        for path in sorted(media_dir.iterdir()):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertNotIn(digest, seen, "duplicate bytes: %s" % path.name)
            seen[digest] = path.name
        media_root = media_dir.resolve()
        for name in ("full.fcpxml", "vo-spine.fcpxml"):
            root = ET.parse(str(self.media_out / name)).getroot()
            for rep in root.iter("media-rep"):
                src = rep.get("src")
                self.assertTrue(src.startswith("file://"), src)
                target = _uri_to_path(src).resolve()
                self.assertTrue(target.is_file(), str(target))
                self.assertTrue(target.is_relative_to(media_root), str(target))
        full = ET.parse(str(self.media_out / "full.fcpxml")).getroot()
        assets = _asset_map(full)
        formats = _format_map(full)
        mov_assets = [a for a in assets.values()
                      if a.find("media-rep") is not None and
                      (a.find("media-rep").get("src") or "").endswith(".mov")]
        self.assertEqual(len(mov_assets), 6)
        durations = {_parse_time(formats[a.get("format")].get("frameDuration"))
                     for a in mov_assets}
        self.assertEqual(durations, {Fraction(1001, 30000), Fraction(1, 25),
                                     Fraction(1001, 24000)})
        expected = {clip_duration(5, "30000/1001"),
                    clip_duration(6, "30000/1001"),
                    clip_duration(5, "25/1"),
                    clip_duration(6, "25/1"),
                    clip_duration(5, "24000/1001"),
                    clip_duration(6, "24000/1001")}
        actual = {_parse_time(a.get("duration")) for a in mov_assets}
        self.assertEqual(actual, expected)
        starts = [_parse_time(a.get("start", "0s")) for a in mov_assets]
        self.assertEqual(sum(1 for s in starts if s == Fraction(3600, 1)), 1)

    def test_vo_spine_has_tone_on_spine(self):
        self._require_media()
        root = ET.parse(str(self.media_out / "vo-spine.fcpxml")).getroot()
        events = root.findall("event")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].get("name"), "Check voiceover")
        self.assertEqual([c.tag for c in events[0]], ["project"])
        assets = _asset_map(root)
        spine = events[0].find("project").find("sequence").find("spine")
        items = list(spine)
        self.assertEqual(len(items), 1)
        bed = items[0]
        self.assertEqual(bed.tag, "asset-clip")
        self.assertEqual(bed.get("offset"), "0s")
        self.assertEqual(bed.get("duration"), "20s")
        bed_asset = assets[bed.get("ref")]
        self.assertTrue(bed_asset.find("media-rep").get("src").endswith(".wav"))
        anchored = [c for c in bed if c.get("lane") == "1"]
        self.assertEqual(len(anchored), 3)
        mov_refs = [c for c in anchored
                    if assets[c.get("ref")].find("media-rep").get("src").endswith(".mov")]
        jpg_refs = [c for c in anchored
                    if assets[c.get("ref")].find("media-rep").get("src").endswith(".jpg")]
        self.assertEqual(len(mov_refs), 2)
        self.assertEqual(len(jpg_refs), 1)
        bed_off = _parse_time(bed.get("offset", "0s"))
        bed_end = bed_off + _parse_time(bed.get("duration"))
        spans = []
        for conn in anchored:
            coff = _parse_time(conn.get("offset", "0s"))
            cend = coff + _parse_time(conn.get("duration"))
            self.assertGreaterEqual(coff, bed_off)
            self.assertLessEqual(cend, bed_end)
            spans.append((coff, cend))
        spans.sort()
        for (_, first_end), (second_start, _) in zip(spans, spans[1:]):
            self.assertLessEqual(first_end, second_start)
        chapters = bed.findall("chapter-marker")
        self.assertGreaterEqual(len(chapters), 2)

    def _check_project_times(self, root):
        assets = _asset_map(root)
        formats = _format_map(root)
        for project in root.iter("project"):
            seq = project.find("sequence")
            seq_fd = _parse_time(formats[seq.get("format")].get("frameDuration"))
            spine = seq.find("spine")
            offset = Fraction(0, 1)
            seq_dur = _parse_time(seq.get("duration"))
            for item in spine:
                if item.tag == "gap":
                    off = _parse_time(item.get("offset", "0s"))
                    dur = _parse_time(item.get("duration"))
                    self.assertEqual(off, offset)
                    self.assertEqual(off % seq_fd, 0)
                    self.assertEqual(dur % seq_fd, 0)
                    offset += dur
                    continue
                off = _parse_time(item.get("offset", "0s"))
                dur = _parse_time(item.get("duration"))
                start = _parse_time(item.get("start", "0s"))
                self.assertEqual(off, offset)
                self.assertEqual(off % seq_fd, 0)
                self.assertEqual(dur % seq_fd, 0)
                asset = assets.get(item.get("ref"))
                self.assertIsNotNone(asset, item.get("ref"))
                media_start = _parse_time(asset.get("start", "0s"))
                media_dur = _parse_time(asset.get("duration"))
                self.assertGreaterEqual(start, media_start)
                self.assertLessEqual(start + dur, media_start + media_dur)
                fmt_id = item.get("format") or asset.get("format")
                if asset.get("hasVideo") == "0":
                    clip_fd = seq_fd
                elif fmt_id in formats and formats[fmt_id].get("frameDuration"):
                    clip_fd = _parse_time(formats[fmt_id].get("frameDuration"))
                else:
                    clip_fd = seq_fd
                self.assertEqual((start - media_start) % clip_fd, 0)
                offset += dur
                for conn in [c for c in item if c.tag == "asset-clip"]:
                    coff = _parse_time(conn.get("offset", "0s"))
                    cdur = _parse_time(conn.get("duration"))
                    cstart = _parse_time(conn.get("start", "0s"))
                    self.assertEqual((coff - start) % seq_fd, 0)
                    self.assertEqual(cdur % seq_fd, 0)
                    casset = assets.get(conn.get("ref"))
                    cmedia_start = _parse_time(casset.get("start", "0s"))
                    cmedia_dur = _parse_time(casset.get("duration"))
                    self.assertGreaterEqual(cstart, cmedia_start)
                    self.assertLessEqual(cstart + cdur, cmedia_start + cmedia_dur)
                for ann in [c for c in item if c.tag in
                            ("keyword", "marker", "chapter-marker", "rating")]:
                    at = _parse_time(ann.get("start", "0s"))
                    self.assertEqual((at - start) % clip_fd, 0)
                    if ann.tag in ("marker", "chapter-marker"):
                        self.assertEqual(_parse_time(ann.get("duration")), clip_fd)
            self.assertEqual(offset, seq_dur)

    def test_sequence_times_are_whole_frames(self):
        plans = [self.export_out / "project-only.fcpxml"]
        if self.have_ffmpeg:
            plans = [self.media_out / "full.fcpxml",
                     self.media_out / "vo-spine.fcpxml"] + plans
        for path in plans:
            with self.subTest(sample=path.name):
                self._check_project_times(ET.parse(str(path)).getroot())

    def test_replace_noop_equals_export(self):
        data = (self.export_out / "replace-noop.fcpxml").read_bytes()
        self.assertEqual(data, self.baseline_bytes)
        _assert_trees_equal(self, ET.fromstring(data),
                            ET.fromstring(self.baseline_bytes))

    def test_replace_annotations_only_adds(self):
        from tests.fcp_samples import ADDED_NOTE_LINE, ADDED_PREFIX

        annotated = ET.parse(str(self.export_out / "replace-annotations.fcpxml")).getroot()
        original = ET.fromstring(self.baseline_bytes)
        firsts = []
        for event in original.findall("library/event") + original.findall("event"):
            for clip in event.findall("asset-clip"):
                info = [c for c in _browser_clips(original) if c[1] is clip]
                if info:
                    firsts.append((event.get("name"), clip))
                    break
        self.assertEqual(len(firsts), 2)
        for name, _ in firsts:
            with self.subTest(event=name):
                event = next(e for e in
                             annotated.findall("library/event") + annotated.findall("event")
                             if e.get("name") == name)
                clip = [c for c in _browser_clips(annotated) if c[0] is event][0][1]
                added = [c for c in clip
                         if (c.get("value") or "").startswith(ADDED_PREFIX) or
                         (c.get("name") or "").startswith(ADDED_PREFIX)]
                self.assertEqual(len(added), 5)
                note = clip.find("note")
                self.assertIsNotNone(note)
                self.assertTrue((note.text or "").split("\n")[-1] == ADDED_NOTE_LINE)
        scrubbed = ET.fromstring(
            (self.export_out / "replace-annotations.fcpxml").read_bytes())
        def _is_added(element):
            return (element.get("value") or "").startswith(ADDED_PREFIX) or \
                (element.get("name") or "").startswith(ADDED_PREFIX)

        for event in scrubbed.findall("library/event") + scrubbed.findall("event"):
            for clip in event.findall("asset-clip"):
                if not any(_is_added(c) for c in clip):
                    continue
                for child in [c for c in list(clip) if _is_added(c)]:
                    clip.remove(child)
                note = clip.find("note")
                if note is not None and (note.text or "").endswith("\n" + ADDED_NOTE_LINE):
                    note.text = note.text[:-(len(ADDED_NOTE_LINE) + 1)]
                    if note.text == "":
                        clip.remove(note)
                elif note is not None and note.text == ADDED_NOTE_LINE:
                    clip.remove(note)
        _assert_trees_equal(self, scrubbed, original)

    def test_project_only_reuses_export_assets(self):
        from tests.fcp_samples import ADDED_PREFIX

        root = ET.parse(str(self.export_out / "project-only.fcpxml")).getroot()
        original = ET.fromstring(self.baseline_bytes)
        self.assertIsNone(root.find("library"))
        self.assertEqual(root.findall("event"), [])
        projects = root.findall("project")
        self.assertEqual(len(projects), 2)
        self.assertEqual(projects[0].get("name"), "Check rough cut")
        self.assertEqual(projects[1].get("name"), "Check section selects")
        for project in projects:
            self.assertFalse(project.get("uid"))
            self.assertFalse(project.get("modDate"))
        rough_spine = projects[0].find("sequence").find("spine")
        first_spine = [c for c in rough_spine if c.tag == "asset-clip"][0]
        self.assertTrue(any(
            (k.get("value") or "").startswith(ADDED_PREFIX)
            for k in first_spine.findall("keyword")))
        self.assertTrue(any(
            (m.get("value") or "").startswith(ADDED_PREFIX)
            for m in first_spine.findall("marker")))
        rough_names = [c.get("name") for c in rough_spine if c.tag == "asset-clip"]
        select_spine = projects[1].find("sequence").find("spine")
        select_names = [c.get("name") for c in select_spine if c.tag == "asset-clip"]
        self.assertEqual(rough_names, select_names)
        for rclip, sclip in zip(
                [c for c in rough_spine if c.tag == "asset-clip"],
                [c for c in select_spine if c.tag == "asset-clip"]):
            self.assertEqual(rclip.get("ref"), sclip.get("ref"))
            rstart = _parse_time(rclip.get("start", "0s"))
            rdur = _parse_time(rclip.get("duration"))
            sstart = _parse_time(sclip.get("start", "0s"))
            sdur = _parse_time(sclip.get("duration"))
            self.assertGreaterEqual(sstart, rstart)
            self.assertLessEqual(sstart + sdur, rstart + rdur)
            self.assertLess(sdur, rdur)
        inside = set()
        for project in projects:
            for clip in project.iter("asset-clip"):
                inside.add(id(clip))
        for clip in root.iter("asset-clip"):
            self.assertIn(id(clip), inside)
        old_uids = {a.get("uid") for a in _asset_map(original).values()}
        old_srcs = {r.get("src") for r in original.iter("media-rep")}
        for asset in _asset_map(root).values():
            self.assertIn(asset.get("uid"), old_uids)
        for rep in root.iter("media-rep"):
            self.assertIn(rep.get("src"), old_srcs)

    def test_reads_fcpxmld_export(self):
        bundle = self.tmp / "x.fcpxmld"
        bundle.mkdir(exist_ok=True)
        (bundle / "Info.fcpxml").write_bytes(self.baseline_bytes)
        first = self.tmp / "from-file"
        second = self.tmp / "from-bundle"
        self.assertEqual(_run_main(
            ["--output", str(first), "--from-export", str(self.fixture)]), 0)
        self.assertEqual(_run_main(
            ["--output", str(second), "--from-export", str(bundle)]), 0)
        for name in ("replace-noop.fcpxml", "replace-annotations.fcpxml",
                     "project-only.fcpxml"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def _snapshot(self, folder):
        folder = Path(folder)
        return {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
                for p in folder.iterdir()}

    def _assert_refusal(self, proc, path_arg):
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn(path_arg, proc.stderr)
        self.assertEqual(len(proc.stderr.strip().splitlines()), 1)

    def test_refuses_existing_output(self):
        target = self.tmp / "refuse-nonempty"
        target.mkdir(exist_ok=True)
        (target / "keep.txt").write_text("keep")
        (target / ".hidden").write_text("hidden")
        before = self._snapshot(target)
        proc = subprocess.run(
            [sys.executable, "-m", "tests.fcp_samples", "--output", str(target)],
            cwd=str(self.repo_root), capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self._assert_refusal(proc, str(target))
        self.assertEqual(self._snapshot(target), before)
        cases = []
        inner = self.repo_root / ("tmp-fcp-refuse-%d" % os.getpid())
        cases.append(("repo-inner", str(inner), inner.exists()))
        bundled = self.tmp / "x.fcpbundle" / "out"
        cases.append(("fcpbundle", str(bundled), (self.tmp / "x.fcpbundle").exists()))
        afile = self.tmp / "refuse-file.txt"
        afile.write_text("data")
        file_before = (afile.stat().st_size, afile.stat().st_mtime_ns)
        cases.append(("existing-file", str(afile), None))
        try:
            for label, path_arg, _ in cases:
                with self.subTest(case=label):
                    proc = subprocess.run(
                        [sys.executable, "-m", "tests.fcp_samples",
                         "--output", path_arg],
                        cwd=str(self.repo_root), capture_output=True, text=True,
                        timeout=120,
                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                    self._assert_refusal(proc, path_arg)
                    if label == "repo-inner":
                        self.assertFalse(inner.exists())
                    elif label == "fcpbundle":
                        self.assertFalse((self.tmp / "x.fcpbundle").exists())
                    else:
                        self.assertEqual(
                            (afile.stat().st_size, afile.stat().st_mtime_ns), file_before)
                        self.assertEqual(afile.read_text(), "data")
        finally:
            if inner.exists():
                shutil.rmtree(str(inner), ignore_errors=True)
            if (self.tmp / "x.fcpbundle").exists():
                shutil.rmtree(str(self.tmp / "x.fcpbundle"), ignore_errors=True)

    def test_export_is_not_changed(self):
        work = self.tmp / "export-copy.fcpxml"
        work.write_bytes(self.baseline_bytes)
        before = (work.read_bytes(), work.stat().st_mtime_ns)
        out = self.tmp / "export-unchanged-run"
        self.assertEqual(_run_main(
            ["--output", str(out), "--from-export", str(work)]), 0)
        self.assertEqual(work.read_bytes(), before[0])
        self.assertEqual(work.stat().st_mtime_ns, before[1])


if __name__ == "__main__":
    unittest.main()
