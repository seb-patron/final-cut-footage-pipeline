"""Tests for the synthetic media fixture maker.

Every test uses makers from :mod:`tests.media_fixtures`, so with ffmpeg
missing each test is reported as skipped (with the ffmpeg reason), never
failed. Probing goes through ffprobe/exiftool command lines with the same
skip guards.
"""

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from tests import media_fixtures
from tests.media_fixtures import (
    CARD_CREATION_DATES,
    CARD_FILES,
    DJI_MAKE,
    DJI_MODEL,
    IPHONE_LOCATION,
    IPHONE_MAKE,
    IPHONE_MODEL,
    _build_clip,
    make_audio,
    make_card_tree,
    make_clip,
    make_still,
)

_UTC = timezone.utc


def _require_ffprobe():
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise unittest.SkipTest(
            media_fixtures.FFMPEG_SKIP_REASON + " but ffprobe is missing")
    return ffprobe


def _require_exiftool():
    exiftool = shutil.which("exiftool")
    if exiftool is None:
        raise unittest.SkipTest(
            media_fixtures.EXIFTOOL_SKIP_REASON + " but it is missing")
    return exiftool


def _probe(path):
    ffprobe = _require_ffprobe()
    completed = subprocess.run(
        [ffprobe, "-hide_banner", "-v", "error",
         "-show_format", "-show_streams", "-of", "json", str(path)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=300)
    return json.loads(completed.stdout.decode("utf-8"))


def _video_stream(probed):
    for stream in probed["streams"]:
        if stream.get("codec_type") == "video":
            return stream
    raise AssertionError("no video stream in %r" % (probed,))


def _parse_instant(text):
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


class TestMedia(unittest.TestCase):
    def test_clip_rate_and_timecode(self):
        clips = {}
        for rate in ("30000/1001", "25/1", "24000/1001"):
            clips[rate] = make_clip(rate=rate,
                                    start_timecode="01:00:00:00")
        for rate, clip in clips.items():
            with self.subTest(rate=rate):
                video = _video_stream(_probe(clip))
                self.assertEqual(video["r_frame_rate"], rate)
                self.assertEqual(video["tags"]["timecode"],
                                 "01:00:00:00")
                self.assertEqual(clip.suffix, ".mov")

    def test_clip_whole_frame_durations(self):
        cases = (
            ("30000/1001", 1, 30, Fraction(1001, 1000)),
            ("25/1", 1, 25, Fraction(1, 1)),
            # Not frame-aligned: round(1.01 x 24000/1001) is 24, so the
            # clip is 24 frames = 1001/1000 s (a "-t 1.01" trim would
            # instead stretch the container to 1.01 s of audio). A
            # distinct rate keeps this case's cache entry separate from
            # the 1-second cases above, which share one file per rate
            # once their requests round to the same frame count.
            ("24000/1001", 1.01, 24, Fraction(1001, 1000)),
        )
        for rate, seconds, want_frames, want_duration in cases:
            with self.subTest(rate=rate, seconds=seconds):
                self.assertEqual(
                    media_fixtures.clip_frames(seconds, rate), want_frames)
                self.assertEqual(
                    media_fixtures.clip_duration(seconds, rate),
                    want_duration)
                clip = make_clip(seconds=seconds, rate=rate)
                probed = _probe(clip)
                video = _video_stream(probed)
                if "nb_frames" in video:
                    got_frames = int(video["nb_frames"])
                else:
                    ffprobe = _require_ffprobe()
                    completed = subprocess.run(
                        [ffprobe, "-hide_banner", "-v", "error",
                         "-select_streams", "v:0", "-count_frames",
                         "-show_entries", "stream=nb_read_frames",
                         "-of", "json", str(clip)],
                        check=True, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, timeout=300)
                    counted = json.loads(
                        completed.stdout.decode("utf-8"))
                    got_frames = int(
                        counted["streams"][0]["nb_read_frames"])
                self.assertEqual(got_frames, want_frames)
                half_frame = float(
                    Fraction(1, 1) / media_fixtures._parse_rate(rate)) / 2
                for label, text in (
                        ("video", video["duration"]),
                        ("format", probed["format"]["duration"])):
                    with self.subTest(part=label):
                        self.assertLessEqual(
                            abs(float(text) - float(want_duration)),
                            half_frame)
                # Audio rule: the fixture encodes 48 kHz PCM trimmed to
                # exactly round(duration x 48000) samples, so the audio
                # duration matches the wanted duration to within one
                # sample (plus ffprobe microsecond rounding). One AAC
                # codec frame (1024 samples ~= 21 ms) already exceeds
                # this, which is why the fixture uses PCM: AAC
                # priming/padding stretched the audio and container by
                # ~23 ms on some ffmpeg builds.
                audio_tolerance = 1.5 / 48000
                for stream in probed["streams"]:
                    if stream.get("codec_type") == "audio":
                        with self.subTest(part="audio"):
                            self.assertLessEqual(
                                abs(float(stream["duration"])
                                    - float(want_duration)),
                                audio_tolerance)

    def test_clip_creation_date(self):
        moment = datetime(2020, 1, 2, 3, 4, 5, tzinfo=_UTC)
        clip = make_clip(creation_time=moment)
        tags = _probe(clip)["format"]["tags"]
        for part in tags["creation_time"].split(";"):
            with self.subTest(part=part):
                self.assertEqual(_parse_instant(part), moment)
        self.assertEqual(_parse_instant(tags["com.apple.quicktime.creationdate"]),
                         moment)
        bare = make_clip(creation_time=None)
        bare_tags = _probe(bare)["format"]["tags"]
        self.assertNotIn("creation_time", bare_tags)
        self.assertNotIn("com.apple.quicktime.creationdate", bare_tags)
        self.assertNotIn("com.apple.quicktime.make", bare_tags)
        self.assertNotIn("com.apple.quicktime.model", bare_tags)
        self.assertNotIn("com.apple.quicktime.location.ISO6709", bare_tags)

    def test_card_tree_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            card = make_card_tree(Path(directory) / "card")
            found = sorted(
                path.relative_to(card).as_posix() for path in card.rglob("*")
                if path.is_file())
            self.assertEqual(found, [
                "DCIM/100APPLE/IMG_0001.MOV",
                "DCIM/100APPLE/IMG_0002.MOV",
                "DCIM/100APPLE/IMG_0003.MOV",
                "DCIM/DJI_001/DJI_20200102100000_0001_D.MP4",
                "DCIM/DJI_001/DJI_20200102101500_0002_D.MP4",
                "DCIM/DJI_001/DJI_20200103090000_0003_D.MP4",
            ])
            self.assertEqual(tuple(found), tuple(sorted(CARD_FILES)))
            dji_tags = _probe(
                card / "DCIM/DJI_001/DJI_20200102100000_0001_D.MP4"
            )["format"]["tags"]
            self.assertEqual(dji_tags["com.apple.quicktime.make"], DJI_MAKE)
            self.assertEqual(dji_tags["com.apple.quicktime.make"], "DJI")
            self.assertEqual(dji_tags["com.apple.quicktime.model"], DJI_MODEL)
            iphone_tags = _probe(
                card / "DCIM/100APPLE/IMG_0001.MOV")["format"]["tags"]
            self.assertEqual(iphone_tags["com.apple.quicktime.make"],
                             IPHONE_MAKE)
            self.assertEqual(iphone_tags["com.apple.quicktime.make"], "Apple")
            self.assertIn("iPhone",
                          iphone_tags["com.apple.quicktime.model"])
            self.assertEqual(iphone_tags["com.apple.quicktime.model"],
                             IPHONE_MODEL)
            self.assertEqual(iphone_tags["com.apple.quicktime.location.ISO6709"],
                             IPHONE_LOCATION)
            for relative in CARD_FILES:
                with self.subTest(path=relative):
                    target = card / relative
                    when = CARD_CREATION_DATES[relative]
                    self.assertEqual(int(target.stat().st_mtime),
                                     int(when.timestamp()))
                    tags = _probe(target)["format"]["tags"]
                    for part in tags["creation_time"].split(";"):
                        self.assertEqual(_parse_instant(part), when)

    def test_still_size_and_exif_date(self):
        plain = make_still(size=(64, 48))
        video = _video_stream(_probe(plain))
        self.assertEqual((video["width"], video["height"]), (64, 48))
        moment = datetime(2020, 1, 2, 3, 4, 5)
        stamped = make_still(size=(64, 48), exif_datetime=moment)
        video = _video_stream(_probe(stamped))
        self.assertEqual((video["width"], video["height"]), (64, 48))
        exiftool = _require_exiftool()
        for tag in ("DateTimeOriginal", "CreateDate"):
            completed = subprocess.run(
                [exiftool, "-s", "-s", "-s", "-" + tag, str(stamped)],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=300)
            self.assertEqual(completed.stdout.decode("utf-8").strip(),
                             "2020:01:02 03:04:05")

    def test_same_args_same_path_and_bytes(self):
        first = make_clip(seconds=1, rate="30000/1001", size=(64, 36),
                          start_timecode="01:00:00:00",
                          creation_time=datetime(2020, 1, 2, 3, 4, 5,
                                                 tzinfo=_UTC),
                          make="DJI", model=DJI_MODEL,
                          location=IPHONE_LOCATION,
                          audio="tone", container="mov")
        second = make_clip(seconds=1, rate="30000/1001", size=(64, 36),
                           start_timecode="01:00:00:00",
                           creation_time=datetime(2020, 1, 2, 3, 4, 5,
                                                  tzinfo=_UTC),
                           make="DJI", model=DJI_MODEL,
                           location=IPHONE_LOCATION,
                           audio="tone", container="mov")
        self.assertEqual(first, second)
        wanted = hashlib.sha256(first.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            fresh = Path(directory) / "fresh.mov"
            _build_clip(fresh, seconds=1.0, rate="30000/1001",
                        size=(64, 36), start_timecode="01:00:00:00",
                        creation_time=datetime(2020, 1, 2, 3, 4, 5,
                                               tzinfo=_UTC),
                        make="DJI", model=DJI_MODEL,
                        location=IPHONE_LOCATION,
                        audio="tone", container="mov")
            rebuilt = hashlib.sha256(fresh.read_bytes()).hexdigest()
        self.assertEqual(wanted, rebuilt)
        tone = make_audio(seconds=1, kind="tone", sample_rate=48000)
        self.assertEqual(tone.suffix, ".wav")
        silence = make_audio(seconds=1, kind="silent", sample_rate=48000)
        self.assertNotEqual(tone.read_bytes(), silence.read_bytes())


if __name__ == "__main__":
    unittest.main()
