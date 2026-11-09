"""Tests for fcp_pipeline.callouts spoken-callout tagging."""

import contextlib
import io
import json
import os
import socket
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

from fcp_pipeline.callouts import (
    ALONE_PAUSE,
    CalloutError,
    find_callouts,
    find_takes,
    main,
    normalize_word,
    tag_callouts,
)
from fcp_pipeline.catalog import (
    Catalog,
    CatalogError,
    MediaRecord,
    TagRecord,
    TranscriptSegmentRecord,
    TranscriptWordRecord,
)
from fcp_pipeline.trip import load_trip


def project_text():
    """Build the invented project.toml shared by every test."""
    return (
        'title = "Invented Callout Survey"\n'
        "\n"
        "[pipeline]\n"
        'timezone = "UTC"\n'
        "start_date = 2020-06-15\n"
    )


def make_trip(parent):
    """Write project.toml under a fresh trip folder, returning the folder."""
    trip = Path(parent) / "trip"
    trip.mkdir()
    (trip / "project.toml").write_text(project_text(), encoding="utf-8")
    return trip


def fixture_clips():
    """Load the invented clips from the shared JSON fixture."""
    path = (
        Path(__file__).parent / "fixtures" / "transcripts" / "callouts.json"
    )
    with open(path, encoding="utf-8") as handle:
        return {
            clip["name"]: clip for clip in json.load(handle)["clips"]
        }


def segment_records(clip):
    """Turn one fixture clip's segments into catalog records."""
    records = []
    for position, segment in enumerate(clip["segments"]):
        words = []
        for number, item in enumerate(segment["words"]):
            words.append(
                TranscriptWordRecord(
                    idx=number,
                    start=Fraction(str(item["start"])),
                    stop=Fraction(str(item["end"])),
                    word=item["word"],
                    probability=None,
                )
            )
        records.append(
            TranscriptSegmentRecord(
                idx=position,
                start=Fraction(str(segment["start"])),
                stop=Fraction(str(segment["end"])),
                text=segment["text"],
                words=words,
            )
        )
    return records


def add_clip(catalog, tmp, clip, *, frame_rate=None, duration=None):
    """Insert one fixture clip with its transcript rows, returning its id."""
    rate = (
        frame_rate
        if frame_rate is not None
        else Fraction(str(clip.get("frame_rate", "30/1")))
    )
    length = (
        duration
        if duration is not None
        else Fraction(str(clip.get("duration", "60")))
    )
    media_id = catalog.put_media(
        MediaRecord(
            asset_id=f"test-{clip['name']}",
            kind="clip",
            path=f"/invented/{clip['name']}.mov",
            duration=length,
            frame_rate=rate,
        )
    )
    json_path = os.path.join(str(tmp), f"{clip['name']}.json")
    txt_path = os.path.join(str(tmp), f"{clip['name']}.txt")
    catalog.add_transcript_set(
        media_id,
        files=[(json_path, "ab" * 32), (txt_path, "cd" * 32)],
        language="en",
        engine="invented-engine",
        created="2020-06-15T08:30:00Z",
        segments=segment_records(clip),
    )
    return media_id


def add_word_clip(
    catalog,
    tmp,
    name,
    segments,
    *,
    frame_rate=None,
    duration=None,
):
    """Insert a hand-built clip, returning its id."""
    return add_clip(
        catalog,
        tmp,
        {
            "name": name,
            "duration": "60",
            "frame_rate": "30/1",
            "segments": segments,
        },
        frame_rate=frame_rate,
        duration=duration,
    )


def word_segment(words, *, idx=0, text="invented words"):
    """Build one fixture-style segment from (word, start, end) triples."""
    return {
        "start": words[0][1],
        "end": words[-1][2],
        "text": text,
        "words": [
            {"word": word, "start": start, "end": end}
            for word, start, end in words
        ],
    }


def tags_by_slot(catalog, media_id):
    """Map (folder, start, duration) to tag for one media row."""
    return {
        (tag.folder, tag.start, tag.duration): tag
        for tag in catalog.list_tags(media_id)
    }


def entry_by_asset(result, asset_id):
    """Find one run entry by asset id."""
    for entry in result.entries:
        if entry.asset_id == asset_id:
            return entry
    raise AssertionError(f"missing entry {asset_id}")


class TestCallouts(unittest.TestCase):
    def setUp(self):
        self._socket_patch = mock.patch(
            "socket.socket",
            side_effect=AssertionError("socket must not be used"),
        )
        self._connect_patch = mock.patch(
            "socket.create_connection",
            side_effect=AssertionError("socket must not be used"),
        )
        self._socket_patch.start()
        self._connect_patch.start()
        self.addCleanup(self._socket_patch.stop)
        self.addCleanup(self._connect_patch.stop)

    def test_mark_places_marker_at_word(self):
        clips = fixture_clips()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_clip(catalog, tmp, clips["mark-only"])
                result = tag_callouts(catalog, settings)
                entry = entry_by_asset(result, "test-mark-only")
                self.assertEqual(entry.status, "tagged")
                self.assertEqual(entry.stored, 1)
                self.assertEqual(entry.callouts, 1)
                markers = [
                    tag
                    for tag in catalog.list_tags(media_id)
                    if tag.folder == "Marker"
                ]
                self.assertEqual(len(markers), 1)
                (marker,) = markers
                self.assertEqual(marker.value, "Marker")
                self.assertEqual(marker.start, Fraction(151, 30))
                self.assertEqual(marker.duration, Fraction(1, 30))
                self.assertEqual(marker.source, "callout")
                self.assertIsNone(marker.confidence)
                self.assertEqual(marker.status, "plain")
                self.assertEqual(
                    [
                        tag
                        for tag in catalog.list_tags(media_id)
                        if tag.folder in ("Rating", "Take")
                    ],
                    [],
                )
        with self.subTest("non-decimal frame rate snaps to whole frames"):
            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                settings = load_trip(make_trip(tmp))
                with Catalog.open(settings) as catalog:
                    media_id = add_clip(
                        catalog,
                        tmp,
                        clips["mark-only"],
                        frame_rate=Fraction(30000, 1001),
                    )
                    tag_callouts(catalog, settings)
                    (marker,) = [
                        tag
                        for tag in catalog.list_tags(media_id)
                        if tag.folder == "Marker"
                    ]
                    frame_rate = Fraction(30000, 1001)
                    self.assertEqual(
                        (marker.start * frame_rate).denominator, 1
                    )
                    self.assertEqual(
                        (marker.duration * frame_rate).denominator, 1
                    )
                    self.assertEqual(marker.duration, frame_rate ** -1)
                    self.assertNotEqual(
                        marker.start, Fraction(101, 20)
                    )

    def test_keeper_favorites_take(self):
        clips = fixture_clips()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_clip(catalog, tmp, clips["keeper"])
                result = tag_callouts(catalog, settings)
                entry = entry_by_asset(result, "test-keeper")
                self.assertEqual(entry.status, "tagged")
                self.assertEqual(entry.stored, 1)
                favorites = [
                    tag
                    for tag in catalog.list_tags(media_id)
                    if tag.folder == "Rating" and tag.value == "Favorite"
                ]
                self.assertEqual(len(favorites), 1)
                (favorite,) = favorites
                self.assertEqual(favorite.start, Fraction(1, 1))
                self.assertEqual(favorite.duration, Fraction(36, 5))
                self.assertEqual(
                    favorite.start + favorite.duration, Fraction(41, 5)
                )
                self.assertEqual(favorite.source, "callout")
                self.assertIsNone(favorite.confidence)
                self.assertEqual(favorite.status, "plain")
                callout_at = Fraction(88, 10)
                self.assertTrue(
                    callout_at < favorite.start
                    or callout_at >= favorite.start + favorite.duration
                )

    def test_trash_rejects_take(self):
        clips = fixture_clips()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_clip(catalog, tmp, clips["trash"])
                result = tag_callouts(catalog, settings)
                entry = entry_by_asset(result, "test-trash")
                self.assertEqual(entry.status, "tagged")
                self.assertEqual(entry.stored, 1)
                rejects = [
                    tag
                    for tag in catalog.list_tags(media_id)
                    if tag.folder == "Rating" and tag.value == "Reject"
                ]
                self.assertEqual(len(rejects), 1)
                (reject,) = rejects
                self.assertEqual(reject.start, Fraction(1, 1))
                self.assertEqual(reject.duration, Fraction(36, 5))
                self.assertEqual(
                    reject.start + reject.duration, Fraction(41, 5)
                )
                self.assertEqual(reject.source, "callout")

    def test_again_supersedes_previous_take(self):
        clips = fixture_clips()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_clip(catalog, tmp, clips["again"])
                result = tag_callouts(catalog, settings)
                entry = entry_by_asset(result, "test-again")
                self.assertEqual(entry.status, "tagged")
                self.assertEqual(entry.stored, 2)
                self.assertEqual(entry.callouts, 2)
                by_slot = tags_by_slot(catalog, media_id)
                superseded = by_slot.get(
                    ("Take", Fraction(1, 1), Fraction(2, 1))
                )
                self.assertIsNotNone(superseded)
                self.assertEqual(superseded.value, "Superseded")
                self.assertEqual(superseded.source, "callout")
                favorite = by_slot.get(
                    ("Rating", Fraction(9, 2), Fraction(5, 2))
                )
                self.assertIsNotNone(favorite)
                self.assertEqual(favorite.value, "Favorite")
                first_end = superseded.start + superseded.duration
                self.assertLessEqual(first_end, favorite.start)

    def test_word_in_sentence_ignored(self):
        clips = fixture_clips()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_clip(catalog, tmp, clips["chatter"])
                result = tag_callouts(catalog, settings)
                entry = entry_by_asset(result, "test-chatter")
                self.assertEqual(entry.status, "no-callouts")
                self.assertEqual(entry.stored, 0)
                self.assertEqual(entry.callouts, 0)
                self.assertEqual(catalog.list_tags(media_id), [])
        with self.subTest("mark alone in its own segment is a callout"):
            segments = [
                TranscriptSegmentRecord(
                    idx=0,
                    start=Fraction(0, 1),
                    stop=Fraction(1, 1),
                    text="mark",
                    words=[
                        TranscriptWordRecord(
                            idx=0,
                            start=Fraction(1, 5),
                            stop=Fraction(1, 2),
                            word="mark",
                            probability=None,
                        )
                    ],
                )
            ]
            found = find_callouts(segments)
            self.assertEqual([call.kind for call in found], ["mark"])
        with self.subTest("mark with pauses on both sides is a callout"):
            segments = [
                TranscriptSegmentRecord(
                    idx=0,
                    start=Fraction(0, 1),
                    stop=Fraction(10, 1),
                    text="invented Mark harbor",
                    words=[
                        TranscriptWordRecord(
                            idx=0,
                            start=Fraction(1, 1),
                            stop=Fraction(2, 1),
                            word="invented",
                            probability=None,
                        ),
                        TranscriptWordRecord(
                            idx=1,
                            start=Fraction(5, 2),
                            stop=Fraction(3, 1),
                            word="Mark",
                            probability=None,
                        ),
                        TranscriptWordRecord(
                            idx=2,
                            start=Fraction(7, 2),
                            stop=Fraction(4, 1),
                            word="harbor",
                            probability=None,
                        ),
                    ],
                )
            ]
            found = find_callouts(segments)
            self.assertEqual([call.kind for call in found], ["mark"])
            self.assertEqual(
                normalize_word(found[0].word), "mark"
            )
        with self.subTest("mark with short pauses is not a callout"):
            segments = [
                TranscriptSegmentRecord(
                    idx=0,
                    start=Fraction(0, 1),
                    stop=Fraction(10, 1),
                    text="invented Mark harbor",
                    words=[
                        TranscriptWordRecord(
                            idx=0,
                            start=Fraction(1, 1),
                            stop=Fraction(2, 1),
                            word="invented",
                            probability=None,
                        ),
                        TranscriptWordRecord(
                            idx=1,
                            start=Fraction(23, 10),
                            stop=Fraction(26, 10),
                            word="Mark",
                            probability=None,
                        ),
                        TranscriptWordRecord(
                            idx=2,
                            start=Fraction(29, 10),
                            stop=Fraction(4, 1),
                            word="harbor",
                            probability=None,
                        ),
                    ],
                )
            ]
            self.assertEqual(find_callouts(segments), [])

    def test_all_four_in_one_clip(self):
        clips = fixture_clips()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_clip(catalog, tmp, clips["all-four"])
                result = tag_callouts(catalog, settings)
                entry = entry_by_asset(result, "test-all-four")
                self.assertEqual(entry.status, "tagged")
                self.assertEqual(entry.stored, 4)
                self.assertEqual(entry.callouts, 4)
                tags = catalog.list_tags(media_id)
                self.assertEqual(len(tags), 4)
                by_value = {}
                for tag in tags:
                    by_value.setdefault(
                        (tag.folder, tag.value), []
                    ).append(tag)
                self.assertEqual(
                    len(by_value.get(("Marker", "Marker"), [])), 1
                )
                self.assertEqual(
                    len(by_value.get(("Rating", "Favorite"), [])), 1
                )
                self.assertEqual(
                    len(by_value.get(("Rating", "Reject"), [])), 1
                )
                self.assertEqual(
                    len(by_value.get(("Take", "Superseded"), [])), 1
                )
                (marker,) = by_value[("Marker", "Marker")]
                (favorite,) = by_value[("Rating", "Favorite")]
                self.assertGreaterEqual(marker.start, favorite.start)
                self.assertLess(
                    marker.start, favorite.start + favorite.duration
                )

    def test_pause_splits_takes(self):
        clips = fixture_clips()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_clip(catalog, tmp, clips["pause"])
                tag_callouts(catalog, settings)
                (favorite,) = [
                    tag
                    for tag in catalog.list_tags(media_id)
                    if tag.folder == "Rating"
                ]
                self.assertEqual(favorite.start, Fraction(9, 2))
                self.assertEqual(favorite.duration, Fraction(3, 2))
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_clip(catalog, tmp, clips["pause"])
                tag_callouts(
                    catalog, settings, gap=Fraction(3, 1)
                )
                (favorite,) = [
                    tag
                    for tag in catalog.list_tags(media_id)
                    if tag.folder == "Rating"
                ]
                self.assertEqual(favorite.start, Fraction(1, 1))
                self.assertEqual(favorite.duration, Fraction(5, 1))
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_clip(catalog, tmp, clips["pause-exact"])
                tag_callouts(catalog, settings)
                (favorite,) = [
                    tag
                    for tag in catalog.list_tags(media_id)
                    if tag.folder == "Rating"
                ]
                self.assertEqual(favorite.start, Fraction(1, 1))
                self.assertEqual(favorite.duration, Fraction(4, 1))
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                add_clip(catalog, tmp, clips["keeper"])
                with self.assertRaises(CalloutError):
                    tag_callouts(
                        catalog, settings, gap=Fraction(0, 1)
                    )
                with self.assertRaises(CalloutError):
                    tag_callouts(
                        catalog, settings, gap=Fraction(-1, 1)
                    )
                with self.assertRaises(CalloutError):
                    find_takes([], [], gap=Fraction(0, 1))
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            trip = make_trip(tmp)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            "--trip",
                            os.fspath(trip),
                            "--gap",
                            "0",
                        ]
                    )
            self.assertEqual(raised.exception.code, 2)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            "--trip",
                            os.fspath(trip),
                            "--gap",
                            "-1",
                        ]
                    )
            self.assertEqual(raised.exception.code, 2)

    def test_callout_with_no_take_stores_nothing(self):
        clips = fixture_clips()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_clip(catalog, tmp, clips["no-take"])
                result = tag_callouts(catalog, settings)
                entry = entry_by_asset(result, "test-no-take")
                self.assertEqual(
                    [
                        tag
                        for tag in catalog.list_tags(media_id)
                        if tag.folder == "Rating"
                    ],
                    [],
                )
                self.assertTrue(
                    any(
                        note.startswith("no take before keeper")
                        for note in entry.notes
                    ),
                    entry.notes,
                )
                superseded = [
                    tag
                    for tag in catalog.list_tags(media_id)
                    if tag.folder == "Take"
                ]
                self.assertEqual(len(superseded), 1)
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_clip(
                    catalog, tmp, clips["lonely-keeper"]
                )
                result = tag_callouts(catalog, settings)
                entry = entry_by_asset(result, "test-lonely-keeper")
                self.assertEqual(catalog.list_tags(media_id), [])
                self.assertTrue(
                    any(
                        note.startswith("no take before keeper")
                        for note in entry.notes
                    ),
                    entry.notes,
                )

    def test_long_pause_before_ending_callout_stores_nothing(self):
        speech = word_segment(
            [("hello", "0.0", "0.5"), ("world", "0.6", "1.0")],
            text="hello world",
        )
        far = word_segment([("Keeper.", "4.0", "4.4")], text="Keeper.")
        near = word_segment([("Keeper.", "2.5", "2.9")], text="Keeper.")
        gap = Fraction(2, 1)
        with self.subTest("pause longer than gap leaves keeper with no take"):
            records = segment_records({"segments": [speech, far]})
            callouts = find_callouts(records)
            self.assertEqual(
                [call.kind for call in callouts], ["keeper"]
            )
            takes = find_takes(records, callouts, gap=gap)
            self.assertEqual(
                [
                    (take.start, take.stop, take.end_kind)
                    for take in takes
                ],
                [(Fraction(0, 1), Fraction(1, 1), None)],
            )
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_word_clip(
                    catalog, tmp, "far-keeper", [speech, far]
                )
                result = tag_callouts(catalog, settings, gap=gap)
                entry = entry_by_asset(result, "test-far-keeper")
                self.assertEqual(entry.stored, 0)
                self.assertEqual(
                    [
                        tag
                        for tag in catalog.list_tags(media_id)
                        if tag.folder == "Rating"
                    ],
                    [],
                )
                self.assertTrue(
                    any(
                        note.startswith("no take before keeper")
                        for note in entry.notes
                    ),
                    entry.notes,
                )
        with self.subTest("pause within gap keeps the keeper take"):
            records = segment_records({"segments": [speech, near]})
            callouts = find_callouts(records)
            self.assertEqual(
                [call.kind for call in callouts], ["keeper"]
            )
            takes = find_takes(records, callouts, gap=gap)
            self.assertEqual(
                [
                    (take.start, take.stop, take.end_kind)
                    for take in takes
                ],
                [(Fraction(0, 1), Fraction(1, 1), "keeper")],
            )
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_word_clip(
                    catalog, tmp, "near-keeper", [speech, near]
                )
                result = tag_callouts(catalog, settings, gap=gap)
                entry = entry_by_asset(result, "test-near-keeper")
                self.assertEqual(entry.stored, 1)
                (favorite,) = [
                    tag
                    for tag in catalog.list_tags(media_id)
                    if tag.folder == "Rating" and tag.value == "Favorite"
                ]
                self.assertEqual(favorite.start, Fraction(0, 1))
                self.assertEqual(favorite.duration, Fraction(1, 1))

    def test_standalone_boundary_exact_pause(self):
        self.assertEqual(ALONE_PAUSE, Fraction(2, 5))

        def segments_with(before, after):
            mark_start = Fraction(2, 1) + before
            mark_stop = mark_start + Fraction(1, 2)
            following_start = mark_stop + after
            return [
                TranscriptSegmentRecord(
                    idx=0,
                    start=Fraction(1, 1),
                    stop=following_start + Fraction(1, 1),
                    text="invented Mark harbor",
                    words=[
                        TranscriptWordRecord(
                            idx=0,
                            start=Fraction(1, 1),
                            stop=Fraction(2, 1),
                            word="invented",
                            probability=None,
                        ),
                        TranscriptWordRecord(
                            idx=1,
                            start=mark_start,
                            stop=mark_stop,
                            word="Mark",
                            probability=None,
                        ),
                        TranscriptWordRecord(
                            idx=2,
                            start=following_start,
                            stop=following_start + Fraction(1, 1),
                            word="harbor",
                            probability=None,
                        ),
                    ],
                )
            ]

        exact = Fraction(2, 5)
        just_short = Fraction(39, 100)
        with self.subTest("exactly 0.4 s on both sides is a callout"):
            found = find_callouts(segments_with(exact, exact))
            self.assertEqual([call.kind for call in found], ["mark"])
        with self.subTest("0.39 s before is not a callout"):
            self.assertEqual(
                find_callouts(segments_with(just_short, exact)), []
            )
        with self.subTest("0.39 s after is not a callout"):
            self.assertEqual(
                find_callouts(segments_with(exact, just_short)), []
            )

    def test_gap_accepts_only_plain_decimal(self):
        with tempfile.TemporaryDirectory() as raw:
            trip = make_trip(Path(raw))
            for bad in ("1/2", "-1", "0", "inf", "nan", "2\n"):
                with self.subTest(gap=bad):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            main(
                                [
                                    "--trip",
                                    os.fspath(trip),
                                    "--gap",
                                    bad,
                                ]
                            )
                    self.assertEqual(raised.exception.code, 2)
            for good in ("2", "2.5", "0.5"):
                with self.subTest(gap=good):
                    with contextlib.redirect_stdout(io.StringIO()):
                        code = main(
                            [
                                "--trip",
                                os.fspath(trip),
                                "--gap",
                                good,
                            ]
                        )
                    self.assertEqual(code, 0)

    def test_no_safe_write_dependency(self):
        import fcp_pipeline.callouts as callouts_module

        self.assertFalse(hasattr(callouts_module, "SafeWriteError"))
        with tempfile.TemporaryDirectory() as raw:
            trip = make_trip(Path(raw))
            with mock.patch.object(
                Catalog,
                "open",
                side_effect=CatalogError("invented refusal"),
            ):
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    code = main(["--trip", os.fspath(trip)])
                self.assertEqual(code, 1)
                self.assertIn("error:", err.getvalue())
            with mock.patch.object(
                Catalog, "open", side_effect=OSError("invented failure")
            ):
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    code = main(["--trip", os.fspath(trip)])
                self.assertEqual(code, 1)
                self.assertIn("error:", err.getvalue())

    def test_snapping_uses_whole_frames_and_clip_end(self):
        clips = fixture_clips()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_clip(catalog, tmp, clips["snap-end"])
                tag_callouts(catalog, settings)
                (favorite,) = [
                    tag
                    for tag in catalog.list_tags(media_id)
                    if tag.folder == "Rating"
                ]
                self.assertEqual(
                    favorite.start + favorite.duration, Fraction(60, 1)
                )
                self.assertEqual(favorite.duration, Fraction(5, 1))
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_clip(catalog, tmp, clips["snap-zero"])
                result = tag_callouts(catalog, settings)
                entry = entry_by_asset(result, "test-snap-zero")
                self.assertEqual(catalog.list_tags(media_id), [])
                self.assertEqual(entry.stored, 0)
        for numerator, denominator in (
            (24, 1),
            (30, 1),
            (30000, 1001),
        ):
            with self.subTest(frame_rate=f"{numerator}/{denominator}"):
                frame_rate = Fraction(numerator, denominator)
                with tempfile.TemporaryDirectory() as raw:
                    tmp = Path(raw)
                    settings = load_trip(make_trip(tmp))
                    with Catalog.open(settings) as catalog:
                        media_id = add_clip(
                            catalog,
                            tmp,
                            clips["keeper"],
                            frame_rate=frame_rate,
                        )
                        tag_callouts(catalog, settings)
                        for tag in catalog.list_tags(media_id):
                            self.assertIsInstance(tag.start, Fraction)
                            self.assertIsInstance(
                                tag.duration, Fraction
                            )
                            self.assertEqual(
                                (tag.start * frame_rate).denominator, 1
                            )
                            self.assertEqual(
                                (tag.duration * frame_rate).denominator, 1
                            )
                            self.assertGreater(tag.duration, 0)

    def test_existing_higher_source_kept(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_word_clip(
                    catalog,
                    tmp,
                    "blocked",
                    [
                        word_segment(
                            [
                                ("Invented", "1.0", "1.4"),
                                ("harbor", "1.5", "1.7"),
                                ("dawn.", "1.75", "2.0"),
                            ],
                            text="Invented harbor dawn.",
                        ),
                        word_segment(
                            [("Mark.", "2.5", "2.9")], text="Mark."
                        ),
                        word_segment(
                            [
                                ("quiet", "3.3", "3.6"),
                                ("water.", "3.7", "4.0"),
                            ],
                            text="quiet water.",
                        ),
                        word_segment(
                            [("Keeper.", "4.6", "5.0")], text="Keeper."
                        ),
                    ],
                )
                frame_rate = Fraction(30, 1)
                frame = Fraction(1, 1) / frame_rate
                self.assertTrue(
                    catalog.put_tag(
                        TagRecord(
                            media_id=media_id,
                            folder="Marker",
                            value="Invented locked marker",
                            start=Fraction(5, 2),
                            duration=frame,
                            source="editor",
                            confidence=None,
                            status="locked",
                        )
                    )
                )
                self.assertTrue(
                    catalog.put_tag(
                        TagRecord(
                            media_id=media_id,
                            folder="Rating",
                            value="Invented slate favorite",
                            start=Fraction(1, 1),
                            duration=Fraction(3, 1),
                            source="slate",
                            confidence=None,
                            status="plain",
                        )
                    )
                )
                before = catalog.list_tags(media_id)
                result = tag_callouts(catalog, settings)
                entry = entry_by_asset(result, "test-blocked")
                self.assertEqual(catalog.list_tags(media_id), before)
                self.assertEqual(entry.stored, 0)
                self.assertTrue(
                    any("kept editor tag" in note for note in entry.notes),
                    entry.notes,
                )
                self.assertTrue(
                    any("kept slate tag" in note for note in entry.notes),
                    entry.notes,
                )
                rerun = tag_callouts(catalog, settings)
                self.assertEqual(
                    catalog.list_tags(media_id), before
                )
                self.assertEqual(
                    len(entry_by_asset(rerun, "test-blocked").notes), 2
                )
        with self.subTest("an older callout tag is replaced, not doubled"):
            clips = fixture_clips()
            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                settings = load_trip(make_trip(tmp))
                with Catalog.open(settings) as catalog:
                    media_id = add_clip(catalog, tmp, clips["mark-only"])
                    frame = Fraction(1, 30)
                    self.assertTrue(
                        catalog.put_tag(
                            TagRecord(
                                media_id=media_id,
                                folder="Marker",
                                value="Invented stale marker",
                                start=Fraction(151, 30),
                                duration=frame,
                                source="callout",
                                confidence=None,
                                status="plain",
                            )
                        )
                    )
                    tag_callouts(catalog, settings)
                    tags = catalog.list_tags(media_id)
                    self.assertEqual(len(tags), 1)
                    self.assertEqual(tags[0].value, "Marker")
                    tag_callouts(catalog, settings)
                    again = catalog.list_tags(media_id)
                    self.assertEqual(len(again), 1)
                    self.assertEqual(again[0].id, tags[0].id)
                    self.assertEqual(again[0].value, "Marker")

    def test_dry_run_and_run_record(self):
        clips = fixture_clips()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                media_id = add_clip(catalog, tmp, clips["keeper"])
                dry = tag_callouts(catalog, settings, dry_run=True)
                dry_entry = entry_by_asset(dry, "test-keeper")
                self.assertEqual(dry_entry.status, "would-tag")
                self.assertEqual(dry_entry.stored, 0)
                self.assertTrue(dry.outcome.startswith("would-tag 1, "))
                self.assertEqual(catalog.list_tags(media_id), [])
                self.assertEqual(catalog.list_runs(), [])
                result = tag_callouts(catalog, settings)
                self.assertEqual(
                    result.outcome,
                    "tagged 1, no-callouts 0, "
                    "no-transcript 0, skipped 0",
                )
                runs = catalog.list_runs()
                self.assertEqual(len(runs), 1)
                self.assertEqual(runs[0].step, "callouts")
                self.assertEqual(runs[0].outcome, result.outcome)
                self.assertFalse(runs[0].dry_run)

    def test_no_frame_rate_skipped(self):
        clips = fixture_clips()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings = load_trip(make_trip(tmp))
            with Catalog.open(settings) as catalog:
                unrated_id = catalog.put_media(
                    MediaRecord(
                        asset_id="test-no-frame-rate",
                        kind="clip",
                        path="/invented/no-frame-rate.mov",
                        duration=Fraction(60, 1),
                        frame_rate=None,
                    )
                )
                json_path = os.path.join(str(tmp), "no-frame-rate.json")
                txt_path = os.path.join(str(tmp), "no-frame-rate.txt")
                catalog.add_transcript_set(
                    unrated_id,
                    files=[(json_path, "ab" * 32), (txt_path, "cd" * 32)],
                    language="en",
                    engine="invented-engine",
                    created="2020-06-15T08:30:00Z",
                    segments=segment_records(clips["keeper"]),
                )
                add_clip(
                    catalog,
                    tmp,
                    {
                        "name": "empty",
                        "duration": "60",
                        "frame_rate": "30/1",
                        "segments": [],
                    },
                )
                silent_id = catalog.put_media(
                    MediaRecord(
                        asset_id="test-silent",
                        kind="clip",
                        path="/invented/silent.mov",
                        duration=Fraction(60, 1),
                        frame_rate=Fraction(30, 1),
                    )
                )
                result = tag_callouts(catalog, settings)
                skipped = entry_by_asset(result, "test-no-frame-rate")
                self.assertEqual(skipped.status, "skipped")
                self.assertEqual(
                    skipped.reason, "no frame rate in the catalog"
                )
                self.assertEqual(
                    catalog.list_tags(unrated_id), []
                )
                self.assertEqual(
                    entry_by_asset(result, "test-empty").status,
                    "no-transcript",
                )
                self.assertEqual(
                    entry_by_asset(result, "test-silent").status,
                    "no-transcript",
                )

    def test_read_only_and_no_network(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve() / "project"
            footage = root / "footage"
            footage.mkdir(parents=True)
            (root / "project.toml").write_text(
                project_text(), encoding="utf-8"
            )
            clip_path = footage / "invented-harbor.mov"
            clip_path.write_bytes(b"\x00" * 1024)
            before = {}
            for dirpath, _dirnames, filenames in os.walk(root):
                for filename in filenames:
                    full = Path(dirpath) / filename
                    info = full.stat()
                    before[full.resolve()] = (
                        info.st_size,
                        info.st_mtime_ns,
                    )
            before_dirs = {path for path in root.rglob("*")}
            settings = load_trip(root)
            catalog = Catalog.open(settings)
            try:
                media_id = catalog.put_media(
                    MediaRecord(
                        asset_id="test-invented-harbor",
                        kind="clip",
                        path=os.fspath(clip_path),
                        duration=Fraction(60, 1),
                        frame_rate=Fraction(30, 1),
                    )
                )
                clips = fixture_clips()
                json_path = os.path.join(str(root), "keeper.json")
                txt_path = os.path.join(str(root), "keeper.txt")
                catalog.add_transcript_set(
                    media_id,
                    files=[(json_path, "ab" * 32), (txt_path, "cd" * 32)],
                    language="en",
                    engine="invented-engine",
                    created="2020-06-15T08:30:00Z",
                    segments=segment_records(clips["keeper"]),
                )
                opened = []
                real_open = open

                def spy(path, *args, **kwargs):
                    opened.append(os.fspath(path))
                    return real_open(path, *args, **kwargs)

                with mock.patch("builtins.open", spy):
                    result = tag_callouts(catalog, settings)
                self.assertEqual(
                    entry_by_asset(
                        result, "test-invented-harbor"
                    ).status,
                    "tagged",
                )
            finally:
                catalog.close()
            after = {}
            for dirpath, _dirnames, filenames in os.walk(root):
                for filename in filenames:
                    full = Path(dirpath) / filename
                    info = full.stat()
                    after[full.resolve()] = (
                        info.st_size,
                        info.st_mtime_ns,
                    )
            pipeline_dir = (root / ".pipeline").resolve()
            for path, details in after.items():
                if path not in before:
                    self.assertTrue(
                        path.is_relative_to(pipeline_dir), path
                    )
                elif not path.is_relative_to(pipeline_dir):
                    self.assertEqual(details, before[path])
            self.assertEqual(
                {
                    path.relative_to(root)
                    for path in root.rglob("*")
                    if not path.resolve().is_relative_to(pipeline_dir)
                },
                {
                    path.relative_to(root)
                    for path in before_dirs
                    if not path.resolve().is_relative_to(pipeline_dir)
                },
            )
            footage_names = {
                os.fspath(path) for path in (clip_path.resolve(),)
            }
            for path in opened:
                self.assertNotIn(path, footage_names)
                self.assertFalse(path.endswith(".json"), path)
                self.assertFalse(path.endswith(".txt"), path)

    def test_main_exit_codes(self):
        with tempfile.TemporaryDirectory() as raw:
            missing = Path(raw) / "missing"
            missing.mkdir()
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = main(["--trip", os.fspath(missing)])
            self.assertEqual(code, 1)
            self.assertIn("error:", err.getvalue())
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            trip = make_trip(tmp)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["--trip", os.fspath(trip)])
            self.assertEqual(code, 0)
            self.assertEqual(
                out.getvalue().strip(),
                "tagged 0, no-callouts 0, no-transcript 0, skipped 0",
            )
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            trip = make_trip(tmp)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(
                        ["--trip", os.fspath(trip), "--gap", "abc"]
                    )
            self.assertEqual(raised.exception.code, 2)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main([])
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
