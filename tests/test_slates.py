"""Tests for fcp_pipeline.slates spoken-slate tagging."""

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

from fcp_pipeline.catalog import (
    Catalog,
    MediaRecord,
    TagRecord,
    TranscriptSegmentRecord,
)
from fcp_pipeline.slates import (
    FILM_STOCKS,
    SlateError,
    build_vocabulary,
    find_slate,
    main,
    match_word,
    parse_slate,
    tag_slates,
)
from fcp_pipeline.trip import load_trip

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "transcripts" / "slates.json"
)

PROJECT_TOML = """title = "Invented Slate Survey"

[pipeline]
timezone = "UTC"
start_date = 2020-06-15

[[pipeline.places]]
name = "Moonlit Cove"
lat = 12.5
lon = 45.25
radius_m = 500

[[pipeline.places]]
name = "Paper Lantern Bay"
lat = 12.6
lon = 45.3
radius_m = 500
"""


def fixture_entries():
    """Load the invented slate clips and segments."""
    return {
        entry["name"]: entry
        for entry in json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["clips"]
    }


def write_trip(parent):
    """Write project.toml under a fresh trip folder, returning the folder."""
    trip = Path(parent) / "trip"
    trip.mkdir()
    (trip / "project.toml").write_text(PROJECT_TOML, encoding="utf-8")
    return trip


def add_clip(catalog, tmp, name, *, duration="60", camera="Invented Cam A"):
    """Insert one invented clip row, returning its id."""
    return catalog.put_media(
        MediaRecord(
            asset_id=f"slate-{name}",
            kind="clip",
            path=f"{tmp}/footage/{name}.mov",
            camera=camera,
            duration=Fraction(duration),
        )
    )


def segment_records(entry):
    """Build catalog segment records from a fixture entry."""
    return [
        TranscriptSegmentRecord(
            idx=index,
            start=Fraction(item["start"]),
            stop=Fraction(item["end"]),
            text=item["text"],
            words=[],
        )
        for index, item in enumerate(entry["segments"])
    ]


def add_transcript(catalog, tmp, media_id, entry):
    """Store one fixture entry as a .json/.txt transcript set."""
    name = entry["name"]
    catalog.add_transcript_set(
        media_id,
        files=[
            (f"{tmp}/transcripts/{name}.json", "ab" * 32),
            (f"{tmp}/transcripts/{name}.txt", "cd" * 32),
        ],
        language="en",
        engine="invented-engine",
        created="2020-06-15T08:30:00Z",
        segments=segment_records(entry),
    )


def add_segments(catalog, tmp, media_id, name, items):
    """Store inline segments as a .json/.txt transcript set."""
    catalog.add_transcript_set(
        media_id,
        files=[
            (f"{tmp}/transcripts/{name}.json", "ab" * 32),
            (f"{tmp}/transcripts/{name}.txt", "cd" * 32),
        ],
        language="en",
        engine="invented-engine",
        created="2020-06-15T08:30:00Z",
        segments=[
            TranscriptSegmentRecord(
                idx=index,
                start=Fraction(item["start"]),
                stop=Fraction(item["end"]),
                text=item["text"],
                words=[],
            )
            for index, item in enumerate(items)
        ],
    )


def load_clips(catalog, tmp, names):
    """Insert clips and their fixture transcripts, returning ids by name."""
    entries = fixture_entries()
    media_ids = {}
    for name in names:
        entry = entries[name]
        media_id = add_clip(catalog, tmp, name, duration=entry["duration"])
        add_transcript(catalog, tmp, media_id, entry)
        media_ids[name] = media_id
    return media_ids


def tags_by_folder(catalog, media_id):
    """Map folder name to tag for one media row."""
    return {tag.folder: tag for tag in catalog.list_tags(media_id)}


def camera_vocabulary(catalog, settings):
    """Vocabulary with the invented slate camera listed."""
    return build_vocabulary(catalog, settings, extra={"camera": ["F3"]})


class TestSlates(unittest.TestCase):
    def setUp(self):
        socket_patch = mock.patch.object(
            socket,
            "socket",
            side_effect=AssertionError("slates step used the network"),
        )
        socket_patch.start()
        self.addCleanup(socket_patch.stop)
        connector_patch = mock.patch.object(
            socket,
            "create_connection",
            side_effect=AssertionError("slates step used the network"),
        )
        connector_patch.start()
        self.addCleanup(connector_patch.stop)

    def test_full_slate_gives_five_keywords(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp))
            with Catalog.in_memory() as catalog:
                media_ids = load_clips(catalog, tmp, ["full"])
                result = tag_slates(
                    catalog, settings, vocabulary=camera_vocabulary(catalog, settings)
                )
                (entry,) = result.entries
                self.assertEqual(entry.status, "tagged")
                self.assertEqual(entry.stored, 6)
                self.assertEqual(
                    entry.text,
                    "Day 3, Moonlit Cove, B-roll, Portra 400 on the F3.",
                )
                tags = tags_by_folder(catalog, media_ids["full"])
                self.assertEqual(
                    set(tags),
                    {"Day", "Place", "Shot", "Stock", "Camera", "Slate"},
                )
                expected = {
                    "Day": "Day 3",
                    "Place": "Moonlit Cove",
                    "Shot": "B-roll",
                    "Stock": "Portra 400",
                    "Camera": "F3",
                    "Slate": "Day 3, Moonlit Cove, B-roll, Portra 400 on the F3.",
                }
                for folder, value in expected.items():
                    with self.subTest(folder=folder):
                        tag = tags[folder]
                        self.assertEqual(tag.value, value)
                        self.assertEqual(tag.source, "slate")
                        self.assertEqual(tag.status, "plain")
                        self.assertIsNone(tag.start)
                        self.assertIsNone(tag.duration)
                        self.assertIsNone(tag.confidence)

    def test_misheard_stock_fuzzy_matched(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp))
            with Catalog.in_memory() as catalog:
                media_ids = load_clips(catalog, tmp, ["misheard"])
                result = tag_slates(
                    catalog, settings, vocabulary=camera_vocabulary(catalog, settings)
                )
                (entry,) = result.entries
                self.assertEqual(entry.status, "tagged")
                stock = tags_by_folder(catalog, media_ids["misheard"])["Stock"]
                self.assertEqual(stock.value, "Portra 400")
                self.assertEqual(stock.status, "plain")
        pinned = [
            (("Porter 400", ["Portra 400", "Portra 800"]), "Portra 400"),
            (("Moonlight Cove", ["Moonlit Cove"]), "Moonlit Cove"),
            (("F 3", ["F3"]), "F3"),
            (("Portra 400", ["Portra 800"]), None),
            (("Tri X 400", ["Tri-X 400"]), "Tri-X 400"),
            (("abcde", ["abcdf", "abcdg"]), None),
            (("abcdef", ["abcdefg", "abcdex"]), "abcdefg"),
        ]
        for (heard, candidates), expected in pinned:
            with self.subTest(heard=heard):
                self.assertEqual(match_word(heard, candidates), expected)

    def test_unknown_place_becomes_check_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp))
            with Catalog.in_memory() as catalog:
                media_ids = load_clips(
                    catalog, tmp, ["unknown-place", "mystery-stock"]
                )
                result = tag_slates(
                    catalog, settings, vocabulary=camera_vocabulary(catalog, settings)
                )
                self.assertEqual(
                    [entry.status for entry in result.entries],
                    ["tagged", "tagged"],
                )
                strange = tags_by_folder(catalog, media_ids["unknown-place"])
                self.assertEqual(strange["Place"].value, "Willow Quarry")
                self.assertEqual(strange["Place"].status, "check")
                self.assertEqual(strange["Day"].value, "Day 1")
                self.assertEqual(strange["Day"].status, "plain")
                self.assertEqual(strange["Shot"].value, "B-roll")
                self.assertEqual(strange["Shot"].status, "plain")
                stock = tags_by_folder(catalog, media_ids["mystery-stock"])[
                    "Stock"
                ]
                self.assertEqual(stock.value, "mystery film 999")
                self.assertEqual(stock.status, "check")

    def test_no_slate_no_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp))
            with Catalog.in_memory() as catalog:
                media_ids = load_clips(catalog, tmp, ["chatter"])
                quiet = add_clip(catalog, tmp, "untranscribed")
                result = tag_slates(
                    catalog, settings, vocabulary=camera_vocabulary(catalog, settings)
                )
                self.assertEqual(
                    [entry.status for entry in result.entries],
                    ["no-slate", "no-transcript"],
                )
                self.assertEqual(catalog.list_tags(media_ids["chatter"]), [])
                self.assertEqual(catalog.list_tags(quiet), [])

    def test_mid_clip_slate_tags_whole_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp))
            with Catalog.in_memory() as catalog:
                media_ids = load_clips(catalog, tmp, ["mid-clip"])
                result = tag_slates(
                    catalog, settings, vocabulary=camera_vocabulary(catalog, settings)
                )
                (entry,) = result.entries
                self.assertEqual(entry.status, "tagged")
                for tag in catalog.list_tags(media_ids["mid-clip"]):
                    self.assertIsNone(tag.start)
                    self.assertIsNone(tag.duration)
                slate = find_slate(
                    segment_records(fixture_entries()["mid-clip"])
                )
                self.assertIsNotNone(slate)
                assert slate is not None
                text, moment = slate
                self.assertEqual(
                    text, "Day 4, Moonlit Cove, B-roll, Portra 400 on the F3."
                )
                self.assertEqual(moment, Fraction(41))

    def test_existing_higher_source_kept(self):
        slate = "Slate. Day 3, Paper Lantern Bay, Interview, Portra 400 on the F3."
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp))
            with Catalog.in_memory() as catalog:
                media_id = add_clip(catalog, tmp, "kept")
                catalog.put_tag(
                    TagRecord(
                        media_id=media_id,
                        folder="Day",
                        value="Day 2",
                        start=None,
                        duration=None,
                        source="metadata",
                        confidence=None,
                        status="plain",
                    )
                )
                catalog.put_tag(
                    TagRecord(
                        media_id=media_id,
                        folder="Place",
                        value="Moonlit Cove",
                        start=None,
                        duration=None,
                        source="gps",
                        confidence=None,
                        status="plain",
                    )
                )
                catalog.put_tag(
                    TagRecord(
                        media_id=media_id,
                        folder="Shot",
                        value="A-roll",
                        start=None,
                        duration=None,
                        source="editor",
                        confidence=None,
                        status="locked",
                    )
                )
                add_segments(
                    catalog,
                    tmp,
                    media_id,
                    "kept",
                    [{"start": "0.5", "end": "4.0", "text": slate}],
                )
                vocabulary = camera_vocabulary(catalog, settings)
                result = tag_slates(catalog, settings, vocabulary=vocabulary)
                (entry,) = result.entries
                self.assertEqual(entry.status, "tagged")
                self.assertEqual(entry.stored, 3)
                tags = tags_by_folder(catalog, media_id)
                self.assertEqual(tags["Day"].value, "Day 2")
                self.assertEqual(tags["Day"].source, "metadata")
                self.assertEqual(tags["Place"].value, "Moonlit Cove")
                self.assertEqual(tags["Place"].source, "gps")
                self.assertEqual(tags["Shot"].value, "A-roll")
                self.assertEqual(tags["Shot"].status, "locked")
                self.assertEqual(tags["Stock"].value, "Portra 400")
                self.assertEqual(tags["Stock"].source, "slate")
                for folder, source in (
                    ("Day", "metadata"),
                    ("Place", "gps"),
                    ("Shot", "editor"),
                ):
                    with self.subTest(folder=folder):
                        self.assertTrue(
                            any(
                                folder in note and source in note
                                for note in entry.notes
                            ),
                            entry.notes,
                        )
                before = [
                    (tag.folder, tag.value, tag.source, tag.status)
                    for tag in catalog.list_tags(media_id)
                ]
                rerun = tag_slates(catalog, settings, vocabulary=vocabulary)
                self.assertEqual(rerun.entries[0].status, "tagged")
                after = [
                    (tag.folder, tag.value, tag.source, tag.status)
                    for tag in catalog.list_tags(media_id)
                ]
                self.assertEqual(before, after)
                add_segments(
                    catalog,
                    tmp,
                    media_id,
                    "kept-2",
                    [
                        {
                            "start": "0.5",
                            "end": "4.0",
                            "text": "Slate. Day 9, Paper Lantern Bay, "
                            "Interview, Ektar 100 on the F3.",
                        }
                    ],
                )
                tag_slates(catalog, settings, vocabulary=vocabulary)
                retagged = tags_by_folder(catalog, media_id)
                self.assertEqual(retagged["Stock"].value, "Ektar 100")
                self.assertEqual(retagged["Stock"].source, "slate")
                self.assertEqual(len(catalog.list_tags(media_id)), len(before))

    def test_parse_slate_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp))
            with Catalog.in_memory() as catalog:
                vocabulary = camera_vocabulary(catalog, settings)
                cases = [
                    ("day three", [("Day", "Day 3", "plain")]),
                    ("Day 12", [("Day", "Day 12", "plain")]),
                    (
                        "Portra 400 on the F3",
                        [
                            ("Stock", "Portra 400", "plain"),
                            ("Camera", "F3", "plain"),
                        ],
                    ),
                    (
                        "Portra 400 on F3",
                        [
                            ("Stock", "Portra 400", "plain"),
                            ("Camera", "F3", "plain"),
                        ],
                    ),
                    (
                        "Day 3, Day 5",
                        [("Day", "Day 3", "plain")],
                    ),
                    ("B-roll.", [("Shot", "B-roll", "plain")]),
                    (
                        "Day 1, Willow Quarry, Hazy Barn.",
                        [
                            ("Day", "Day 1", "plain"),
                            ("Place", "Willow Quarry", "check"),
                        ],
                    ),
                    ("", []),
                    ("... , ; .", []),
                ]
                for text, expected in cases:
                    with self.subTest(text=text):
                        self.assertEqual(
                            [
                                (part.folder, part.value, part.status)
                                for part in parse_slate(text, vocabulary)
                            ],
                            expected,
                        )
                leftovers = add_clip(catalog, tmp, "leftovers")
                add_segments(
                    catalog,
                    tmp,
                    leftovers,
                    "leftovers",
                    [
                        {
                            "start": "0.5",
                            "end": "4.0",
                            "text": "Slate. Day 1, Willow Quarry, Hazy Barn.",
                        }
                    ],
                )
                result = tag_slates(catalog, settings, vocabulary=vocabulary)
                (entry,) = result.entries
                self.assertIn("ignored: Hazy Barn", entry.notes)
                single = add_clip(catalog, tmp, "single-shot")
                add_segments(
                    catalog,
                    tmp,
                    single,
                    "single-shot",
                    [{"start": "0.5", "end": "2.0", "text": "Slate. B-roll."}],
                )
                shot_result = tag_slates(catalog, settings, vocabulary=vocabulary)
                shot_tags = tags_by_folder(catalog, single)
                self.assertEqual(set(shot_tags), {"Shot", "Slate"})
                self.assertEqual(shot_tags["Shot"].value, "B-roll")
                self.assertEqual(shot_result.entries[1].stored, 2)

    def test_vocabulary_extra_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_trip(tmp)
            settings = load_trip(trip)
            with Catalog.in_memory() as catalog:
                vocabulary = build_vocabulary(
                    catalog,
                    settings,
                    extra={"place": ["Willow Quarry"], "camera": ["F3"]},
                )
                self.assertEqual(
                    vocabulary["place"][:2],
                    ("Moonlit Cove", "Paper Lantern Bay"),
                )
                self.assertEqual(vocabulary["place"][-1], "Willow Quarry")
                self.assertEqual(
                    list(vocabulary["stock"][: len(FILM_STOCKS)]),
                    list(FILM_STOCKS),
                )
                self.assertEqual(vocabulary["camera"][-1], "F3")
                for bad in (
                    {"place": "Moonlit Cove"},
                    {"place": ["Moonlit Cove", 7]},
                    {"place": [""]},
                    {"orbit": ["Moonlit Cove"]},
                    ["Moonlit Cove"],
                ):
                    with self.subTest(extra=bad):
                        with self.assertRaises(SlateError):
                            build_vocabulary(catalog, settings, extra=bad)
            bad_file = Path(tmp) / "vocabulary.json"
            bad_file.write_text(
                '{"place": "Moonlit Cove"}', encoding="utf-8"
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(
                    ["--trip", str(trip), "--vocabulary", str(bad_file)]
                )
            self.assertEqual(code, 1)
            self.assertIn("error:", stderr.getvalue())
            self.assertIn(str(bad_file), stderr.getvalue())

    def test_dry_run_and_run_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp))
            with Catalog.in_memory() as catalog:
                media_ids = load_clips(catalog, tmp, ["full"])
                vocabulary = camera_vocabulary(catalog, settings)
                dry = tag_slates(
                    catalog, settings, vocabulary=vocabulary, dry_run=True
                )
                (preview,) = dry.entries
                self.assertEqual(preview.status, "would-tag")
                self.assertEqual(preview.stored, 0)
                self.assertIn("would-tag 1", dry.outcome)
                self.assertEqual(catalog.list_tags(media_ids["full"]), [])
                self.assertEqual(catalog.list_runs(), [])
                result = tag_slates(catalog, settings, vocabulary=vocabulary)
                self.assertEqual(result.entries[0].status, "tagged")
                runs = catalog.list_runs()
                self.assertEqual(len(runs), 1)
                self.assertEqual(runs[0].step, "slates")
                self.assertEqual(runs[0].outcome, result.outcome)
                self.assertFalse(runs[0].dry_run)

    def test_read_only_and_no_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_trip(tmp)
            footage = Path(tmp) / "footage"
            footage.mkdir()
            (footage / "clip-a.mov").write_bytes(b"invented-placeholder-bytes")
            with Catalog.in_memory() as catalog:
                media_ids = load_clips(catalog, tmp, ["full"])

                def snapshot():
                    return {
                        path.name: (
                            path.stat().st_size,
                            path.stat().st_mtime_ns,
                        )
                        for path in sorted(footage.iterdir())
                    }

                before = snapshot()
                opened = []
                real_open = open

                def spy_open(path, *args, **kwargs):
                    opened.append(os.fspath(path))
                    return real_open(path, *args, **kwargs)

                with mock.patch("builtins.open", side_effect=spy_open):
                    settings = load_trip(trip)
                    vocabulary = camera_vocabulary(catalog, settings)
                    tag_slates(catalog, settings, vocabulary=vocabulary)
                self.assertEqual(snapshot(), before)
                transcript_paths = {
                    f"{tmp}/transcripts/full.json",
                    f"{tmp}/transcripts/full.txt",
                }
                self.assertFalse(transcript_paths & set(opened))
                self.assertEqual(set(opened), {str(trip / "project.toml")})
                self.assertNotEqual(catalog.list_tags(media_ids["full"]), [])

    def test_main_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(main(["--trip", f"{tmp}/missing-trip"]), 1)
            trip = write_trip(tmp)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["--trip", str(trip)]), 0)
            self.assertEqual(
                stdout.getvalue(), "tagged 0, no-slate 0, no-transcript 0\n"
            )
            with self.assertRaises(SystemExit) as raised:
                main([])
            self.assertEqual(raised.exception.code, 2)

    def test_second_slate_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp))
            with Catalog.in_memory() as catalog:
                media_ids = load_clips(catalog, tmp, ["two-slates"])
                result = tag_slates(
                    catalog, settings, vocabulary=camera_vocabulary(catalog, settings)
                )
                (entry,) = result.entries
                self.assertEqual(entry.status, "tagged")
                self.assertIn("second slate ignored", entry.notes)
                tags = tags_by_folder(catalog, media_ids["two-slates"])
                self.assertEqual(tags["Day"].value, "Day 5")
                self.assertEqual(
                    tags["Slate"].value, "Day 5, Moonlit Cove, B-roll."
                )

    def test_lone_slate_word_gives_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp))
            with Catalog.in_memory() as catalog:
                media_ids = load_clips(catalog, tmp, ["lone-slate"])
                result = tag_slates(
                    catalog, settings, vocabulary=camera_vocabulary(catalog, settings)
                )
                (entry,) = result.entries
                self.assertEqual(entry.status, "no-slate")
                self.assertEqual(
                    entry.reason, "slate word with no recognizable parts"
                )
                self.assertEqual(catalog.list_tags(media_ids["lone-slate"]), [])

    def test_find_slate_positions(self):
        def segments(*items):
            return [
                TranscriptSegmentRecord(
                    idx=index,
                    start=Fraction(start),
                    stop=Fraction(end),
                    text=text,
                    words=[],
                )
                for index, (start, end, text) in enumerate(items)
            ]

        with self.subTest("after a sentence end"):
            slate = find_slate(
                segments(("0.5", "3.0", "Wind howls. Slate. Day 3, Moonlit Cove."))
            )
            self.assertIsNotNone(slate)
            assert slate is not None
            self.assertEqual(slate[0], "Day 3, Moonlit Cove.")
            self.assertEqual(slate[1], Fraction("0.5"))
        with self.subTest("mid-sentence slate word is not a slate"):
            self.assertIsNone(
                find_slate(segments(("0.5", "2.0", "I will slate the shot later.")))
            )
        with self.subTest("lowercase slate word counts"):
            slate = find_slate(segments(("0.5", "2.0", "slate. Day 3.")))
            self.assertIsNotNone(slate)
            assert slate is not None
            self.assertEqual(slate[0], "Day 3.")
        with self.subTest("lone slate borrows the next segment"):
            slate = find_slate(
                segments(("0.5", "1.1", "Slate."), ("1.2", "2.0", "Day 3."))
            )
            self.assertIsNotNone(slate)
            assert slate is not None
            self.assertEqual(slate, ("Day 3.", Fraction("0.5")))
        with self.subTest("lone slate too far from the next segment"):
            slate = find_slate(
                segments(("0.5", "1.1", "Slate."), ("10.0", "11.0", "Day 3."))
            )
            self.assertIsNotNone(slate)
            assert slate is not None
            self.assertEqual(slate, ("", Fraction("0.5")))


if __name__ == "__main__":
    unittest.main()
