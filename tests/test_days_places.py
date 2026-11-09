"""Tests for fcp_pipeline.days_places day, place and gear assignment."""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fcp_pipeline.catalog import Catalog, MediaRecord, TagRecord
from fcp_pipeline.days_places import (
    DaysPlacesError,
    _validate_places,
    assign_days_places,
    main,
    place_clusters,
)
from fcp_pipeline.trip import load_trip


def trip_text(
    *,
    timezone="UTC",
    start_date="2020-06-15",
    clock_offsets=None,
    places=None,
):
    """Build an invented project.toml for a temporary trip folder."""
    lines = [
        'title = "Invented Survey"',
        "",
        "[pipeline]",
        f'timezone = "{timezone}"',
        f"start_date = {start_date}",
        "",
    ]
    if clock_offsets:
        lines.append("[pipeline.clock_offsets]")
        for camera, offset in clock_offsets.items():
            lines.append(f'"{camera}" = "{offset}"')
        lines.append("")
    for place in places or []:
        lines.append("[[pipeline.places]]")
        for key, value in place.items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
        lines.append("")
    return "\n".join(lines)


def write_trip(parent, text):
    """Write project.toml under a fresh trip folder, returning the folder."""
    trip = Path(parent) / "trip"
    trip.mkdir()
    (trip / "project.toml").write_text(text, encoding="utf-8")
    return trip


def add_media(
    catalog,
    name,
    *,
    kind="clip",
    capture="2020-06-16T00:01:10Z",
    source="quicktime",
    camera=None,
    lat=None,
    lon=None,
):
    """Insert one invented media row, returning its id."""
    return catalog.put_media(
        MediaRecord(
            asset_id=f"test-{name}",
            kind=kind,
            path=f"/invented/{name}.mov",
            capture_time=capture,
            capture_time_source=source,
            camera=camera,
            gps_lat=lat,
            gps_lon=lon,
        )
    )


def tags_by_folder(catalog, media_id):
    """Map folder name to tag for one media row."""
    return {tag.folder: tag for tag in catalog.list_tags(media_id)}


def row_by_asset(catalog, asset_id):
    """Find one media row by asset id."""
    for row in catalog.list_media():
        if row.asset_id == asset_id:
            return row
    raise AssertionError(f"missing media row {asset_id}")


class TestDays(unittest.TestCase):
    def test_clock_offset_moves_day(self):
        text = trip_text(clock_offsets={"invented-cam-a": "+00:03:10"})
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp, text))
            with Catalog.in_memory() as catalog:
                shifted = add_media(
                    catalog,
                    "shifted",
                    capture="2020-06-15T23:58:00Z",
                    camera="invented-cam-a",
                )
                plain = add_media(
                    catalog,
                    "plain",
                    capture="2020-06-15T23:58:00Z",
                    camera="invented-cam-b",
                )
                result = assign_days_places(catalog, settings)
                self.assertEqual(row_by_asset(catalog, "test-shifted").day, 2)
                self.assertEqual(
                    tags_by_folder(catalog, shifted)["Day"].value, "Day 2"
                )
                self.assertEqual(row_by_asset(catalog, "test-plain").day, 1)
                self.assertEqual(
                    tags_by_folder(catalog, plain)["Day"].value, "Day 1"
                )
                self.assertEqual(result.assigned, 2)
        text = trip_text(clock_offsets={"invented-cam-a": "-00:03:10"})
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp, text))
            with Catalog.in_memory() as catalog:
                early = add_media(
                    catalog,
                    "early",
                    capture="2020-06-16T00:01:00Z",
                    camera="invented-cam-a",
                )
                assign_days_places(catalog, settings)
                self.assertEqual(row_by_asset(catalog, "test-early").day, 1)
                self.assertEqual(
                    tags_by_folder(catalog, early)["Day"].value, "Day 1"
                )

    def test_timezone_applied(self):
        try:
            ZoneInfo("Pacific/Auckland")
        except ZoneInfoNotFoundError:
            self.skipTest("time zone database lacks Pacific/Auckland")
        capture = "2020-06-15T23:30:00Z"
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(
                write_trip(tmp, trip_text(timezone="Pacific/Auckland"))
            )
            with Catalog.in_memory() as catalog:
                media_id = add_media(catalog, "far-east", capture=capture)
                result = assign_days_places(catalog, settings)
                self.assertEqual(row_by_asset(catalog, "test-far-east").day, 2)
                self.assertEqual(
                    tags_by_folder(catalog, media_id)["Day"].value, "Day 2"
                )
                self.assertEqual(result.assigned, 1)
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp, trip_text(timezone="UTC")))
            with Catalog.in_memory() as catalog:
                add_media(catalog, "prime", capture=capture)
                assign_days_places(catalog, settings)
                self.assertEqual(row_by_asset(catalog, "test-prime").day, 1)
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp, trip_text(timezone="Not/AZone")))
            with Catalog.in_memory() as catalog:
                media_id = add_media(catalog, "lost", capture=capture)
                before = list(catalog.list_media())
                with self.assertRaises(DaysPlacesError) as raised:
                    assign_days_places(catalog, settings)
                self.assertIn("[pipeline].timezone", str(raised.exception))
                self.assertEqual(list(catalog.list_media()), before)
                self.assertEqual(catalog.list_tags(media_id), [])

    def test_gps_inside_radius_tags_place(self):
        places = [
            {
                "name": "Invented Cove",
                "lat": 12.5,
                "lon": 45.25,
                "radius_m": 500,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp, trip_text(places=places)))
            with Catalog.in_memory() as catalog:
                near = add_media(
                    catalog,
                    "near",
                    capture="2020-06-16T10:00:00Z",
                    lat=12.5,
                    lon=45.2501,
                )
                far = add_media(
                    catalog,
                    "far",
                    capture="2020-06-16T10:00:00Z",
                    lat=12.518,
                    lon=45.25,
                )
                assign_days_places(catalog, settings)
                self.assertEqual(
                    row_by_asset(catalog, "test-near").place, "Invented Cove"
                )
                place_tag = tags_by_folder(catalog, near)["Place"]
                self.assertEqual(place_tag.value, "Invented Cove")
                self.assertEqual(place_tag.source, "gps")
                self.assertEqual(place_tag.status, "plain")
                self.assertIsNone(row_by_asset(catalog, "test-far").place)
                self.assertNotIn("Place", tags_by_folder(catalog, far))
        with self.subTest("nearest of two overlapping places wins"):
            overlapping = [
                {
                    "name": "Invented North Cove",
                    "lat": 12.5,
                    "lon": 45.25,
                    "radius_m": 500,
                },
                {
                    "name": "Invented South Cove",
                    "lat": 12.5,
                    "lon": 45.249,
                    "radius_m": 500,
                },
            ]
            with tempfile.TemporaryDirectory() as tmp:
                settings = load_trip(
                    write_trip(tmp, trip_text(places=overlapping))
                )
                with Catalog.in_memory() as catalog:
                    add_media(
                        catalog,
                        "between",
                        capture="2020-06-16T10:00:00Z",
                        lat=12.5,
                        lon=45.2492,
                    )
                    assign_days_places(catalog, settings)
                    self.assertEqual(
                        row_by_asset(catalog, "test-between").place,
                        "Invented South Cove",
                    )
        with self.subTest("exact tie keeps the first-listed place"):
            twins = [
                {
                    "name": "Invented First Cove",
                    "lat": 12.5,
                    "lon": 45.25,
                    "radius_m": 500,
                },
                {
                    "name": "Invented Second Cove",
                    "lat": 12.5,
                    "lon": 45.25,
                    "radius_m": 500,
                },
            ]
            with tempfile.TemporaryDirectory() as tmp:
                settings = load_trip(write_trip(tmp, trip_text(places=twins)))
                with Catalog.in_memory() as catalog:
                    add_media(
                        catalog,
                        "twin",
                        capture="2020-06-16T10:00:00Z",
                        lat=12.5,
                        lon=45.2501,
                    )
                    assign_days_places(catalog, settings)
                    self.assertEqual(
                        row_by_asset(catalog, "test-twin").place,
                        "Invented First Cove",
                    )
        with self.subTest("farther containing place wins over nearer miss"):
            varied = [
                {
                    "name": "Invented Near Cove",
                    "lat": 12.5,
                    "lon": 45.25,
                    "radius_m": 100,
                },
                {
                    "name": "Invented Far Cove",
                    "lat": 12.5,
                    "lon": 45.26,
                    "radius_m": 2000,
                },
            ]
            with tempfile.TemporaryDirectory() as tmp:
                settings = load_trip(
                    write_trip(tmp, trip_text(places=varied))
                )
                with Catalog.in_memory() as catalog:
                    add_media(
                        catalog,
                        "edge",
                        capture="2020-06-16T10:00:00Z",
                        lat=12.5,
                        lon=45.25138,
                    )
                    assign_days_places(catalog, settings)
                    self.assertEqual(
                        row_by_asset(catalog, "test-edge").place,
                        "Invented Far Cove",
                    )
        with self.subTest("missing radius raises naming entry and key"):
            broken = [
                {
                    "name": "Invented Cove",
                    "lat": 12.5,
                    "lon": 45.25,
                    "radius_m": 500,
                },
                {"name": "Invented Second Cove", "lat": 12.6, "lon": 45.3},
            ]
            with tempfile.TemporaryDirectory() as tmp:
                settings = load_trip(
                    write_trip(tmp, trip_text(places=broken))
                )
                with Catalog.in_memory() as catalog:
                    media_id = add_media(
                        catalog, "stray", capture="2020-06-16T10:00:00Z"
                    )
                    with self.assertRaises(DaysPlacesError) as raised:
                        assign_days_places(catalog, settings)
                    self.assertIn("1", str(raised.exception))
                    self.assertIn("radius_m", str(raised.exception))
                    self.assertIsNone(row_by_asset(catalog, "test-stray").day)
                    self.assertEqual(catalog.list_tags(media_id), [])

    def test_outside_places_makes_cluster(self):
        places = [
            {
                "name": "Invented Cove",
                "lat": 12.5,
                "lon": 45.25,
                "radius_m": 500,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp, trip_text(places=places)))
            with Catalog.in_memory() as catalog:
                add_media(
                    catalog,
                    "a1",
                    capture="2020-06-16T10:00:00Z",
                    lat=11.0,
                    lon=44.0,
                )
                add_media(
                    catalog,
                    "a2",
                    capture="2020-06-16T10:20:00Z",
                    lat=11.0005,
                    lon=44.0005,
                )
                add_media(
                    catalog,
                    "b",
                    capture="2020-06-16T11:00:00Z",
                    lat=11.03,
                    lon=44.0,
                )
                add_media(
                    catalog,
                    "c",
                    capture="2020-06-16T10:05:00Z",
                    lat=12.5,
                    lon=45.2501,
                )
                add_media(
                    catalog,
                    "d",
                    capture="2020-06-17T10:00:00Z",
                    lat=11.0001,
                    lon=44.0001,
                )
                assign_days_places(catalog, settings)
                self.assertEqual(row_by_asset(catalog, "test-c").place,
                                 "Invented Cove")
                clusters = place_clusters(catalog, settings)
                self.assertEqual(len(clusters), 3)
                pair, lone, other_day = clusters
                self.assertEqual((pair.day, pair.clip_count), (2, 2))
                self.assertAlmostEqual(pair.centre_lat, 11.00025, places=6)
                self.assertAlmostEqual(pair.centre_lon, 44.00025, places=6)
                self.assertEqual(pair.first_time, "2020-06-16T10:00:00")
                self.assertEqual(pair.last_time, "2020-06-16T10:20:00")
                self.assertEqual(pair.asset_ids, ("test-a1", "test-a2"))
                self.assertEqual((lone.day, lone.clip_count), (2, 1))
                self.assertEqual(lone.asset_ids, ("test-b",))
                self.assertEqual((other_day.day, other_day.clip_count), (3, 1))
                self.assertEqual(other_day.asset_ids, ("test-d",))
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp, trip_text(places=places)))
            with Catalog.in_memory() as catalog:
                for name, moment in (
                    ("e1", "2020-06-16T10:00:00Z"),
                    ("e2", "2020-06-16T10:10:00Z"),
                    ("e3", "2020-06-16T10:20:00Z"),
                ):
                    add_media(
                        catalog,
                        name,
                        capture=moment,
                        lat=11.0,
                        lon=44.0,
                    )
                assign_days_places(catalog, settings)
                clusters = place_clusters(catalog, settings)
                self.assertEqual(len(clusters), 1)
                self.assertEqual(clusters[0].clip_count, 3)

    def test_clusters_use_corrected_local_time(self):
        try:
            ZoneInfo("Pacific/Auckland")
        except ZoneInfoNotFoundError:
            self.skipTest("time zone database lacks Pacific/Auckland")
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(
                write_trip(
                    tmp,
                    trip_text(
                        timezone="Pacific/Auckland",
                        clock_offsets={"invented-cam-a": "+00:03:10"},
                    ),
                )
            )
            with Catalog.in_memory() as catalog:
                add_media(
                    catalog,
                    "auckland",
                    capture="2020-06-15T23:58:00Z",
                    camera="invented-cam-a",
                    lat=11.0,
                    lon=44.0,
                )
                assign_days_places(catalog, settings)
                self.assertEqual(
                    row_by_asset(catalog, "test-auckland").day, 2
                )
                clusters = place_clusters(catalog, settings)
                self.assertEqual(len(clusters), 1)
                self.assertEqual(
                    clusters[0].first_time, "2020-06-16T12:01:10"
                )
                self.assertEqual(
                    clusters[0].last_time, "2020-06-16T12:01:10"
                )
                self.assertNotEqual(
                    clusters[0].first_time, "2020-06-15T23:58:00"
                )
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(
                write_trip(
                    tmp,
                    trip_text(
                        clock_offsets={"invented-cam-a": "+02:00:00"}
                    ),
                )
            )
            with Catalog.in_memory() as catalog:
                add_media(
                    catalog,
                    "early",
                    capture="2020-06-16T10:00:00Z",
                    camera="invented-cam-a",
                    lat=11.0,
                    lon=44.0,
                )
                add_media(
                    catalog,
                    "late",
                    capture="2020-06-16T11:00:00Z",
                    camera="invented-cam-b",
                    lat=11.0,
                    lon=44.0,
                )
                assign_days_places(catalog, settings)
                clusters = place_clusters(catalog, settings)
                self.assertEqual(len(clusters), 1)
                self.assertEqual(
                    clusters[0].first_time, "2020-06-16T11:00:00"
                )
                self.assertEqual(
                    clusters[0].last_time, "2020-06-16T12:00:00"
                )
                self.assertEqual(
                    clusters[0].asset_ids, ("test-late", "test-early")
                )

    def test_huge_integers_rejected_with_truncated_message(self):
        limit_before = sys.get_int_max_str_digits()
        # Built by shifting, not by parsing, so the default digit limit
        # is never touched: too wide for a float, but printable.
        huge = 1 << 2000
        huge_digits = str(huge)
        self.assertGreater(huge.bit_length(), 1024)
        with self.subTest("latitude"):
            with self.assertRaises(DaysPlacesError) as raised:
                _validate_places(
                    [
                        {
                            "name": "Invented Cove",
                            "lat": huge,
                            "lon": 0.0,
                            "radius_m": 500,
                        }
                    ]
                )
            message = str(raised.exception)
            self.assertIn("place entry 0", message)
            self.assertIn("lat", message)
            self.assertNotIn(huge_digits, message)
            self.assertLess(len(message), 500)
        with self.subTest("radius"):
            with self.assertRaises(DaysPlacesError) as raised:
                _validate_places(
                    [
                        {
                            "name": "Invented Cove",
                            "lat": 0.0,
                            "lon": 0.0,
                            "radius_m": huge,
                        }
                    ]
                )
            message = str(raised.exception)
            self.assertIn("place entry 0", message)
            self.assertIn("radius_m", message)
            self.assertNotIn(huge_digits, message)
            self.assertLess(len(message), 500)
        with self.subTest("cli latitude"):
            with tempfile.TemporaryDirectory() as tmp:
                digits = "9" * 5001
                text = (
                    trip_text()
                    + '[[pipeline.places]]\nname = "Invented Cove"\n'
                    + f"lat = {digits}\nlon = 0\nradius_m = 500\n"
                )
                trip = write_trip(tmp, text)
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    code = main(["--trip", os.fspath(trip)])
                output = err.getvalue()
                self.assertEqual(code, 1)
                self.assertNotIn("Traceback", output)
                self.assertIn("project.toml", output)
                self.assertIn(os.fspath(trip), output)
                self.assertIn("too long", output)
                self.assertIn("place entry 0", output)
                self.assertIn("lat", output)
                self.assertNotIn(digits, output)
                self.assertLess(len(output), 500)
        with self.subTest("cli radius"):
            with tempfile.TemporaryDirectory() as tmp:
                digits = "9" * 5001
                text = (
                    trip_text()
                    + '[[pipeline.places]]\nname = "Invented Cove"\n'
                    + f"lat = 0\nlon = 0\nradius_m = {digits}\n"
                )
                trip = write_trip(tmp, text)
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    code = main(["--trip", os.fspath(trip)])
                output = err.getvalue()
                self.assertEqual(code, 1)
                self.assertNotIn("Traceback", output)
                self.assertIn("project.toml", output)
                self.assertIn(os.fspath(trip), output)
                self.assertIn("too long", output)
                self.assertIn("place entry 0", output)
                self.assertIn("radius_m", output)
                self.assertNotIn(digits, output)
                self.assertLess(len(output), 500)
        self.assertEqual(sys.get_int_max_str_digits(), limit_before)

    def test_unverified_time_gives_check_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp, trip_text()))
            with Catalog.in_memory() as catalog:
                filed = add_media(
                    catalog,
                    "filed",
                    capture="2020-06-16T08:00:00Z",
                    source="file",
                )
                taped = add_media(
                    catalog,
                    "taped",
                    capture="2020-06-16T08:00:00Z",
                    source="quicktime",
                )
                result = assign_days_places(catalog, settings)
                filed_tag = tags_by_folder(catalog, filed)["Day"]
                self.assertEqual(filed_tag.value, "Day 2")
                self.assertEqual(filed_tag.status, "check")
                self.assertEqual(
                    tags_by_folder(catalog, taped)["Day"].status, "plain"
                )
                self.assertEqual(result.checked, 1)

    def test_gear_keyword_from_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp, trip_text()))
            with Catalog.in_memory() as catalog:
                named = add_media(
                    catalog,
                    "named",
                    camera="Invented Maker Model One",
                )
                unnamed = add_media(catalog, "unnamed")
                assign_days_places(catalog, settings)
                gear = tags_by_folder(catalog, named)["Gear"]
                self.assertEqual(gear.value, "Invented Maker Model One")
                self.assertEqual(gear.source, "metadata")
                self.assertEqual(gear.status, "plain")
                self.assertNotIn("Gear", tags_by_folder(catalog, unnamed))

    def test_before_start_and_no_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp, trip_text()))
            with Catalog.in_memory() as catalog:
                early = add_media(
                    catalog,
                    "early",
                    capture="2020-06-14T12:00:00Z",
                    camera="invented-cam-a",
                )
                add_media(catalog, "blank", capture=None, source=None)
                add_media(
                    catalog,
                    "memo",
                    kind="memo",
                    capture="2020-06-16T10:00:00Z",
                    camera="invented-cam-a",
                    lat=11.0,
                    lon=44.0,
                )
                blank_before = row_by_asset(catalog, "test-blank")
                memo_before = row_by_asset(catalog, "test-memo")
                result = assign_days_places(catalog, settings)
                early_row = row_by_asset(catalog, "test-early")
                self.assertIsNone(early_row.day)
                self.assertNotIn("Day", tags_by_folder(catalog, early))
                self.assertEqual(
                    [entry.status for entry in result.entries],
                    ["before-start", "no-time"],
                )
                self.assertEqual(result.before_start, 1)
                self.assertEqual(result.no_time, 1)
                self.assertEqual(row_by_asset(catalog, "test-blank"),
                                 blank_before)
                self.assertEqual(row_by_asset(catalog, "test-memo"),
                                 memo_before)
                memo_id = memo_before.id
                assert memo_id is not None
                self.assertEqual(catalog.list_tags(memo_id), [])
                self.assertNotIn(
                    "test-memo",
                    [entry.asset_id for entry in result.entries],
                )

    def test_rerun_is_stable(self):
        places = [
            {
                "name": "Invented Cove",
                "lat": 12.5,
                "lon": 45.25,
                "radius_m": 500,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_trip(tmp, trip_text(places=places)))
            with Catalog.in_memory() as catalog:
                first_id = add_media(
                    catalog,
                    "first",
                    capture="2020-06-16T10:00:00Z",
                    camera="invented-cam-a",
                    lat=12.5,
                    lon=45.2501,
                )
                second_id = add_media(
                    catalog,
                    "second",
                    capture="2020-06-16T11:00:00Z",
                    camera="invented-cam-a",
                    lat=12.5,
                    lon=45.2502,
                )
                assign_days_places(catalog, settings)
                rows_after_first = list(catalog.list_media())
                tags_after_first = {
                    media_id: catalog.list_tags(media_id)
                    for media_id in (first_id, second_id)
                }
                second = assign_days_places(catalog, settings)
                self.assertEqual(list(catalog.list_media()), rows_after_first)
                for media_id in (first_id, second_id):
                    folders = sorted(
                        tag.folder for tag in catalog.list_tags(media_id)
                    )
                    self.assertEqual(folders, ["Day", "Gear", "Place"])
                    self.assertEqual(
                        catalog.list_tags(media_id),
                        tags_after_first[media_id],
                    )
                self.assertTrue(
                    all(entry.kept_locked == 0 for entry in second.entries)
                )
                locked = TagRecord(
                    media_id=first_id,
                    folder="Day",
                    value="Day 9",
                    start=None,
                    duration=None,
                    source="editor",
                    confidence=None,
                    status="locked",
                )
                self.assertTrue(catalog.put_tag(locked))
                third = assign_days_places(catalog, settings)
                self.assertEqual(
                    tags_by_folder(catalog, first_id)["Day"].value, "Day 9"
                )
                kept = {
                    entry.asset_id: entry.kept_locked
                    for entry in third.entries
                }
                self.assertEqual(kept["test-first"], 1)
                self.assertEqual(kept["test-second"], 0)

    def test_writes_only_the_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "project"
            media_dir = root / "media"
            media_dir.mkdir(parents=True)
            files = []
            for name in ("clip-a.mov", "clip-b.mov", "clip-c.mov"):
                path = media_dir / name
                path.write_bytes(b"\x00" * 64)
                files.append(path)
            (root / "project.toml").write_text(
                trip_text(), encoding="utf-8"
            )
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
                captures = (
                    "2020-06-16T10:00:00Z",
                    "2020-06-16T10:10:00Z",
                    "2020-06-16T10:20:00Z",
                )
                with mock.patch(
                    "socket.socket",
                    side_effect=AssertionError("socket must not be used"),
                ):
                    for path, capture in zip(files, captures):
                        catalog.put_media(
                            MediaRecord(
                                asset_id=f"test-{path.stem}",
                                kind="clip",
                                path=os.fspath(path),
                                capture_time=capture,
                                capture_time_source="quicktime",
                                camera="invented-cam-a",
                                gps_lat=11.0,
                                gps_lon=44.0,
                            )
                        )
                    assign_days_places(catalog, settings)
            finally:
                catalog.close()
            after = {}
            for dirpath, _dirnames, filenames in os.walk(root):
                for filename in filenames:
                    full = Path(dirpath) / filename
                    info = full.stat()
                    after[full.resolve()] = (info.st_size, info.st_mtime_ns)
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
            with mock.patch(
                "socket.socket",
                side_effect=AssertionError("socket must not be used"),
            ):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = main(["--trip", os.fspath(root)])
            self.assertEqual(code, 0)
            lines = out.getvalue().splitlines()
            self.assertTrue(
                any(
                    line.startswith("cluster day 2: 3 clips")
                    for line in lines
                ),
                lines,
            )
            self.assertIn(
                "assigned 3, before-start 0, no-time 0, check 0", lines
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "project"
            root.mkdir()
            (root / "project.toml").write_text(
                trip_text(timezone="Not/AZone"), encoding="utf-8"
            )
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = main(["--trip", os.fspath(root)])
            self.assertEqual(code, 1)
            self.assertIn("[pipeline].timezone", err.getvalue())


if __name__ == "__main__":
    unittest.main()
