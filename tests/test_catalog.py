"""Tests for fcp_pipeline.catalog, the project's catalog database."""

import csv
import hashlib
import io
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

from fcp_pipeline.catalog import (
    ASSETS_TSV_COLUMNS,
    SCHEMA_VERSION,
    Catalog,
    CatalogError,
    ExportRecord,
    MediaRecord,
    RunRecord,
    TagRecord,
    TranscriptRecord,
    TranscriptSegmentRecord,
    TranscriptWordRecord,
    export_assets_tsv,
)
from fcp_pipeline.safe_write import SafeWriteError
from fcp_pipeline.trip import load_trip

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "fcp_pipeline" / "schema.sql"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "templates" / "assets.tsv"


def split_statements(sql):
    """Split SQL text the same way the catalog does."""
    return [part.strip() for part in sql.split(";") if part.strip()]


def dump_schema(connection):
    """(sqlite_master rows, {table: PRAGMA table_info rows}) for comparison."""
    master = connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY name"
    ).fetchall()
    tables = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    ]
    info = {}
    for table in tables:
        info[table] = connection.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
    return (master, info)


class TestCatalog(unittest.TestCase):
    """Each project lives in a temporary folder with invented values."""

    def make_project(self, tmp: Path, name="project"):
        """Build a trip folder whose media/library paths are temp folders."""
        media = tmp / "media"
        footage = media / "footage"
        photos = media / "photos"
        assets = media / "assets"
        library = tmp / "libs" / "EditLib"
        card = tmp / "cards" / "CARD-A"
        for folder in (footage, photos, assets, library, card):
            folder.mkdir(parents=True)
        (footage / "clip-a.mov").write_bytes(b"clip-bytes-a")
        (photos / "photo-a.jpg").write_bytes(b"photo-bytes-a")
        (assets / "asset-a.txt").write_bytes(b"asset-bytes-a")
        (library / "library.fcplibrary").write_bytes(b"library-bytes")
        project = tmp / name
        project.mkdir(parents=True, exist_ok=True)
        (project / "project.toml").write_text(
            "title = \"Invented Catalog Survey\"\n"
            "[final_cut]\n"
            f"library = \"{library.as_posix()}\"\n"
            "events = [\"Invented Event\"]\n"
            "[media]\n"
            f"footage = [\"{footage.as_posix()}\"]\n"
            f"photos = [\"{photos.as_posix()}\"]\n"
            f"assets = \"{assets.as_posix()}\"\n"
            "[pipeline]\n"
            "timezone = \"UTC\"\n"
            "start_date = 2020-06-15\n",
            encoding="utf-8",
        )
        places = {
            "tmp": tmp,
            "media": media,
            "footage": footage,
            "photos": photos,
            "assets": assets,
            "library": library,
            "card": card,
            "project": project,
        }
        return load_trip(project), places

    def full_media(self, path: Path, asset_id="asset-1"):
        """A media record with every field set, matching the file on disk."""
        info = os.stat(path)
        return MediaRecord(
            asset_id=asset_id,
            kind="clip",
            path=os.fspath(path),
            size=info.st_size,
            mtime_ns=info.st_mtime_ns,
            inode=info.st_ino,
            camera="invented-cam-a",
            duration=Fraction(1001, 1000),
            frame_rate=Fraction(30000, 1001),
            start_timecode=Fraction(12, 1),
            capture_time="2020-06-15T08:30:00Z",
            capture_time_source="camera-clock",
            gps_lat=0.5,
            gps_lon=-140.25,
            day=1,
            place="Invented Cove",
            fcp_event="Invented Event",
            fcp_event_uid="event-uid-1",
            fcp_asset_uid="asset-uid-1",
        )

    def test_reopen_keeps_rows(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            media_path = places["footage"] / "clip-a.mov"
            with Catalog.open(settings) as catalog:
                self.assertEqual(catalog.schema_version, SCHEMA_VERSION)
                media_id = catalog.put_media(self.full_media(media_path))
                catalog.put_tag(
                    TagRecord(
                        media_id=media_id,
                        folder="Place",
                        value="Invented Cove",
                        start=Fraction(1, 3),
                        duration=Fraction(1001, 30000),
                        source="metadata",
                        confidence=0.9,
                        status="plain",
                    )
                )
                catalog.add_transcript(
                    TranscriptRecord(
                        media_id=media_id,
                        path="transcripts/clip-a.txt",
                        sha256="ab" * 32,
                        language="en",
                        engine="invented-engine",
                        created="2020-06-15T08:30:00Z",
                    )
                )
                run_id = catalog.start_run("scan", dry_run=False)
                catalog.finish_run(run_id, "ok")
                catalog.add_export(run_id, "edit/001-cut.fcpxml", "cd" * 32)
                before = (
                    catalog.list_media(),
                    catalog.list_tags(media_id),
                    catalog.list_transcripts(media_id),
                    catalog.list_runs(),
                    catalog.list_exports(),
                )
            with Catalog.open(settings) as catalog:
                self.assertEqual(catalog.schema_version, SCHEMA_VERSION)
                after = (
                    catalog.list_media(),
                    catalog.list_tags(media_id),
                    catalog.list_transcripts(media_id),
                    catalog.list_runs(),
                    catalog.list_exports(),
                )
            self.assertEqual(before, after)

    def test_migrates_older_schema(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            media_path = places["footage"] / "clip-a.mov"
            pipeline = places["project"] / ".pipeline"
            pipeline.mkdir()
            db_path = pipeline / "catalog.sqlite"
            schema = SCHEMA_PATH.read_text(encoding="utf-8")
            v1 = [
                s
                for s in split_statements(schema)
                if "exports" not in s.lower()
                and "transcript_segments" not in s.lower()
                and "transcript_words" not in s.lower()
            ]
            self.assertTrue(all("exports" not in s.lower() for s in v1))
            self.assertTrue(
                all("transcript_segments" not in s.lower() for s in v1)
            )
            self.assertTrue(
                all("transcript_words" not in s.lower() for s in v1)
            )
            connection = sqlite3.connect(os.fspath(db_path))
            try:
                for statement in v1:
                    connection.execute(statement)
                info = os.stat(media_path)
                connection.execute(
                    "INSERT INTO media (asset_id, kind, path, size, mtime_ns,"
                    " inode, camera, duration) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "asset-1",
                        "clip",
                        os.path.abspath(os.fspath(media_path)),
                        info.st_size,
                        info.st_mtime_ns,
                        info.st_ino,
                        "invented-cam-a",
                        "1001/1000",
                    ),
                )
                connection.execute(
                    "INSERT INTO tags (media_id, folder, value, source,"
                    " status) VALUES (?, ?, ?, ?, ?)",
                    (1, "Place", "Invented Cove", "metadata", "plain"),
                )
                connection.execute(
                    "INSERT INTO schema_version (version) VALUES (1)"
                )
                connection.commit()
            finally:
                connection.close()
            with Catalog.open(settings) as catalog:
                self.assertEqual(catalog.schema_version, SCHEMA_VERSION)
                media = catalog.get_media(media_path)
                self.assertIsNotNone(media)
                assert media is not None
                self.assertEqual(media.duration, Fraction(1001, 1000))
                self.assertEqual(
                    [(t.folder, t.value) for t in catalog.list_tags(media.id)],
                    [("Place", "Invented Cove")],
                )
                migrated = dump_schema(catalog._connection)
            with Catalog.in_memory() as fresh_memory:
                memory_schema = dump_schema(fresh_memory._connection)
            with tempfile.TemporaryDirectory() as raw2:
                settings2, _ = self.make_project(Path(raw2))
                with Catalog.open(settings2) as fresh:
                    fresh_file = dump_schema(fresh._connection)
            self.assertEqual(migrated, fresh_file)
            with self.subTest("in-memory schema equals the fresh file's"):
                self.assertEqual(memory_schema, fresh_file)

    def test_migrates_v2_to_v3(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            media_path = places["footage"] / "clip-a.mov"
            pipeline = places["project"] / ".pipeline"
            pipeline.mkdir()
            db_path = pipeline / "catalog.sqlite"
            schema = SCHEMA_PATH.read_text(encoding="utf-8")
            v2 = [
                s
                for s in split_statements(schema)
                if "transcript_segments" not in s.lower()
                and "transcript_words" not in s.lower()
            ]
            self.assertTrue(
                all("transcript_segments" not in s.lower() for s in v2)
            )
            self.assertTrue(
                all("transcript_words" not in s.lower() for s in v2)
            )
            connection = sqlite3.connect(os.fspath(db_path))
            try:
                for statement in v2:
                    connection.execute(statement)
                info = os.stat(media_path)
                connection.execute(
                    "INSERT INTO media (asset_id, kind, path, size, mtime_ns,"
                    " inode, camera) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        "asset-1",
                        "clip",
                        os.path.abspath(os.fspath(media_path)),
                        info.st_size,
                        info.st_mtime_ns,
                        info.st_ino,
                        "invented-cam-a",
                    ),
                )
                connection.execute(
                    "INSERT INTO transcripts (media_id, path, language,"
                    " engine, created) VALUES (?, ?, ?, ?, ?)",
                    (
                        1,
                        "transcripts/clip-a.json",
                        "en",
                        "invented-engine",
                        "2020-06-15T08:30:00Z",
                    ),
                )
                connection.execute(
                    "INSERT INTO schema_version (version) VALUES (2)"
                )
                connection.commit()
            finally:
                connection.close()
            with Catalog.open(settings) as catalog:
                self.assertEqual(catalog.schema_version, SCHEMA_VERSION)
                media = catalog.get_media(media_path)
                self.assertIsNotNone(media)
                assert media is not None
                self.assertEqual(
                    [
                        (t.path, t.language)
                        for t in catalog.list_transcripts(media.id)
                    ],
                    [("transcripts/clip-a.json", "en")],
                )
                migrated = dump_schema(catalog._connection)
            with Catalog.in_memory() as fresh_memory:
                memory_schema = dump_schema(fresh_memory._connection)
            with tempfile.TemporaryDirectory() as raw2:
                settings2, _ = self.make_project(Path(raw2))
                with Catalog.open(settings2) as fresh:
                    fresh_file = dump_schema(fresh._connection)
            self.assertEqual(migrated, fresh_file)
            with self.subTest("in-memory schema equals the fresh file's"):
                self.assertEqual(memory_schema, fresh_file)

    def test_transcript_set_round_trip(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            media_path = places["footage"] / "clip-a.mov"
            with Catalog.open(settings) as catalog:
                media_id = catalog.put_media(self.full_media(media_path))
                segments = [
                    TranscriptSegmentRecord(
                        idx=0,
                        start=Fraction(0, 1),
                        stop=Fraction(5, 2),
                        text="invented hello there",
                        words=[
                            TranscriptWordRecord(
                                idx=0,
                                start=Fraction(0, 1),
                                stop=Fraction(2, 5),
                                word="invented",
                                probability=None,
                            ),
                            TranscriptWordRecord(
                                idx=1,
                                start=Fraction(1, 2),
                                stop=Fraction(9, 10),
                                word="hello",
                                probability=0.8,
                            ),
                        ],
                    )
                ]
                ids = catalog.add_transcript_set(
                    media_id,
                    files=[
                        ("transcripts/clip-a.json", "ab" * 32),
                        ("transcripts/clip-a.srt", "cd" * 32),
                        ("transcripts/clip-a.txt", "ef" * 32),
                    ],
                    language="en",
                    engine="invented-engine",
                    created="2020-06-15T08:30:00Z",
                    segments=segments,
                )
                self.assertEqual(len(ids), 3)
                rows = catalog.list_transcripts(media_id)
                self.assertEqual([row.id for row in rows], ids)
                self.assertEqual(
                    [row.path for row in rows],
                    [
                        "transcripts/clip-a.json",
                        "transcripts/clip-a.srt",
                        "transcripts/clip-a.txt",
                    ],
                )
                back = catalog.list_transcript_segments(ids[0])
                self.assertEqual(len(back), 1)
                (segment,) = back
                self.assertEqual(segment.idx, 0)
                self.assertEqual(segment.start, Fraction(0, 1))
                self.assertEqual(segment.stop, Fraction(5, 2))
                self.assertEqual(segment.text, "invented hello there")
                self.assertEqual(len(segment.words), 2)
                first, second = segment.words
                self.assertEqual(first.word, "invented")
                self.assertIsNone(first.probability)
                self.assertEqual(second.start, Fraction(1, 2))
                self.assertEqual(second.stop, Fraction(9, 10))
                self.assertEqual(second.probability, 0.8)
                self.assertEqual(
                    catalog.list_transcript_segments(ids[1]), []
                )
                with self.subTest("a rerun adds rows without touching earlier ones"):
                    again = catalog.add_transcript_set(
                        media_id,
                        files=[
                            ("transcripts/clip-a.json", "ab" * 32),
                            ("transcripts/clip-a.srt", "cd" * 32),
                            ("transcripts/clip-a.txt", "ef" * 32),
                        ],
                        language="en",
                        engine="invented-engine",
                        created="2020-06-15T08:30:00Z",
                        segments=segments,
                    )
                    self.assertEqual(
                        [row.id for row in catalog.list_transcripts(media_id)],
                        ids + again,
                    )
                    self.assertEqual(
                        catalog.list_transcript_segments(ids[0]), back
                    )

    def test_transcript_set_is_atomic(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            media_path = places["footage"] / "clip-a.mov"
            with Catalog.open(settings) as catalog:
                media_id = catalog.put_media(self.full_media(media_path))

                def counts():
                    connection = catalog._connection
                    return tuple(
                        connection.execute(
                            f"SELECT COUNT(*) FROM {table}"
                        ).fetchone()[0]
                        for table in (
                            "transcripts",
                            "transcript_segments",
                            "transcript_words",
                        )
                    )

                self.assertEqual(counts(), (0, 0, 0))
                bad = [
                    TranscriptSegmentRecord(
                        idx=0,
                        start=Fraction(0, 1),
                        stop=Fraction(5, 2),
                        text="invented words",
                        words=[
                            TranscriptWordRecord(
                                idx=0,
                                start=Fraction(0, 1),
                                stop=Fraction(2, 5),
                                word="invented",
                                probability=None,
                            ),
                            TranscriptWordRecord(
                                idx=1,
                                start=0.5,
                                stop=Fraction(9, 10),
                                word="bad",
                                probability=None,
                            ),
                        ],
                    )
                ]
                with self.assertRaises(CatalogError):
                    catalog.add_transcript_set(
                        media_id,
                        files=[
                            ("transcripts/clip-a.json", "ab" * 32),
                            ("transcripts/clip-a.srt", "cd" * 32),
                            ("transcripts/clip-a.txt", "ef" * 32),
                        ],
                        language="en",
                        engine="invented-engine",
                        created="2020-06-15T08:30:00Z",
                        segments=bad,
                    )
                self.assertEqual(counts(), (0, 0, 0))
                self.assertEqual(catalog.list_transcripts(media_id), [])

    def test_refuses_newer_schema(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            pipeline = places["project"] / ".pipeline"
            pipeline.mkdir()
            db_path = pipeline / "catalog.sqlite"
            connection = sqlite3.connect(os.fspath(db_path))
            try:
                connection.execute(
                    "CREATE TABLE schema_version (version INTEGER NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO schema_version (version) VALUES (99)"
                )
                connection.commit()
            finally:
                connection.close()
            before_bytes = db_path.read_bytes()
            before_mtime = os.stat(db_path).st_mtime_ns
            with self.assertRaises(CatalogError) as caught:
                Catalog.open(settings)
            message = str(caught.exception)
            self.assertIn("99", message)
            self.assertIn(str(SCHEMA_VERSION), message)
            self.assertEqual(db_path.read_bytes(), before_bytes)
            self.assertEqual(os.stat(db_path).st_mtime_ns, before_mtime)
            self.assertEqual(sorted(os.listdir(pipeline)), ["catalog.sqlite"])

    def make_version_fixture(self, tmp: Path, versions):
        """Build a project whose database holds only crafted version rows."""
        settings, places = self.make_project(tmp)
        pipeline = places["project"] / ".pipeline"
        pipeline.mkdir()
        db_path = pipeline / "catalog.sqlite"
        connection = sqlite3.connect(os.fspath(db_path))
        try:
            connection.execute(
                "CREATE TABLE schema_version (version INTEGER NOT NULL)"
            )
            for version in versions:
                connection.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (version,),
                )
            connection.commit()
        finally:
            connection.close()
        return settings, db_path

    def test_refuses_two_version_rows(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, db_path = self.make_version_fixture(tmp, [1, 99])
            before = hashlib.sha256(db_path.read_bytes()).hexdigest()
            with self.assertRaises(CatalogError):
                Catalog.open(settings)
            self.assertEqual(
                hashlib.sha256(db_path.read_bytes()).hexdigest(), before
            )

    def test_refuses_duplicate_version_rows(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, db_path = self.make_version_fixture(tmp, [1, 1])
            before = hashlib.sha256(db_path.read_bytes()).hexdigest()
            with self.assertRaises(CatalogError):
                Catalog.open(settings)
            self.assertEqual(
                hashlib.sha256(db_path.read_bytes()).hexdigest(), before
            )

    def test_refuses_text_version(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, db_path = self.make_version_fixture(tmp, ["v2"])
            before = hashlib.sha256(db_path.read_bytes()).hexdigest()
            with self.assertRaises(CatalogError) as caught:
                Catalog.open(settings)
            self.assertIn("v2", str(caught.exception))
            self.assertEqual(
                hashlib.sha256(db_path.read_bytes()).hexdigest(), before
            )

    def test_schema_version_holds_one_row(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, _ = self.make_project(tmp)
            with Catalog.open(settings) as catalog:
                with self.assertRaises(sqlite3.IntegrityError):
                    catalog._connection.execute(
                        "INSERT INTO schema_version (version) VALUES (99)"
                    )
                catalog._connection.rollback()
                self.assertEqual(
                    catalog._connection.execute(
                        "SELECT COUNT(*) FROM schema_version"
                    ).fetchone()[0],
                    1,
                )

    def test_refuses_bad_values_and_foreign_files(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            media_path = places["footage"] / "clip-a.mov"
            with Catalog.open(settings) as catalog:
                with self.subTest("unknown kind"):
                    bad = self.full_media(media_path, asset_id="bad-kind")
                    bad.kind = "sequence"
                    with self.assertRaises(CatalogError):
                        catalog.put_media(bad)
                with self.subTest("bad capture time"):
                    bad = self.full_media(media_path, asset_id="bad-time")
                    bad.capture_time = "June 15th"
                    with self.assertRaises(CatalogError):
                        catalog.put_media(bad)
                with self.subTest("unknown tag source and status"):
                    media_id = catalog.put_media(self.full_media(media_path))
                    with self.assertRaises(CatalogError):
                        catalog.put_tag(
                            TagRecord(
                                media_id=media_id,
                                folder="Shot",
                                value="wide",
                                source="vibes",
                                status="plain",
                            )
                        )
                    with self.assertRaises(CatalogError):
                        catalog.put_tag(
                            TagRecord(
                                media_id=media_id,
                                folder="Shot",
                                value="wide",
                                source="ai",
                                status="archived",
                            )
                        )
                with self.subTest("bad confidence"):
                    with self.assertRaises(CatalogError):
                        catalog.put_tag(
                            TagRecord(
                                media_id=media_id,
                                folder="Shot",
                                value="wide",
                                source="ai",
                                confidence=1.5,
                                status="plain",
                            )
                        )
                with self.subTest("bad sha256"):
                    run_id = catalog.start_run("scan", dry_run=False)
                    with self.assertRaises(CatalogError):
                        catalog.add_export(run_id, "edit/001-cut.fcpxml", "ZZ")
                with self.subTest("bad dry_run"):
                    with self.assertRaises(CatalogError):
                        catalog.start_run("scan", dry_run=2)
                with self.subTest("unknown run"):
                    with self.assertRaises(CatalogError):
                        catalog.finish_run(999, "ok")
                    with self.assertRaises(CatalogError):
                        catalog.add_export(999, "edit/001-cut.fcpxml", "ab" * 32)
                with self.subTest("non-rational stored value"):
                    catalog._connection.execute(
                        "UPDATE media SET duration = 'fast' WHERE id = ?",
                        (media_id,),
                    )
                    try:
                        with self.assertRaises(CatalogError):
                            catalog.get_media(media_path)
                    finally:
                        catalog._connection.rollback()
            with self.subTest("tables without a version are not a catalog"):
                project = places["project"]
                pipeline = project / ".pipeline"
                foreign = pipeline / "catalog.sqlite"
                if os.path.lexists(foreign):
                    os.remove(foreign)
                connection = sqlite3.connect(os.fspath(foreign))
                try:
                    connection.execute("CREATE TABLE notes (id INTEGER)")
                    connection.commit()
                finally:
                    connection.close()
                with self.assertRaises(CatalogError):
                    Catalog.open(settings)
                os.remove(foreign)
            with self.subTest("empty file gets the current schema"):
                pipeline = places["project"] / ".pipeline"
                foreign = pipeline / "catalog.sqlite"
                foreign.write_bytes(b"")
                with Catalog.open(settings) as catalog:
                    self.assertEqual(catalog.schema_version, SCHEMA_VERSION)
            with self.subTest("hard-linked catalog is refused"):
                pipeline = places["project"] / ".pipeline"
                foreign = pipeline / "catalog.sqlite"
                extra = pipeline / "catalog-alias.sqlite"
                os.link(foreign, extra)
                try:
                    with self.assertRaises(SafeWriteError):
                        Catalog.open(settings)
                finally:
                    os.remove(extra)
            with self.subTest("non-regular catalog is refused"):
                pipeline = places["project"] / ".pipeline"
                foreign = pipeline / "catalog.sqlite"
                saved = pipeline / "catalog-saved.sqlite"
                os.rename(foreign, saved)
                try:
                    foreign.mkdir()
                    with self.assertRaises(SafeWriteError):
                        Catalog.open(settings)
                finally:
                    foreign.rmdir()
                    os.rename(saved, foreign)

    def test_unchanged_detects_mtime_change(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            media_path = places["footage"] / "clip-a.mov"
            with Catalog.open(settings) as catalog:
                catalog.put_media(self.full_media(media_path))
                self.assertTrue(catalog.unchanged(media_path))
                info = os.stat(media_path)
                os.utime(
                    media_path,
                    ns=(info.st_atime_ns, info.st_mtime_ns + 2_000_000_000),
                )
                self.assertFalse(catalog.unchanged(media_path))
                with self.subTest("changed size"):
                    media_path.write_bytes(b"clip-bytes-a-plus-more")
                    self.assertFalse(catalog.unchanged(media_path))
                with self.subTest("replaced file"):
                    media_path.unlink()
                    media_path.write_bytes(b"clip-bytes-a")
                    self.assertFalse(catalog.unchanged(media_path))
                with self.subTest("deleted file"):
                    media_path.unlink()
                    self.assertFalse(catalog.unchanged(media_path))
                with self.subTest("unknown path"):
                    self.assertFalse(
                        catalog.unchanged(places["footage"] / "missing.mov")
                    )

    def test_rational_round_trip(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            media_path = places["footage"] / "clip-a.mov"
            record = self.full_media(media_path)
            record.duration = Fraction(1001, 1000)
            record.frame_rate = Fraction(30000, 1001)
            with Catalog.open(settings) as catalog:
                media_id = catalog.put_media(record)
                back = catalog.get_media(media_path)
                assert back is not None
                self.assertEqual(back.duration, Fraction(1001, 1000))
                self.assertIsInstance(back.duration, Fraction)
                self.assertEqual(back.frame_rate, Fraction(30000, 1001))
                self.assertIsInstance(back.frame_rate, Fraction)
                catalog.put_tag(
                    TagRecord(
                        media_id=media_id,
                        folder="Shot",
                        value="wide",
                        start=Fraction(1, 3),
                        duration=Fraction(1001, 30000),
                        source="ai",
                        status="plain",
                    )
                )
                (tag,) = catalog.list_tags(media_id)
                self.assertEqual(tag.start, Fraction(1, 3))
                self.assertEqual(tag.duration, Fraction(1001, 30000))
            with Catalog.in_memory() as memory:
                memory_id = memory.put_media(record)
                memory_back = memory.get_media(media_path)
                assert memory_back is not None
                self.assertEqual(memory_back.duration, Fraction(1001, 1000))
                self.assertEqual(memory_back.frame_rate, Fraction(30000, 1001))
                bad_path = places["footage"] / "clip-b.mov"
                bad_path.write_bytes(b"clip-bytes-b")
                bad = self.full_media(bad_path, asset_id="asset-2")
                bad.duration = 3.5  # type: ignore[assignment]
                with self.assertRaises(CatalogError):
                    memory.put_media(bad)
                self.assertIsNone(memory.get_media(bad_path))
                flagged = self.full_media(bad_path, asset_id="asset-3")
                flagged.duration = True  # type: ignore[assignment]
                with self.assertRaises(CatalogError):
                    memory.put_media(flagged)
                self.assertIsNone(memory.get_media(bad_path))
                _ = memory_id

    def test_locked_tag_survives_ai_update(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            media_path = places["footage"] / "clip-a.mov"
            with Catalog.open(settings) as catalog:
                media_id = catalog.put_media(self.full_media(media_path))
                catalog.put_tag(
                    TagRecord(
                        media_id=media_id,
                        folder="Place",
                        value="Invented Cove",
                        source="editor",
                        status="locked",
                    )
                )
                self.assertFalse(
                    catalog.put_tag(
                        TagRecord(
                            media_id=media_id,
                            folder="Place",
                            value="Invented Bay",
                            source="ai",
                            status="plain",
                        )
                    )
                )
                (tag,) = catalog.list_tags(media_id)
                self.assertEqual(tag.value, "Invented Cove")
                self.assertEqual(tag.source, "editor")
                self.assertEqual(tag.status, "locked")
                with self.subTest("unlocked ai tag is replaced by a new ai value"):
                    catalog.put_tag(
                        TagRecord(
                            media_id=media_id,
                            folder="Shot",
                            value="wide",
                            source="ai",
                            status="plain",
                        )
                    )
                    (first,) = [
                        t for t in catalog.list_tags(media_id)
                        if t.folder == "Shot"
                    ]
                    self.assertTrue(
                        catalog.put_tag(
                            TagRecord(
                                media_id=media_id,
                                folder="Shot",
                                value="tight",
                                source="ai",
                                status="plain",
                            )
                        )
                    )
                    (second,) = [
                        t for t in catalog.list_tags(media_id)
                        if t.folder == "Shot"
                    ]
                    self.assertEqual(second.value, "tight")
                    self.assertEqual(second.id, first.id)
                with self.subTest("editor update replaces an editor-locked tag"):
                    self.assertTrue(
                        catalog.put_tag(
                            TagRecord(
                                media_id=media_id,
                                folder="Place",
                                value="Invented Point",
                                source="editor",
                                status="locked",
                            )
                        )
                    )
                    (tag,) = [
                        t for t in catalog.list_tags(media_id)
                        if t.folder == "Place"
                    ]
                    self.assertEqual(tag.value, "Invented Point")
                with self.subTest("metadata-locked tag is not replaced by gps"):
                    catalog.put_tag(
                        TagRecord(
                            media_id=media_id,
                            folder="Day",
                            value="day-one",
                            source="metadata",
                            status="locked",
                        )
                    )
                    self.assertFalse(
                        catalog.put_tag(
                            TagRecord(
                                media_id=media_id,
                                folder="Day",
                                value="day-two",
                                source="gps",
                                status="plain",
                            )
                        )
                    )
                    (tag,) = [
                        t for t in catalog.list_tags(media_id)
                        if t.folder == "Day"
                    ]
                    self.assertEqual(tag.value, "day-one")

    def test_assets_tsv_header_matches_template(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            media_path = places["footage"] / "clip-a.mov"
            with Catalog.open(settings) as catalog:
                record = self.full_media(media_path)
                record.camera = "invented-cam-a"
                catalog.put_media(record)
                other_path = places["footage"] / "clip-b.mov"
                other_path.write_bytes(b"clip-bytes-b")
                other = self.full_media(other_path, asset_id="asset-2")
                other.duration = Fraction(2, 3)
                other.frame_rate = None
                other.camera = None
                other.fcp_event = None
                other.fcp_asset_uid = None
                catalog.put_media(other)
                third_path = places["footage"] / "clip-c.mov"
                third_path.write_bytes(b"clip-bytes-c")
                third = self.full_media(third_path, asset_id="asset-3")
                third.duration = None
                third.frame_rate = Fraction(25, 1)
                third.camera = None
                catalog.put_media(third)
                text = export_assets_tsv(catalog)
            fixture = FIXTURE_PATH.read_text(encoding="utf-8")
            self.assertEqual(
                text.splitlines()[0].split("\t"),
                fixture.strip().split("\t"),
            )
            self.assertEqual(
                text.splitlines()[0].split("\t"), list(ASSETS_TSV_COLUMNS)
            )
            rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
            self.assertEqual(rows[0], list(ASSETS_TSV_COLUMNS))
            self.assertEqual(len(rows), 4)
            row = dict(zip(rows[0], rows[1]))
            rounded = dict(zip(rows[0], rows[2]))
            unknown = dict(zip(rows[0], rows[3]))
            self.assertEqual(row["asset_id"], "asset-1")
            self.assertEqual(row["source_path"], os.path.abspath(media_path))
            self.assertEqual(row["source_filename"], "clip-a.mov")
            self.assertEqual(row["fps_num"], "30000")
            self.assertEqual(row["fps_den"], "1001")
            self.assertEqual(row["duration_seconds"], "1.001")
            self.assertEqual(row["camera"], "invented-cam-a")
            self.assertEqual(row["library_asset_id"], "asset-uid-1")
            self.assertEqual(row["library_event"], "Invented Event")
            for column in (
                "timebase",
                "event",
                "audio_channels",
                "availability",
                "review_method",
                "reviewed_ranges",
                "notes",
                "library_registration_status",
                "browser_playback_verified",
                "used_in_projects",
            ):
                self.assertEqual(row[column], "")
            self.assertEqual(rounded["asset_id"], "asset-2")
            self.assertEqual(rounded["duration_seconds"], "0.666667")
            self.assertEqual(rounded["fps_num"], "")
            self.assertEqual(rounded["fps_den"], "")
            self.assertEqual(rounded["camera"], "")
            self.assertEqual(rounded["library_asset_id"], "")
            self.assertEqual(rounded["library_event"], "")
            self.assertEqual(unknown["asset_id"], "asset-3")
            self.assertEqual(unknown["duration_seconds"], "")
            self.assertEqual(unknown["fps_num"], "25")
            self.assertEqual(unknown["fps_den"], "1")

    def test_refuses_symlinked_pipeline(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            footage = places["footage"]
            project = places["project"]
            before = sorted(
                os.path.relpath(p, footage) for p in footage.rglob("*")
            )
            (project / ".pipeline").symlink_to(footage, target_is_directory=True)
            with self.assertRaises(SafeWriteError):
                Catalog.open(settings)
            self.assertEqual(
                sorted(os.path.relpath(p, footage) for p in footage.rglob("*")),
                before,
            )
            self.assertFalse(
                os.path.lexists(project / ".pipeline" / "catalog.sqlite")
            )
            with self.subTest("symlink to another folder inside the project"):
                (project / ".pipeline").unlink()
                other = project / "other"
                other.mkdir()
                (other / "note.txt").write_text("invented", encoding="utf-8")
                (project / ".pipeline").symlink_to(other, target_is_directory=True)
                with self.assertRaises(SafeWriteError):
                    Catalog.open(settings)
                self.assertFalse(
                    os.path.lexists(project / ".pipeline" / "catalog.sqlite")
                )
                self.assertFalse(os.path.lexists(other / "catalog.sqlite"))

    def test_refuses_pipeline_swapped_before_connect(self):
        import fcp_pipeline.catalog as catalog_module

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            project = places["project"]
            pipeline = project / ".pipeline"
            with Catalog.open(settings):
                pass
            db_path = pipeline / "catalog.sqlite"
            self.assertTrue(db_path.is_file())
            real_bytes = db_path.read_bytes()
            decoy = tmp / "decoy-footage"
            decoy.mkdir()
            (decoy / "clip-a.mov").write_bytes(b"decoy-bytes-a")
            (decoy / "clip-b.mov").write_bytes(b"decoy-bytes-b")

            def snapshot(folder: Path):
                return {
                    entry.name: (
                        entry.stat().st_size,
                        hashlib.sha256(entry.read_bytes()).hexdigest(),
                    )
                    for entry in sorted(folder.iterdir())
                    if entry.is_file() and not entry.is_symlink()
                }

            before = snapshot(decoy)
            self.assertEqual(len(before), 2)
            real_connect = catalog_module.sqlite3.connect
            swapped = []
            backup = project / ".pipeline-real"

            def swapping_connect(*args, **kwargs):
                if kwargs.get("uri"):
                    return real_connect(*args, **kwargs)
                if not swapped:
                    os.rename(pipeline, backup)
                    pipeline.symlink_to(decoy, target_is_directory=True)
                    swapped.append(True)
                return real_connect(*args, **kwargs)

            try:
                with mock.patch.object(
                    catalog_module.sqlite3,
                    "connect",
                    side_effect=swapping_connect,
                ):
                    with self.assertRaises(SafeWriteError):
                        Catalog.open(settings)
            finally:
                if pipeline.is_symlink():
                    pipeline.unlink()
                if backup.is_dir() and not pipeline.exists():
                    os.rename(backup, pipeline)
            self.assertTrue(swapped, "the .pipeline swap never happened")
            self.assertTrue(
                (decoy / "catalog.sqlite").is_file(),
                "the refused open never reached the swapped folder",
            )
            after = snapshot(decoy)
            self.assertEqual(
                {name: after[name] for name in before},
                before,
                "pre-existing decoy files changed",
            )
            self.assertEqual(db_path.read_bytes(), real_bytes)

    def test_refuses_protected_overlap(self):
        cases = (
            "footage",
            "photos",
            "assets",
            "library",
            "card-is-project",
            "fcpbundle",
        )
        for case in cases:
            with self.subTest(case):
                with tempfile.TemporaryDirectory() as raw:
                    tmp = Path(raw)
                    if case == "footage":
                        root = tmp / "footage"
                        root.mkdir()
                        project = root / "project"
                        footage, photos = root, tmp / "photos"
                        photos.mkdir()
                        assets, library = tmp / "assets", tmp / "EditLib"
                        assets.mkdir()
                        library.mkdir()
                    elif case == "photos":
                        root = tmp / "photos"
                        root.mkdir()
                        project = root / "project"
                        footage, photos = tmp / "footage", root
                        footage.mkdir()
                        assets, library = tmp / "assets", tmp / "EditLib"
                        assets.mkdir()
                        library.mkdir()
                    elif case == "assets":
                        root = tmp / "assets"
                        root.mkdir()
                        project = root / "project"
                        footage, photos = tmp / "footage", tmp / "photos"
                        footage.mkdir()
                        photos.mkdir()
                        assets, library = root, tmp / "EditLib"
                        library.mkdir()
                    elif case == "library":
                        root = tmp / "EditLib"
                        root.mkdir()
                        project = root / "project"
                        footage, photos = tmp / "footage", tmp / "photos"
                        footage.mkdir()
                        photos.mkdir()
                        assets, library = tmp / "assets", root
                        assets.mkdir()
                    elif case == "card-is-project":
                        project = tmp / "project"
                        footage, photos = tmp / "footage", tmp / "photos"
                        footage.mkdir()
                        photos.mkdir()
                        assets, library = tmp / "assets", tmp / "EditLib"
                        assets.mkdir()
                        library.mkdir()
                    else:
                        root = tmp / "Example.FCPBundle"
                        root.mkdir()
                        project = root / "project"
                        footage, photos = tmp / "footage", tmp / "photos"
                        footage.mkdir()
                        photos.mkdir()
                        assets, library = tmp / "assets", tmp / "EditLib"
                        assets.mkdir()
                        library.mkdir()
                    project.mkdir(parents=True)
                    (project / "project.toml").write_text(
                        "title = \"Invented Overlap Survey\"\n"
                        "[final_cut]\n"
                        f"library = \"{library.as_posix()}\"\n"
                        "events = [\"Invented Event\"]\n"
                        "[media]\n"
                        f"footage = [\"{footage.as_posix()}\"]\n"
                        f"photos = [\"{photos.as_posix()}\"]\n"
                        f"assets = \"{assets.as_posix()}\"\n"
                        "[pipeline]\n"
                        "timezone = \"UTC\"\n"
                        "start_date = 2020-06-15\n",
                        encoding="utf-8",
                    )
                    settings = load_trip(project)
                    cards = [project] if case == "card-is-project" else []
                    with self.assertRaises(SafeWriteError):
                        Catalog.open(settings, cards=cards)
                    self.assertFalse(os.path.lexists(project / ".pipeline"))

    def test_writes_only_in_pipeline(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            project = places["project"]
            media_path = places["footage"] / "clip-a.mov"

            def snapshot():
                found = set()
                for base, dirs, files in os.walk(project, followlinks=False):
                    for name in dirs + files:
                        found.add(
                            os.path.relpath(os.path.join(base, name), project)
                        )
                return found

            before = snapshot()
            with Catalog.open(settings) as catalog:
                media_id = catalog.put_media(self.full_media(media_path))
                catalog.put_tag(
                    TagRecord(
                        media_id=media_id,
                        folder="Place",
                        value="Invented Cove",
                        source="metadata",
                        status="plain",
                    )
                )
                catalog.add_transcript(
                    TranscriptRecord(
                        media_id=media_id,
                        path="transcripts/clip-a.txt",
                        created="2020-06-15T08:30:00Z",
                    )
                )
                run_id = catalog.start_run("scan", dry_run=True)
                catalog.finish_run(run_id, "ok")
                catalog.add_export(run_id, "edit/001-cut.fcpxml", "ef" * 32)
            after = snapshot()
            new = sorted(after - before)
            self.assertTrue(new, "expected new files under .pipeline/")
            for relpath in new:
                self.assertTrue(
                    relpath == ".pipeline"
                    or relpath.startswith(".pipeline" + os.sep),
                    f"write outside .pipeline/: {relpath}",
                )
            self.assertEqual(
                sorted(os.listdir(project / ".pipeline")), ["catalog.sqlite"]
            )

    def test_export_command(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            settings, places = self.make_project(tmp)
            project = places["project"]
            media_path = places["footage"] / "clip-a.mov"
            base = [sys.executable, "-m", "fcp_pipeline.catalog"]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

            def run(*args):
                return subprocess.run(
                    [*base, *args],
                    cwd=str(ROOT),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

            missing = run("export", "--tsv", "assets", "--trip", str(project))
            self.assertEqual(missing.returncode, 1)
            self.assertTrue(missing.stderr.strip(), "one line on stderr")
            self.assertEqual(len(missing.stderr.strip().splitlines()), 1)
            self.assertFalse(os.path.lexists(project / ".pipeline"))
            with Catalog.open(settings) as catalog:
                catalog.put_media(self.full_media(media_path))
            printed = run("export", "--tsv", "assets", "--trip", str(project))
            self.assertEqual(printed.returncode, 0)
            rows = list(csv.reader(io.StringIO(printed.stdout), delimiter="\t"))
            self.assertEqual(rows[0], list(ASSETS_TSV_COLUMNS))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1][0], "asset-1")
            target = project / "edit" / "assets.tsv"
            written = run(
                "export",
                "--tsv",
                "assets",
                "--trip",
                str(project),
                "--output",
                "edit/assets.tsv",
            )
            self.assertEqual(written.returncode, 0)
            first_bytes = target.read_bytes()
            self.assertTrue(first_bytes.startswith(b"asset_id\t"))
            repeat = run(
                "export",
                "--tsv",
                "assets",
                "--trip",
                str(project),
                "--output",
                "edit/assets.tsv",
            )
            self.assertEqual(repeat.returncode, 1)
            self.assertEqual(target.read_bytes(), first_bytes)
            refused = run(
                "export",
                "--tsv",
                "assets",
                "--trip",
                str(project),
                "--output",
                "outline.md",
            )
            self.assertEqual(refused.returncode, 1)
            self.assertFalse(os.path.lexists(project / "outline.md"))
            usage = run("export", "--tsv", "bogus", "--trip", str(project))
            self.assertEqual(usage.returncode, 2)
            bare = run()
            self.assertEqual(bare.returncode, 2)


if __name__ == "__main__":
    unittest.main()
