"""The project's catalog: one SQLite file under ``.pipeline/``.

:class:`Catalog` keeps ``.pipeline/catalog.sqlite`` with the standard
library's :mod:`sqlite3`: media rows, tags, transcripts, runs and exports.
Every time and rate is an exact :class:`~fractions.Fraction`, never a
rounded decimal. The schema version upgrades older files and refuses newer
ones untouched. :func:`export_assets_tsv` renders the editor's ``assets.tsv``
template, and :func:`main` runs the ``export`` command. Importing this
module opens no database and touches no file.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sqlite3
import stat as stat_module
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from fcp_pipeline.safe_write import SafeWriteError, SafeWriter
from fcp_pipeline.trip import TripSettingsError, load_trip

__all__ = [
    "ASSETS_TSV_COLUMNS",
    "Catalog",
    "CatalogError",
    "ExportRecord",
    "MediaRecord",
    "RunRecord",
    "SOURCE_RANK",
    "TagRecord",
    "TranscriptRecord",
    "export_assets_tsv",
    "main",
]

#: Current schema version. Version 1 is the same catalog without the
#: exports table; version 2 adds it.
SCHEMA_VERSION = 2

#: Allowed media kinds, tag sources (lowest to highest rank) and tag statuses.
KINDS = ("clip", "still", "memo", "voiceover")
SOURCES = ("metadata", "gps", "slate", "callout", "vision-text", "ai", "editor")
SOURCE_RANK = ("ai", "vision-text", "callout", "slate", "gps", "metadata", "editor")
STATUSES = ("plain", "check", "locked", "deleted")

#: Header of the editor's assets.tsv template, in order.
ASSETS_TSV_COLUMNS = (
    "asset_id",
    "source_path",
    "source_filename",
    "duration_seconds",
    "fps_num",
    "fps_den",
    "timebase",
    "camera",
    "event",
    "audio_channels",
    "availability",
    "review_method",
    "reviewed_ranges",
    "notes",
    "library_asset_id",
    "library_event",
    "library_registration_status",
    "browser_playback_verified",
    "used_in_projects",
)

_CATALOG_FILENAME = "catalog.sqlite"
_PIPELINE_DIRNAME = ".pipeline"

_RATIONAL_RE = re.compile(r"(-?\d+)/(\d+)")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


class CatalogError(ValueError):
    """Raised for bad catalog values, bad schema versions and non-catalogs.

    The message names the path, key or value and the rule it broke, never
    file contents.
    """


@dataclass
class MediaRecord:
    """One media row; every column is a field, ``id`` is None before insert."""

    id: int | None = None
    asset_id: str | None = None
    kind: str | None = None
    path: str | None = None
    size: int | None = None
    mtime_ns: int | None = None
    inode: int | None = None
    camera: str | None = None
    duration: Fraction | None = None
    frame_rate: Fraction | None = None
    start_timecode: Fraction | None = None
    capture_time: str | None = None
    capture_time_source: str | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    day: int | None = None
    place: str | None = None
    fcp_event: str | None = None
    fcp_event_uid: str | None = None
    fcp_asset_uid: str | None = None


@dataclass
class TagRecord:
    """One tag row; ``start`` and ``duration`` both None means the whole clip."""

    id: int | None = None
    media_id: int | None = None
    folder: str | None = None
    value: str | None = None
    start: Fraction | None = None
    duration: Fraction | None = None
    source: str | None = None
    confidence: float | None = None
    status: str | None = None


@dataclass
class TranscriptRecord:
    """One transcript row."""

    id: int | None = None
    media_id: int | None = None
    path: str | None = None
    sha256: str | None = None
    language: str | None = None
    engine: str | None = None
    created: str | None = None


@dataclass
class RunRecord:
    """One run row."""

    id: int | None = None
    step: str | None = None
    started: str | None = None
    finished: str | None = None
    dry_run: bool | None = None
    outcome: str | None = None


@dataclass
class ExportRecord:
    """One export row."""

    id: int | None = None
    run_id: int | None = None
    path: str | None = None
    sha256: str | None = None
    read_at: str | None = None


# -- small helpers ----------------------------------------------------


def _utcnow() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_utc(value: str | None, *, name: str) -> str | None:
    """Validate an optional UTC timestamp, returning it unchanged."""
    if value is None:
        return None
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise CatalogError(
            f"bad {name} {value!r}: must be UTC like 2020-06-15T08:30:00Z"
        )
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise CatalogError(
            f"bad {name} {value!r}: must be UTC like 2020-06-15T08:30:00Z"
        ) from None
    return value


def _to_fraction_text(value: Fraction | int | None, *, name: str) -> str | None:
    """Store a rational as ``N/D`` in lowest terms with ``D > 0``.

    Accepts :class:`~fractions.Fraction` and :class:`int` (an int comes
    back as ``Fraction`` on read); ``None`` means unknown. A float or
    bool raises :class:`CatalogError` before anything is written.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        raise CatalogError(
            f"bad {name} {value!r}: must be a Fraction or int, never a float or bool"
        )
    frac = value if isinstance(value, Fraction) else Fraction(value, 1)
    return f"{frac.numerator}/{frac.denominator}"


def _from_fraction_text(text: str | None, *, name: str) -> Fraction | None:
    """Read a stored ``N/D`` rational back into a :class:`Fraction`.

    A stored value that is not ``N/D`` raises :class:`CatalogError`.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        raise CatalogError(
            f"bad stored {name} {text!r}: must be TEXT like '30000/1001'"
        )
    match = _RATIONAL_RE.fullmatch(text)
    if match is None:
        raise CatalogError(
            f"bad stored {name} {text!r}: must be TEXT like '30000/1001'"
        )
    numerator, denominator = int(match.group(1)), int(match.group(2))
    if denominator <= 0:
        raise CatalogError(
            f"bad stored {name} {text!r}: denominator must be above zero"
        )
    return Fraction(numerator, denominator)


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise CatalogError(f"bad {name} {value!r}: must be a string")
    return value


def _optional_text(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, name=name)


def _require_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CatalogError(f"bad {name} {value!r}: must be an int, never a float or bool")
    return value


def _optional_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, name=name)


def _optional_float(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogError(
            f"bad {name} {value!r}: must be a number, never a bool"
        )
    return float(value)


# -- schema text (read lazily: importing this module touches no file) --

_schema_cache: str | None = None


def _schema_sql() -> str:
    """The complete current schema, read from ``schema.sql`` next to this file."""
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = Path(__file__).with_name("schema.sql").read_text(
            encoding="utf-8"
        )
    return _schema_cache


def _split_statements(sql: str) -> list[str]:
    """Split SQL text into single statements, dropping empty parts."""
    return [part.strip() for part in sql.split(";") if part.strip()]


def _upgrades() -> dict[int, str]:
    """Ordered mapping from each version to the SQL upgrading the one before it.

    Version 1 is the current schema without the exports table, so the
    version 2 upgrade is the exact exports-table statement sliced out of
    ``schema.sql``: a file upgraded with it ends up byte-identical in
    ``sqlite_master`` to a fresh catalog.
    """
    schema = _schema_sql()
    marker = "CREATE TABLE exports"
    start = schema.index(marker)
    return {2: schema[start:].strip()}


def _fresh_statements() -> list[str]:
    return _split_statements(_schema_sql())


def _open_connection(path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open one connection with foreign keys on and temp tables in memory."""
    if read_only:
        uri = Path(os.fspath(path)).as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
    else:
        connection = sqlite3.connect(os.fspath(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA temp_store = MEMORY")
    if not read_only:
        connection.execute("PRAGMA journal_mode = DELETE")
    return connection


def _probe_version(db_path: Path) -> tuple[str, int | None]:
    """Read the schema version through a read-only connection.

    Returns ``("missing" | "empty" | "unversioned" | "ok", version)`` where
    the version is only set for ``"ok"``. The read-only connection writes
    nothing: no journal, no new file.
    """
    connection = _open_connection(db_path, read_only=True)
    try:
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if not names:
            return ("empty", None)
        if "schema_version" not in names:
            return ("unversioned", None)
        row = connection.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            raise CatalogError(
                f"bad catalog at {db_path}: schema_version holds no row"
            )
        version = row[0]
        if isinstance(version, bool) or not isinstance(version, int):
            raise CatalogError(
                f"bad catalog at {db_path}: schema version {version!r} is not an int"
            )
        return ("ok", version)
    finally:
        connection.close()


# -- the catalog ------------------------------------------------------


class Catalog:
    """One open catalog: ``.pipeline/catalog.sqlite`` or private memory."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        path: Path | None,
        schema_version: int,
    ) -> None:
        self._connection = connection
        self._path = path
        self._schema_version = schema_version
        self._closed = False

    @classmethod
    def open(cls, settings, *, cards=(), create: bool = True) -> Catalog:
        """Open ``<project>/.pipeline/catalog.sqlite`` for a loaded trip.

        ``settings`` is the :class:`TripSettings` that
        :func:`fcp_pipeline.trip.load_trip` returns; ``cards`` is any card
        volumes a caller hands in to protect. Every path check below runs
        before SQLite opens anything, and any refusal raises
        :class:`SafeWriteError` leaving nothing behind: no ``.pipeline/``
        folder, no file.
        """
        project_abs = Path(os.path.abspath(os.fspath(settings.folder)))
        project = project_abs.resolve()
        if not project.is_dir():
            raise SafeWriteError(
                f"refused to open catalog in {project_abs}: project folder "
                f"{project} is not an existing folder (the catalog lives only "
                "in the project's own .pipeline/ folder)"
            )
        pipeline = project / _PIPELINE_DIRNAME
        if os.path.lexists(pipeline):
            if pipeline.is_symlink():
                raise SafeWriteError(
                    f"refused to open catalog in {pipeline}: .pipeline is a "
                    "symlink (the catalog lives only in the project's own "
                    ".pipeline/ folder)"
                )
            if not pipeline.is_dir():
                raise SafeWriteError(
                    f"refused to open catalog in {pipeline}: .pipeline exists "
                    "and is not a folder"
                )
        db_path = pipeline / _CATALOG_FILENAME
        db_resolved = db_path.resolve()

        protected: list[tuple[str, Path]] = []

        def add(label: str, value: object) -> None:
            if value is None:
                return
            try:
                text = os.fspath(value)
            except TypeError:
                raise SafeWriteError(
                    f"refused to open catalog at {db_path}: protected "
                    f"{label} value {value!r} is not a path"
                ) from None
            if not text:
                return
            root = Path(text)
            if not root.is_absolute():
                root = project_abs / root
            protected.append((label, root.resolve()))

        for entry in settings.footage:
            add("[media] footage", entry)
        for entry in settings.photos:
            add("[media] photos", entry)
        add("[media] assets", settings.assets)
        add("[final_cut] library", settings.library)
        for card in cards:
            add("card", card)
        for label, root in protected:
            if db_resolved == root or db_resolved.is_relative_to(root):
                raise SafeWriteError(
                    f"refused to open catalog at {db_path}: inside protected "
                    f"{label} path {root} (the catalog lives only in the "
                    "project's own .pipeline/ folder)"
                )
        for part in db_resolved.parts:
            if part.lower().endswith(".fcpbundle"):
                raise SafeWriteError(
                    f"refused to open catalog at {db_path}: inside a "
                    "*.fcpbundle bundle (the catalog never lives in a "
                    "Final Cut bundle)"
                )

        if not os.path.lexists(pipeline):
            if not create:
                raise CatalogError(
                    f"no catalog at {db_path}: the file does not exist "
                    "(nothing was created)"
                )
            os.mkdir(pipeline)
            if pipeline.is_symlink() or not pipeline.is_dir():
                raise SafeWriteError(
                    f"refused to open catalog in {pipeline}: .pipeline is "
                    "not a folder"
                )
        if os.path.lexists(db_path):
            if os.path.islink(db_path):
                raise SafeWriteError(
                    f"refused to open catalog at {db_path}: the catalog file "
                    "is a symlink (it must be a regular file)"
                )
            info = os.lstat(db_path)
            if not stat_module.S_ISREG(info.st_mode):
                raise SafeWriteError(
                    f"refused to open catalog at {db_path}: the catalog file "
                    "is not a regular file"
                )
            if info.st_nlink != 1:
                raise SafeWriteError(
                    f"refused to open catalog at {db_path}: the catalog file "
                    f"has {info.st_nlink} hard links, expected exactly one"
                )
            state, version = _probe_version(db_path)
            if state == "unversioned":
                raise CatalogError(
                    f"bad catalog at {db_path}: the file holds tables but no "
                    "schema_version table, so it is not a catalog"
                )
            if state == "ok" and version is not None and version > SCHEMA_VERSION:
                raise CatalogError(
                    f"bad catalog at {db_path}: schema version {version} is "
                    f"newer than this catalog code's version {SCHEMA_VERSION}; "
                    "the file was left untouched"
                )
            connection = _open_connection(db_path)
            catalog = cls(connection, db_path, version or SCHEMA_VERSION)
            try:
                if state == "empty":
                    catalog._create_fresh()
                elif state == "ok" and (version or SCHEMA_VERSION) < SCHEMA_VERSION:
                    catalog._upgrade(version or 0)
            except Exception:
                catalog.close()
                raise
            return catalog
        if not create:
            raise CatalogError(
                f"no catalog at {db_path}: the file does not exist "
                "(nothing was created)"
            )
        connection = _open_connection(db_path)
        catalog = cls(connection, db_path, SCHEMA_VERSION)
        try:
            catalog._create_fresh()
        except Exception:
            catalog.close()
            raise
        return catalog

    @classmethod
    def in_memory(cls) -> Catalog:
        """Open a private in-memory catalog with the same schema."""
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA temp_store = MEMORY")
        catalog = cls(connection, None, SCHEMA_VERSION)
        try:
            catalog._create_fresh()
        except Exception:
            catalog.close()
            raise
        return catalog

    # -- setup ------------------------------------------------------

    def _create_fresh(self) -> None:
        """Write the current schema and version row in one transaction."""
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            for statement in _fresh_statements():
                self._connection.execute(statement)
            self._connection.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        self._schema_version = SCHEMA_VERSION

    def _upgrade(self, version: int) -> None:
        """Upgrade one older version to the current schema in one transaction."""
        upgrades = _upgrades()
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            while version < SCHEMA_VERSION:
                step = upgrades.get(version + 1)
                if step is None:
                    raise CatalogError(
                        f"bad catalog at {self._path}: no upgrade from schema "
                        f"version {version} to {SCHEMA_VERSION}"
                    )
                for statement in _split_statements(step):
                    self._connection.execute(statement)
                version += 1
            self._connection.execute(
                "UPDATE schema_version SET version = ?", (SCHEMA_VERSION,)
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        self._schema_version = SCHEMA_VERSION

    # -- lifecycle --------------------------------------------------

    @property
    def path(self) -> Path | None:
        """The database file, or None for an in-memory catalog."""
        return self._path

    @property
    def schema_version(self) -> int:
        """The schema version this catalog was opened with."""
        return self._schema_version

    def close(self) -> None:
        """Close the catalog's connection; safe to call twice."""
        if not self._closed:
            self._closed = True
            self._connection.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _ensure_open(self) -> sqlite3.Connection:
        if self._closed:
            raise CatalogError("catalog is closed")
        return self._connection

    # -- media ------------------------------------------------------

    @staticmethod
    def _media_params(record: MediaRecord) -> tuple:
        asset_id = _require_text(record.asset_id, name="asset_id")
        if not asset_id:
            raise CatalogError("bad asset_id '': must be a non-empty string")
        if record.kind not in KINDS:
            raise CatalogError(
                f"bad kind {record.kind!r}: must be one of {', '.join(KINDS)}"
            )
        raw_path = os.fspath(record.path) if record.path is not None else None
        if not isinstance(raw_path, str) or not raw_path:
            raise CatalogError(
                f"bad path {record.path!r}: must be a non-empty path"
            )
        stored_path = os.path.abspath(raw_path)
        return (
            asset_id,
            record.kind,
            stored_path,
            _optional_int(record.size, name="size"),
            _optional_int(record.mtime_ns, name="mtime_ns"),
            _optional_int(record.inode, name="inode"),
            _optional_text(record.camera, name="camera"),
            _to_fraction_text(record.duration, name="duration"),
            _to_fraction_text(record.frame_rate, name="frame_rate"),
            _to_fraction_text(record.start_timecode, name="start_timecode"),
            _check_utc(record.capture_time, name="capture_time"),
            _optional_text(record.capture_time_source, name="capture_time_source"),
            _optional_float(record.gps_lat, name="gps_lat"),
            _optional_float(record.gps_lon, name="gps_lon"),
            _optional_int(record.day, name="day"),
            _optional_text(record.place, name="place"),
            _optional_text(record.fcp_event, name="fcp_event"),
            _optional_text(record.fcp_event_uid, name="fcp_event_uid"),
            _optional_text(record.fcp_asset_uid, name="fcp_asset_uid"),
        )

    @staticmethod
    def _media_from_row(row: sqlite3.Row) -> MediaRecord:
        return MediaRecord(
            id=row["id"],
            asset_id=row["asset_id"],
            kind=row["kind"],
            path=row["path"],
            size=row["size"],
            mtime_ns=row["mtime_ns"],
            inode=row["inode"],
            camera=row["camera"],
            duration=_from_fraction_text(row["duration"], name="duration"),
            frame_rate=_from_fraction_text(row["frame_rate"], name="frame_rate"),
            start_timecode=_from_fraction_text(
                row["start_timecode"], name="start_timecode"
            ),
            capture_time=row["capture_time"],
            capture_time_source=row["capture_time_source"],
            gps_lat=row["gps_lat"],
            gps_lon=row["gps_lon"],
            day=row["day"],
            place=row["place"],
            fcp_event=row["fcp_event"],
            fcp_event_uid=row["fcp_event_uid"],
            fcp_asset_uid=row["fcp_asset_uid"],
        )

    def put_media(self, record: MediaRecord) -> int:
        """Insert a media row, or replace the row with the same path.

        Replacing keeps the old row's id. Every value is validated before
        anything is written. Returns the row id.
        """
        params = self._media_params(record)
        connection = self._ensure_open()
        try:
            existing = connection.execute(
                "SELECT id FROM media WHERE path = ?", (params[2],)
            ).fetchone()
            if existing is not None:
                connection.execute(
                    "UPDATE media SET asset_id = ?, kind = ?, path = ?, "
                    "size = ?, mtime_ns = ?, inode = ?, camera = ?, "
                    "duration = ?, frame_rate = ?, start_timecode = ?, "
                    "capture_time = ?, capture_time_source = ?, gps_lat = ?, "
                    "gps_lon = ?, day = ?, place = ?, fcp_event = ?, "
                    "fcp_event_uid = ?, fcp_asset_uid = ? WHERE id = ?",
                    (*params, existing["id"]),
                )
                connection.commit()
                return int(existing["id"])
            cursor = connection.execute(
                "INSERT INTO media (asset_id, kind, path, size, mtime_ns, "
                "inode, camera, duration, frame_rate, start_timecode, "
                "capture_time, capture_time_source, gps_lat, gps_lon, day, "
                "place, fcp_event, fcp_event_uid, fcp_asset_uid) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params,
            )
            connection.commit()
            return int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise CatalogError(
                f"bad media row for path {params[2]!r}: {exc}"
            ) from None

    def get_media(self, path: str | Path) -> MediaRecord | None:
        """Return the media row for ``os.path.abspath(path)``, or None."""
        raw = os.fspath(path)
        if not isinstance(raw, str):
            raise CatalogError(f"bad path {path!r}: must be a path")
        stored = os.path.abspath(raw)
        connection = self._ensure_open()
        row = connection.execute(
            "SELECT * FROM media WHERE path = ?", (stored,)
        ).fetchone()
        if row is None:
            return None
        return self._media_from_row(row)

    def list_media(self) -> list[MediaRecord]:
        """Return every media row in id order."""
        connection = self._ensure_open()
        return [
            self._media_from_row(row)
            for row in connection.execute("SELECT * FROM media ORDER BY id")
        ]

    def unchanged(self, path: str | Path) -> bool:
        """Whether the catalog's row still matches the file on disk.

        True only when the catalog holds a row for
        ``os.path.abspath(path)`` and :func:`os.stat` gives the same size,
        ``st_mtime_ns`` and ``st_ino``. A missing file, a missing row or any
        unknown stored value gives False. Stats the file; never opens it.
        """
        try:
            raw = os.fspath(path)
        except TypeError:
            return False
        if not isinstance(raw, str):
            return False
        stored = os.path.abspath(raw)
        connection = self._ensure_open()
        row = connection.execute(
            "SELECT size, mtime_ns, inode FROM media WHERE path = ?", (stored,)
        ).fetchone()
        if row is None:
            return False
        if row["size"] is None or row["mtime_ns"] is None or row["inode"] is None:
            return False
        try:
            info = os.stat(stored)
        except OSError:
            return False
        return (
            info.st_size == row["size"]
            and info.st_mtime_ns == row["mtime_ns"]
            and info.st_ino == row["inode"]
        )

    # -- tags -------------------------------------------------------

    @staticmethod
    def _tag_params(record: TagRecord) -> tuple:
        media_id = _require_int(record.media_id, name="media_id")
        folder = _require_text(record.folder, name="folder")
        value = _require_text(record.value, name="value")
        if record.source not in SOURCES:
            raise CatalogError(
                f"bad source {record.source!r}: must be one of {', '.join(SOURCES)}"
            )
        if record.status not in STATUSES:
            raise CatalogError(
                f"bad status {record.status!r}: must be one of "
                f"{', '.join(STATUSES)}"
            )
        confidence = record.confidence
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(
                confidence, (int, float)
            ):
                raise CatalogError(
                    f"bad confidence {confidence!r}: must be 0 to 1, or NULL"
                )
            if not 0 <= confidence <= 1:
                raise CatalogError(
                    f"bad confidence {confidence!r}: must be 0 to 1, or NULL"
                )
            confidence = float(confidence)
        return (
            media_id,
            folder,
            value,
            _to_fraction_text(record.start, name="tag start"),
            _to_fraction_text(record.duration, name="tag duration"),
            record.source,
            confidence,
            record.status,
        )

    @staticmethod
    def _tag_from_row(row: sqlite3.Row) -> TagRecord:
        return TagRecord(
            id=row["id"],
            media_id=row["media_id"],
            folder=row["folder"],
            value=row["value"],
            start=_from_fraction_text(row["start"], name="tag start"),
            duration=_from_fraction_text(row["duration"], name="tag duration"),
            source=row["source"],
            confidence=row["confidence"],
            status=row["status"],
        )

    def put_tag(self, tag: TagRecord) -> bool:
        """Insert or replace the tag at the same (media, folder, range).

        With no tag at the same ``(media_id, folder, start, duration)``
        this inserts and returns True. Over a tag that is not locked this
        replaces it, keeping its id, and returns True. Over a locked tag
        this replaces it only when the new source ranks at or above the
        stored one, and otherwise changes nothing and returns False.
        """
        params = self._tag_params(tag)
        media_id, folder, value = params[0], params[1], params[2]
        start_text, duration_text = params[3], params[4]
        source, confidence, status = params[5], params[6], params[7]
        connection = self._ensure_open()
        try:
            existing = connection.execute(
                "SELECT id, source, status FROM tags WHERE media_id = ? "
                "AND folder = ? AND ifnull(start, '') = ifnull(?, '') "
                "AND ifnull(duration, '') = ifnull(?, '')",
                (media_id, folder, start_text, duration_text),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO tags (media_id, folder, value, start, "
                    "duration, source, confidence, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (media_id, folder, value, start_text, duration_text,
                     source, confidence, status),
                )
                connection.commit()
                return True
            if existing["status"] == "locked" and (
                SOURCE_RANK.index(source)
                < SOURCE_RANK.index(existing["source"])
            ):
                return False
            connection.execute(
                "UPDATE tags SET value = ?, source = ?, confidence = ?, "
                "status = ? WHERE id = ?",
                (value, source, confidence, status, existing["id"]),
            )
            connection.commit()
            return True
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise CatalogError(
                f"bad tag for media {media_id}: {exc}"
            ) from None

    def list_tags(self, media_id: int) -> list[TagRecord]:
        """Return every tag for a media row in id order."""
        _require_int(media_id, name="media_id")
        connection = self._ensure_open()
        return [
            self._tag_from_row(row)
            for row in connection.execute(
                "SELECT * FROM tags WHERE media_id = ? ORDER BY id",
                (media_id,),
            )
        ]

    # -- transcripts ------------------------------------------------

    def add_transcript(self, record: TranscriptRecord) -> int:
        """Insert a transcript row; returns its id."""
        media_id = _require_int(record.media_id, name="media_id")
        path = _require_text(record.path, name="transcript path")
        connection = self._ensure_open()
        try:
            cursor = connection.execute(
                "INSERT INTO transcripts (media_id, path, sha256, language, "
                "engine, created) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    media_id,
                    path,
                    _optional_text(record.sha256, name="sha256"),
                    _optional_text(record.language, name="language"),
                    _optional_text(record.engine, name="engine"),
                    _check_utc(record.created, name="created"),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise CatalogError(
                f"bad transcript for media {media_id}: {exc}"
            ) from None

    def list_transcripts(self, media_id: int) -> list[TranscriptRecord]:
        """Return every transcript for a media row in id order."""
        _require_int(media_id, name="media_id")
        connection = self._ensure_open()
        out = []
        for row in connection.execute(
            "SELECT * FROM transcripts WHERE media_id = ? ORDER BY id",
            (media_id,),
        ):
            out.append(
                TranscriptRecord(
                    id=row["id"],
                    media_id=row["media_id"],
                    path=row["path"],
                    sha256=row["sha256"],
                    language=row["language"],
                    engine=row["engine"],
                    created=row["created"],
                )
            )
        return out

    # -- runs and exports -------------------------------------------

    def start_run(self, step: str, *, dry_run: bool | int) -> int:
        """Record a started run; returns its id."""
        if not isinstance(step, str) or not step:
            raise CatalogError(
                f"bad step {step!r}: must be a non-empty string"
            )
        if isinstance(dry_run, bool):
            flag = int(dry_run)
        elif type(dry_run) is int and dry_run in (0, 1):
            flag = dry_run
        else:
            raise CatalogError(
                f"bad dry_run {dry_run!r}: must be True, False, 0 or 1"
            )
        connection = self._ensure_open()
        cursor = connection.execute(
            "INSERT INTO runs (step, started, dry_run) VALUES (?, ?, ?)",
            (step, _utcnow(), flag),
        )
        connection.commit()
        return int(cursor.lastrowid)

    def finish_run(self, run_id: int, outcome: str | None) -> None:
        """Record a run's finish time and outcome."""
        _require_int(run_id, name="run_id")
        if outcome is not None and not isinstance(outcome, str):
            raise CatalogError(
                f"bad outcome {outcome!r}: must be a string or None"
            )
        connection = self._ensure_open()
        cursor = connection.execute(
            "UPDATE runs SET finished = ?, outcome = ? WHERE id = ?",
            (_utcnow(), outcome, run_id),
        )
        connection.commit()
        if cursor.rowcount == 0:
            raise CatalogError(f"bad run_id {run_id!r}: no such run")

    def list_runs(self) -> list[RunRecord]:
        """Return every run in id order."""
        connection = self._ensure_open()
        out = []
        for row in connection.execute("SELECT * FROM runs ORDER BY id"):
            out.append(
                RunRecord(
                    id=row["id"],
                    step=row["step"],
                    started=row["started"],
                    finished=row["finished"],
                    dry_run=bool(row["dry_run"]),
                    outcome=row["outcome"],
                )
            )
        return out

    def add_export(self, run_id: int, path: str, sha256: str) -> int:
        """Record an export of a run; the sha256 is checked before writing."""
        _require_int(run_id, name="run_id")
        _require_text(path, name="export path")
        if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
            raise CatalogError(
                f"bad sha256 {sha256!r}: must be 64 lowercase hex characters"
            )
        connection = self._ensure_open()
        try:
            cursor = connection.execute(
                "INSERT INTO exports (run_id, path, sha256, read_at) "
                "VALUES (?, ?, ?, ?)",
                (run_id, path, sha256, _utcnow()),
            )
            connection.commit()
            return int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise CatalogError(
                f"bad export for run {run_id}: {exc}"
            ) from None

    def list_exports(self) -> list[ExportRecord]:
        """Return every export in id order."""
        connection = self._ensure_open()
        out = []
        for row in connection.execute("SELECT * FROM exports ORDER BY id"):
            out.append(
                ExportRecord(
                    id=row["id"],
                    run_id=row["run_id"],
                    path=row["path"],
                    sha256=row["sha256"],
                    read_at=row["read_at"],
                )
            )
        return out


# -- assets.tsv export ------------------------------------------------


def _format_seconds(value: Fraction) -> str:
    """Render a duration as decimal source seconds.

    Exact when the value needs at most 6 decimal places, otherwise rounded
    half-even to 6 places on the exact fraction; trailing zeros are removed,
    so 1001/1000 gives ``1.001`` and 12 gives ``12``.
    """
    scaled = value * 1_000_000
    whole, rest = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * rest
    if twice > scaled.denominator or (
        twice == scaled.denominator and whole % 2 != 0
    ):
        whole += 1
    sign = "-" if whole < 0 else ""
    digits, frac = divmod(abs(whole), 1_000_000)
    if frac == 0:
        return f"{sign}{digits}"
    return f"{sign}{digits}.{frac:06d}".rstrip("0")


def export_assets_tsv(catalog: Catalog) -> str:
    """Render the editor's assets.tsv template for every media row in id order.

    The header is :data:`ASSETS_TSV_COLUMNS`; each row copies ``asset_id``,
    ``camera``, ``path`` (plus its last component), the exact frame rate and
    the duration as decimal seconds. Every column the catalog does not hold
    is left empty, and so is every unknown value: empty means unknown, never
    false or zero.
    """
    buffer = io.StringIO()
    writer = csv.writer(
        buffer, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL
    )
    writer.writerow(list(ASSETS_TSV_COLUMNS))
    for media in catalog.list_media():
        if media.duration is None:
            duration_text = ""
        else:
            duration_text = _format_seconds(media.duration)
        if media.frame_rate is None:
            fps_num, fps_den = "", ""
        else:
            fps_num, fps_den = (
                str(media.frame_rate.numerator),
                str(media.frame_rate.denominator),
            )
        writer.writerow(
            [
                media.asset_id or "",
                media.path or "",
                os.path.basename(media.path) if media.path else "",
                duration_text,
                fps_num,
                fps_den,
                "",
                media.camera or "",
                "",
                "",
                "",
                "",
                "",
                "",
                media.fcp_asset_uid or "",
                media.fcp_event or "",
                "",
                "",
                "",
            ]
        )
    return buffer.getvalue()


# -- command ----------------------------------------------------------


def _fail(message: str) -> int:
    print(f"catalog: error: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run ``python3 -m fcp_pipeline.catalog export ...``; returns an exit code."""
    parser = argparse.ArgumentParser(
        prog="fcp_pipeline.catalog",
        description="Export the project catalog (.pipeline/catalog.sqlite).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    exporter = subparsers.add_parser("export", help="Write a template TSV.")
    exporter.add_argument(
        "--tsv", choices=("assets",), default="assets", help="Template to export."
    )
    exporter.add_argument("--trip", required=True, help="Project folder.")
    exporter.add_argument("--output", default=None, help="Where to write the TSV.")
    args = parser.parse_args(argv)
    try:
        settings = load_trip(args.trip)
    except TripSettingsError as exc:
        return _fail(str(exc))
    try:
        catalog = Catalog.open(settings, create=False)
    except CatalogError as exc:
        return _fail(str(exc))
    except SafeWriteError as exc:
        return _fail(str(exc))
    with catalog:
        try:
            text = export_assets_tsv(catalog)
        except CatalogError as exc:
            return _fail(str(exc))
        if args.output is None:
            sys.stdout.write(text)
            return 0
        try:
            SafeWriter(settings).write_new(args.output, text)
        except SafeWriteError as exc:
            return _fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
