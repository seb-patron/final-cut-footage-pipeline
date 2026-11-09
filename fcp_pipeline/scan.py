"""The scan step: read every clip's facts into the catalog.

:func:`scan_catalog` probes each ``clip`` row and every clip file under
the footage folders read-only, and records duration, frame rate, start
timecode, camera, GPS and capture time. Unchanged files are skipped; a
file that cannot be read is recorded as unreadable, never dropped.
Footage is only ever read: the scan never opens a footage file for
writing. Importing this module runs no tool and touches no file.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import stat as stat_module
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fcp_pipeline.catalog import Catalog, CatalogError, MediaRecord
from fcp_pipeline.probe import (
    PROBE_TOOLS,
    ProbeError,
    ProbeFacts,
    parse_timecode,
    probe_file,
)
from fcp_pipeline.safe_write import SafeWriteError
from fcp_pipeline.trip import TripSettingsError, load_trip

__all__ = [
    "CAPTURE_FILE",
    "CAPTURE_FILENAME",
    "CAPTURE_QUICKTIME",
    "ScanEntry",
    "ScanResult",
    "capture_time_for",
    "main",
    "parse_timecode",
    "scan_catalog",
]

#: Where the capture time came from: the camera's creation date, a
#: drone-style file name, or the file's own modification time.
CAPTURE_QUICKTIME = "quicktime"
CAPTURE_FILENAME = "filename"
CAPTURE_FILE = "file"

#: Clip extensions the footage walk picks up, matched case-insensitively.
_VIDEO_EXTENSIONS = (".mov", ".mp4", ".m4v")

#: A drone-style timestamp anywhere in the file name, case-insensitive.
_DJI_RE = re.compile(r"DJI_(\d{14})_", re.IGNORECASE)


@dataclass(frozen=True)
class ScanEntry:
    """What the scan did with one clip file."""

    path: str
    resolved: str
    status: str
    audio_streams: int | None = None
    width: int | None = None
    height: int | None = None
    capture_verified: bool | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ScanResult:
    """Every entry of one scan, in the order the files were visited."""

    entries: tuple[ScanEntry, ...] = ()

    @property
    def scanned(self) -> int:
        """How many clips were probed and recorded (new and known)."""
        return sum(
            1 for entry in self.entries if entry.status in ("scanned", "added")
        )

    @property
    def skipped(self) -> int:
        """How many clips were already up to date and left alone."""
        return sum(1 for entry in self.entries if entry.status == "skipped")

    @property
    def unreadable(self) -> int:
        """How many files could not be read."""
        return sum(
            1 for entry in self.entries if entry.status == "unreadable"
        )

    @property
    def unreadable_paths(self) -> tuple[str, ...]:
        """The paths of the files that could not be read, in order."""
        return tuple(
            entry.path
            for entry in self.entries
            if entry.status == "unreadable"
        )


def capture_time_for(
    facts: ProbeFacts, path: str | os.PathLike[str], stat_result: os.stat_result
) -> tuple[str, str, bool]:
    """Return the clip's capture time as ``(UTC text, source, verified)``.

    The camera's creation date wins, then a drone-style file name read
    as UTC, then the file's own modification time, which is flagged
    unverified.
    """
    if facts.quicktime_created is not None:
        return (facts.quicktime_created, CAPTURE_QUICKTIME, True)
    base = os.path.basename(os.fspath(path))
    for match in _DJI_RE.finditer(base):
        digits = match.group(1)
        try:
            moment = datetime(
                int(digits[0:4]),
                int(digits[4:6]),
                int(digits[6:8]),
                int(digits[8:10]),
                int(digits[10:12]),
                int(digits[12:14]),
            )
        except ValueError:
            continue
        return (moment.strftime("%Y-%m-%dT%H:%M:%SZ"), CAPTURE_FILENAME, True)
    seconds, nanos = divmod(stat_result.st_mtime_ns, 1_000_000_000)
    moment = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
        microsecond=nanos // 1000
    )
    return (moment.strftime("%Y-%m-%dT%H:%M:%SZ"), CAPTURE_FILE, False)


def _scan_asset_id(resolved: str) -> str:
    """Invent a stable asset id for a newly found path."""
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return "scan-" + digest


def _resolve_key(path_text: str) -> str:
    """Return the comparison key for a path (resolves symlinks)."""
    try:
        return os.fspath(Path(path_text).resolve())
    except OSError:
        return os.path.abspath(path_text)


def _footage_roots(settings: object) -> list[Path]:
    """Return the existing footage folders for these settings."""
    roots: list[Path] = []
    footage = getattr(settings, "footage", [])
    folder = Path(os.fspath(getattr(settings, "folder")))
    for entry in footage:
        raw = os.fspath(entry)
        root = Path(raw) if os.path.isabs(raw) else folder / raw
        try:
            if root.is_dir():
                roots.append(root)
        except OSError:
            continue
    return roots


def _iter_footage_files(settings: object) -> list[tuple[str, str]]:
    """List ``(path, resolved)`` clip files under the footage folders.

    Every regular file with a clip extension is found recursively,
    including inside folders named ``*.fcpbundle``, in sorted resolved
    path order, once each. Symlinks to files are followed by stat only.
    """
    found: dict[str, str] = {}
    for root in _footage_roots(settings):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for name in sorted(filenames):
                if not name.lower().endswith(_VIDEO_EXTENSIONS):
                    continue
                full = os.path.join(dirpath, name)
                try:
                    info = os.stat(full)
                except OSError:
                    continue
                if not stat_module.S_ISREG(info.st_mode):
                    continue
                key = _resolve_key(full)
                if key not in found:
                    found[key] = full
    return [(found[key], key) for key in sorted(found)]


def _unreadable_row(
    catalog: Catalog, asset_id: str, display: str
) -> None:
    """Record a placeholder row so a later scan retries the file."""
    catalog.put_media(MediaRecord(asset_id=asset_id, kind="clip", path=display))


def scan_catalog(
    catalog: Catalog, settings: object = None, *, probe: object = probe_file
) -> ScanResult:
    """Probe every clip and record its facts in an open catalog.

    Scans each ``clip`` row, then every clip file under the footage
    folders when ``settings`` is given. A missing tool raises
    :class:`ProbeError` naming the tool before anything is marked; a
    file that cannot be read is recorded as unreadable and the run goes
    on. A file that changed while it was probed stops the run, even
    when the probe itself failed.
    """
    for tool in PROBE_TOOLS:
        if shutil.which(tool) is None:
            raise ProbeError(
                f"missing required tool: {tool} is not on PATH "
                f"(install {tool} to run the scan)"
            )

    targets: list[tuple[str, object | None, str]] = []
    seen: set[str] = set()
    rows_by_resolved: dict[str, object] = {}
    for row in catalog.list_media():
        key = _resolve_key(row.path or "")
        if key not in rows_by_resolved:
            rows_by_resolved[key] = row
        if row.kind != "clip":
            continue
        if key in seen:
            continue
        seen.add(key)
        targets.append((row.path or "", row, key))
    if settings is not None:
        for full, key in _iter_footage_files(settings):
            if key in seen:
                continue
            seen.add(key)
            targets.append((full, rows_by_resolved.get(key), key))

    entries: list[ScanEntry] = []
    for display, existing, key in targets:
        if (
            existing is not None
            and getattr(existing, "duration", None) is not None
            and catalog.unchanged(display)
        ):
            entries.append(
                ScanEntry(path=display, resolved=key, status="skipped")
            )
            continue
        try:
            before = os.stat(display)
        except OSError as exc:
            reason = f"cannot stat {display}: {exc}"
            if existing is None:
                _unreadable_row(catalog, _scan_asset_id(key), display)
            entries.append(
                ScanEntry(
                    path=display, resolved=key, status="unreadable", reason=reason
                )
            )
            continue
        probe_error: ProbeError | None = None
        try:
            facts = probe(display)  # type: ignore[operator]
        except ProbeError as exc:
            facts = None
            probe_error = exc
        try:
            after = os.stat(display)
        except OSError as exc:
            raise ProbeError(
                f"file changed during scan: {display}: {exc}"
            ) from None
        if (
            after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise ProbeError(f"file changed during scan: {display}")
        if probe_error is not None:
            if existing is None:
                _unreadable_row(catalog, _scan_asset_id(key), display)
            entries.append(
                ScanEntry(
                    path=display,
                    resolved=key,
                    status="unreadable",
                    reason=str(probe_error),
                )
            )
            continue
        assert facts is not None
        capture_text, source, verified = capture_time_for(facts, display, before)
        parts = [part for part in (facts.make, facts.model) if part]
        camera = " ".join(parts) if parts else None
        if facts.gps_lat is not None and facts.gps_lon is not None:
            gps_lat: float | None = facts.gps_lat
            gps_lon: float | None = facts.gps_lon
        else:
            gps_lat, gps_lon = None, None
        if existing is None:
            catalog.put_media(
                MediaRecord(
                    asset_id=_scan_asset_id(key),
                    kind="clip",
                    path=display,
                    size=before.st_size,
                    mtime_ns=before.st_mtime_ns,
                    inode=before.st_ino,
                    camera=camera,
                    duration=facts.duration,
                    frame_rate=facts.frame_rate,
                    start_timecode=facts.start_timecode,
                    capture_time=capture_text,
                    capture_time_source=source,
                    gps_lat=gps_lat,
                    gps_lon=gps_lon,
                )
            )
            status = "added"
        else:
            catalog.put_media(
                MediaRecord(
                    id=existing.id,
                    asset_id=existing.asset_id,
                    kind=existing.kind,
                    path=existing.path,
                    size=before.st_size,
                    mtime_ns=before.st_mtime_ns,
                    inode=before.st_ino,
                    camera=camera,
                    duration=facts.duration,
                    frame_rate=facts.frame_rate,
                    start_timecode=facts.start_timecode,
                    capture_time=capture_text,
                    capture_time_source=source,
                    gps_lat=gps_lat,
                    gps_lon=gps_lon,
                    day=existing.day,
                    place=existing.place,
                    fcp_event=existing.fcp_event,
                    fcp_event_uid=existing.fcp_event_uid,
                    fcp_asset_uid=existing.fcp_asset_uid,
                )
            )
            status = "scanned"
        entries.append(
            ScanEntry(
                path=display,
                resolved=key,
                status=status,
                audio_streams=facts.audio_streams,
                width=facts.width,
                height=facts.height,
                capture_verified=verified,
                reason=None,
            )
        )
    return ScanResult(entries=tuple(entries))


def main(argv: list[str] | None = None) -> int:
    """Run the scan step for one trip folder."""
    parser = argparse.ArgumentParser(
        prog="fcp_pipeline.scan",
        description="Probe every clip and record its facts in the catalog.",
    )
    parser.add_argument(
        "--trip",
        required=True,
        help="Trip folder holding project.toml.",
    )
    args = parser.parse_args(argv)
    try:
        settings = load_trip(args.trip)
    except TripSettingsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    folder = Path(os.fspath(settings.folder))
    for entry in settings.footage:
        raw = os.fspath(entry)
        root = Path(raw) if os.path.isabs(raw) else folder / raw
        try:
            found = root.is_dir()
        except OSError:
            found = False
        if not found:
            print(
                f"warning: footage folder not found: {raw}", file=sys.stderr
            )
    try:
        catalog = Catalog.open(settings)
    except (CatalogError, SafeWriteError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        try:
            run_id = catalog.start_run("scan", dry_run=False)
        except (CatalogError, SafeWriteError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            result = scan_catalog(catalog, settings)
        except ProbeError as exc:
            try:
                catalog.finish_run(run_id, f"failed: {exc}")
            except (CatalogError, SafeWriteError, OSError):
                pass
            print(f"error: {exc}", file=sys.stderr)
            if shutil.which("ffprobe") is None or shutil.which("exiftool") is None:
                return 3
            return 1
        except (CatalogError, SafeWriteError, OSError) as exc:
            try:
                catalog.finish_run(run_id, f"failed: {exc}")
            except (CatalogError, SafeWriteError, OSError):
                pass
            print(f"error: {exc}", file=sys.stderr)
            return 1
        outcome = (
            f"scanned {result.scanned}, "
            f"skipped {result.skipped}, "
            f"unreadable {result.unreadable}"
        )
        catalog.finish_run(run_id, outcome)
    finally:
        catalog.close()
    for entry in result.entries:
        if entry.status == "unreadable":
            print(f"unreadable: {entry.path}: {entry.reason}")
    print(outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
