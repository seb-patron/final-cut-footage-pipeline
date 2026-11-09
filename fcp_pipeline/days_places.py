"""Assign trip days, places and gear keywords to catalog rows.

For every ``clip`` or ``still`` row with a capture time, the camera's
clock offset and the trip time zone move the capture time before the
trip day is counted (Day 1 is the trip's start date). GPS inside a
trip place names that place; GPS outside every place is left for
:func:`place_clusters` to group into unnamed per-day clusters for the
memo step to name. Results land in the catalog only: media rows gain a
day and a place, and whole-clip Day, Place and Gear tags are written
through the catalog's own API. Importing this module runs no tool,
opens no database and touches no file.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fcp_pipeline.catalog import Catalog, CatalogError, MediaRecord, TagRecord
from fcp_pipeline.safe_write import SafeWriteError
from fcp_pipeline.scan import CAPTURE_FILE
from fcp_pipeline.trip import TripSettingsError, load_trip

__all__ = [
    "CLUSTER_RADIUS_M",
    "DayEntry",
    "DaysPlacesError",
    "DaysPlacesResult",
    "GpsCluster",
    "assign_days_places",
    "main",
    "place_clusters",
]

#: Mean earth radius in metres used for great-circle distances.
EARTH_RADIUS_M = 6371008.8

#: How close two unmatched GPS points must be to share a cluster.
CLUSTER_RADIUS_M = 500.0

#: Media kinds this step considers; memo and voiceover rows stay untouched.
_CONSIDERED_KINDS = ("clip", "still")


class DaysPlacesError(ValueError):
    """Raised for a bad place entry or an unknown trip time zone."""


@dataclass(frozen=True)
class DayEntry:
    """What the step did with one media row."""

    path: str | None
    asset_id: str | None
    status: str
    day: int | None
    place: str | None
    check: bool
    kept_locked: int = 0


@dataclass(frozen=True)
class DaysPlacesResult:
    """Every entry of one run, in catalog order."""

    entries: tuple[DayEntry, ...] = ()

    @property
    def assigned(self) -> int:
        """How many entries got a trip day."""
        return sum(1 for entry in self.entries if entry.status == "assigned")

    @property
    def before_start(self) -> int:
        """How many entries were captured before the trip started."""
        return sum(1 for entry in self.entries if entry.status == "before-start")

    @property
    def no_time(self) -> int:
        """How many entries had no capture time."""
        return sum(1 for entry in self.entries if entry.status == "no-time")

    @property
    def checked(self) -> int:
        """How many entries carry an unverified capture time."""
        return sum(1 for entry in self.entries if entry.check)


@dataclass(frozen=True)
class GpsCluster:
    """Unnamed GPS points on one trip day, near each other.

    Clusters carry no name; naming them is the memo step's job. They
    are recomputed from the catalog on demand, never stored.
    """

    day: int
    centre_lat: float
    centre_lon: float
    first_time: str
    last_time: str
    clip_count: int
    asset_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Place:
    """One validated trip place."""

    name: str
    lat: float
    lon: float
    radius_m: float


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two points."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    half = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, max(0.0, half))))


def _short_number(value: object, limit: int = 80) -> str:
    """Render a numeric setting without echoing thousands of digits.

    Huge integers cannot use ``repr`` under the default string-conversion
    limit; show a truncated prefix instead so error messages stay small.
    """
    try:
        text = repr(value)
    except ValueError:
        get_limit = getattr(sys, "get_int_max_str_digits", None)
        set_limit = getattr(sys, "set_int_max_str_digits", None)
        if get_limit is None or set_limit is None:
            return "<number too large to display>"
        old = get_limit()
        try:
            set_limit(0)
            try:
                text = repr(value)
            except Exception:
                return "<number too large to display>"
        finally:
            try:
                set_limit(old)
            except Exception:
                pass
    if len(text) > limit:
        return text[:limit] + f"...<truncated {len(text)} chars>"
    return text


#: Integers wider than this cannot fit in a float and are rejected before
#: any float conversion. 2**1024 exceeds the float maximum.
_MAX_INT_BITS = 1024


def _validate_places(raw_places: object) -> list[_Place]:
    """Check every place entry, returning validated places in order.

    Nothing is returned halfway: any bad entry raises
    :class:`DaysPlacesError` naming the entry's position and the key.
    """
    if not isinstance(raw_places, list):
        raise DaysPlacesError(
            "bad [pipeline].places: must be an array of "
            "[[pipeline.places]] tables"
        )
    total = len(raw_places)
    validated: list[_Place] = []
    for index, entry in enumerate(raw_places):
        label = f"place entry {index} (entry {index + 1} of {total})"
        if not isinstance(entry, dict):
            raise DaysPlacesError(
                f"bad places {label}: must be a table with "
                "name, lat, lon and radius_m"
            )
        if "name" not in entry:
            raise DaysPlacesError(f"bad places {label}: missing key 'name'")
        name = entry["name"]
        if not isinstance(name, str) or not name:
            raise DaysPlacesError(
                f"bad places {label}: key 'name' must be a non-empty string"
            )
        coords: dict[str, float] = {}
        for key, low, high in (("lat", -90.0, 90.0), ("lon", -180.0, 180.0)):
            if key not in entry:
                raise DaysPlacesError(
                    f"bad places {label}: missing key {key!r}"
                )
            value = entry[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DaysPlacesError(
                    f"bad places {label}: key {key!r} must be a number, "
                    f"got {_short_number(value)}"
                )
            if isinstance(value, int) and value.bit_length() > _MAX_INT_BITS:
                raise DaysPlacesError(
                    f"bad places {label}: key {key!r} must be between "
                    f"{low} and {high}, got {_short_number(value)}"
                )
            try:
                finite = math.isfinite(value)
            except OverflowError:
                raise DaysPlacesError(
                    f"bad places {label}: key {key!r} must be between "
                    f"{low} and {high}, got {_short_number(value)}"
                ) from None
            if not finite or not low <= value <= high:
                raise DaysPlacesError(
                    f"bad places {label}: key {key!r} must be between "
                    f"{low} and {high}, got {_short_number(value)}"
                )
            try:
                coords[key] = float(value)
            except OverflowError:
                raise DaysPlacesError(
                    f"bad places {label}: key {key!r} must be between "
                    f"{low} and {high}, got {_short_number(value)}"
                ) from None
        if "radius_m" not in entry:
            raise DaysPlacesError(
                f"bad places {label}: missing key 'radius_m'"
            )
        radius = entry["radius_m"]
        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            raise DaysPlacesError(
                f"bad places {label}: key 'radius_m' must be a number, "
                f"got {_short_number(radius)}"
            )
        if isinstance(radius, int) and radius.bit_length() > _MAX_INT_BITS:
            raise DaysPlacesError(
                f"bad places {label}: key 'radius_m' must be above 0, "
                f"got {_short_number(radius)}"
            )
        try:
            radius_finite = math.isfinite(radius)
        except OverflowError:
            raise DaysPlacesError(
                f"bad places {label}: key 'radius_m' must be above 0, "
                f"got {_short_number(radius)}"
            ) from None
        if not radius_finite or radius <= 0:
            raise DaysPlacesError(
                f"bad places {label}: key 'radius_m' must be above 0, "
                f"got {_short_number(radius)}"
            )
        try:
            radius_value = float(radius)
        except OverflowError:
            raise DaysPlacesError(
                f"bad places {label}: key 'radius_m' must be above 0, "
                f"got {_short_number(radius)}"
            ) from None
        validated.append(
            _Place(
                name=name,
                lat=coords["lat"],
                lon=coords["lon"],
                radius_m=radius_value,
            )
        )
    return validated


def _load_timezone(name: object) -> ZoneInfo:
    """Return the trip time zone, or raise naming ``[pipeline].timezone``."""
    try:
        return ZoneInfo(name)  # type: ignore[arg-type]
    except (ZoneInfoNotFoundError, ValueError, KeyError, OSError) as exc:
        raise DaysPlacesError(
            f"bad [pipeline].timezone {name!r}: unknown time zone"
        ) from exc


def _corrected_local(
    capture_time: str, offset_seconds: int, tz: ZoneInfo
) -> datetime:
    """Shift a catalog capture time by the clock offset into local time."""
    captured = datetime.strptime(capture_time, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    return (captured + timedelta(seconds=offset_seconds)).astimezone(tz)


def _match_place(
    places: list[_Place], lat: float | None, lon: float | None
) -> str | None:
    """Name the nearest place containing the GPS point, if any."""
    if lat is None or lon is None:
        return None
    best: _Place | None = None
    best_distance = 0.0
    for place in places:
        distance = _haversine_m(place.lat, place.lon, lat, lon)
        if distance <= place.radius_m and (
            best is None or distance < best_distance
        ):
            best = place
            best_distance = distance
    if best is not None:
        return best.name
    return None


def assign_days_places(catalog: Catalog, settings: object) -> DaysPlacesResult:
    """Set each clip's and still's day and place, and write its tags.

    Every tag is whole-clip (start and duration None, confidence None).
    A refused tag stays refused and counts as kept, never forced.
    """
    places = _validate_places(getattr(settings, "places"))
    tz = _load_timezone(getattr(settings, "timezone"))
    start_date = getattr(settings, "start_date")
    offsets = getattr(settings, "clock_offsets") or {}
    entries: list[DayEntry] = []
    for row in catalog.list_media():
        if row.kind not in _CONSIDERED_KINDS:
            continue
        if not row.capture_time:
            entries.append(
                DayEntry(
                    path=row.path,
                    asset_id=row.asset_id,
                    status="no-time",
                    day=None,
                    place=None,
                    check=False,
                    kept_locked=0,
                )
            )
            continue
        if row.camera:
            offset = offsets.get(row.camera, 0)
        else:
            offset = 0
        local = _corrected_local(row.capture_time, offset, tz)
        if (local.date() - start_date).days + 1 < 1:
            status: str = "before-start"
            day: int | None = None
        else:
            status = "assigned"
            day = (local.date() - start_date).days + 1
        place = _match_place(places, row.gps_lat, row.gps_lon)
        check = row.capture_time_source == CAPTURE_FILE
        media_id = catalog.put_media(replace(row, day=day, place=place))
        kept = 0
        if day is not None:
            kept += 0 if catalog.put_tag(
                TagRecord(
                    media_id=media_id,
                    folder="Day",
                    value=f"Day {day}",
                    start=None,
                    duration=None,
                    source="metadata",
                    confidence=None,
                    status="check" if check else "plain",
                )
            ) else 1
        if place is not None:
            kept += 0 if catalog.put_tag(
                TagRecord(
                    media_id=media_id,
                    folder="Place",
                    value=place,
                    start=None,
                    duration=None,
                    source="gps",
                    confidence=None,
                    status="plain",
                )
            ) else 1
        if row.camera:
            kept += 0 if catalog.put_tag(
                TagRecord(
                    media_id=media_id,
                    folder="Gear",
                    value=row.camera,
                    start=None,
                    duration=None,
                    source="metadata",
                    confidence=None,
                    status="plain",
                )
            ) else 1
        entries.append(
            DayEntry(
                path=row.path,
                asset_id=row.asset_id,
                status=status,
                day=day,
                place=place,
                check=check,
                kept_locked=kept,
            )
        )
    return DaysPlacesResult(entries=tuple(entries))


def place_clusters(catalog: Catalog, settings: object) -> list[GpsCluster]:
    """Group unmatched GPS points into unnamed clusters, one list per day.

    Only considered rows with a day, GPS and no place take part. Each
    row joins the first cluster of its day whose centre is within
    :data:`CLUSTER_RADIUS_M`; otherwise it starts a new cluster. Rows
    are sorted and reported by corrected local time (capture time plus
    the camera's clock offset, converted to the trip time zone).
    """
    tz = _load_timezone(getattr(settings, "timezone"))
    offsets = getattr(settings, "clock_offsets") or {}
    by_day: dict[int, list[tuple[datetime, MediaRecord]]] = {}
    for row in catalog.list_media():
        if row.kind not in _CONSIDERED_KINDS:
            continue
        if row.day is None or row.place is not None:
            continue
        if row.gps_lat is None or row.gps_lon is None:
            continue
        if not row.capture_time:
            continue
        if row.camera:
            offset = offsets.get(row.camera, 0)
        else:
            offset = 0
        local = _corrected_local(row.capture_time, offset, tz)
        by_day.setdefault(row.day, []).append((local, row))
    clusters: list[GpsCluster] = []
    for day in sorted(by_day):
        ordered = sorted(
            by_day[day],
            key=lambda pair: (pair[0], pair[1].id or 0),
        )
        members: list[list[tuple[datetime, MediaRecord]]] = []
        for pair in ordered:
            _, row = pair
            target: list[tuple[datetime, MediaRecord]] | None = None
            for group in members:
                count = len(group)
                centre_lat = sum(item.gps_lat or 0.0 for _, item in group) / count
                centre_lon = sum(item.gps_lon or 0.0 for _, item in group) / count
                if (
                    _haversine_m(
                        centre_lat,
                        centre_lon,
                        row.gps_lat or 0.0,
                        row.gps_lon or 0.0,
                    )
                    <= CLUSTER_RADIUS_M
                ):
                    target = group
                    break
            if target is None:
                members.append([pair])
            else:
                target.append(pair)
        for group in members:
            count = len(group)
            centre_lat = sum(item.gps_lat or 0.0 for _, item in group) / count
            centre_lon = sum(item.gps_lon or 0.0 for _, item in group) / count
            first = group[0][0].strftime("%Y-%m-%dT%H:%M:%S")
            last = group[-1][0].strftime("%Y-%m-%dT%H:%M:%S")
            clusters.append(
                GpsCluster(
                    day=day,
                    centre_lat=centre_lat,
                    centre_lon=centre_lon,
                    first_time=first,
                    last_time=last,
                    clip_count=count,
                    asset_ids=tuple(
                        item.asset_id or "" for _, item in group
                    ),
                )
            )
    clusters.sort(key=lambda cluster: (cluster.day, cluster.first_time))
    return clusters


def main(argv: list[str] | None = None) -> int:
    """Run the days-and-places step for one trip folder."""
    parser = argparse.ArgumentParser(
        prog="fcp_pipeline.days_places",
        description=(
            "Assign trip days, places and gear keywords in the catalog."
        ),
    )
    parser.add_argument(
        "--trip",
        required=True,
        help="Trip folder holding project.toml.",
    )
    args = parser.parse_args(argv)
    try:
        settings = load_trip(args.trip)
    except (TripSettingsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        catalog = Catalog.open(settings)
    except (CatalogError, SafeWriteError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        try:
            run_id = catalog.start_run("days-places", dry_run=False)
        except (CatalogError, SafeWriteError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            result = assign_days_places(catalog, settings)
            clusters = place_clusters(catalog, settings)
        except (DaysPlacesError, CatalogError, SafeWriteError, OSError) as exc:
            try:
                catalog.finish_run(run_id, f"failed: {exc}")
            except (CatalogError, SafeWriteError, OSError):
                pass
            print(f"error: {exc}", file=sys.stderr)
            return 1
        outcome = (
            f"assigned {result.assigned}, "
            f"before-start {result.before_start}, "
            f"no-time {result.no_time}, "
            f"check {result.checked}"
        )
        try:
            catalog.finish_run(run_id, outcome)
        except (CatalogError, SafeWriteError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    finally:
        catalog.close()
    for cluster in clusters:
        print(
            f"cluster day {cluster.day}: {cluster.clip_count} clips, "
            f"centre {cluster.centre_lat:.4f},{cluster.centre_lon:.4f}, "
            f"{cluster.first_time} to {cluster.last_time}"
        )
    print(outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
