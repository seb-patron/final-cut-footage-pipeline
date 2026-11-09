"""Read-only media facts for one clip, via ffprobe and exiftool.

:func:`probe_file` runs the two tools and returns a :class:`ProbeFacts`
with exact :class:`~fractions.Fraction` times and rates, never floats.
Importing this module runs no tool and touches no file.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction

__all__ = [
    "PROBE_TOOLS",
    "ProbeError",
    "ProbeFacts",
    "parse_timecode",
    "probe_file",
]

#: External tools the probe needs, in the order they are checked.
PROBE_TOOLS = ("ffprobe", "exiftool")

#: How long one tool call may take before it is abandoned.
_PROBE_TIMEOUT = 120

#: ffprobe argument list (never shell), before the media path.
_FFPROBE_ARGS = ("-v", "error", "-show_format", "-show_streams", "-of", "json")

#: exiftool argument list (read-only: no write option), before the path.
_EXIFTOOL_ARGS = (
    "-j",
    "-n",
    "-api",
    "QuickTimeUTC=1",
    "-Make",
    "-Model",
    "-GPSLatitude",
    "-GPSLongitude",
    "-CreationDate",
    "-CreateDate",
)

_TIMECODE_RE = re.compile(r"(\d+):(\d\d):(\d\d)([:;])(\d\d)")

_EXIF_DATE_RE = re.compile(
    r"^(\d{4}):(\d{2}):(\d{2})[ T]"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d+))?"
    r"\s*([Zz]|[+-]\d{2}:?\d{2})?$"
)


class ProbeError(ValueError):
    """Raised when a clip cannot be probed exactly.

    The message names the missing tool or the unreadable path, never a
    guessed fact.
    """


@dataclass(frozen=True)
class ProbeFacts:
    """Exact facts read from one clip; unknown text fields are None."""

    duration: Fraction | None
    frame_rate: Fraction | None
    start_timecode: Fraction | None
    audio_streams: int
    width: int | None
    height: int | None
    make: str | None
    model: str | None
    gps_lat: float | None
    gps_lon: float | None
    quicktime_created: str | None


def parse_timecode(text: str, rate: Fraction) -> Fraction:
    """Return the seconds of ``HH:MM:SS:FF`` at the exact ``rate``.

    A ``;`` before the frame number means drop frame: at 30000/1001 two
    frames are dropped per minute (four at 60000/1001), except every
    tenth minute. Anything malformed raises :class:`ProbeError`.
    """
    if isinstance(rate, bool) or not isinstance(rate, (Fraction, int)):
        raise ProbeError(
            f"bad rate {rate!r}: must be a Fraction, never a float"
        )
    exact = rate if isinstance(rate, Fraction) else Fraction(rate, 1)
    if exact <= 0:
        raise ProbeError(f"bad rate {rate!r}: must be above zero")
    match = _TIMECODE_RE.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        raise ProbeError(
            f"bad timecode {text!r}: must look like 'HH:MM:SS:FF'"
        )
    hours, minutes, seconds, separator, frames_text = match.groups()
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)
    frames_number = int(frames_text)
    if minutes > 59 or seconds > 59:
        raise ProbeError(
            f"bad timecode {text!r}: minutes and seconds stay below 60"
        )
    nominal = int(round(exact))
    if nominal <= 0 or frames_number >= nominal:
        raise ProbeError(
            f"bad timecode {text!r}: frame number stays below {nominal}"
        )
    frames = (hours * 3600 + minutes * 60 + seconds) * nominal + frames_number
    if separator == ";" and exact in (Fraction(30000, 1001), Fraction(60000, 1001)):
        dropped = 2 if exact == Fraction(30000, 1001) else 4
        total_minutes = hours * 60 + minutes
        frames -= dropped * (total_minutes - total_minutes // 10)
    return Fraction(frames, 1) / exact


def _parse_rational(text: object, *, path: str, name: str) -> Fraction:
    """Parse an ffprobe rational or decimal string exactly."""
    value: Fraction | None = None
    try:
        if isinstance(text, bool):
            raise ValueError("bool is never a rate")
        if isinstance(text, int):
            value = Fraction(text, 1)
        elif isinstance(text, str):
            value = Fraction(text.strip())
        elif isinstance(text, float):
            value = Fraction(str(text))
        else:
            raise ValueError(f"not a number: {text!r}")
    except (ValueError, ZeroDivisionError) as exc:
        raise ProbeError(
            f"cannot read {path}: bad {name} {text!r}"
        ) from exc
    assert value is not None
    if value <= 0:
        raise ProbeError(f"cannot read {path}: bad {name} {text!r}")
    return value


def _parse_exif_date(text: object) -> str | None:
    """Normalize an exiftool date to ``YYYY-MM-DDTHH:MM:SSZ``, or None."""
    if not isinstance(text, str):
        return None
    match = _EXIF_DATE_RE.match(text.strip())
    if match is None:
        return None
    year, month, day, hour, minute, second, _frac, zone = match.groups()
    try:
        moment = datetime(
            int(year), int(month), int(day),
            int(hour), int(minute), int(second),
        )
    except ValueError:
        return None
    if zone not in (None, "Z", "z"):
        sign = 1 if zone[0] == "+" else -1
        digits = zone[1:].replace(":", "")
        try:
            shift = int(digits[:2]) * 60 + int(digits[2:4])
        except ValueError:
            return None
        moment = moment - timedelta(minutes=sign * shift)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_number(value: object) -> float | None:
    """Return a JSON value as a float, or None when it is not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _parse_text(value: object) -> str | None:
    """Return a JSON value as text, or None when missing or blank."""
    if isinstance(value, str) and value != "":
        return value
    return None


def _run_tool(
    exe: str, args: tuple[str, ...], path: str, *, runner: object
) -> tuple[int, str]:
    """Run one tool and return its exit code and standard output."""
    run = runner if runner is not None else subprocess.run
    try:
        completed = run(  # type: ignore[operator]
            [exe, *args, path],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProbeError(f"cannot read {path}: {exc}") from exc
    return completed.returncode, completed.stdout


def probe_file(path: str | os.PathLike[str], *, runner: object = None) -> ProbeFacts:
    """Probe one clip read-only and return its exact facts.

    Runs ``ffprobe`` for duration, frame rate, timecode, stream counts
    and pixel size, and ``exiftool`` for camera, GPS and capture date.
    A missing tool raises :class:`ProbeError` naming the tool; an
    unreadable file raises :class:`ProbeError` naming the path. Missing
    exiftool fields stay None.
    """
    display = os.fspath(path)
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise ProbeError(
            "missing required tool: ffprobe is not on PATH "
            "(install ffprobe to run the scan)"
        )
    exiftool = shutil.which("exiftool")
    if exiftool is None:
        raise ProbeError(
            "missing required tool: exiftool is not on PATH "
            "(install exiftool to run the scan)"
        )

    code, out = _run_tool(ffprobe, _FFPROBE_ARGS, display, runner=runner)
    if code != 0:
        raise ProbeError(f"cannot read {display}: ffprobe could not read it")
    try:
        data = json.loads(out)
    except (ValueError, TypeError) as exc:
        raise ProbeError(
            f"cannot read {display}: ffprobe output did not parse"
        ) from exc
    if not isinstance(data, dict):
        raise ProbeError(
            f"cannot read {display}: ffprobe output did not parse"
        )
    streams = data.get("streams")
    if not isinstance(streams, list):
        raise ProbeError(
            f"cannot read {display}: ffprobe output did not parse"
        )
    video = [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "video"
    ]
    if not video:
        raise ProbeError(f"cannot read {display}: no video stream found")
    first = video[0]
    frame_rate = _parse_rational(
        first.get("r_frame_rate"), path=display, name="frame rate"
    )
    width = first.get("width")
    width = width if isinstance(width, int) and width > 0 else None
    height = first.get("height")
    height = height if isinstance(height, int) and height > 0 else None
    audio_streams = sum(
        1
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "audio"
    )

    duration: Fraction | None = None
    container = data.get("format")
    container_duration = (
        container.get("duration") if isinstance(container, dict) else None
    )
    if container_duration is not None:
        exact_duration = _parse_rational(
            container_duration, path=display, name="duration"
        )
        duration = (
            Fraction(round(exact_duration * frame_rate), 1) / frame_rate
        )

    start_timecode: Fraction | None = None
    tags = first.get("tags")
    timecode_text = tags.get("timecode") if isinstance(tags, dict) else None
    if timecode_text is None and isinstance(container, dict):
        container_tags = container.get("tags")
        timecode_text = (
            container_tags.get("timecode")
            if isinstance(container_tags, dict)
            else None
        )
    if timecode_text is not None:
        if not isinstance(timecode_text, str):
            raise ProbeError(
                f"cannot read {display}: bad timecode {timecode_text!r}"
            )
        start_timecode = parse_timecode(timecode_text, frame_rate)

    code, out = _run_tool(exiftool, _EXIFTOOL_ARGS, display, runner=runner)
    entry: dict = {}
    if code == 0:
        try:
            items = json.loads(out)
        except (ValueError, TypeError):
            items = None
        if isinstance(items, list) and items and isinstance(items[0], dict):
            entry = items[0]

    quicktime_created = _parse_exif_date(entry.get("CreationDate"))
    if quicktime_created is None:
        quicktime_created = _parse_exif_date(entry.get("CreateDate"))

    return ProbeFacts(
        duration=duration,
        frame_rate=frame_rate,
        start_timecode=start_timecode,
        audio_streams=audio_streams,
        width=width,
        height=height,
        make=_parse_text(entry.get("Make")),
        model=_parse_text(entry.get("Model")),
        gps_lat=_parse_number(entry.get("GPSLatitude")),
        gps_lon=_parse_number(entry.get("GPSLongitude")),
        quicktime_created=quicktime_created,
    )
