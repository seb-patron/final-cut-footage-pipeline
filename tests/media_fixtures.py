"""Tiny deterministic synthetic media fixtures built with ffmpeg.

Generators (all cached, see below):

- :func:`make_clip`: a short QuickTime ``.mov`` (or ``.mp4``) clip with an
  exact frame rate, pixel size, whole-frame duration (see below),
  start timecode and a few QuickTime metadata keys. The requested
  seconds are converted to ``round(seconds x rate)`` frames and the
  clip is encoded by frame count, so the actual duration is exactly
  ``frames / rate`` (for example, 1 s at 30000/1001 is 30 frames =
  1001/1000 s); see :func:`clip_frames` and :func:`clip_duration`.
- :func:`make_still`: a single-frame JPEG of a given pixel size, optionally
  stamped with EXIF dates via exiftool.
- :func:`make_audio`: a 16-bit PCM WAV holding a 440 Hz tone or silence.
- :func:`make_card_tree`: a fake DJI-style plus iPhone-style camera card
  tree holding exactly six clips made with :func:`make_clip`.

Every generator raises :class:`unittest.SkipTest` naming ffmpeg when
``ffmpeg`` or ``ffprobe`` is not on ``PATH``, so tests that use these
makers are reported as skipped, not failed, on machines without ffmpeg.
:func:`make_still` raises :class:`unittest.SkipTest` naming exiftool when
an EXIF date is requested but ``exiftool`` is not on ``PATH``.

Determinism: every build uses generated lavfi sources, a single thread
and ffmpeg bitexact flags, and no wall-clock time enters the files, so
the same arguments give byte-identical files. Generated files live in one
folder created with :func:`tempfile.mkdtemp` on first use and removed at
interpreter exit; a repeated call with the same arguments returns the
same path without running ffmpeg again. Callers must treat returned
paths as read-only; :func:`make_card_tree` copies from the cache into
``dest`` so no two tests share writable card files. Nothing is ever
written into the repository, the current working folder, or anywhere
other than that cache folder and ``dest``.

Only the standard library is used. Never point these makers at real
footage: every make, model, date and coordinate here is invented.
"""

import atexit
import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

#: Reason fragment used when the ffmpeg tooling is missing.
FFMPEG_SKIP_REASON = "ffmpeg and ffprobe are required to build synthetic media"

#: Reason fragment used when exiftool is missing.
EXIFTOOL_SKIP_REASON = "exiftool is required to stamp EXIF dates"

# Invented card-tree metadata. The DJI-style file names carry UTC
# timestamps; the matching creation instants are read from those names.
DJI_MAKE = "DJI"
DJI_MODEL = "SyntheticDrone S1"
IPHONE_MAKE = "Apple"
IPHONE_MODEL = "iPhone Synthetic"
IPHONE_LOCATION = "+00.0000-030.0000/"

_UTC = timezone.utc

CARD_DJI_CLIPS = (
    ("DCIM/DJI_001/DJI_20200102100000_0001_D.MP4",
     datetime(2020, 1, 2, 10, 0, 0, tzinfo=_UTC)),
    ("DCIM/DJI_001/DJI_20200102101500_0002_D.MP4",
     datetime(2020, 1, 2, 10, 15, 0, tzinfo=_UTC)),
    ("DCIM/DJI_001/DJI_20200103090000_0003_D.MP4",
     datetime(2020, 1, 3, 9, 0, 0, tzinfo=_UTC)),
)

CARD_IPHONE_CLIPS = (
    ("DCIM/100APPLE/IMG_0001.MOV",
     datetime(2020, 1, 4, 10, 0, 0, tzinfo=_UTC)),
    ("DCIM/100APPLE/IMG_0002.MOV",
     datetime(2020, 1, 4, 10, 15, 0, tzinfo=_UTC)),
    ("DCIM/100APPLE/IMG_0003.MOV",
     datetime(2020, 1, 5, 9, 0, 0, tzinfo=_UTC)),
)

#: The six relative paths :func:`make_card_tree` creates, in order.
CARD_FILES = tuple(path for path, _ in CARD_DJI_CLIPS + CARD_IPHONE_CLIPS)

#: Maps each card relative path to its invented QuickTime creation instant.
CARD_CREATION_DATES = {path: when for path, when in CARD_DJI_CLIPS + CARD_IPHONE_CLIPS}

_CACHE_DIR = None


def _remove_cache_dir():
    global _CACHE_DIR
    directory, _CACHE_DIR = _CACHE_DIR, None
    if directory is not None:
        shutil.rmtree(directory, ignore_errors=True)


def _cache_dir():
    """Return the per-run cache folder, creating it on first use."""
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = Path(tempfile.mkdtemp(prefix="synthetic-media-fixtures-"))
        atexit.register(_remove_cache_dir)
    return _CACHE_DIR


def _require_ffmpeg():
    """Return (ffmpeg, ffprobe) paths or raise SkipTest naming ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise unittest.SkipTest(
            FFMPEG_SKIP_REASON + " but at least one was not found on PATH")
    return ffmpeg, ffprobe


def _require_exiftool():
    """Return the exiftool path or raise SkipTest naming exiftool."""
    exiftool = shutil.which("exiftool")
    if exiftool is None:
        raise unittest.SkipTest(
            EXIFTOOL_SKIP_REASON + " but it was not found on PATH")
    return exiftool


def _cache_key(parts):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode("utf-8") + b"\x00")
    return digest.hexdigest()


def _qt_stamp(moment):
    """Format an aware datetime as a QuickTime creation-date string."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_UTC)
    moment = moment.astimezone(_UTC)
    text = moment.strftime("%Y-%m-%dT%H:%M:%S")
    if moment.microsecond:
        text += ".%06d" % moment.microsecond
    return text + "Z"


def _check_size(size):
    width, height = size
    if (not isinstance(width, int) or not isinstance(height, int)
            or width <= 0 or height <= 0):
        raise ValueError("size must be two positive ints, got %r" % (size,))
    return width, height


def _parse_rate(rate):
    """Return ``rate`` as a :class:`~fractions.Fraction` of frames/second."""
    try:
        parsed = Fraction(str(rate).strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("rate must be 'NUM/DEN' or 'NUM', got %r"
                         % (rate,)) from exc
    if parsed <= 0:
        raise ValueError("rate must be positive, got %r" % (rate,))
    return parsed


def clip_frames(seconds, rate):
    """Return the whole-frame count for a ``seconds``-long clip at ``rate``.

    The requested seconds are converted to ``round(seconds x rate)``
    frames; the result is always at least one frame. Raises
    :class:`ValueError` for non-positive or non-numeric seconds and for
    rates :func:`_parse_rate` rejects.
    """
    try:
        requested = float(seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("seconds must be a positive number, got %r"
                         % (seconds,)) from exc
    if not requested > 0 or requested in (float("inf"), float("-inf")):
        raise ValueError("seconds must be a positive number, got %r"
                         % (seconds,))
    frames = int(round(requested * float(_parse_rate(rate))))
    if frames < 1:
        raise ValueError("seconds %r at rate %r is less than one frame"
                         % (seconds, rate))
    return frames


def clip_duration(seconds, rate):
    """Return the exact clip duration for ``seconds`` at ``rate``.

    The result is ``Fraction(frames, 1) / rate`` where ``frames`` comes
    from :func:`clip_frames`, so for example 1 s at 30000/1001 is
    ``Fraction(1001, 1000)`` seconds (30 frames).
    """
    return Fraction(clip_frames(seconds, rate), 1) / _parse_rate(rate)


def _build_clip(path, seconds, rate, size, start_timecode, creation_time,
                make, model, location, audio, container):
    """Run ffmpeg once to build a clip at ``path`` (no cache lookup).

    The clip holds exactly ``clip_frames(seconds, rate)`` video frames:
    encoding is limited with ``-frames:v`` (plus ``-shortest`` when an
    audio stream is present), never with ``-t``, so the duration is
    exactly ``clip_duration(seconds, rate)``.
    """
    ffmpeg, _ = _require_ffmpeg()
    width, height = size
    frames = clip_frames(seconds, rate)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i",
        "testsrc2=size=%dx%d:rate=%s" % (width, height, rate),
    ]
    if audio == "tone":
        command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    elif audio == "silent":
        command += ["-f", "lavfi", "-i",
                    "anullsrc=sample_rate=48000:channel_layout=mono"]
    command += ["-frames:v", str(frames)]
    if audio is not None:
        command += ["-shortest"]
    command += ["-map", "0:v"]
    if audio is not None:
        command += ["-map", "1:a"]
    command += ["-c:v", "libx264", "-preset", "ultrafast", "-threads", "1",
                "-pix_fmt", "yuv420p"]
    if audio is not None:
        command += ["-c:a", "aac"]
    command += ["-fflags", "+bitexact",
                "-flags:v", "+bitexact", "-flags:a", "+bitexact"]
    if start_timecode is not None:
        command += ["-timecode", start_timecode]
    if creation_time is not None:
        stamp = _qt_stamp(creation_time)
        command += ["-metadata", "creation_time=" + stamp,
                    "-metadata", "com.apple.quicktime.creationdate=" + stamp]
    if make is not None:
        command += ["-metadata", "com.apple.quicktime.make=" + make]
    if model is not None:
        command += ["-metadata", "com.apple.quicktime.model=" + model]
    if location is not None:
        command += ["-metadata",
                    "com.apple.quicktime.location.ISO6709=" + location]
    command += ["-movflags", "use_metadata_tags", str(path)]
    subprocess.run(command, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                   timeout=300)


def make_clip(seconds=1, rate="30000/1001", size=(64, 36),
              start_timecode=None, creation_time=None, make=None,
              model=None, location=None, audio="tone", container="mov"):
    """Build (or reuse) a tiny QuickTime clip and return its path.

    Whole-frame rule: the requested ``seconds`` are converted to
    ``round(seconds x rate)`` frames (see :func:`clip_frames`), the clip
    is encoded by frame count (``-frames:v``, plus ``-shortest`` when an
    audio stream is present; never ``-t``), and the actual duration is
    exactly ``frames / rate`` (see :func:`clip_duration`). For example,
    1 s at 30000/1001 is 30 frames = 1001/1000 s, and 1 s at 25/1 is
    25 frames = 1 s. The clip holds exactly ``rate`` (ffprobe video
    ``r_frame_rate``), ``size`` pixels and ``start_timecode``.
    ``creation_time`` is an aware datetime written as the QuickTime
    creation date; ``None`` writes no creation date at all. ``make``,
    ``model`` and ``location`` (an ISO 6709 string) are written as the
    matching ``com.apple.quicktime.*`` keys; ``None`` writes nothing.
    ``audio`` is ``"tone"`` (440 Hz sine), ``"silent"`` or ``None`` (no
    audio stream). ``container`` is ``"mov"`` or ``"mp4"``.

    The returned path's clip has the frame count and exact duration
    recorded by :func:`clip_frames` and :func:`clip_duration` for the
    same ``seconds`` and ``rate``. Requests that round to the same
    frame count share one cached file, since their encodings are
    identical.
    """
    _require_ffmpeg()
    if container not in ("mov", "mp4"):
        raise ValueError("container must be 'mov' or 'mp4', got %r"
                         % (container,))
    if audio not in ("tone", "silent", None):
        raise ValueError("audio must be 'tone', 'silent' or None, got %r"
                         % (audio,))
    if not isinstance(rate, str) or not rate:
        raise ValueError("rate must be a non-empty string, got %r" % (rate,))
    width, height = _check_size(tuple(size))
    frames = clip_frames(seconds, rate)
    key = _cache_key(("clip", frames, rate, width, height,
                      start_timecode,
                      None if creation_time is None
                      else _qt_stamp(creation_time),
                      make, model, location, audio, container))
    out = _cache_dir() / ("clip-%s.%s" % (key, container))
    if not out.exists():
        _build_clip(out, seconds=float(seconds), rate=rate,
                    size=(width, height), start_timecode=start_timecode,
                    creation_time=creation_time, make=make, model=model,
                    location=location, audio=audio, container=container)
    return out


def make_still(size=(64, 48), exif_datetime=None):
    """Build (or reuse) a JPEG still of ``size`` pixels; return its path.

    ``exif_datetime`` is a naive datetime written with exiftool as EXIF
    DateTimeOriginal and CreateDate.
    """
    _require_ffmpeg()
    if exif_datetime is not None:
        _require_exiftool()
    width, height = _check_size(tuple(size))
    stamp = None if exif_datetime is None else exif_datetime.strftime(
        "%Y:%m:%d %H:%M:%S")
    key = _cache_key(("still", width, height, stamp))
    out = _cache_dir() / ("still-%s.jpg" % key)
    if not out.exists():
        ffmpeg, _ = _require_ffmpeg()
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            "testsrc2=size=%dx%d:rate=1" % (width, height),
            "-frames:v", "1", "-c:v", "mjpeg", "-q:v", "2",
            "-threads", "1",
            "-fflags", "+bitexact", "-flags:v", "+bitexact",
            str(out),
        ]
        subprocess.run(command, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       timeout=300)
        if stamp is not None:
            exiftool = _require_exiftool()
            subprocess.run(
                [exiftool, "-overwrite_original",
                 "-DateTimeOriginal=" + stamp, "-CreateDate=" + stamp,
                 str(out)],
                check=True, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, timeout=300)
    return out


def make_audio(seconds=1, kind="tone", sample_rate=48000):
    """Build (or reuse) a 16-bit PCM WAV; return its path.

    ``kind`` is ``"tone"`` (440 Hz) or ``"silent"``.
    """
    _require_ffmpeg()
    if kind not in ("tone", "silent"):
        raise ValueError("kind must be 'tone' or 'silent', got %r" % (kind,))
    if (not isinstance(sample_rate, int) or sample_rate <= 0):
        raise ValueError("sample_rate must be a positive int, got %r"
                         % (sample_rate,))
    key = _cache_key(("audio", float(seconds), kind, sample_rate))
    out = _cache_dir() / ("audio-%s.wav" % key)
    if not out.exists():
        ffmpeg, _ = _require_ffmpeg()
        if kind == "tone":
            source = "sine=frequency=440:sample_rate=%d" % sample_rate
        else:
            source = ("anullsrc=sample_rate=%d:channel_layout=mono"
                      % sample_rate)
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", source,
            "-t", str(float(seconds)),
            "-c:a", "pcm_s16le",
            "-fflags", "+bitexact", "-flags:a", "+bitexact",
            str(out),
        ]
        subprocess.run(command, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       timeout=300)
    return out


def make_card_tree(dest):
    """Create ``dest`` with a fake DJI-style/iPhone-style card tree.

    Creates the folder ``dest`` (raising :class:`FileExistsError` if it
    already exists) and fills it with exactly the six clips in
    :data:`CARD_FILES`, copied from the clip cache so no two tests share
    writable card files. Each clip's QuickTime creation date is the
    invented 2020 instant in :data:`CARD_CREATION_DATES`, and each file's
    modification time is set to that same instant. Returns ``dest``.
    """
    _require_ffmpeg()
    dest = Path(dest)
    os.makedirs(dest, exist_ok=False)
    for relative in CARD_FILES:
        when = CARD_CREATION_DATES[relative]
        if "DJI_001" in relative:
            cached = make_clip(container="mp4", make=DJI_MAKE,
                               model=DJI_MODEL, creation_time=when)
        else:
            cached = make_clip(container="mov", make=IPHONE_MAKE,
                               model=IPHONE_MODEL, location=IPHONE_LOCATION,
                               creation_time=when)
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
        stamp = when.timestamp()
        os.utime(target, (stamp, stamp))
    return dest
