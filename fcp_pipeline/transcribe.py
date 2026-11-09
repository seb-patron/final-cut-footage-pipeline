"""The transcription step: transcribe catalog clips locally on the Mac.

:func:`transcribe_catalog` visits every ``clip`` row in the catalog and,
for each one with an audio stream, runs a local speech-to-text backend
(:mod:`fcp_pipeline.backends.whisper_local`, an optional install) with
word timestamps and the trip's gear and place words as the first prompt.
It stores the segments and words in the catalog and writes one
``.srt``, one ``.txt`` and one word-level ``.json`` per clip into the
project's ``transcripts/`` folder through the safe write layer.

Clips with no audio and clips already transcribed in their current state
are skipped; a crash resumes clip by clip on the next run. No audio
leaves the Mac: this module imports no network code and the backend
itself opens no socket. Footage is only ever read. Importing this module
runs no tool and touches no file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fcp_pipeline.backends import whisper_local
from fcp_pipeline.catalog import (
    Catalog,
    CatalogError,
    TranscriptSegmentRecord,
    TranscriptWordRecord,
)
from fcp_pipeline.safe_write import SafeWriteError, SafeWriter
from fcp_pipeline.trip import TripSettingsError, load_trip

__all__ = [
    "TranscribeEntry",
    "TranscribeError",
    "TranscribeResult",
    "build_initial_prompt",
    "clip_has_audio",
    "main",
    "transcribe_catalog",
]

#: Longest initial prompt handed to the backend, in characters.
PROMPT_LIMIT = 600

#: How long one ffprobe audio check may take, in seconds.
_FFPROBE_TIMEOUT = 60


class TranscribeError(ValueError):
    """Raised when a clip's audio cannot be checked (ffprobe is missing)."""


def clip_has_audio(path: str | os.PathLike[str], *, runner=None) -> bool:
    """Whether ``path`` holds any audio stream, checked read-only.

    Runs ``ffprobe -v error -select_streams a -show_entries
    stream=index -of csv=p=0 <path>`` with list arguments (never a
    shell); ``runner`` defaults to :func:`subprocess.run` with captured
    output, text mode, a timeout and no standard input. True when the
    output holds any non-empty line. A missing ffprobe raises
    :class:`TranscribeError` naming ffprobe. The file is only read.
    """
    target = os.fspath(path)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        target,
    ]
    if shutil.which("ffprobe") is None:
        raise TranscribeError(
            "missing required tool: ffprobe is not on PATH "
            "(install ffprobe to check clip audio)"
        )
    run = subprocess.run if runner is None else runner
    try:
        completed = run(
            command,
            capture_output=True,
            text=True,
            timeout=_FFPROBE_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        raise TranscribeError(
            "missing required tool: ffprobe is not on PATH "
            "(install ffprobe to check clip audio)"
        ) from None
    except OSError as exc:
        raise TranscribeError(f"cannot run ffprobe for {target}: {exc}") from None
    output = completed.stdout
    if not isinstance(output, str):
        return False
    return any(line.strip() for line in output.splitlines())


def _render_prompt(names: list[tuple[str, str]]) -> str:
    """Render gear and place names as one prompt sentence."""
    gear = [name for kind, name in names if kind == "gear"]
    places = [name for kind, name in names if kind == "place"]
    parts = []
    if gear:
        parts.append("Gear: " + ", ".join(gear))
    if places:
        parts.append("Places: " + ", ".join(places))
    if not parts:
        return ""
    return ". ".join(parts) + "."


def build_initial_prompt(catalog: Catalog, settings: object) -> str:
    """Build the backend's first prompt from gear and place words.

    The camera names are the distinct non-empty ``media.camera`` values
    in the catalog (sorted), and the place names come from
    ``settings.places`` in order; each part with no names is left out.
    The text is cut at the last whole name keeping it at or under 600
    characters, so the gear and place words bias the recognizer toward
    the words the trip actually uses.
    """
    cameras = sorted(
        {row.camera for row in catalog.list_media() if row.camera}
    )
    places: list[str] = []
    for entry in getattr(settings, "places", []) or []:
        if isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str) and name:
                places.append(name)
    names: list[tuple[str, str]] = [("gear", name) for name in cameras]
    names += [("place", name) for name in places]
    while names:
        text = _render_prompt(names)
        if len(text) <= PROMPT_LIMIT:
            return text
        names.pop()
    return ""


def _safe_stem(path_text: str) -> str:
    """The clip file's stem with unsafe characters replaced by ``_``."""
    stem = Path(path_text).stem
    return "".join(
        char if (char.isalnum() or char in ".-_") else "_" for char in stem
    )


def _clip_key(abs_path: str, size: int, mtime_ns: int, model: str) -> str:
    """Short state code for a clip, so a changed clip or model renames output."""
    digest = hashlib.sha256(
        f"{abs_path}\0{size}\0{mtime_ns}\0{model}".encode("utf-8")
    ).hexdigest()
    return digest[:8]


def _free_base(transcripts_dir: str, stem: str, key: str) -> tuple[str, list[str]]:
    """Pick a base name none of whose three files exists yet.

    After a crash between writing and recording, leftover files may sit
    next to no catalog record; the next run then uses ``<base>-2``,
    ``-3`` and so on, never touching the leftovers.
    """
    number = 0
    while True:
        suffix = "" if number == 0 else f"-{number}"
        base = f"{stem}-{key}{suffix}"
        paths = [
            os.path.join(transcripts_dir, base + extension)
            for extension in (".json", ".srt", ".txt")
        ]
        if not any(os.path.lexists(candidate) for candidate in paths):
            return base, paths
        number = 2 if number == 0 else number + 1


def _is_up_to_date(
    catalog: Catalog, media_id: int, transcripts_dir: str, stem: str, key: str
) -> bool:
    """Whether a catalog row already records this clip's current state.

    Accepts the expected ``stem-key.json`` path and any suffixed
    ``stem-key-2.json``-style path from a run that followed a
    crash-leftover, as long as the recorded file still exists.
    """
    expected = os.path.join(transcripts_dir, f"{stem}-{key}.json")
    suffixed = re.compile(re.escape(f"{stem}-{key}") + r"(-\d+)?\.json\Z")
    for record in catalog.list_transcripts(media_id):
        if record.path == expected and os.path.exists(record.path):
            return True
        if (
            os.path.dirname(record.path) == transcripts_dir
            and suffixed.fullmatch(os.path.basename(record.path)) is not None
            and os.path.exists(record.path)
        ):
            return True
    return False


def _format_seconds(value) -> float:
    """Render an exact fraction of seconds as a JSON number at millisecond resolution."""
    return float(value)


def _srt_timestamp(value) -> str:
    """Render an exact fraction of seconds as ``HH:MM:SS,mmm`` (hours uncapped)."""
    total_ms = int(value * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _utcnow() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class TranscribeEntry:
    """What the step did with one clip file."""

    path: str
    status: str
    reason: str | None = None
    files: tuple[str, ...] = ()


@dataclass(frozen=True)
class TranscribeResult:
    """Every entry of one transcription run, in catalog id order."""

    entries: tuple[TranscribeEntry, ...] = ()

    @property
    def transcribed(self) -> int:
        """How many clips were transcribed by this run."""
        return sum(1 for entry in self.entries if entry.status == "transcribed")

    @property
    def skipped(self) -> int:
        """How many clips were already up to date and left alone."""
        return sum(1 for entry in self.entries if entry.status == "skipped")

    @property
    def no_audio(self) -> int:
        """How many clips hold no audio stream."""
        return sum(1 for entry in self.entries if entry.status == "no-audio")

    @property
    def stale(self) -> int:
        """How many clips changed since the scan and were left alone."""
        return sum(1 for entry in self.entries if entry.status == "stale")

    @property
    def failed(self) -> int:
        """How many clips failed without stopping the run."""
        return sum(1 for entry in self.entries if entry.status == "failed")

    @property
    def would_transcribe(self) -> int:
        """How many clips a dry run would transcribe."""
        return sum(
            1 for entry in self.entries if entry.status == "would-transcribe"
        )

    @property
    def outcome(self) -> str:
        """One-line summary, such as ``transcribed 2, skipped 1, ...``."""
        text = (
            f"transcribed {self.transcribed}, skipped {self.skipped}, "
            f"no-audio {self.no_audio}, stale {self.stale}, "
            f"failed {self.failed}"
        )
        if self.would_transcribe:
            text = f"would-transcribe {self.would_transcribe}, " + text
        return text


def _build_outputs(model: str, transcription) -> tuple[bytes, bytes, bytes]:
    """Render the ``.json``, ``.srt`` and ``.txt`` bytes for one answer."""
    segments = []
    for segment in transcription.segments:
        segments.append(
            {
                "start": _format_seconds(segment.start),
                "end": _format_seconds(segment.stop),
                "text": segment.text,
                "words": [
                    {
                        "word": word.word,
                        "start": _format_seconds(word.start),
                        "end": _format_seconds(word.stop),
                        "probability": word.probability,
                    }
                    for word in segment.words
                ],
            }
        )
    payload = {
        "version": 1,
        "engine": whisper_local.ENGINE,
        "model": model,
        "language": transcription.language,
        "segments": segments,
    }
    json_blob = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    cues = []
    for index, segment in enumerate(transcription.segments, start=1):
        cues.append(
            f"{index}\n{_srt_timestamp(segment.start)} --> "
            f"{_srt_timestamp(segment.stop)}\n{segment.text.strip()}"
        )
    srt_blob = (
        ("\n\n".join(cues) + "\n").encode("utf-8") if cues else b""
    )
    texts = [segment.text.strip() for segment in transcription.segments]
    txt_blob = ("\n".join(texts) + "\n").encode("utf-8") if texts else b""
    return json_blob, srt_blob, txt_blob


def _build_records(transcription) -> list:
    """Convert a backend answer into catalog segment records."""
    records = []
    for index, segment in enumerate(transcription.segments):
        records.append(
            TranscriptSegmentRecord(
                idx=index,
                start=segment.start,
                stop=segment.stop,
                text=segment.text,
                words=[
                    TranscriptWordRecord(
                        idx=word_index,
                        start=word.start,
                        stop=word.stop,
                        word=word.word,
                        probability=word.probability,
                    )
                    for word_index, word in enumerate(segment.words)
                ],
            )
        )
    return records


def transcribe_catalog(
    catalog: Catalog,
    settings: object,
    *,
    backend=None,
    audio_probe=None,
    dry_run: bool = False,
    created: str | None = None,
) -> TranscribeResult:
    """Transcribe every catalog clip with audio into transcripts and the catalog.

    ``backend`` is a callable ``backend(path, *, model,
    initial_prompt)`` (defaulting to the local mlx-whisper backend) and
    ``audio_probe`` is a callable ``audio_probe(path)`` (defaulting to
    :func:`clip_has_audio`); ``created`` is the UTC text recorded on the
    new catalog rows. Each clip is visited in catalog id order: a clip
    that changed since the scan is ``stale``, one with no audio is
    ``no-audio``, one already recorded in its current state is
    ``skipped``, and the rest are transcribed. A failed backend call or
    a refused write marks that clip ``failed`` and the run goes on; a
    crash (:class:`BaseException`) is never swallowed. With
    ``dry_run=True`` nothing is imported or written, the backend is
    never called, and each clip that would be transcribed is reported
    as ``would-transcribe``. The run itself is recorded with
    ``start_run("transcribe")`` and ``finish_run`` except in dry runs,
    which add no catalog rows at all.
    """
    model = getattr(settings, "transcribe_model", None) or (
        whisper_local.DEFAULT_MODEL
    )
    if created is None:
        created = _utcnow()
    rows = [row for row in catalog.list_media() if row.kind == "clip"]
    probe = audio_probe if audio_probe is not None else clip_has_audio
    transcripts_dir = os.path.abspath(os.fspath(settings.transcripts))

    def check(row) -> tuple[TranscribeEntry | None, str | None]:
        """Terminal entry, or None with the clip's state key."""
        path = row.path or ""
        if not catalog.unchanged(path):
            return (
                TranscribeEntry(
                    path=path,
                    status="stale",
                    reason="changed since the scan; run the scan again",
                ),
                None,
            )
        if not probe(path):
            return TranscribeEntry(path=path, status="no-audio"), None
        try:
            info = os.stat(path)
        except OSError:
            return (
                TranscribeEntry(
                    path=path,
                    status="stale",
                    reason="changed since the scan; run the scan again",
                ),
                None,
            )
        key = _clip_key(
            os.path.abspath(path), info.st_size, info.st_mtime_ns, model
        )
        if _is_up_to_date(catalog, row.id, transcripts_dir, _safe_stem(path), key):
            return TranscribeEntry(path=path, status="skipped"), None
        if dry_run:
            return TranscribeEntry(path=path, status="would-transcribe"), None
        return None, key

    if dry_run:
        dry_entries: list[TranscribeEntry] = []
        for row in rows:
            dry_entry, _ = check(row)
            assert dry_entry is not None
            dry_entries.append(dry_entry)
        return TranscribeResult(entries=tuple(dry_entries))
    run_id = catalog.start_run("transcribe", dry_run=False)
    try:
        prompt: str | None = None
        entries: list[TranscribeEntry] = []
        active = backend if backend is not None else whisper_local.transcribe_file
        for row in rows:
            entry, pending = check(row)
            if entry is not None:
                entries.append(entry)
                continue
            assert pending is not None
            key = pending
            path = row.path or ""
            try:
                if prompt is None:
                    prompt = build_initial_prompt(catalog, settings)
                transcription = active(
                    path, model=model, initial_prompt=prompt
                )
                blobs = _build_outputs(model, transcription)
                _, paths = _free_base(transcripts_dir, _safe_stem(path), key)
                writer = SafeWriter(settings)
                written: list[str] = []
                for target, blob in zip(paths, blobs):
                    written.append(os.fspath(writer.write_new(target, blob)))
                hashes = [
                    hashlib.sha256(blob).hexdigest() for blob in blobs
                ]
                catalog.add_transcript_set(
                    row.id,
                    files=list(zip(written, hashes)),
                    language=transcription.language,
                    engine=f"{whisper_local.ENGINE}:{model}",
                    created=created,
                    segments=_build_records(transcription),
                )
                entries.append(
                    TranscribeEntry(
                        path=path, status="transcribed", files=tuple(written)
                    )
                )
            except Exception as exc:
                entries.append(
                    TranscribeEntry(
                        path=path,
                        status="failed",
                        reason=str(exc) or type(exc).__name__,
                    )
                )
    except TranscribeError as exc:
        catalog.finish_run(run_id, f"failed: {exc}")
        raise
    result = TranscribeResult(entries=tuple(entries))
    catalog.finish_run(run_id, result.outcome)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the transcription step for one trip folder."""
    parser = argparse.ArgumentParser(
        prog="fcp_pipeline.transcribe",
        description="Transcribe catalog clips locally into transcripts/.",
    )
    parser.add_argument(
        "--trip",
        required=True,
        help="Trip folder holding project.toml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be transcribed without writing anything.",
    )
    args = parser.parse_args(argv)
    try:
        settings = load_trip(args.trip)
    except TripSettingsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not args.dry_run and not whisper_local.available():
        print(f"error: {whisper_local.INSTALL_HINT}", file=sys.stderr)
        return 2
    try:
        catalog = Catalog.open(settings)
    except (CatalogError, SafeWriteError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        try:
            result = transcribe_catalog(
                catalog, settings, dry_run=args.dry_run
            )
        except TranscribeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
        except (CatalogError, SafeWriteError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    finally:
        catalog.close()
    for entry in result.entries:
        if entry.status in ("failed", "stale") and entry.reason is not None:
            print(f"{entry.status}: {entry.path}: {entry.reason}")
    print(result.outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
