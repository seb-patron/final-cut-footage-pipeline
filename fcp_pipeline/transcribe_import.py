"""Import mlx-whisper JSON results the editor already made.

:func:`import_results` reads one folder of ``.json`` result files, matches
each to a catalog clip by file name and duration (within one frame), and
stores every match exactly as if the transcribe step had just produced
it: the three files in ``transcripts/`` and the catalog rows from
:func:`Catalog.add_transcript_set`. Results that match nothing are
reported with a reason. Nothing is re-run: no speech model, no ffprobe,
no network, no footage read. Importing this module runs no tool, opens
no database and reads no file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from fcp_pipeline import transcribe
from fcp_pipeline.backends import whisper_local
from fcp_pipeline.catalog import Catalog, CatalogError
from fcp_pipeline.safe_write import SafeWriteError, SafeWriter

__all__ = [
    "ENGINE_NAME",
    "MAX_RESULT_BYTES",
    "STEP",
    "ImportEntry",
    "ImportResult",
    "ResultFormatError",
    "import_results",
    "parse_result",
]

#: Run row step recorded for an import run.
STEP = "transcribe-import"

#: Catalog engine recorded on imported transcript rows.
ENGINE_NAME = "mlx-whisper:imported"

#: A result file larger than this is reported without being read fully.
MAX_RESULT_BYTES = 20_000_000

#: Wrapped batch-result statuses that count as a finished success.
_SUCCESS_STATUSES = frozenset(
    {
        "success",
        "successful",
        "succeeded",
        "completed",
        "complete",
        "ok",
        "done",
    }
)


class ResultFormatError(ValueError):
    """Raised when a result is not a usable mlx-whisper result."""


@dataclass(frozen=True)
class ImportEntry:
    """What the import step did with one result file."""

    name: str
    status: str
    reason: str | None = None
    path: str | None = None
    files: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImportResult:
    """Every entry of one import run, in sorted result-name order."""

    entries: tuple[ImportEntry, ...] = ()

    @property
    def imported(self) -> int:
        """How many results were stored as new transcripts."""
        return sum(1 for entry in self.entries if entry.status == "imported")

    @property
    def skipped(self) -> int:
        """How many results were already recorded and left alone."""
        return sum(1 for entry in self.entries if entry.status == "skipped")

    @property
    def unmatched(self) -> int:
        """How many results matched no clip and were only listed."""
        return sum(1 for entry in self.entries if entry.status == "unmatched")

    @property
    def stale(self) -> int:
        """How many results named a clip that changed since the scan."""
        return sum(1 for entry in self.entries if entry.status == "stale")

    @property
    def failed(self) -> int:
        """How many results failed to store without stopping the run."""
        return sum(1 for entry in self.entries if entry.status == "failed")

    @property
    def would_import(self) -> int:
        """How many results a dry run would store."""
        return sum(
            1 for entry in self.entries if entry.status == "would-import"
        )

    @property
    def outcome(self) -> str:
        """One-line summary, such as ``imported 2, skipped 1, ...``."""
        text = (
            f"imported {self.imported}, skipped {self.skipped}, "
            f"unmatched {self.unmatched}, stale {self.stale}, "
            f"failed {self.failed}"
        )
        if self.would_import:
            text = f"would-import {self.would_import}, " + text
        return text


def _utcnow() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(value: object, *, name: str) -> Fraction:
    """Turn a JSON seconds value into an exact :class:`Fraction`.

    Accepts finite ints and floats (booleans are refused); NaN and
    Infinity are refused. Anything else raises :class:`ResultFormatError`
    naming ``name`` and the rule.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultFormatError(
            f"bad {name}: must be a finite number of seconds, "
            "never a bool, NaN or Infinity"
        )
    if not math.isfinite(float(value)):
        raise ResultFormatError(
            f"bad {name}: must be a finite number of seconds, "
            "never a bool, NaN or Infinity"
        )
    return whisper_local._to_fraction(value)


def _parse_duration(value: object, *, name: str) -> Fraction:
    """Turn a JSON duration value into an exact :class:`Fraction`.

    The value must be a finite number above zero; booleans, NaN,
    Infinity and non-positive numbers raise :class:`ResultFormatError`
    naming ``name`` and the rule.
    """
    duration = _parse_time(value, name=name)
    if duration <= 0:
        raise ResultFormatError(
            f"bad {name}: must be a finite number of seconds above 0"
        )
    return duration


def _parse_word(raw: object, *, segment_label: str, index: int) -> whisper_local.Word:
    """Turn one JSON word object into a :class:`Word`."""
    label = f"{segment_label}.words[{index}]"
    if not isinstance(raw, dict):
        raise ResultFormatError(
            f'bad {label}: must be an object with "word", "start" and "end"'
        )
    word_text = raw.get("word")
    if not isinstance(word_text, str):
        raise ResultFormatError(f"bad {label}.word: must be a string")
    start = _parse_time(raw.get("start"), name=f"{label}.start")
    stop = _parse_time(raw.get("end"), name=f"{label}.end")
    probability = raw.get("probability")
    if probability is not None:
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0 <= float(probability) <= 1
        ):
            raise ResultFormatError(
                f"bad {label}.probability: must be None or a number "
                "from 0 to 1"
            )
        probability = float(probability)
    return whisper_local.Word(
        start=start, stop=stop, word=word_text, probability=probability
    )


def _parse_segment(raw: object, *, index: int) -> whisper_local.Segment:
    """Turn one JSON segment object into a :class:`Segment`."""
    label = f"segments[{index}]"
    if not isinstance(raw, dict):
        raise ResultFormatError(
            f'bad {label}: must be an object with "start", "end" and "text"'
        )
    start = _parse_time(raw.get("start"), name=f"{label}.start")
    stop = _parse_time(raw.get("end"), name=f"{label}.end")
    if start < 0 or stop < start:
        raise ResultFormatError(
            f"bad {label}.start and {label}.end: must satisfy "
            "0 <= start <= end"
        )
    text = raw.get("text")
    if not isinstance(text, str):
        raise ResultFormatError(f"bad {label}.text: must be a string")
    words_raw = raw.get("words")
    if words_raw is None:
        words: tuple[whisper_local.Word, ...] = ()
    elif not isinstance(words_raw, list):
        raise ResultFormatError(
            f"bad {label}.words: must be a list of word objects"
        )
    else:
        words = tuple(
            _parse_word(item, segment_label=label, index=word_index)
            for word_index, item in enumerate(words_raw)
        )
    return whisper_local.Segment(
        start=start, stop=stop, text=text, words=words
    )


def parse_result(data: object) -> tuple[whisper_local.Transcription, Fraction | None]:
    """Turn one decoded JSON result into a transcription and a duration.

    Accepts the plain JSON mlx-whisper writes (a top-level object with
    ``segments`` and an optional top-level ``duration``) and the wrapped
    batch shape (extra ``asset``, ``status`` and ``result`` keys, with
    the duration in ``asset.duration_seconds`` or, failing that, the
    last segment's end). Other keys (``tokens``, ``temperature``,
    ``avg_logprob`` and the like) are ignored. Times become exact
    :class:`Fraction` values at millisecond resolution. Anything else
    raises :class:`ResultFormatError` naming the key and the rule, never
    the file's text. The duration is None when no source carries one.
    """
    if not isinstance(data, dict):
        raise ResultFormatError(
            'bad result: must be an object with a "segments" list'
        )
    nested = data.get("result")
    nested_map = nested if isinstance(nested, dict) else {}
    if "segments" in data:
        raw_segments = data["segments"]
    elif "segments" in nested_map:
        raw_segments = nested_map["segments"]
    else:
        raise ResultFormatError(
            'bad segments: must be a list of segment objects with '
            '"start", "end" and "text"'
        )
    if not isinstance(raw_segments, list):
        raise ResultFormatError(
            'bad segments: must be a list of segment objects with '
            '"start", "end" and "text"'
        )
    segments = tuple(
        _parse_segment(item, index=index)
        for index, item in enumerate(raw_segments)
    )
    language = data.get("language")
    if not isinstance(language, str):
        nested_language = nested_map.get("language")
        language = (
            nested_language if isinstance(nested_language, str) else None
        )
    duration: Fraction | None = None
    if "duration" in data and data["duration"] is not None:
        duration = _parse_duration(data["duration"], name="duration")
    else:
        asset = data.get("asset")
        if (
            isinstance(asset, dict)
            and "duration_seconds" in asset
            and asset["duration_seconds"] is not None
        ):
            duration = _parse_duration(
                asset["duration_seconds"], name="asset.duration_seconds"
            )
        elif (
            isinstance(asset, dict)
            or (
                isinstance(data.get("status"), str)
                and data["status"].casefold() in _SUCCESS_STATUSES
            )
        ) and segments:
            duration = segments[-1].stop
    return whisper_local.Transcription(
        language=language, segments=segments
    ), duration


def _result_status(data: dict) -> str | None:
    """The wrapped batch status when it is not a success value."""
    status = data.get("status")
    if not isinstance(status, str):
        return None
    if status.casefold() in _SUCCESS_STATUSES:
        return None
    return status


def import_results(
    catalog: Catalog,
    settings: object,
    folder: str | os.PathLike[str],
    *,
    dry_run: bool = False,
    created: str | None = None,
) -> ImportResult:
    """Import one folder of existing Whisper JSON results into the catalog.

    ``folder`` must hold the result files directly (``.json`` by name,
    case-insensitive, not recursive); each is matched to a catalog clip
    by file name and a duration within one frame, and every match is
    stored exactly as if the transcribe step had just produced it.
    With ``dry_run=True`` nothing is written, no catalog row is added
    and no run is recorded; each result that would be stored is
    reported as ``would-import``.
    """
    folder_text = os.fspath(folder)
    if not os.path.isdir(folder_text):
        raise ValueError(f"import folder not found: {folder_text}")
    names = sorted(
        name
        for name in os.listdir(folder_text)
        if name.lower().endswith(".json")
        and os.path.isfile(os.path.join(folder_text, name))
    )
    clips = [row for row in catalog.list_media() if row.kind == "clip"]
    model = getattr(settings, "transcribe_model", None) or (
        whisper_local.DEFAULT_MODEL
    )
    transcripts_dir = os.path.abspath(os.fspath(settings.transcripts))
    stamp = created or _utcnow()

    def unmatched(name: str, reason: str) -> ImportEntry:
        """An entry for a result that matched nothing, with its reason."""
        return ImportEntry(name=name, status="unmatched", reason=reason)

    def process(name: str, claimed: dict[int, str]) -> ImportEntry:
        """Match and store one result file, never raising for its data."""
        full = os.path.join(folder_text, name)
        try:
            size = os.path.getsize(full)
        except OSError as exc:
            return unmatched(name, f"cannot read result file: {exc}")
        if size > MAX_RESULT_BYTES:
            return unmatched(
                name,
                f"result file is larger than the {MAX_RESULT_BYTES}-byte "
                "size limit",
            )
        try:
            with open(full, "rb") as handle:
                blob = handle.read(MAX_RESULT_BYTES + 1)
        except OSError as exc:
            return unmatched(name, f"cannot read result file: {exc}")
        if len(blob) > MAX_RESULT_BYTES:
            return unmatched(
                name,
                f"result file is larger than the {MAX_RESULT_BYTES}-byte "
                "size limit",
            )
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            return unmatched(name, "result is not valid UTF-8 JSON")
        try:
            data = json.loads(text)
        except ValueError:
            return unmatched(name, "result is not valid JSON")
        if isinstance(data, dict):
            status = _result_status(data)
            if status is not None:
                return unmatched(
                    name, f"result status {status!r} is not a success"
                )
        try:
            transcription, duration = parse_result(data)
        except ResultFormatError as exc:
            return unmatched(name, str(exc))
        if duration is None:
            return unmatched(name, "no duration in the result")
        stem = name[:-5]
        folded = stem.casefold()
        candidates = [
            row
            for row in clips
            if row.path is not None
            and (
                Path(row.path).stem.casefold() == folded
                or Path(row.path).name.casefold() == folded
            )
        ]
        if not candidates:
            return unmatched(name, f"no clip named {stem} in the catalog")
        timed = [
            row
            for row in candidates
            if row.duration is not None and row.frame_rate
        ]
        if not timed:
            return unmatched(
                name, "clip has no duration or frame rate in the catalog"
            )
        matches = [
            row
            for row in timed
            if abs(duration - row.duration) <= 1 / row.frame_rate
        ]
        if len(matches) > 1:
            return unmatched(name, "more than one clip matches")
        if not matches:
            closest = min(
                timed, key=lambda row: abs(duration - row.duration)
            )
            return unmatched(
                name,
                f"duration {float(duration):.3f} s differs from the "
                f"clip's {float(closest.duration):.3f} s by more than "
                "one frame",
            )
        (match,) = matches
        if match.id in claimed:
            return unmatched(
                name, f"clip already matched by {claimed[match.id]}"
            )
        claimed[match.id] = name
        path = match.path or ""
        try:
            stale = not catalog.unchanged(path)
            if stale:
                info = None
                up_to_date = False
            else:
                info = os.stat(path)
                key = transcribe._clip_key(
                    os.path.abspath(path),
                    info.st_size,
                    info.st_mtime_ns,
                    model,
                )
                up_to_date = transcribe._is_up_to_date(
                    catalog,
                    match.id,
                    transcripts_dir,
                    transcribe._safe_stem(path),
                    key,
                )
        except (CatalogError, OSError, ArithmeticError) as exc:
            return ImportEntry(
                name=name,
                status="failed",
                reason=str(exc) or type(exc).__name__,
                path=match.path,
            )
        if stale:
            return ImportEntry(
                name=name,
                status="stale",
                reason="changed since the scan; run the scan again",
                path=match.path,
            )
        if up_to_date:
            return ImportEntry(
                name=name, status="skipped", path=match.path
            )
        if dry_run:
            return ImportEntry(
                name=name, status="would-import", path=match.path
            )
        try:
            blobs = transcribe._build_outputs("imported", transcription)
            _, paths = transcribe._free_base(
                transcripts_dir, transcribe._safe_stem(path), key
            )
            writer = SafeWriter(settings)
            written = [
                os.fspath(writer.write_new(target, blob))
                for target, blob in zip(paths, blobs)
            ]
            hashes = [
                hashlib.sha256(blob).hexdigest() for blob in blobs
            ]
            catalog.add_transcript_set(
                match.id,
                files=list(zip(written, hashes)),
                language=transcription.language,
                engine=ENGINE_NAME,
                created=stamp,
                segments=transcribe._build_records(transcription),
            )
        except (SafeWriteError, CatalogError, OSError) as exc:
            return ImportEntry(
                name=name,
                status="failed",
                reason=str(exc) or type(exc).__name__,
                path=match.path,
            )
        return ImportEntry(
            name=name,
            status="imported",
            path=match.path,
            files=tuple(written),
        )

    if dry_run:
        claimed: dict[int, str] = {}
        return ImportResult(
            entries=tuple(process(name, claimed) for name in names)
        )
    run_id = catalog.start_run(STEP, dry_run=False)
    result = ImportResult(())
    try:
        claimed: dict[int, str] = {}
        result = ImportResult(
            entries=tuple(process(name, claimed) for name in names)
        )
    finally:
        catalog.finish_run(run_id, result.outcome)
    return result
