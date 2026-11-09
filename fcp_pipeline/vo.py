"""The voiceover transcription step: transcribe the voiceover recording.

:func:`transcribe_voiceover` visits every voiceover recording that
``project.toml`` points to (``[media].voiceover``, a path or an array of
paths; the audio stays where it is and is never copied, moved or
changed) and saves one word-timed
``transcripts/<name>-<code>.json`` per recording through the safe
write layer, plus a catalog row of kind ``"voiceover"`` with its
transcript, segments and words. The default provider is local
mlx-whisper on this Mac and nothing is uploaded. The hosted audio
transcription (the editor's own API credits) runs only when
``[pipeline.ai.vo_audio]`` asks for it, through the network gate,
asking for word times. Word times are required: a hosted answer
without them falls back to a local run, and a local answer without
them fails that recording; no time is ever invented. A recording
already done (same SHA-256) is skipped without any call; when the
transcript file is there but the catalog has no row for it, the row is
repaired from that validated file instead and reported as recorded.
Nothing calls
this step yet except its own command line; importing this module runs
no tool, opens no database and reads no file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from fcp_pipeline.ai import AiConfigError, audio_openai, step_config
from fcp_pipeline.backends import whisper_local
from fcp_pipeline.catalog import (
    Catalog,
    CatalogError,
    MediaRecord,
    TranscriptSegmentRecord,
    TranscriptWordRecord,
)
from fcp_pipeline.memo import (
    MAX_JSON_BYTES,
    MAX_UPLOAD_BYTES,
    _safe_name,
    _sha256_file,
)
from fcp_pipeline.net import NetworkGate, OfflineError, set_offline
from fcp_pipeline.safe_write import SafeWriteError, SafeWriter
from fcp_pipeline.transcribe import _build_records, build_initial_prompt
from fcp_pipeline.trip import TripSettingsError, load_trip

__all__ = [
    "VoEntry",
    "VoError",
    "VoResult",
    "main",
    "recordings",
    "transcribe_voiceover",
]


class VoError(ValueError):
    """Raised when a voiceover recording cannot be read.

    The message names the path or key and the rule it broke, never
    recording contents.
    """


@dataclass(frozen=True)
class VoEntry:
    """What the voiceover step did with one recording file."""

    name: str
    status: str
    reason: str | None = None
    engine: str | None = None
    file: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class VoResult:
    """Every entry of one voiceover run, in recording order."""

    entries: tuple[VoEntry, ...] = ()

    @property
    def transcribed(self) -> int:
        """How many recordings were transcribed by this run."""
        return sum(1 for entry in self.entries if entry.status == "transcribed")

    @property
    def skipped(self) -> int:
        """How many recordings were already done and left alone."""
        return sum(1 for entry in self.entries if entry.status == "skipped")

    @property
    def recorded(self) -> int:
        """How many recordings regained their catalog row from file."""
        return sum(1 for entry in self.entries if entry.status == "recorded")

    @property
    def failed(self) -> int:
        """How many recordings failed without stopping the run."""
        return sum(1 for entry in self.entries if entry.status == "failed")

    @property
    def would_transcribe(self) -> int:
        """How many recordings a dry run would transcribe."""
        return sum(
            1 for entry in self.entries if entry.status == "would-transcribe"
        )

    @property
    def outcome(self) -> str:
        """One-line summary, such as ``transcribed 1, skipped 1, ...``."""
        text = (
            f"transcribed {self.transcribed}, skipped {self.skipped}, "
            f"failed {self.failed}"
        )
        if self.recorded:
            text += f", recorded {self.recorded}"
        if self.would_transcribe:
            text = f"would-transcribe {self.would_transcribe}, " + text
        return text


def recordings(settings) -> list[Path]:
    """Return the voiceover recording paths from ``[media].voiceover``.

    Reads ``settings.project_toml`` again and returns the
    ``[media].voiceover`` value as paths: a string or an array of
    strings, each non-empty; a relative path is resolved against
    ``settings.folder``. An absent key gives []. Any other type raises
    :class:`VoError` naming ``[media].voiceover``.
    """
    with open(os.fspath(settings.project_toml), "rb") as handle:
        data = tomllib.load(handle)
    media = data.get("media", {})
    if not isinstance(media, dict):
        return []
    value = media.get("voiceover", None)
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise VoError(
            "project.toml key '[media].voiceover' must be a path string or "
            "an array of path strings"
        )
    base = Path(os.fspath(settings.folder))
    paths: list[Path] = []
    for item in values:
        if not isinstance(item, str) or not item:
            raise VoError(
                "project.toml key '[media].voiceover' must be a path string "
                "or an array of path strings, each non-empty"
            )
        candidate = Path(item)
        if candidate.is_absolute():
            paths.append(candidate)
        else:
            paths.append(base / item)
    return paths


def _find_done_file(settings, digest: str):
    """Return the done voiceover file holding this SHA-256, or None.

    The result is ``(path, blob, document)``: the file's path string,
    its raw bytes, and its parsed JSON document. Only a file directly
    in ``transcripts/`` whose role is ``"voiceover"`` and whose
    recording SHA-256 matches counts; anything else is ignored.
    """
    folder = Path(os.fspath(settings.transcripts))
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    for name in sorted(names):
        if not name.endswith(".json"):
            continue
        candidate = folder / name
        try:
            if os.path.islink(candidate):
                continue
            if not os.path.isfile(candidate):
                continue
            if os.path.getsize(candidate) > MAX_JSON_BYTES:
                continue
            with open(candidate, "rb") as handle:
                blob = handle.read()
            document = json.loads(blob.decode("utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(document, dict):
            continue
        if document.get("role") != "voiceover":
            continue
        recording = document.get("recording")
        if not isinstance(recording, dict):
            continue
        sha = recording.get("sha256")
        if isinstance(sha, str) and sha == digest:
            return (os.fspath(candidate), bytes(blob), document)
    return None


def _already_done(settings, digest: str) -> bool:
    """Whether a voiceover file directly in ``transcripts/`` holds this SHA-256."""
    return _find_done_file(settings, digest) is not None


def _checked_file_time(value: object) -> Fraction | None:
    """Millisecond Fraction for a transcript-file time, or None if bad.

    The absolute bound is checked before any float conversion or
    ``math.isfinite`` call, so a huge int never reaches ``float()``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if abs(value) > audio_openai.MAX_SECONDS:
        return None
    if not math.isfinite(value):
        return None
    if value < 0:
        return None
    return Fraction(int(round(float(value) * 1000)), 1000)


def _repair_records(document: object):
    """Build catalog segment records from a done-file document.

    Returns the records, or None when the document is not a valid
    voiceover transcript (wrong shape, bad times, or no words).
    """
    if not isinstance(document, dict):
        return None
    segments_raw = document.get("segments")
    if not isinstance(segments_raw, list) or not segments_raw:
        return None
    if not isinstance(document.get("engine"), str) or not document.get(
        "engine"
    ):
        return None
    if not isinstance(document.get("model"), str) or not document.get("model"):
        return None
    language = document.get("language")
    if language is not None and not isinstance(language, str):
        return None
    records = []
    total = 0
    for index, item in enumerate(segments_raw):
        if not isinstance(item, dict):
            return None
        start = _checked_file_time(item.get("start"))
        stop = _checked_file_time(item.get("end"))
        if start is None or stop is None or not start <= stop:
            return None
        text = item.get("text")
        if not isinstance(text, str):
            return None
        words_raw = item.get("words")
        if not isinstance(words_raw, list):
            return None
        words = []
        for word_index, entry in enumerate(words_raw):
            if not isinstance(entry, dict):
                return None
            word = entry.get("word")
            if not isinstance(word, str) or not word.strip():
                return None
            word_start = _checked_file_time(entry.get("start"))
            word_stop = _checked_file_time(entry.get("end"))
            if word_start is None or word_stop is None:
                return None
            if not word_start <= word_stop:
                return None
            probability = entry.get("probability")
            if probability is not None:
                if isinstance(probability, bool) or not isinstance(
                    probability, (int, float)
                ):
                    return None
                if not 0 <= probability <= 1:
                    return None
                probability = float(probability)
                if not math.isfinite(probability):
                    return None
            words.append(
                TranscriptWordRecord(
                    idx=word_index,
                    start=word_start,
                    stop=word_stop,
                    word=word.strip(),
                    probability=probability,
                )
            )
        total += len(words)
        records.append(
            TranscriptSegmentRecord(
                idx=index, start=start, stop=stop, text=text, words=words
            )
        )
    if total < 1:
        return None
    return records


def _repair_missing_row(catalog: Catalog, settings, text, digest, found,
                        created: str | None = None):
    """Record the catalog row for an existing done-file, or skip.

    When the catalog already holds a row for the file, returns a
    ``"skipped"`` entry. Otherwise validates the file and inserts its
    catalog row without touching the file, returning a ``"recorded"``
    entry; an invalid file or a refused insert returns ``"failed"``.
    """
    name = Path(text).name
    path_text, blob, document = found
    try:
        media = catalog.get_media(os.path.abspath(text))
    except CatalogError:
        media = None
    file_sha = hashlib.sha256(bytes(blob)).hexdigest()
    if media is not None and getattr(media, "kind", None) == "voiceover":
        try:
            transcripts = catalog.list_transcripts(media.id)
        except (CatalogError, OSError, ValueError):
            transcripts = []
        for transcript in transcripts:
            if transcript.path == path_text or transcript.sha256 == file_sha:
                return VoEntry(name=name, status="skipped")
    records = _repair_records(document)
    if records is None:
        return VoEntry(
            name=name,
            status="failed",
            reason="the existing transcript file is invalid",
        )
    try:
        info = os.stat(text)
    except OSError as exc:
        return VoEntry(name=name, status="failed", reason=str(exc))
    language = document.get("language")
    engine = document.get("engine")
    model = document.get("model")
    asset = "vo-" + hashlib.sha256(
        os.fspath(Path(text).resolve()).encode("utf-8")
    ).hexdigest()[:16]
    try:
        media_id = catalog.put_media(
            MediaRecord(
                asset_id=asset,
                kind="voiceover",
                path=os.path.abspath(text),
                size=info.st_size,
                mtime_ns=info.st_mtime_ns,
                inode=info.st_ino,
            )
        )
        catalog.add_transcript_set(
            media_id,
            files=[(path_text, file_sha)],
            language=language,
            engine=f"{engine}:{model}",
            created=created or _utcnow(),
            segments=records,
        )
    except (CatalogError, SafeWriteError, OSError) as exc:
        return VoEntry(
            name=name,
            status="failed",
            reason=str(exc) or type(exc).__name__,
        )
    return VoEntry(
        name=name, status="recorded", engine=engine, file=path_text
    )


def _inside_output(settings, path_text: str) -> bool:
    """Whether the recording sits inside a pipeline-written folder.

    Both the link's own location (its parent resolved, its name kept)
    and its fully resolved target are compared against the resolved
    ``transcripts/`` and ``.pipeline/`` folders, so a symlink inside
    ``transcripts/`` that points outside and one outside that points
    into ``transcripts/`` both count, whatever ``TMPDIR`` resolves to.
    """
    roots: list[Path] = []
    for folder in (
        getattr(settings, "transcripts", None),
        getattr(settings, "pipeline", None),
    ):
        if folder is None:
            continue
        try:
            roots.append(Path(os.fspath(folder)).resolve())
        except OSError:
            continue
    if not roots:
        return False
    given = Path(os.path.abspath(path_text))
    candidates: list[Path] = [given]
    try:
        candidates.append(given.parent.resolve() / given.name)
    except OSError:
        pass
    try:
        candidates.append(Path(path_text).resolve())
    except OSError:
        pass
    for candidate in candidates:
        for root in roots:
            if candidate == root or root in candidate.parents:
                return True
    return False


def _seconds(value) -> float:
    """Render an exact time as JSON seconds, rounded to 3 decimals."""
    return round(float(value), 3)


def _utcnow() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _has_word_times(transcription) -> bool:
    """Whether every word has finite times with 0 <= start <= end.

    The transcript must also hold at least one word. Segments with an
    empty words list are kept as they are; no time is created for them.
    """
    total = 0
    for segment in transcription.segments:
        for word in segment.words:
            total += 1
            try:
                finite = math.isfinite(float(word.start)) and math.isfinite(
                    float(word.stop)
                )
                ordered = 0 <= word.start <= word.stop
            except (ArithmeticError, TypeError, ValueError):
                return False
            if not finite or not ordered:
                return False
    return total >= 1


def _hosted_to_transcription(reply):
    """Place hosted words into segments as a local-style transcription.

    Each word goes into the last segment whose start is at or before
    the word's start (the first segment when none), in order, as a
    :class:`whisper_local.Word` with probability None.
    """
    buckets: list[list] = [[] for _ in reply.segments]
    for word in reply.words:
        pick = 0
        for index, segment in enumerate(reply.segments):
            if segment.start <= word.start:
                pick = index
        buckets[pick].append(
            whisper_local.Word(
                start=word.start,
                stop=word.end,
                word=word.word,
                probability=None,
            )
        )
    return whisper_local.Transcription(
        language=reply.language,
        segments=tuple(
            whisper_local.Segment(
                start=segment.start,
                stop=segment.end,
                text=segment.text,
                words=tuple(buckets[index]),
            )
            for index, segment in enumerate(reply.segments)
        ),
    )


def _failed_local(name: str, exc: Exception, notes: list[str]):
    """Build the failed entry for a local backend error."""
    if isinstance(exc, whisper_local.BackendMissing):
        reason = whisper_local.INSTALL_HINT
    else:
        reason = str(exc) or type(exc).__name__
    return VoEntry(
        name=name, status="failed", reason=reason, notes=tuple(notes)
    )


def transcribe_voiceover(
    catalog: Catalog,
    settings,
    *,
    dry_run: bool = False,
    api_client=None,
    local_backend=None,
    created: str | None = None,
) -> VoResult:
    """Transcribe every voiceover recording into ``transcripts/``.

    Visits the recordings from :func:`recordings` in order. A path that
    is not an existing regular file is ``"failed"`` with
    ``"recording not found"``; a recording inside the pipeline-written
    folders is ``"failed"`` with
    ``"recording is inside the project's output folders"``. A recording
    whose SHA-256 already sits in a voiceover file is ``"skipped"``;
    when that file's catalog row is missing the row is repaired from
    the validated file and the entry is ``"recorded"`` instead.
    With ``dry_run`` the entry is ``"would-transcribe"`` and nothing
    else happens. Provider ``"local"`` transcribes locally; provider
    ``"openai"`` calls the hosted transcription through the network
    gate and falls back to local when offline, when the recording is
    over the hosted size limit, or when the hosted answer has no usable
    word times. After either path the transcript must hold at least one
    word with finite times, else the recording is ``"failed"`` with
    ``"no word times in the answer"`` and nothing is written. The
    document is written with the safe write layer; existing files are
    never overwritten or deleted. The run is recorded with
    ``start_run("vo-transcribe")`` and ``finish_run`` in a finally,
    except in dry runs, which record nothing. ``local_backend`` is a
    callable ``backend(path, *, model, initial_prompt)`` standing in for
    the local backend in tests, and ``api_client`` is the hosted client
    standing in for the gate's own client.
    """
    config = step_config(settings, "vo_audio")
    found = recordings(settings)
    if dry_run:
        entries: list[VoEntry] = []
        for raw in found:
            text = os.fspath(raw)
            name = Path(text).name
            if not os.path.isfile(text):
                entries.append(
                    VoEntry(
                        name=name, status="failed", reason="recording not found"
                    )
                )
                continue
            if _inside_output(settings, text):
                entries.append(
                    VoEntry(
                        name=name,
                        status="failed",
                        reason="recording is inside the project's "
                        "output folders",
                    )
                )
                continue
            try:
                digest = _sha256_file(text)
            except OSError as exc:
                entries.append(
                    VoEntry(name=name, status="failed", reason=str(exc))
                )
                continue
            if _already_done(settings, digest):
                entries.append(VoEntry(name=name, status="skipped"))
                continue
            entries.append(VoEntry(name=name, status="would-transcribe"))
        return VoResult(entries=tuple(entries))
    run_id = catalog.start_run("vo-transcribe", dry_run=False)
    try:
        local_model = (
            getattr(settings, "transcribe_model", None)
            or whisper_local.DEFAULT_MODEL
        )
        prompt: str | None = None
        done: list[VoEntry] = []
        for raw in found:
            entry, prompt = _transcribe_one(
                catalog,
                settings,
                raw,
                config,
                local_model,
                api_client=api_client,
                local_backend=local_backend,
                prompt=prompt,
                created=created,
            )
            done.append(entry)
        result = VoResult(entries=tuple(done))
    except BaseException as exc:
        catalog.finish_run(run_id, f"stopped: {type(exc).__name__}")
        raise
    catalog.finish_run(run_id, result.outcome)
    return result


def _transcribe_one(
    catalog: Catalog,
    settings,
    raw,
    config,
    local_model: str,
    *,
    api_client=None,
    local_backend=None,
    prompt: str | None = None,
    created: str | None = None,
) -> tuple[VoEntry, str | None]:
    """Transcribe one recording, returning its entry and prompt.

    The local prompt is built once with
    :func:`fcp_pipeline.transcribe.build_initial_prompt` and handed back
    for the next recording.
    """
    text = os.fspath(raw)
    name = Path(text).name
    if not os.path.isfile(text):
        return (
            VoEntry(name=name, status="failed", reason="recording not found"),
            prompt,
        )
    if _inside_output(settings, text):
        return (
            VoEntry(
                name=name,
                status="failed",
                reason="recording is inside the project's output folders",
            ),
            prompt,
        )
    try:
        digest = _sha256_file(text)
    except OSError as exc:
        return (
            VoEntry(name=name, status="failed", reason=str(exc)),
            prompt,
        )
    found = _find_done_file(settings, digest)
    if found is not None:
        return (
            _repair_missing_row(
                catalog, settings, text, digest, found, created
            ),
            prompt,
        )

    notes: list[str] = []
    engine: str | None = None
    model_used: str | None = None
    transcription = None

    def run_local():
        """Run the local backend once, keeping its word times."""
        nonlocal prompt, engine, model_used, transcription
        active = local_backend or whisper_local.transcribe_file
        if prompt is None:
            prompt = build_initial_prompt(catalog, settings)
        transcription = active(text, model=local_model, initial_prompt=prompt)
        engine = whisper_local.ENGINE
        model_used = local_model

    if config.provider == "local":
        try:
            run_local()
        except Exception as exc:
            return (
                _failed_local(name, exc, notes),
                prompt,
            )
    else:
        gate = NetworkGate(settings, "vo_audio")
        if gate.offline:
            notes.append("offline: transcribed locally")
            try:
                run_local()
            except Exception as exc:
                return (
                    _failed_local(name, exc, notes),
                    prompt,
                )
        else:
            try:
                oversize = os.path.getsize(text) > MAX_UPLOAD_BYTES
            except OSError:
                oversize = False
            if oversize:
                notes.append(
                    f"{name}: over the hosted size limit; transcribed locally"
                )
                try:
                    run_local()
                except Exception as exc:
                    return (
                        _failed_local(name, exc, notes),
                        prompt,
                    )
            else:
                try:
                    reply = audio_openai.transcribe_audio(
                        gate, config.model, text, client=api_client, words=True
                    )
                except OfflineError:
                    notes.append("offline: transcribed locally")
                    try:
                        run_local()
                    except Exception as exc:
                        return (
                            _failed_local(name, exc, notes),
                            prompt,
                        )
                except audio_openai.AudioFormatError:
                    notes.append(
                        f"{name}: the hosted answer had no usable word times "
                        "(AudioFormatError); transcribed locally"
                    )
                    try:
                        run_local()
                    except Exception as exc:
                        return (
                            _failed_local(name, exc, notes),
                            prompt,
                        )
                except Exception as exc:
                    notes.append(
                        f"{name}: the hosted transcription failed "
                        f"({type(exc).__name__}); transcribed locally"
                    )
                    try:
                        run_local()
                    except Exception as nested:
                        return (
                            _failed_local(name, nested, notes),
                            prompt,
                        )
                else:
                    transcription = _hosted_to_transcription(reply)
                    engine = "openai"
                    model_used = config.model

    assert transcription is not None
    if not _has_word_times(transcription):
        return (
            VoEntry(
                name=name,
                status="failed",
                reason="no word times in the answer",
                notes=tuple(notes),
            ),
            prompt,
        )
    saved = []
    for segment in transcription.segments:
        saved.append(
            {
                "start": _seconds(segment.start),
                "end": _seconds(segment.stop),
                "text": segment.text,
                "words": [
                    {
                        "word": word.word,
                        "start": _seconds(word.start),
                        "end": _seconds(word.stop),
                        "probability": word.probability,
                    }
                    for word in segment.words
                ],
            }
        )
    document = {
        "version": 1,
        "role": "voiceover",
        "recording": {
            "path": os.path.abspath(text),
            "sha256": digest,
        },
        "engine": engine,
        "model": model_used,
        "language": transcription.language,
        "segments": saved,
    }
    blob = (json.dumps(document, indent=2) + "\n").encode("utf-8")
    stem = _safe_name(text)
    first = Path(os.fspath(settings.transcripts)) / f"{stem}-{digest[:8]}.json"
    target = first
    number = 2
    while os.path.lexists(target):
        target = (
            Path(os.fspath(settings.transcripts))
            / f"{stem}-{digest[:8]}-{number}.json"
        )
        number += 1
    try:
        writer = SafeWriter(settings)
        written = writer.write_new(target, blob)
    except (SafeWriteError, OSError) as exc:
        return (
            VoEntry(
                name=name,
                status="failed",
                reason=str(exc),
                notes=tuple(notes),
            ),
            prompt,
        )
    try:
        info = os.stat(text)
    except OSError as exc:
        return (
            VoEntry(
                name=name,
                status="failed",
                reason=str(exc),
                notes=tuple(notes),
            ),
            prompt,
        )
    asset = "vo-" + hashlib.sha256(
        os.fspath(Path(text).resolve()).encode("utf-8")
    ).hexdigest()[:16]
    try:
        media_id = catalog.put_media(
            MediaRecord(
                asset_id=asset,
                kind="voiceover",
                path=os.path.abspath(text),
                size=info.st_size,
                mtime_ns=info.st_mtime_ns,
                inode=info.st_ino,
            )
        )
        file_sha = hashlib.sha256(bytes(blob)).hexdigest()
        catalog.add_transcript_set(
            media_id,
            files=[(os.fspath(written), file_sha)],
            language=transcription.language,
            engine=f"{engine}:{model_used}",
            created=created or _utcnow(),
            segments=_build_records(transcription),
        )
    except (CatalogError, SafeWriteError, OSError) as exc:
        return (
            VoEntry(
                name=name,
                status="failed",
                reason=str(exc) or type(exc).__name__,
                notes=tuple(notes),
            ),
            prompt,
        )
    return (
        VoEntry(
            name=name,
            status="transcribed",
            engine=engine,
            file=os.fspath(written),
            notes=tuple(notes),
        ),
        prompt,
    )


def main(argv: list[str] | None = None) -> int:
    """Transcribe the voiceover recording(s) into ``transcripts/``."""
    parser = argparse.ArgumentParser(
        prog="fcp_pipeline.vo",
        description="Transcribe the voiceover recording into transcripts/.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    transcribe = subparsers.add_parser(
        "transcribe", help="Transcribe the voiceover recording."
    )
    transcribe.add_argument(
        "--trip",
        required=True,
        help="Trip folder holding project.toml.",
    )
    transcribe.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be transcribed without writing anything.",
    )
    transcribe.add_argument(
        "--offline",
        action="store_true",
        help="Refuse every network call before connecting.",
    )
    args = parser.parse_args(argv)
    if args.offline:
        set_offline(True)
    try:
        settings = load_trip(args.trip)
    except TripSettingsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        catalog = Catalog.open(settings)
    except (CatalogError, SafeWriteError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        try:
            result = transcribe_voiceover(
                catalog, settings, dry_run=args.dry_run
            )
        except AiConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except VoError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except (CatalogError, SafeWriteError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    finally:
        catalog.close()
    for entry in result.entries:
        for note in entry.notes:
            print(f"warning: {note}", file=sys.stderr)
        if entry.status == "failed" and entry.reason is not None:
            print(f"{entry.status}: {entry.name}: {entry.reason}")
    print(result.outcome)
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
