"""The trip memo step: transcribe the editor's memo recording.

:func:`transcribe_memos` visits every memo recording that
``project.toml`` points to (``[media].memo``, a path or an array of
paths; the audio stays where it is and is never copied into the
project) and saves one ``memo/<name>.json`` per recording through the
safe write layer: the transcript with segment times plus a reference to
the recording (its path and SHA-256). The default provider is local
mlx-whisper on this Mac; the hosted audio transcription (the editor's
own API credits) runs only when ``[pipeline.ai.memo_audio]`` asks for
it, through the network gate. A hosted run falls back to local when
offline, when the call fails, or when the answer has no segment times.
A recording already done (same SHA-256) is skipped without any call,
and a recording over the hosted size limit is split at silences into
temporary pieces outside the project. Nothing calls this step yet
except its own command line; importing this module runs no tool, opens
no database and reads no file.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from fcp_pipeline.ai import AiConfigError, audio_openai, step_config
from fcp_pipeline.backends import whisper_local
from fcp_pipeline.catalog import Catalog, CatalogError
from fcp_pipeline.net import NetworkGate, OfflineError, set_offline
from fcp_pipeline.safe_write import SafeWriteError, SafeWriter
from fcp_pipeline.transcribe import build_initial_prompt
from fcp_pipeline.trip import TripSettingsError, load_trip

__all__ = [
    "MAX_JSON_BYTES",
    "MAX_UPLOAD_BYTES",
    "SILENCE_PEAK",
    "MemoEntry",
    "MemoError",
    "MemoResult",
    "main",
    "plan_chunks",
    "recordings",
    "split_recording",
    "transcribe_memos",
]

#: Recordings bigger than this are split before any hosted upload, under
#: the hosted service's own 25 MB limit.
MAX_UPLOAD_BYTES = 24_000_000
#: Memo files bigger than this are never treated as done recordings.
MAX_JSON_BYTES = 5_000_000
#: A 50 ms window counts as silent when its peak stays below this.
SILENCE_PEAK = 300

#: Sample rate of the decoded split pieces, in Hz.
_DECODE_RATE = 16_000
#: Silence window length, in milliseconds.
_WINDOW_MS = 50
#: Shortest silence run that offers a cut point, in windows (0.5 s).
_MIN_SILENCE_WINDOWS = 10
#: How long one ffmpeg decode may take before it is abandoned, in seconds.
_FFMPEG_TIMEOUT = 300
#: Name of the decoded scratch file inside a split folder.
_DECODED_NAME = "decoded.wav"


class MemoError(ValueError):
    """Raised when a memo recording cannot be read, split or decoded.

    The message names the path, key or tool call and the rule it broke,
    never recording contents.
    """


@dataclass(frozen=True)
class MemoEntry:
    """What the memo step did with one recording file."""

    name: str
    status: str
    reason: str | None = None
    engine: str | None = None
    file: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoResult:
    """Every entry of one memo run, in recording order."""

    entries: tuple[MemoEntry, ...] = ()

    @property
    def transcribed(self) -> int:
        """How many recordings were transcribed by this run."""
        return sum(1 for entry in self.entries if entry.status == "transcribed")

    @property
    def skipped(self) -> int:
        """How many recordings were already done and left alone."""
        return sum(1 for entry in self.entries if entry.status == "skipped")

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
        if self.would_transcribe:
            text = f"would-transcribe {self.would_transcribe}, " + text
        return text


def recordings(settings) -> list[Path]:
    """Return the memo recording paths from ``[media].memo``.

    Reads ``settings.project_toml`` again and returns the ``[media].memo``
    value as paths: a string or an array of strings, each non-empty; a
    relative path is resolved against ``settings.folder``. An absent key
    gives []. Any other type raises :class:`MemoError` naming
    ``[media].memo``.
    """
    with open(os.fspath(settings.project_toml), "rb") as handle:
        data = tomllib.load(handle)
    media = data.get("media", {})
    if not isinstance(media, dict):
        return []
    value = media.get("memo", None)
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise MemoError(
            "project.toml key '[media].memo' must be a path string or an "
            "array of path strings"
        )
    base = Path(os.fspath(settings.folder))
    paths: list[Path] = []
    for item in values:
        if not isinstance(item, str) or not item:
            raise MemoError(
                "project.toml key '[media].memo' must be a path string or "
                "an array of path strings, each non-empty"
            )
        candidate = Path(item)
        if candidate.is_absolute():
            paths.append(candidate)
        else:
            paths.append(base / item)
    return paths


def _sha256_file(path: str) -> str:
    """Hash a file by reading it in blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _already_done(settings, digest: str) -> bool:
    """Whether any memo file directly in ``memo/`` holds this SHA-256."""
    folder = Path(os.fspath(settings.memo))
    try:
        names = os.listdir(folder)
    except OSError:
        return False
    for name in names:
        if not name.endswith(".json"):
            continue
        candidate = folder / name
        try:
            if not os.path.isfile(candidate):
                continue
            if os.path.getsize(candidate) > MAX_JSON_BYTES:
                continue
            with open(candidate, "rb") as handle:
                document = json.loads(handle.read().decode("utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(document, dict):
            continue
        recording = document.get("recording")
        if not isinstance(recording, dict):
            continue
        sha = recording.get("sha256")
        if isinstance(sha, str) and sha == digest:
            return True
    return False


def _split_plan(
    samples, sample_rate: int, max_bytes: int
) -> tuple[list[tuple[int, int]], list[int]]:
    """Split sample ranges and report the hard-cut positions.

    Returns ``(ranges, hard_cuts)`` where ``ranges`` holds
    ``(start, stop)`` sample offsets covering the audio in order and
    ``hard_cuts`` holds the cut positions, in samples, where no silence
    offered a cut and the piece was cut at the size limit instead.
    """
    max_samples = (max_bytes - 44) // 2
    if max_samples <= 0:
        raise MemoError(
            f"cannot split audio: max_bytes {max_bytes!r} leaves no room "
            "for samples past the 44-byte header"
        )
    total = len(samples)
    if total <= max_samples:
        return ([(0, total)], [])
    window = max(sample_rate * _WINDOW_MS // 1000, 1)
    count = (total + window - 1) // window
    silent: list[bool] = []
    for index in range(count):
        start = index * window
        stop = min(start + window, total)
        peak = 0
        for value in samples[start:stop]:
            magnitude = abs(value)
            if magnitude > peak:
                peak = magnitude
            if peak >= SILENCE_PEAK:
                break
        silent.append(peak < SILENCE_PEAK)
    cuts: list[int] = []
    index = 0
    while index < count:
        if not silent[index]:
            index += 1
            continue
        stop = index
        while stop < count and silent[stop]:
            stop += 1
        if stop - index >= _MIN_SILENCE_WINDOWS:
            cuts.append((index * window + stop * window) // 2)
        index = stop
    ranges: list[tuple[int, int]] = []
    hard: list[int] = []
    start = 0
    while total - start > max_samples:
        limit = start + max_samples
        pick: int | None = None
        for cut in cuts:
            if start < cut <= limit:
                pick = cut
        if pick is None:
            pick = limit
            hard.append(pick)
        ranges.append((start, pick))
        start = pick
    ranges.append((start, total))
    return (ranges, hard)


def plan_chunks(samples, sample_rate: int, max_bytes: int) -> list[tuple[int, int]]:
    """Split ``samples`` (an ``array('h')``) into piece ranges.

    ``max_samples`` is ``(max_bytes - 44) // 2``; silence is a run of
    consecutive 50 ms windows whose peak absolute amplitude stays below
    :data:`SILENCE_PEAK` lasting at least 0.5 s, and each such run offers
    one cut point at its middle sample. Greedy from the start of the
    audio: while more than ``max_samples`` remain, cut at the largest cut
    point in ``(start, start + max_samples]``, or at
    ``start + max_samples`` when there is none (a hard cut). Every piece
    is at most ``max_samples``; audio that fits gives one piece.
    """
    ranges, _ = _split_plan(samples, sample_rate, max_bytes)
    return [(int(first), int(last)) for first, last in ranges]


def _read_decoded(decoded: str) -> tuple[array.array, int]:
    """Read decoded mono 16-bit PCM samples and their rate."""
    with wave.open(decoded, "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2 or channels != 1:
        raise MemoError(
            f"cannot split audio from {decoded}: expected mono 16-bit PCM"
        )
    samples: array.array = array.array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    return (samples, rate)


class _SplitPieces(list):
    """The pieces of one split recording, with hard-cut positions.

    Behaves exactly like a list of ``(piece path, offset)`` pairs; the
    ``hard_cuts`` attribute additionally holds the cut positions, in
    samples, where no silence offered a cut and the piece was cut at the
    size limit instead, so the caller can note them without re-reading
    the decoded audio.
    """

    def __init__(
        self, items: list[tuple[str, Fraction]], hard_cuts: list[int]
    ) -> None:
        super().__init__(items)
        self.hard_cuts = hard_cuts


def split_recording(
    path, folder, max_bytes: int, *, runner=None
) -> list[tuple[str, Fraction]]:
    """Split a long recording into temporary WAV pieces.

    Decodes ``path`` once to a mono 16 kHz 16-bit PCM WAV in ``folder``
    with ``ffmpeg -nostdin -v error -y -i <path> -vn -ac 1 -ar 16000
    -c:a pcm_s16le <folder>/decoded.wav`` (a list of arguments with a
    timeout, never a shell), reads it with the standard wave and array
    modules (byte-swapped on big-endian machines), asks
    :func:`plan_chunks` for the pieces, and writes each piece as
    ``chunk-001.wav``, ``chunk-002.wav``, ... in ``folder`` (a 44-byte
    header plus 2 bytes per sample). Returns a list of ``(piece path,
    offset as Fraction(start_sample, 16000))`` in order. ``folder`` is a
    temporary folder the caller removes; this function writes nothing
    anywhere else.
    """
    if shutil.which("ffmpeg") is None:
        raise MemoError("ffmpeg is needed to split a long recording")
    source = os.fspath(path)
    scratch = os.fspath(folder)
    decoded = os.path.join(scratch, _DECODED_NAME)
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        source,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        decoded,
    ]
    call = subprocess.run if runner is None else runner
    try:
        completed = call(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_FFMPEG_TIMEOUT,
            check=False,
        )
    except FileNotFoundError:
        raise MemoError(
            "ffmpeg is needed to split a long recording"
        ) from None
    except OSError as exc:
        raise MemoError(f"cannot run ffmpeg for {source}: {exc}") from None
    if getattr(completed, "returncode", 1) != 0:
        error = getattr(completed, "stderr", b"")
        if isinstance(error, bytes):
            detail = error.decode("utf-8", "replace").strip().splitlines()
        else:
            detail = str(error).strip().splitlines()
        tail = detail[-1].strip() if detail else "no error output"
        raise MemoError(
            f"ffmpeg decode failed for {source} "
            f"(exit {completed.returncode}): {tail}"
        )
    samples, _rate = _read_decoded(decoded)
    ranges, hard = _split_plan(samples, _DECODE_RATE, max_bytes)
    items: list[tuple[str, Fraction]] = []
    for number, (first, last) in enumerate(ranges, start=1):
        piece = os.path.join(scratch, f"chunk-{number:03d}.wav")
        with wave.open(piece, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(_DECODE_RATE)
            handle.writeframes(samples[first:last].tobytes())
        items.append((piece, Fraction(first, _DECODE_RATE)))
    return _SplitPieces(items, hard)


def _seconds(value) -> float:
    """Render an exact time as JSON seconds, rounded to 3 decimals."""
    return round(float(value), 3)


def _safe_name(path_text: str) -> str:
    """File stem with every unusual character replaced by ``_``."""
    stem = Path(path_text).stem
    return "".join(
        char if (char.isalnum() or char in ".-_") else "_" for char in stem
    )


def transcribe_memos(
    catalog: Catalog,
    settings,
    *,
    dry_run: bool = False,
    api_client=None,
    local_backend=None,
    created: str | None = None,
) -> MemoResult:
    """Transcribe every memo recording into ``memo/<name>.json``.

    Visits the recordings from :func:`recordings` in order. A path that
    is not an existing regular file is ``"failed"`` with
    ``"recording not found"``. A recording whose SHA-256 already sits in
    a memo file is ``"skipped"``. With ``dry_run`` the entry is
    ``"would-transcribe"`` and nothing else happens. Provider
    ``"local"`` transcribes locally; provider ``"openai"`` calls the
    hosted transcription through the network gate and falls back to
    local when offline (noted ``"offline: transcribed locally"``) or
    when the hosted path fails (noted ``"<name>: the hosted
    transcription failed (<ExceptionClassName>); transcribed locally"``).
    The document is written with the safe write layer; existing memo
    files are never overwritten or deleted. The run is recorded with
    ``start_run("memo-transcribe")`` and ``finish_run`` in a finally,
    except in dry runs, which record nothing. ``local_backend`` is a
    callable ``backend(path, *, model, initial_prompt)`` standing in for
    the local backend in tests, and ``api_client`` is the hosted client
    standing in for the gate's own client.
    """
    config = step_config(settings, "memo_audio")
    found = recordings(settings)
    if dry_run:
        entries: list[MemoEntry] = []
        for raw in found:
            text = os.fspath(raw)
            name = Path(text).name
            if not os.path.isfile(text):
                entries.append(
                    MemoEntry(
                        name=name, status="failed", reason="recording not found"
                    )
                )
                continue
            try:
                digest = _sha256_file(text)
            except OSError as exc:
                entries.append(
                    MemoEntry(name=name, status="failed", reason=str(exc))
                )
                continue
            if _already_done(settings, digest):
                entries.append(MemoEntry(name=name, status="skipped"))
                continue
            entries.append(MemoEntry(name=name, status="would-transcribe"))
        return MemoResult(entries=tuple(entries))
    run_id = catalog.start_run("memo-transcribe", dry_run=False)
    try:
        local_model = (
            getattr(settings, "transcribe_model", None)
            or whisper_local.DEFAULT_MODEL
        )
        prompt: str | None = None
        done: list[MemoEntry] = []
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
            )
            done.append(entry)
        result = MemoResult(entries=tuple(done))
    except BaseException as exc:
        catalog.finish_run(run_id, f"stopped: {type(exc).__name__}")
        raise
    catalog.finish_run(run_id, result.outcome)
    return result


def _local_text(language, segments) -> tuple[str | None, list[dict]]:
    """Convert a local backend answer to a language and JSON segments."""
    saved = []
    for segment in segments:
        saved.append(
            {
                "start": _seconds(segment.start),
                "end": _seconds(segment.stop),
                "text": segment.text,
            }
        )
    return (language, saved)


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
) -> tuple[MemoEntry, str | None]:
    """Transcribe one recording, returning its entry and prompt.

    The local prompt is built once with
    :func:`fcp_pipeline.transcribe.build_initial_prompt` and handed back
    for the next recording.
    """
    text = os.fspath(raw)
    name = Path(text).name
    if not os.path.isfile(text):
        return (
            MemoEntry(name=name, status="failed", reason="recording not found"),
            prompt,
        )
    try:
        digest = _sha256_file(text)
    except OSError as exc:
        return (
            MemoEntry(name=name, status="failed", reason=str(exc)),
            prompt,
        )
    if _already_done(settings, digest):
        return (MemoEntry(name=name, status="skipped"), prompt)

    notes: list[str] = []
    engine: str | None = None
    model_used: str | None = None
    language: str | None = None
    saved: list[dict] = []
    chunks = 1

    def run_local():
        """Run the local backend once, converting its answer."""
        nonlocal prompt, engine, model_used, language, saved
        active = local_backend or whisper_local.transcribe_file
        if prompt is None:
            prompt = build_initial_prompt(catalog, settings)
        answer = active(text, model=local_model, initial_prompt=prompt)
        engine = whisper_local.ENGINE
        model_used = local_model
        language, saved = _local_text(answer.language, answer.segments)

    if config.provider == "local":
        try:
            run_local()
        except Exception as exc:
            return (
                _failed_local(name, exc, notes),
                prompt,
            )
    else:
        gate = NetworkGate(settings, "memo_audio")
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
                size = os.path.getsize(text)
                if size <= MAX_UPLOAD_BYTES:
                    reply = audio_openai.transcribe_audio(
                        gate, config.model, text, client=api_client
                    )
                    language = reply.language
                    saved = [
                        {
                            "start": _seconds(segment.start),
                            "end": _seconds(segment.end),
                            "text": segment.text,
                        }
                        for segment in reply.segments
                    ]
                    chunks = 1
                else:
                    with tempfile.TemporaryDirectory() as scratch:
                        pieces = split_recording(
                            text, scratch, MAX_UPLOAD_BYTES
                        )
                        replies = [
                            audio_openai.transcribe_audio(
                                gate, config.model, piece, client=api_client
                            )
                            for piece, _offset in pieces
                        ]
                        offsets = [offset for _piece, offset in pieces]
                        combined: list[dict] = []
                        for reply, offset in zip(replies, offsets):
                            if reply.language is not None and language is None:
                                language = reply.language
                            for segment in reply.segments:
                                combined.append(
                                    {
                                        "start": _seconds(
                                            segment.start + offset
                                        ),
                                        "end": _seconds(segment.end + offset),
                                        "text": segment.text,
                                    }
                                )
                        saved = combined
                        chunks = len(pieces)
                        hard_cuts = list(
                            getattr(pieces, "hard_cuts", [])
                        )
                        for cut_sample in hard_cuts:
                            cut = float(
                                Fraction(cut_sample, _DECODE_RATE)
                            )
                            notes.append(
                                "no silence found; " f"cut at {cut} s"
                            )
                engine = "openai"
                model_used = config.model
            except OfflineError:
                notes.append("offline: transcribed locally")
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

    document = {
        "version": 1,
        "recording": {
            "path": os.path.abspath(text),
            "sha256": digest,
        },
        "engine": engine,
        "model": model_used,
        "language": language,
        "chunks": chunks,
        "segments": saved,
    }
    blob = (json.dumps(document, indent=2) + "\n").encode("utf-8")
    stem = _safe_name(text)
    first = settings.memo / f"{stem}.json"
    target = first
    if os.path.lexists(target):
        target = settings.memo / f"{stem}-{digest[:8]}.json"
        if os.path.lexists(target):
            return (
                MemoEntry(
                    name=name,
                    status="failed",
                    reason="target already exists",
                    notes=tuple(notes),
                ),
                prompt,
            )
    try:
        writer = SafeWriter(settings)
        written = writer.write_new(target, blob)
    except (SafeWriteError, OSError) as exc:
        return (
            MemoEntry(
                name=name,
                status="failed",
                reason=str(exc),
                notes=tuple(notes),
            ),
            prompt,
        )
    return (
        MemoEntry(
            name=name,
            status="transcribed",
            engine=engine,
            file=os.fspath(written),
            notes=tuple(notes),
        ),
        prompt,
    )


def _failed_local(name: str, exc: Exception, notes: list[str]):
    """Build the failed entry for a local backend error."""
    if isinstance(exc, whisper_local.BackendMissing):
        reason = whisper_local.INSTALL_HINT
    else:
        reason = str(exc) or type(exc).__name__
    return MemoEntry(
        name=name, status="failed", reason=reason, notes=tuple(notes)
    )


def main(argv: list[str] | None = None) -> int:
    """Transcribe the trip memo recording(s) into ``memo/``."""
    parser = argparse.ArgumentParser(
        prog="fcp_pipeline.memo",
        description="Transcribe the trip memo recording into memo/.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    transcribe = subparsers.add_parser(
        "transcribe", help="Transcribe the trip memo recording."
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
            result = transcribe_memos(
                catalog, settings, dry_run=args.dry_run
            )
        except AiConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except MemoError as exc:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
