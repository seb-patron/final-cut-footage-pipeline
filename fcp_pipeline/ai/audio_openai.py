"""Hosted audio transcription through the network gate.

:func:`transcribe_audio` uploads one audio file to the hosted audio
transcription service named by the caller (the editor's own API credits)
and returns the parsed answer. Every call runs through
:class:`fcp_pipeline.net.NetworkGate`, so an offline gate refuses before
any key is read, any client is built, or the file is opened. The audio is
read only for this upload; nothing is cached or written. The openai SDK
is built only by the gate inside the call, so importing this module needs
neither the SDK nor the network. Times are exact
:class:`~fractions.Fraction` values at millisecond resolution.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from fractions import Fraction

__all__ = [
    "AudioFormatError",
    "AudioSegment",
    "AudioTranscript",
    "build_request",
    "parse_transcription",
    "transcribe_audio",
]


class AudioFormatError(ValueError):
    """Raised when a hosted transcription answer breaks the shape rules.

    The message names the key and the rule it broke, never answer text.
    """


@dataclass(frozen=True)
class AudioSegment:
    """One transcribed segment with its exact time range."""

    start: Fraction
    end: Fraction
    text: str


@dataclass(frozen=True)
class AudioTranscript:
    """One hosted answer: its language and its segments in order."""

    language: str | None
    segments: tuple[AudioSegment, ...]


def build_request(model: str) -> dict:
    """Return the hosted transcription request body for ``model``.

    The answer must carry per-segment times, so the request pins the
    verbose JSON shape with segment-level times.
    """
    return {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"],
    }


def _answer_dict(raw: object) -> dict:
    """Return ``raw`` as a plain dict, through ``model_dump`` when needed."""
    if isinstance(raw, dict):
        return raw
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        data = dump()
        if isinstance(data, dict):
            return data
    raise AudioFormatError(
        "answer: must be a dict or an object with model_dump()"
    )


def _segment_dict(raw: object, index: int) -> dict:
    """Return one raw segment as a plain dict, through ``model_dump``."""
    if isinstance(raw, dict):
        return raw
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        data = dump()
        if isinstance(data, dict):
            return data
    raise AudioFormatError(
        f"segments[{index}]: must be an object with start, end and text"
    )


def _to_fraction(value: object, index: int, key: str) -> Fraction:
    """Turn a JSON number of seconds into an exact millisecond Fraction."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AudioFormatError(
            f"segments[{index}].{key}: must be a finite number of seconds"
        )
    if not math.isfinite(value):
        raise AudioFormatError(
            f"segments[{index}].{key}: must be a finite number of seconds"
        )
    return Fraction(round(float(value) * 1000), 1000)


def parse_transcription(raw: object) -> AudioTranscript:
    """Parse a hosted verbose transcription answer.

    Accepts a plain dict or an object with ``model_dump()``. The answer
    needs a ``"segments"`` list of objects with finite numeric
    ``"start"`` and ``"end"`` (``0 <= start <= end``; booleans are not
    numbers and NaN or Infinity are refused) and a string ``"text"``
    (kept stripped of surrounding spaces; empty text is kept). Takes
    ``"language"`` when it is a string, else None. Times become exact
    :class:`Fraction` values at millisecond resolution. Anything else,
    including a missing or empty ``"segments"`` key (the chosen model
    returned no segment times), raises :class:`AudioFormatError` naming
    the key and the rule, never the text.
    """
    answer = _answer_dict(raw)
    segments_raw = answer.get("segments")
    if not isinstance(segments_raw, list) or not segments_raw:
        raise AudioFormatError(
            "segments: must be a non-empty list of segments"
        )
    segments: list[AudioSegment] = []
    for index, item in enumerate(segments_raw):
        piece = _segment_dict(item, index)
        start = _to_fraction(piece.get("start"), index, "start")
        end = _to_fraction(piece.get("end"), index, "end")
        if not (Fraction(0, 1) <= start <= end):
            raise AudioFormatError(
                f"segments[{index}]: times must satisfy 0 <= start <= end"
            )
        text = piece.get("text")
        if not isinstance(text, str):
            raise AudioFormatError(
                f"segments[{index}].text: must be a string"
            )
        segments.append(
            AudioSegment(start=start, end=end, text=text.strip())
        )
    language = answer.get("language")
    if not isinstance(language, str):
        language = None
    return AudioTranscript(language=language, segments=tuple(segments))


def transcribe_audio(gate, model: str, audio_path, *, client=None) -> AudioTranscript:
    """Upload one audio file and return its parsed hosted transcript.

    Reads the file size, then runs ``gate.call("openai", model,
    [("audio", size)], do_request)`` where ``do_request`` is built inside
    the gate, so an offline gate refuses before any key is read, any
    client is built, or the file is opened. The request takes ``client``
    (a test double) or the gate's own client, opens the audio file in
    binary mode, and calls
    ``client.audio.transcriptions.create(file=handle, **build_request(model))``;
    a result with ``model_dump()`` is converted to a dict first. The
    answer then goes through :func:`parse_transcription`.
    """
    size = os.path.getsize(os.fspath(audio_path))
    target = os.fspath(audio_path)

    def do_request():
        # Built inside the gate: an offline gate refuses before any key
        # is read, any client is built, or the file is opened.
        backend = client if client is not None else gate.openai_client()
        with open(target, "rb") as handle:
            raw = backend.audio.transcriptions.create(
                file=handle, **build_request(model)
            )
        dump = getattr(raw, "model_dump", None)
        if callable(dump):
            raw = dump()
        return raw

    answer = gate.call("openai", model, [("audio", size)], do_request)
    return parse_transcription(answer)
