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
    "AudioWord",
    "build_request",
    "parse_transcription",
    "transcribe_audio",
]

#: Times larger than this, in seconds, are refused with AudioFormatError.
#: The bound is checked on the raw answer value, before any float
#: conversion or multiplication, so huge ints never reach float().
MAX_SECONDS = 1_000_000


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
class AudioWord:
    """One transcribed word with its exact time range."""

    start: Fraction
    end: Fraction
    word: str


@dataclass(frozen=True)
class AudioTranscript:
    """One hosted answer: its language, segments and words in order."""

    language: str | None
    segments: tuple[AudioSegment, ...]
    words: tuple[AudioWord, ...] = ()


def build_request(model: str, *, words: bool = False) -> dict:
    """Return the hosted transcription request body for ``model``.

    The answer must carry per-segment times, so the request pins the
    verbose JSON shape with segment-level times. With ``words`` true it
    also asks for word-level times.
    """
    granularities = ["segment", "word"] if words else ["segment"]
    return {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities": granularities,
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


def _checked_fraction(value: object, label: str) -> Fraction:
    """Turn a JSON number of seconds into an exact millisecond Fraction.

    ``label`` names the key in refusal messages. A value larger than
    :data:`MAX_SECONDS` is refused like a non-finite one; the bound is
    checked on the raw value, before any float conversion or
    multiplication, so a huge int never reaches ``float()``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AudioFormatError(
            f"{label}: must be a finite number of seconds"
        )
    if value > MAX_SECONDS or not math.isfinite(value):
        raise AudioFormatError(
            f"{label}: must be a finite number of seconds"
        )
    return Fraction(round(float(value) * 1000), 1000)


def _to_fraction(value: object, index: int, key: str) -> Fraction:
    """Turn a JSON number of seconds into an exact millisecond Fraction."""
    return _checked_fraction(value, f"segments[{index}].{key}")


def _word_dict(raw: object, index: int) -> dict:
    """Return one raw word as a plain dict, through ``model_dump()``."""
    if isinstance(raw, dict):
        return raw
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        data = dump()
        if isinstance(data, dict):
            return data
    raise AudioFormatError(
        f"words[{index}]: must be an object with start, end and word"
    )


def parse_transcription(
    raw: object, *, require_words: bool = False
) -> AudioTranscript:
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
    the key and the rule, never the text. With ``require_words`` true
    the answer's top-level ``"words"`` must be a non-empty list of
    objects with a non-empty string ``"word"`` (kept stripped) and
    finite numeric ``"start"`` and ``"end"`` with
    ``0 <= start <= end``; a missing, empty or broken list raises
    :class:`AudioFormatError`. Without it the ``"words"`` key is
    ignored.
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
    words: list[AudioWord] = []
    if require_words:
        words_raw = answer.get("words")
        if not isinstance(words_raw, list) or not words_raw:
            raise AudioFormatError(
                "words: must be a non-empty list of words"
            )
        for index, item in enumerate(words_raw):
            piece = _word_dict(item, index)
            start = _checked_fraction(
                piece.get("start"), f"words[{index}].start"
            )
            end = _checked_fraction(piece.get("end"), f"words[{index}].end")
            if not (Fraction(0, 1) <= start <= end):
                raise AudioFormatError(
                    f"words[{index}]: times must satisfy 0 <= start <= end"
                )
            text = piece.get("word")
            if not isinstance(text, str) or not text.strip():
                raise AudioFormatError(
                    f"words[{index}].word: must be a non-empty string"
                )
            words.append(
                AudioWord(start=start, end=end, word=text.strip())
            )
    return AudioTranscript(
        language=language, segments=tuple(segments), words=tuple(words)
    )


def transcribe_audio(
    gate, model: str, audio_path, *, client=None, words: bool = False
) -> AudioTranscript:
    """Upload one audio file and return its parsed hosted transcript.

    Reads the file size, then runs ``gate.call("openai", model,
    [("audio", size)], do_request)`` where ``do_request`` is built inside
    the gate, so an offline gate refuses before any key is read, any
    client is built, or the file is opened. The request takes ``client``
    (a test double) or the gate's own client, opens the audio file in
    binary mode, and calls
    ``client.audio.transcriptions.create(file=handle, **build_request(model))``;
    a result with ``model_dump()`` is converted to a dict first. The
    answer then goes through :func:`parse_transcription`. With ``words``
    true the request asks for word times and the answer must carry them.
    """
    size = os.path.getsize(os.fspath(audio_path))
    target = os.fspath(audio_path)

    def do_request():
        # Built inside the gate: an offline gate refuses before any key
        # is read, any client is built, or the file is opened.
        backend = client if client is not None else gate.openai_client()
        with open(target, "rb") as handle:
            raw = backend.audio.transcriptions.create(
                file=handle, **build_request(model, words=words)
            )
        dump = getattr(raw, "model_dump", None)
        if callable(dump):
            raw = dump()
        return raw

    answer = gate.call("openai", model, [("audio", size)], do_request)
    return parse_transcription(answer, require_words=words)
