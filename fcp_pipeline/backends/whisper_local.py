"""The local mlx-whisper speech-to-text backend.

:func:`transcribe_file` runs transcription on this Mac through the
optional ``mlx_whisper`` package, which is imported lazily inside the
function so importing this module never needs it installed and never
touches the network itself. All times are exact
:class:`~fractions.Fraction` values at millisecond resolution.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from fractions import Fraction

__all__ = [
    "DEFAULT_MODEL",
    "ENGINE",
    "INSTALL_HINT",
    "BackendMissing",
    "Segment",
    "Transcription",
    "Word",
    "available",
    "transcribe_file",
]

#: The Hugging Face id of whisper-large-v3-turbo for mlx-whisper.
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
#: Engine name recorded in transcript files and catalog rows.
ENGINE = "mlx-whisper"
#: Message telling the editor how to install the optional package.
INSTALL_HINT = (
    "install the optional transcription package with: pip install mlx-whisper"
)


class BackendMissing(RuntimeError):
    """Raised when the optional mlx-whisper package is not installed."""


@dataclass(frozen=True)
class Word:
    """One transcribed word with its exact time range."""

    start: Fraction
    stop: Fraction
    word: str
    probability: float | None


@dataclass(frozen=True)
class Segment:
    """One transcribed segment with its words in order."""

    start: Fraction
    stop: Fraction
    text: str
    words: tuple[Word, ...]


@dataclass(frozen=True)
class Transcription:
    """One backend answer: the language and its segments in order."""

    language: str | None
    segments: tuple[Segment, ...]


def available() -> bool:
    """Whether the optional ``mlx_whisper`` package can be imported."""
    return importlib.util.find_spec("mlx_whisper") is not None


def _to_fraction(value: object) -> Fraction:
    """Turn float seconds into an exact :class:`Fraction` at millisecond resolution."""
    return Fraction(round(float(value) * 1000), 1000)


def transcribe_file(
    path: str | os.PathLike[str], *, model: str, initial_prompt: str
) -> Transcription:
    """Transcribe one clip file locally with mlx-whisper.

    Imports ``mlx_whisper`` inside this function (raising
    :class:`BackendMissing` when it is absent) and calls
    ``mlx_whisper.transcribe`` with word timestamps, converting the
    returned dict to :class:`Transcription`. This function itself never
    opens a socket.
    """
    try:
        import mlx_whisper
    except ImportError:
        raise BackendMissing(INSTALL_HINT) from None
    result = mlx_whisper.transcribe(
        os.fspath(path),
        path_or_hf_repo=model,
        word_timestamps=True,
        initial_prompt=initial_prompt,
        verbose=None,
    )
    language = result.get("language")
    segments: list[Segment] = []
    for raw_segment in result.get("segments", ()):
        words: list[Word] = []
        for raw_word in raw_segment.get("words", ()):
            probability = raw_word.get("probability")
            words.append(
                Word(
                    start=_to_fraction(raw_word["start"]),
                    stop=_to_fraction(raw_word["end"]),
                    word=raw_word["word"],
                    probability=None if probability is None else float(probability),
                )
            )
        segments.append(
            Segment(
                start=_to_fraction(raw_segment["start"]),
                stop=_to_fraction(raw_segment["end"]),
                text=raw_segment["text"],
                words=tuple(words),
            )
        )
    return Transcription(language=language, segments=tuple(segments))


if __name__ == "__main__":
    raise SystemExit("the whisper backend has no command line")
