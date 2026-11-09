"""Turn spoken callouts into catalog markers, ratings and takes.

Reads the word-timed transcripts already in the catalog and turns four
spoken callouts into callout-source tags: ``mark`` gives a one-frame
marker at that word; ``keeper`` gives a favorite range over the take it
ends; ``trash`` gives a reject range over that take; ``again`` ends the
take and marks it superseded so the next take is preferred.

A callout word counts only when it stands alone as its own short
utterance, never inside an ordinary sentence (``I'll mark it later`` is
ignored). Every time snaps to the clip's whole frames as exact
fractions. Higher-ranked and locked tags are never replaced.

Reads only the catalog and writes only catalog tags: no speech model,
no network, no footage read. Importing this module runs no tool,
opens no database and touches no file.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from fractions import Fraction

from fcp_pipeline.catalog import (
    SOURCE_RANK,
    Catalog,
    CatalogError,
    TagRecord,
    TranscriptSegmentRecord,
)
from fcp_pipeline.trip import TripSettingsError, load_trip

__all__ = [
    "ALONE_PAUSE",
    "CALLOUT_WORDS",
    "DEFAULT_GAP",
    "MARKER_FOLDER",
    "RATING_FOLDER",
    "TAKE_FOLDER",
    "Callout",
    "CalloutEntry",
    "CalloutError",
    "CalloutResult",
    "STEP",
    "SOURCE",
    "find_callouts",
    "find_takes",
    "main",
    "normalize_word",
    "tag_callouts",
]

#: Step name recorded on catalog runs.
STEP = "callouts"

#: Tag source written by this step.
SOURCE = "callout"

#: The four spoken callouts this step understands.
CALLOUT_WORDS = ("mark", "keeper", "trash", "again")

#: Default longest pause allowed inside one take, in seconds.
DEFAULT_GAP = Fraction(2)

#: Shortest silence around a callout word inside a segment, in seconds.
ALONE_PAUSE = Fraction(2, 5)

#: Tag folder holding the one-frame callout marker.
MARKER_FOLDER = "Marker"

#: Tag folder holding favorite and reject ranges.
RATING_FOLDER = "Rating"

#: Tag folder holding the superseded take.
TAKE_FOLDER = "Take"

#: Callout kinds that end the take before them.
_END_KINDS = ("keeper", "trash", "again")

#: --gap accepts only a plain decimal number of seconds.
_GAP_RE = re.compile(r"[0-9]+(\.[0-9]+)?$")

#: The (folder, value) stored for each callout kind.
_TAGS_FOR_KIND = {
    "mark": (MARKER_FOLDER, "Marker"),
    "keeper": (RATING_FOLDER, "Favorite"),
    "trash": (RATING_FOLDER, "Reject"),
    "again": (TAKE_FOLDER, "Superseded"),
}


class CalloutError(ValueError):
    """Raised for bad input to the callouts step."""


@dataclass(frozen=True)
class Callout:
    """One recognized callout word with its exact time range."""

    word: str
    kind: str
    start: Fraction
    stop: Fraction


@dataclass(frozen=True)
class Take:
    """One run of speech, ending at a callout, a pause or the clip end."""

    start: Fraction
    stop: Fraction
    end_kind: str | None


@dataclass(frozen=True)
class CalloutEntry:
    """What the step did with one clip."""

    path: str
    asset_id: str
    status: str
    reason: str | None = None
    stored: int = 0
    callouts: int = 0
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalloutResult:
    """Every entry of one callouts run, in catalog id order."""

    entries: tuple[CalloutEntry, ...] = ()

    @property
    def tagged(self) -> int:
        """How many clips got callout tags."""
        return sum(1 for entry in self.entries if entry.status == "tagged")

    @property
    def no_callouts(self) -> int:
        """How many clips held no callout word."""
        return sum(
            1 for entry in self.entries if entry.status == "no-callouts"
        )

    @property
    def no_transcript(self) -> int:
        """How many clips held no word-timed transcript."""
        return sum(
            1 for entry in self.entries if entry.status == "no-transcript"
        )

    @property
    def skipped(self) -> int:
        """How many clips held no frame rate to snap to."""
        return sum(1 for entry in self.entries if entry.status == "skipped")

    @property
    def would_tag(self) -> int:
        """How many clips a dry run would tag."""
        return sum(
            1 for entry in self.entries if entry.status == "would-tag"
        )

    @property
    def outcome(self) -> str:
        """One-line summary of the run."""
        text = (
            f"tagged {self.tagged}, no-callouts {self.no_callouts}, "
            f"no-transcript {self.no_transcript}, skipped {self.skipped}"
        )
        if self.would_tag:
            text = f"would-tag {self.would_tag}, " + text
        return text


def normalize_word(text: str) -> str:
    """Lowercase a word with leading and trailing punctuation removed."""
    if not isinstance(text, str):
        raise CalloutError(f"bad word {text!r}: must be a string")
    lowered = text.casefold().strip()
    start = 0
    end = len(lowered)
    while start < end and not lowered[start].isalnum():
        start += 1
    while end > start and not lowered[end - 1].isalnum():
        end -= 1
    return lowered[start:end]


def _flattened_words(
    segments: list[TranscriptSegmentRecord],
) -> list[tuple[int, object]]:
    """Flatten segments' words in order, skipping wordless segments."""
    flat: list[tuple[int, object]] = []
    for position, segment in enumerate(segments):
        for word in segment.words or []:
            flat.append((position, word))
    return flat


def _edge_indexes(flat: list[tuple[int, object]]) -> tuple[set[int], set[int]]:
    """Indexes of each segment's first and last word in the flat list."""
    firsts: set[int] = set()
    lasts: set[int] = set()
    seen: set[int] = set()
    for index, (position, _word) in enumerate(flat):
        if position not in seen:
            firsts.add(index)
            seen.add(position)
    seen.clear()
    for index in range(len(flat) - 1, -1, -1):
        position = flat[index][0]
        if position not in seen:
            lasts.add(index)
            seen.add(position)
    return firsts, lasts


def find_callouts(
    segments: list[TranscriptSegmentRecord],
) -> list[Callout]:
    """Find the standalone callout words in a clip's stored segments.

    A word is a callout when its normalized text is one of
    :data:`CALLOUT_WORDS` and it stands alone: it is the first word of
    its segment, or the pause before it is at least :data:`ALONE_PAUSE`,
    and it is the last word of its segment, or the pause after it is at
    least :data:`ALONE_PAUSE`.
    """
    flat = _flattened_words(segments)
    firsts, lasts = _edge_indexes(flat)
    found: list[Callout] = []
    for index, (_position, record) in enumerate(flat):
        if record.start is None or record.stop is None:
            continue
        kind = normalize_word(record.word)
        if kind not in CALLOUT_WORDS:
            continue
        if index in firsts:
            before_ok = True
        else:
            previous = flat[index - 1][1]
            before_ok = (
                record.start is not None
                and previous.stop is not None
                and record.start - previous.stop >= ALONE_PAUSE
            )
        if index in lasts:
            after_ok = True
        else:
            following = flat[index + 1][1]
            after_ok = (
                record.stop is not None
                and following.start is not None
                and following.start - record.stop >= ALONE_PAUSE
            )
        if before_ok and after_ok:
            found.append(
                Callout(
                    word=record.word,
                    kind=kind,
                    start=record.start,
                    stop=record.stop,
                )
            )
    return found


def _require_gap(gap: Fraction) -> None:
    """Refuse anything but a positive fraction gap."""
    if not isinstance(gap, Fraction) or gap <= 0:
        raise CalloutError(f"bad gap {gap!r}: must be a Fraction above 0")


def _match_kinds(
    flat: list[tuple[int, object]], callouts: list[Callout]
) -> list[str | None]:
    """Mark each flat word with its recognized callout kind, if any."""
    remaining: dict[tuple[object, object, str], int] = {}
    for callout in callouts:
        key = (callout.start, callout.stop, callout.kind)
        remaining[key] = remaining.get(key, 0) + 1
    kinds: list[str | None] = []
    for _position, record in flat:
        kind: str | None = None
        if isinstance(record.word, str):
            normalized = normalize_word(record.word)
            if normalized in CALLOUT_WORDS:
                key = (record.start, record.stop, normalized)
                if remaining.get(key, 0) > 0:
                    remaining[key] -= 1
                    kind = normalized
        kinds.append(kind)
    return kinds


def _split_chunks(
    flat: list[tuple[int, object]], kinds: list[str | None], gap: Fraction
) -> list[tuple[int, int, str | None]]:
    """Split flat words into chunks at pauses and take-ending callouts.

    Returns ``(start, stop, end_kind)`` triples in order, where
    ``end_kind`` is the callout word that ended the chunk or None. The
    pause check runs before the callout check, so when the pause before
    a take-ending callout is longer than ``gap`` the take ends at that
    pause and the callout opens a new chunk with no take before it.
    """
    chunks: list[tuple[int, int, str | None]] = []
    start = 0
    index = 0
    total = len(flat)
    while index < total:
        if index > start:
            previous = flat[index - 1][1]
            current = flat[index][1]
            if (
                previous.stop is not None
                and current.start is not None
                and current.start - previous.stop > gap
            ):
                chunks.append((start, index, None))
                start = index
                index += 1
                continue
        kind = kinds[index]
        if kind in _END_KINDS:
            chunks.append((start, index, kind))
            start = index + 1
            index += 1
            continue
        index += 1
    chunks.append((start, total, None))
    return chunks


def _speech_bounds(
    flat: list[tuple[int, object]],
    kinds: list[str | None],
    start: int,
    stop: int,
) -> tuple[Fraction, Fraction] | None:
    """Bound the speech (non-callout) words of one chunk, if any."""
    first: Fraction | None = None
    last: Fraction | None = None
    for index in range(start, stop):
        if kinds[index] is not None:
            continue
        record = flat[index][1]
        if first is None:
            first = record.start
        last = record.stop
    if first is None or last is None:
        return None
    return (first, last)


def find_takes(
    segments: list[TranscriptSegmentRecord],
    callouts: list[Callout],
    gap: Fraction = DEFAULT_GAP,
) -> list[Take]:
    """Split a clip's speech into takes between callouts and long pauses.

    A take is the run of speech (non-callout) words between boundaries:
    the start of the clip, a recognized ``keeper``, ``trash`` or
    ``again`` callout (each ends the take before it), or a pause
    strictly longer than ``gap`` between two consecutive words. A
    ``mark`` callout never ends a take. When the pause before a
    take-ending callout is longer than ``gap``, the take ends at that
    pause and the callout has no take before it. A chunk with no speech
    words is not a take.
    """
    _require_gap(gap)
    flat = _flattened_words(segments)
    kinds = _match_kinds(flat, callouts)
    takes: list[Take] = []
    for start, stop, end_kind in _split_chunks(flat, kinds, gap):
        bounds = _speech_bounds(flat, kinds, start, stop)
        if bounds is None:
            continue
        takes.append(Take(start=bounds[0], stop=bounds[1], end_kind=end_kind))
    return takes


def _snap_down(value: Fraction, frame: Fraction) -> Fraction:
    """Snap a time down to a whole frame."""
    return math.floor(value / frame) * frame


def _snap_up(value: Fraction, frame: Fraction) -> Fraction:
    """Snap a time up to a whole frame."""
    return math.ceil(value / frame) * frame


def _display_time(value: Fraction) -> str:
    """Render a note timestamp without float noise."""
    return f"{float(value):g}"


def _process_clip(
    catalog: Catalog, row: object, gap: Fraction, *, dry_run: bool
) -> CalloutEntry:
    """Tag one clip's callouts, or report what a dry run would tag."""
    path = row.path or ""
    asset_id = row.asset_id or ""
    records = [
        record
        for record in catalog.list_transcripts(row.id)
        if (record.path or "").endswith(".json")
    ]
    if not records:
        return CalloutEntry(path=path, asset_id=asset_id, status="no-transcript")
    segments = catalog.list_transcript_segments(records[-1].id)
    if not segments:
        return CalloutEntry(path=path, asset_id=asset_id, status="no-transcript")
    frame_rate = row.frame_rate
    if frame_rate is None or frame_rate <= 0:
        return CalloutEntry(
            path=path,
            asset_id=asset_id,
            status="skipped",
            reason="no frame rate in the catalog",
        )
    callouts = find_callouts(segments)
    if not callouts:
        return CalloutEntry(path=path, asset_id=asset_id, status="no-callouts")
    frame = Fraction(1, 1) / frame_rate
    flat = _flattened_words(segments)
    kinds = _match_kinds(flat, callouts)
    chunk_start_at: dict[int, int] = {}
    for start, stop, end_kind in _split_chunks(flat, kinds, gap):
        if end_kind in _END_KINDS:
            chunk_start_at[stop] = start
    slots: dict[tuple[object, object, object], tuple[object, object]] = {}
    for tag in catalog.list_tags(row.id):
        slots[(tag.folder, tag.start, tag.duration)] = (
            tag.source,
            tag.status,
        )
    notes: list[str] = []
    stored = 0
    for index, (_position, record) in enumerate(flat):
        kind = kinds[index]
        if kind is None:
            continue
        folder, value = _TAGS_FOR_KIND[kind]
        if kind == "mark":
            start = _snap_down(record.start, frame)
            length = frame
        else:
            chunk_start = chunk_start_at.get(index)
            if chunk_start is None:
                notes.append(
                    f"no take before {kind} at {_display_time(record.start)}"
                )
                continue
            bounds = _speech_bounds(flat, kinds, chunk_start, index)
            if bounds is None:
                notes.append(
                    f"no take before {kind} at {_display_time(record.start)}"
                )
                continue
            start = _snap_down(bounds[0], frame)
            end = _snap_up(bounds[1], frame)
            if row.duration is not None and end > row.duration:
                end = row.duration
            length = end - start
            if length <= 0:
                continue
        hit = slots.get((folder, start, length))
        if hit is not None:
            source, status = hit
            if status in ("locked", "deleted") or SOURCE_RANK.index(
                source
            ) > SOURCE_RANK.index(SOURCE):
                notes.append(f"{folder}: kept {source} tag")
                continue
        if dry_run:
            continue
        if catalog.put_tag(
            TagRecord(
                media_id=row.id,
                folder=folder,
                value=value,
                start=start,
                duration=length,
                source=SOURCE,
                confidence=None,
                status="plain",
            )
        ):
            stored += 1
        slots[(folder, start, length)] = (SOURCE, "plain")
    status = "would-tag" if dry_run else "tagged"
    return CalloutEntry(
        path=path,
        asset_id=asset_id,
        status=status,
        stored=stored,
        callouts=len(callouts),
        notes=tuple(notes),
    )


def tag_callouts(
    catalog: Catalog,
    settings: object,
    *,
    gap: Fraction = DEFAULT_GAP,
    dry_run: bool = False,
) -> CalloutResult:
    """Store callout tags for every clip, or report a dry run.

    Each ``clip`` row's newest ``.json`` transcript supplies the word
    times; tags go through :meth:`Catalog.put_tag` only, and a locked,
    deleted or higher-source tag in the same slot is kept instead.
    """
    _require_gap(gap)
    if dry_run:
        entries = [
            _process_clip(catalog, row, gap, dry_run=True)
            for row in catalog.list_media()
            if row.kind == "clip"
        ]
        return CalloutResult(entries=tuple(entries))
    run_id = catalog.start_run(STEP, dry_run=False)
    result: CalloutResult | None = None
    try:
        entries = [
            _process_clip(catalog, row, gap, dry_run=False)
            for row in catalog.list_media()
            if row.kind == "clip"
        ]
        result = CalloutResult(entries=tuple(entries))
        return result
    finally:
        catalog.finish_run(
            run_id, result.outcome if result is not None else "failed"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the callouts step for one trip folder."""
    parser = argparse.ArgumentParser(
        prog="fcp_pipeline.callouts",
        description=(
            "Turn spoken callouts into catalog markers, ratings and takes."
        ),
    )
    parser.add_argument(
        "--trip",
        required=True,
        help="Trip folder holding project.toml.",
    )
    parser.add_argument(
        "--gap",
        default="2",
        help="Longest pause inside one take, in seconds (default 2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be tagged without writing anything.",
    )
    args = parser.parse_args(argv)
    if _GAP_RE.fullmatch(str(args.gap)) is None:
        parser.error(
            f"bad --gap {args.gap!r}: must be a number of seconds above 0"
        )
    try:
        gap = Fraction(str(args.gap))
    except (ValueError, ZeroDivisionError):
        parser.error(
            f"bad --gap {args.gap!r}: must be a number of seconds above 0"
        )
    if gap <= 0:
        parser.error(
            f"bad --gap {args.gap!r}: must be a number of seconds above 0"
        )
    try:
        settings = load_trip(args.trip)
    except TripSettingsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        catalog = Catalog.open(settings)
    except (CatalogError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        try:
            result = tag_callouts(
                catalog, settings, gap=gap, dry_run=args.dry_run
            )
        except CalloutError as exc:
            parser.error(f"bad --gap {args.gap!r}: {exc}")
        except (CatalogError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    finally:
        catalog.close()
    for entry in result.entries:
        for note in entry.notes:
            print(f"note: {entry.path}: {note}")
    print(result.outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
