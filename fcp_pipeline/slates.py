"""Find spoken slates in stored transcripts and tag their parts.

A spoken slate is the sentence a clip starts (or breaks) with, shaped
like ``Slate. Day 3, <place>, B-roll, Portra 400 on the F3``. The word
``"slate"`` counts only when announcing punctuation or a recognized
part follows it, so an ordinary ``Slate roofs last long.`` is not a
slate and stores nothing. This step
reads the transcripts already in the catalog (run transcribe first) and
listens to nothing: for each clip it finds the slate sentence, turns
each recognized part into a whole-clip catalog tag with source
``"slate"``, and saves the slate sentence itself as a ``"Slate"`` tag.
Place, stock and camera words are fuzzy-matched to the trip's own words
so a misheard ``"Porter 400"`` becomes ``"Portra 400"``; a part that
cannot be matched is kept as heard with status ``"check"``. Tags from
metadata, GPS or the editor, and locked tags, are never replaced.
Importing this module runs no tool, opens no database and touches no
file.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass
from fractions import Fraction

from fcp_pipeline.catalog import SOURCE_RANK, Catalog, CatalogError, TagRecord
from fcp_pipeline.trip import TripSettingsError, load_trip

__all__ = [
    "CAMERA_FOLDER",
    "DAY_FOLDER",
    "FILM_STOCKS",
    "FUZZY_CUTOFF",
    "MAX_SLATE_CHARS",
    "PLACE_FOLDER",
    "SHOT_FOLDER",
    "SHOT_TYPES",
    "SLATE_FOLDER",
    "SLATE_REACH",
    "SOURCE",
    "STEP",
    "STOCK_FOLDER",
    "SlateEntry",
    "SlateError",
    "SlatePart",
    "SlateResult",
    "build_vocabulary",
    "find_slate",
    "main",
    "match_word",
    "normalize",
    "parse_slate",
    "tag_slates",
]

#: Step name recorded in the catalog's runs table.
STEP = "slates"

#: Tag source this step writes.
SOURCE = "slate"

#: Tag folders this step writes.
DAY_FOLDER = "Day"
PLACE_FOLDER = "Place"
SHOT_FOLDER = "Shot"
STOCK_FOLDER = "Stock"
CAMERA_FOLDER = "Camera"
SLATE_FOLDER = "Slate"

#: Gear folder written by the days-and-places step, read here for cameras.
_GEAR_FOLDER = "Gear"

#: Minimum SequenceMatcher ratio for a fuzzy word match.
FUZZY_CUTOFF = 0.8

#: How far after a lone "Slate." the next segment may start to count.
SLATE_REACH = Fraction(5)

#: Longest slate sentence kept; anything past it is cut off.
MAX_SLATE_CHARS = 300

#: Longest heard stock or camera spelling kept; anything past it is cut.
_MAX_HEARD_CHARS = 80

#: Shot words this step recognizes, in canonical spelling.
SHOT_TYPES = (
    "A-roll",
    "B-roll",
    "Interview",
    "Establishing",
    "Insert",
    "Pickup",
    "Time-lapse",
)

#: Generic film stock names this step recognizes, in canonical spelling.
FILM_STOCKS = (
    "Portra 160",
    "Portra 400",
    "Portra 800",
    "Ektar 100",
    "Gold 200",
    "UltraMax 400",
    "Tri-X 400",
    "HP5 Plus 400",
    "Vision3 500T",
    "Vision3 250D",
    "Cinestill 800T",
    "Velvia 50",
    "Provia 100F",
)

#: Vocabulary keys, in the order they are documented.
_VOCAB_KEYS = ("place", "stock", "camera", "shot")

#: Reason stored when a slate word parses to no recognized part.
_NO_PARTS_REASON = "slate word with no recognizable parts"


class SlateError(ValueError):
    """Raised for bad input to the slates step."""


@dataclass(frozen=True)
class SlatePart:
    """One recognized slate piece: a folder, its value and its status."""

    folder: str
    value: str
    status: str


@dataclass(frozen=True)
class SlateEntry:
    """What the step did with one clip."""

    path: str
    asset_id: str
    status: str
    reason: str | None = None
    stored: int = 0
    text: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SlateResult:
    """Every entry of one run, in catalog order."""

    entries: tuple[SlateEntry, ...] = ()

    @property
    def tagged(self) -> int:
        """How many clips got slate tags."""
        return sum(1 for entry in self.entries if entry.status == "tagged")

    @property
    def no_slate(self) -> int:
        """How many clips hold no usable slate."""
        return sum(1 for entry in self.entries if entry.status == "no-slate")

    @property
    def no_transcript(self) -> int:
        """How many clips hold no transcript to read."""
        return sum(
            1 for entry in self.entries if entry.status == "no-transcript"
        )

    @property
    def would_tag(self) -> int:
        """How many clips a dry run would tag."""
        return sum(1 for entry in self.entries if entry.status == "would-tag")

    @property
    def outcome(self) -> str:
        """One summary line for the run."""
        line = (
            f"tagged {self.tagged}, no-slate {self.no_slate}, "
            f"no-transcript {self.no_transcript}"
        )
        if self.would_tag:
            line = f"would-tag {self.would_tag}, " + line
        return line


def normalize(text: str) -> str:
    """Casefold text, collapsing every non-alphanumeric run to one space."""
    folded = text.casefold()
    spaced = "".join(char if char.isalnum() else " " for char in folded)
    return re.sub(r"\s+", " ", spaced).strip()


_DIGIT_RUN_RE = re.compile(r"\d+")


def match_word(
    heard: str, candidates, *, cutoff: float = FUZZY_CUTOFF
) -> str | None:
    """Fuzzy-match a heard word against the trip's own spellings.

    Both sides compare on keys (the normalized text with spaces
    removed). The digit runs of both keys must match exactly, so
    ``"Portra 400"`` never matches ``"Portra 800"``, and the
    :class:`~difflib.SequenceMatcher` ratio must reach ``cutoff``.
    Returns the single best candidate as given, or None when nothing
    reaches the cutoff or two different candidates tie for best. An
    exact key match always wins.
    """
    heard_key = normalize(heard).replace(" ", "")
    heard_digits = _DIGIT_RUN_RE.findall(heard_key)
    best: str | None = None
    best_ratio = 0.0
    tied = False
    for candidate in candidates:
        candidate_key = normalize(candidate).replace(" ", "")
        if _DIGIT_RUN_RE.findall(candidate_key) != heard_digits:
            continue
        if candidate_key == heard_key:
            return candidate
        ratio = difflib.SequenceMatcher(
            None, heard_key, candidate_key
        ).ratio()
        if ratio < cutoff:
            continue
        if ratio > best_ratio:
            best = candidate
            best_ratio = ratio
            tied = False
        elif ratio == best_ratio and candidate != best:
            tied = True
    if best is None or tied:
        return None
    return best


def _dedupe_keep_first(values) -> tuple[str, ...]:
    """Drop repeats, keeping the first spelling of each word."""
    seen: set[str] = set()
    kept: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        key = normalize(value).replace(" ", "")
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(value)
    return tuple(kept)


def _check_extra(extra) -> dict[str, list[str]]:
    """Validate the ``extra`` vocabulary mapping, returning it unchanged."""
    if not isinstance(extra, dict):
        raise SlateError(
            "bad vocabulary extra: must be a mapping with keys "
            "'place', 'stock', 'camera' and 'shot'"
        )
    for key, value in extra.items():
        if key not in _VOCAB_KEYS:
            raise SlateError(
                f"bad vocabulary extra key {key!r}: must be one of "
                "'place', 'stock', 'camera' and 'shot'"
            )
        if not isinstance(value, list):
            raise SlateError(
                f"bad vocabulary extra key {key!r}: must be a list of "
                "non-empty strings"
            )
        for item in value:
            if not isinstance(item, str) or not item:
                raise SlateError(
                    f"bad vocabulary extra key {key!r}: must be a list of "
                    "non-empty strings"
                )
    return extra


def build_vocabulary(catalog, settings, extra=None) -> dict[str, tuple[str, ...]]:
    """Collect the place, stock, camera and shot words for a trip.

    Places come from ``settings.places`` names first, then the catalog's
    media places, then existing ``"Place"`` tags; stocks are the generic
    film list plus existing ``"Stock"`` tags; cameras are the catalog's
    media cameras plus existing ``"Gear"`` and ``"Camera"`` tags; shots
    are the built-in shot list. ``extra`` lists are appended, never a
    replacement. Duplicates keep the first spelling.
    """
    checked = _check_extra(extra) if extra is not None else {}
    raw_places = getattr(settings, "places", []) or []
    settings_places = [
        entry["name"]
        for entry in raw_places
        if isinstance(entry, dict)
        and isinstance(entry.get("name"), str)
        and entry["name"]
    ]
    media_places: list[str] = []
    media_cameras: list[str] = []
    place_tags: list[str] = []
    stock_tags: list[str] = []
    camera_tags: list[str] = []
    for row in catalog.list_media():
        if isinstance(row.place, str) and row.place:
            media_places.append(row.place)
        if isinstance(row.camera, str) and row.camera:
            media_cameras.append(row.camera)
        for tag in catalog.list_tags(row.id):
            if not isinstance(tag.value, str) or not tag.value:
                continue
            if tag.folder == PLACE_FOLDER:
                place_tags.append(tag.value)
            elif tag.folder == STOCK_FOLDER:
                stock_tags.append(tag.value)
            elif tag.folder in (_GEAR_FOLDER, CAMERA_FOLDER):
                camera_tags.append(tag.value)
    return {
        "place": _dedupe_keep_first(
            [*settings_places, *media_places, *place_tags, *checked.get("place", [])]
        ),
        "stock": _dedupe_keep_first(
            [*FILM_STOCKS, *stock_tags, *checked.get("stock", [])]
        ),
        "camera": _dedupe_keep_first(
            [*media_cameras, *camera_tags, *checked.get("camera", [])]
        ),
        "shot": _dedupe_keep_first(
            [*SHOT_TYPES, *checked.get("shot", [])]
        ),
    }


_SLATE_WORD_RE = re.compile(r"\bslate\b", re.IGNORECASE)

#: What may stand right before a slate word inside a segment's text.
_SENTENCE_ENDS = (". ", "! ", "? ")

#: Punctuation (and space) stripped straight after the slate word.
_LEADING_PUNCT_RE = re.compile(r"^[\s.,!?;:'\"()\[\]\-–—]+")

#: Punctuation right after the word "slate" that announces a slate:
#: a sentence break, a comma or a colon.
_SLATE_ANNOUNCE_PUNCT = (".", "!", "?", ",", ":")

#: Folders that count as recognized slate parts. A leftover Place tag
#: stands only beside one of these; anything else stores nothing.
_RECOGNIZED_FOLDERS = ("Day", "Shot", "Stock", "Camera")


def _starts_with_recognized_part(text: str, vocabulary) -> bool:
    """Whether slate text opens with a genuinely recognized slate part.

    Only the first event counts: the part must stand directly after
    the word "slate". Leftover and Place parts never count, and neither
    do unmatched stock or camera spellings kept as heard.
    """
    for kind, payload in _parse_events(text, vocabulary):
        if kind != "part":
            return False
        return (
            payload.folder in _RECOGNIZED_FOLDERS
            and payload.status == "plain"
        )
    return False


def _slate_occurrences(
    segments, vocabulary=None
) -> list[tuple[str, Fraction]]:
    """Every slate start in a clip: its sentence text and its start time.

    The word "slate" counts only when announcing punctuation (a
    sentence break, a comma or a colon) follows it, or when a
    recognized day, shot, stock or camera stands directly after it;
    "slate" used as an ordinary word ("Slate roofs last long.") is
    skipped. Without a vocabulary only Day N can announce this way.
    """
    found: list[tuple[str, Fraction]] = []
    total = len(segments)
    for index, segment in enumerate(segments):
        text = segment.text or ""
        for match in _SLATE_WORD_RE.finditer(text):
            start = match.start()
            if start == 0 or not text[:start].strip():
                valid = True
            elif start >= 2 and text[start - 2 : start] in _SENTENCE_ENDS:
                valid = True
            else:
                continue
            after = text[match.end() :]
            if after[:1] in _SLATE_ANNOUNCE_PUNCT:
                announced = True
            else:
                stripped = _LEADING_PUNCT_RE.sub("", after).strip()
                announced = not stripped or _starts_with_recognized_part(
                    stripped, vocabulary
                )
            if not announced:
                continue
            slate = _LEADING_PUNCT_RE.sub("", after).strip()
            slate = slate[:MAX_SLATE_CHARS]
            if not slate and index + 1 < total:
                following = segments[index + 1]
                if (
                    following.start is not None
                    and segment.stop is not None
                    and following.start - segment.stop <= SLATE_REACH
                ):
                    slate = (following.text or "").strip()[:MAX_SLATE_CHARS]
            if segment.start is not None:
                moment = segment.start
            else:
                moment = Fraction(0)
            found.append((slate, moment))
    return found


def find_slate(segments, vocabulary=None) -> tuple[str, Fraction] | None:
    """Return a clip's first slate sentence and its start time, or None.

    A slate starts at the word ``"slate"`` (case-insensitive, whole
    word) at the start of a segment's text or right after a sentence
    end inside it, when announcing punctuation (a sentence break, a
    comma or a colon) or a recognized day, shot, stock or camera
    follows the word. The text runs from that word on, less the word
    itself and any punctuation straight after it; when nothing is left
    the text is the next segment's text, if it starts soon enough.
    """
    occurrences = _slate_occurrences(segments, vocabulary)
    if not occurrences:
        return None
    return occurrences[0]


_NUMBER_WORDS: dict[str, int] = {}
for _number, _word in enumerate(
    ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
    1,
):
    _NUMBER_WORDS[_word] = _number
for _number, _word in enumerate(
    [
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ],
    10,
):
    _NUMBER_WORDS[_word] = _number
_NUMBER_WORDS["twenty"] = 20
_NUMBER_WORDS["thirty"] = 30
for _tens_word, _base in (("twenty", 20), ("thirty", 30)):
    for _number, _word in enumerate(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
        1,
    ):
        _NUMBER_WORDS[f"{_tens_word} {_word}"] = _base + _number
_NUMBER_WORDS["thirty one"] = 31

_DAY_RE = re.compile(
    r"day\s+(\d{1,3}|"
    + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
    + r")\b"
)

#: Slate pieces split at commas, semicolons and full stops.
_PIECE_SPLIT_RE = re.compile(r"[,;.]+")

#: The last " on the " (else the last " on ") starts the camera piece.
_ON_SPLIT_RE = re.compile(r"\s+on\s+the\s+|\s+on\s+", re.IGNORECASE)


def _vocabulary_lists(vocabulary) -> tuple[list, list, list, list]:
    """Return the place, stock, camera and shot lists from a vocabulary."""
    source = vocabulary or {}
    return (
        list(source.get("place", ()) or ()),
        list(source.get("stock", ()) or ()),
        list(source.get("camera", ()) or ()),
        list(source.get("shot", ()) or ()),
    )


def _parse_events(text: str, vocabulary) -> list:
    """Split slate text into an ordered event list.

    Each event is ``("part", SlatePart)`` or ``("ignored", text)``: the
    first leftover piece becomes a ``"check"`` Place part, but only
    beside a genuinely recognized day, shot, stock or camera part;
    without one it stays ``"ignored"``, like every further leftover.
    """
    place_list, stock_list, camera_list, shot_list = _vocabulary_lists(
        vocabulary
    )
    events: list = []
    leftover_seen = False
    leftover_index: int | None = None
    for raw_piece in _PIECE_SPLIT_RE.split(text):
        piece = raw_piece.strip()
        if not piece:
            continue
        stock_heard: str | None = None
        camera_heard: str | None = None
        on_matches = list(_ON_SPLIT_RE.finditer(piece))
        if on_matches:
            last = on_matches[-1]
            left = piece[: last.start()].strip()
            right = piece[last.end() :].strip()
            if left and right:
                stock_heard = left
                camera_heard = right
        if stock_heard is not None and camera_heard is not None:
            stock_hit = match_word(stock_heard, stock_list)
            if stock_hit is not None:
                events.append(
                    ("part", SlatePart(STOCK_FOLDER, stock_hit, "plain"))
                )
            else:
                events.append(
                    (
                        "part",
                        SlatePart(
                            STOCK_FOLDER,
                            stock_heard[:_MAX_HEARD_CHARS],
                            "check",
                        ),
                    )
                )
            camera_hit = match_word(camera_heard, camera_list)
            if camera_hit is not None:
                events.append(
                    ("part", SlatePart(CAMERA_FOLDER, camera_hit, "plain"))
                )
            else:
                events.append(
                    (
                        "part",
                        SlatePart(
                            CAMERA_FOLDER,
                            camera_heard[:_MAX_HEARD_CHARS],
                            "check",
                        ),
                    )
                )
            continue
        day_match = _DAY_RE.search(normalize(piece))
        if day_match is not None:
            token = day_match.group(1)
            if token.isdigit():
                number = int(token)
            else:
                number = _NUMBER_WORDS[token]
            events.append(
                ("part", SlatePart(DAY_FOLDER, f"Day {number}", "plain"))
            )
            continue
        shot_hit = match_word(piece, shot_list)
        if shot_hit is not None:
            events.append(("part", SlatePart(SHOT_FOLDER, shot_hit, "plain")))
            continue
        place_hit = match_word(piece, place_list)
        if place_hit is not None:
            events.append(("part", SlatePart(PLACE_FOLDER, place_hit, "plain")))
            continue
        if not leftover_seen:
            leftover_seen = True
            leftover_index = len(events)
            events.append(("part", SlatePart(PLACE_FOLDER, piece, "check")))
        else:
            events.append(("ignored", piece))
    if leftover_index is not None and not any(
        kind == "part"
        and payload.folder in _RECOGNIZED_FOLDERS
        and payload.status == "plain"
        for kind, payload in events
    ):
        _kind, payload = events[leftover_index]
        events[leftover_index] = ("ignored", payload.value)
    return events


def parse_slate(text: str, vocabulary) -> list[SlatePart]:
    """Turn slate text into one part per folder, the first spelling wins."""
    parts: list[SlatePart] = []
    seen: set[str] = set()
    for kind, payload in _parse_events(text, vocabulary):
        if kind != "part":
            continue
        if payload.folder in seen:
            continue
        seen.add(payload.folder)
        parts.append(payload)
    return parts


def _outranks_slate(source: str) -> bool:
    """Whether an existing tag source beats a spoken slate."""
    return SOURCE_RANK.index(source) > SOURCE_RANK.index(SOURCE)


def _tag_clip(catalog, row, vocabulary, *, dry_run: bool) -> SlateEntry:
    """Tag one clip's slate, returning its entry."""
    notes: list[str] = []
    records = catalog.list_transcripts(row.id)
    newest_json = None
    for record in records:
        if (record.path or "").endswith(".json"):
            newest_json = record
    if newest_json is None:
        return SlateEntry(
            path=row.path, asset_id=row.asset_id, status="no-transcript"
        )
    segments = catalog.list_transcript_segments(newest_json.id)
    if not segments:
        return SlateEntry(
            path=row.path, asset_id=row.asset_id, status="no-transcript"
        )
    occurrences = _slate_occurrences(segments, vocabulary)
    if not occurrences:
        return SlateEntry(path=row.path, asset_id=row.asset_id, status="no-slate")
    text, _moment = occurrences[0]
    if len(occurrences) > 1:
        notes.append("second slate ignored")
    planned: list[SlatePart] = []
    seen: set[str] = set()
    for kind, payload in _parse_events(text, vocabulary):
        if kind == "ignored":
            notes.append(f"ignored: {payload}")
            continue
        if payload.folder in seen:
            notes.append(f"duplicate {payload.folder} ignored: {payload.value}")
            continue
        seen.add(payload.folder)
        planned.append(payload)
    if not any(
        part.folder in _RECOGNIZED_FOLDERS and part.status == "plain"
        for part in planned
    ):
        return SlateEntry(
            path=row.path,
            asset_id=row.asset_id,
            status="no-slate",
            reason=_NO_PARTS_REASON,
            stored=0,
            text=text or None,
            notes=tuple(notes),
        )
    planned.append(SlatePart(SLATE_FOLDER, text, "plain"))
    stored = 0
    if not dry_run:
        slots = {
            (tag.folder, tag.start, tag.duration): tag
            for tag in catalog.list_tags(row.id)
        }
        for part in planned:
            old = slots.get((part.folder, None, None))
            if old is not None and (
                old.status in ("locked", "deleted") or _outranks_slate(old.source)
            ):
                notes.append(f"{part.folder} kept {old.source} tag")
                continue
            if catalog.put_tag(
                TagRecord(
                    media_id=row.id,
                    folder=part.folder,
                    value=part.value,
                    start=None,
                    duration=None,
                    source=SOURCE,
                    confidence=None,
                    status=part.status,
                )
            ):
                stored += 1
        status = "tagged"
    else:
        status = "would-tag"
    return SlateEntry(
        path=row.path,
        asset_id=row.asset_id,
        status=status,
        reason=None,
        stored=stored,
        text=text,
        notes=tuple(notes),
    )


def tag_slates(
    catalog, settings, *, vocabulary=None, dry_run: bool = False
) -> SlateResult:
    """Find every clip's spoken slate and store its parts as slate tags.

    Every tag is whole-clip with source ``"slate"`` and goes in through
    :meth:`catalog.put_tag` only; a tag already held by a locked tag or
    a higher-ranked source is kept and noted. Day, Place, Shot, Stock
    and Camera tags never change any media row. With ``dry_run`` True
    nothing is written and no run is recorded.
    """
    vocab = build_vocabulary(catalog, settings) if vocabulary is None else vocabulary
    if dry_run:
        entries = tuple(
            _tag_clip(catalog, row, vocab, dry_run=True)
            for row in catalog.list_media()
            if row.kind == "clip"
        )
        return SlateResult(entries=entries)
    run_id = catalog.start_run(STEP, dry_run=False)
    outcome = "failed"
    try:
        entries = tuple(
            _tag_clip(catalog, row, vocab, dry_run=False)
            for row in catalog.list_media()
            if row.kind == "clip"
        )
        result = SlateResult(entries=entries)
        outcome = result.outcome
        return result
    finally:
        catalog.finish_run(run_id, outcome)


def _load_vocabulary_file(path: str) -> dict[str, list[str]]:
    """Read a ``--vocabulary`` JSON file into an ``extra`` mapping."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise SlateError(
            f"bad vocabulary file '{path}': cannot read it: {exc}"
        ) from None
    except ValueError as exc:
        raise SlateError(
            f"bad vocabulary file '{path}': invalid JSON: {exc}"
        ) from None
    if not isinstance(data, dict):
        raise SlateError(
            f"bad vocabulary file '{path}': must be a JSON object with keys "
            "'place', 'stock', 'camera' and 'shot'"
        )
    for key, value in data.items():
        if key not in _VOCAB_KEYS:
            raise SlateError(
                f"bad vocabulary file '{path}': unknown key {key!r}"
            )
        if not isinstance(value, list):
            raise SlateError(
                f"bad vocabulary file '{path}': key {key!r} must be an "
                "array of strings"
            )
        for item in value:
            if not isinstance(item, str) or not item:
                raise SlateError(
                    f"bad vocabulary file '{path}': key {key!r} must be an "
                    "array of non-empty strings"
                )
    return data


def main(argv: list[str] | None = None) -> int:
    """Run the slates step for one trip folder."""
    parser = argparse.ArgumentParser(
        prog="fcp_pipeline.slates",
        description=(
            "Find spoken slates in stored transcripts and tag their parts."
        ),
    )
    parser.add_argument(
        "--trip",
        required=True,
        help="Trip folder holding project.toml.",
    )
    parser.add_argument(
        "--vocabulary",
        default=None,
        help="JSON file with place, stock, camera and shot word lists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be tagged without writing anything.",
    )
    args = parser.parse_args(argv)
    try:
        settings = load_trip(args.trip)
    except (TripSettingsError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    extra = None
    if args.vocabulary is not None:
        try:
            extra = _load_vocabulary_file(args.vocabulary)
        except (SlateError, OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    try:
        catalog = Catalog.open(settings)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    result: SlateResult | None = None
    try:
        try:
            if extra is not None:
                vocabulary = build_vocabulary(catalog, settings, extra=extra)
            else:
                vocabulary = None
            result = tag_slates(
                catalog, settings, vocabulary=vocabulary, dry_run=args.dry_run
            )
        except (SlateError, CatalogError, OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    finally:
        catalog.close()
    assert result is not None
    for entry in result.entries:
        for note in entry.notes:
            print(f"note: {entry.path}: {note}")
    print(result.outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
