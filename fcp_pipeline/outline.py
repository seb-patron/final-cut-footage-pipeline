"""Read the editor's outline.md into numbered sections.

:func:`parse_outline` turns an outline's level-2 ATX headings (or level-1
headings when the outline has no level-2 ones) into :class:`Section`
objects, each carrying its wish lines and notes. Section ids come from the
heading text, so reordering the outline keeps every id and changes only
the numbers. A malformed outline raises :class:`OutlineError` naming the
offending line. Nothing here writes anything: the pipeline never edits
outline.md.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

#: A bullet marker: "- ", "* " or "+ " at the start of the stripped line.
BULLET_MARKERS = ("- ", "* ", "+ ")

_HEADING_RE = re.compile(r"^(#{1,6})(?=[ \t]|$)")
_CLOSING_HASHES_RE = re.compile(r"[ \t]+#+[ \t]*$")
_NUMBERED_TITLE_RE = re.compile(r"^\d+[.)][ \t]+")


class OutlineError(ValueError):
    """Raised when an outline cannot be read into sections.

    The message starts with ``"<name> line <N>:"`` naming the offending
    line. :attr:`line` is that 1-based line number.
    """

    def __init__(self, message: str, *, line: int = 1, name: str = "outline.md"):
        super().__init__(message)
        self.name = name
        self.line = line


@dataclass(frozen=True)
class Section:
    """One numbered section of the outline, in heading order."""

    number: int
    id: str
    title: str
    line: int
    wishes: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        """"<number> <title>", for example "1 Cold open"."""
        return f"{self.number} {self.title}"


def _parse_heading(line: str) -> tuple[int, str] | None:
    """Return ``(level, raw_text)`` for an ATX heading line, else None.

    One to six ``#`` characters at the start of the line, then a space,
    a tab, or the end of the line. ``##Title`` and ``####### x`` are
    not headings.
    """
    match = _HEADING_RE.match(line)
    if match is None:
        return None
    return (len(match.group(1)), line[len(match.group(1)):])


def _clean_title(raw: str) -> str:
    """Turn raw heading text into a section title.

    Surrounding spaces go, then a closing run of ``#`` characters
    (spaced off the text, as in ``## Cold open ##``), then one leading
    section number written as digits followed by ``.`` or ``)`` and a
    space (so ``3. Cold open`` becomes ``Cold open`` while
    ``10 things I packed`` keeps its number).
    """
    text = raw.strip()
    if not text or not text.strip(" \t#"):
        return ""
    text = _CLOSING_HASHES_RE.sub("", text).strip()
    numbered = _NUMBERED_TITLE_RE.match(text)
    if numbered is not None:
        text = text[numbered.end():].strip()
    return text


def _slugify(title: str) -> str:
    """Turn a title into an id: lowercase, each run of non-letters and
    non-digits becomes one hyphen, hyphens trimmed from the ends."""
    words = []
    current: list[str] = []
    for char in title.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return "-".join(words)


def parse_outline(text: str, name: str = "outline.md") -> list[Section]:
    """Read outline text into numbered sections, in heading order.

    When the outline has any level-2 heading, the level-2 headings are
    the sections and level-1 headings are ignored; otherwise the
    level-1 headings are the sections. Every bullet line (``- ``,
    ``* `` or ``+ ``, indented or not) up to the next section heading
    is a wish of that section; other non-blank lines there (deeper
    headings and prose) are its notes. Blank lines are dropped.
    """
    lines = text.splitlines()
    levels = [_parse_heading(line) for line in lines]
    section_level = 2 if any(found and found[0] == 2 for found in levels) else 1
    if not any(found and found[0] == section_level for found in levels):
        raise OutlineError(
            f"{name} line 1: no section headings found", line=1, name=name
        )

    sections: list[Section] = []
    seen: dict[str, int] = {}
    current: Section | None = None
    wishes: list[str] = []
    notes: list[str] = []

    def finish() -> None:
        nonlocal current
        if current is not None:
            sections.append(
                Section(
                    number=current.number,
                    id=current.id,
                    title=current.title,
                    line=current.line,
                    wishes=tuple(wishes),
                    notes=tuple(notes),
                )
            )
            current = None
        wishes.clear()
        notes.clear()

    for lineno, line in enumerate(lines, start=1):
        found = levels[lineno - 1]
        if found is not None and found[0] == section_level:
            finish()
            title = _clean_title(found[1])
            if not title:
                raise OutlineError(
                    f"{name} line {lineno}: section heading has an empty title",
                    line=lineno,
                    name=name,
                )
            section_id = _slugify(title)
            if section_id in seen:
                raise OutlineError(
                    f"{name} line {lineno}: duplicate section id "
                    f"{section_id!r} (first used on line {seen[section_id]})",
                    line=lineno,
                    name=name,
                )
            seen[section_id] = lineno
            current = Section(
                number=len(sections) + 1,
                id=section_id,
                title=title,
                line=lineno,
            )
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[:2] in BULLET_MARKERS:
            if current is None:
                raise OutlineError(
                    f"{name} line {lineno}: bullet line before "
                    "the first section heading",
                    line=lineno,
                    name=name,
                )
            wishes.append(stripped[2:].strip())
        elif current is None:
            continue
        elif found is not None and found[0] < section_level:
            continue
        else:
            notes.append(stripped)

    finish()
    return sections


def read_outline(path: str | Path) -> list[Section]:
    """Read the outline file at ``path`` as UTF-8 into sections.

    A byte-order mark is allowed. The file's name is used in error
    messages. The file is never written.
    """
    filepath = Path(path)
    return parse_outline(filepath.read_text(encoding="utf-8-sig"), filepath.name)
