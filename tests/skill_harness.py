"""Offline check and dry-run harness for pipeline skills.

A skill is a folder ``skills/<name>/`` holding ``SKILL.md``. The file
starts with a four-line front matter block (``---``, a ``name:`` line, a
``description:`` line in either order, ``---``) and continues with plain
steps. Runnable commands live in fenced code blocks tagged ``sh``,
``bash``, ``shell`` or untagged; every other fenced block is an example
and is never run.

Only the standard library is used here, plus :mod:`fcp_pipeline.cli`
for command parsing and :mod:`tests.media_fixtures` for the synthetic
trip. Importing this module touches nothing: every read, write and
subprocess happens inside a function call.
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fcp_pipeline.cli import build_parser
from tests import media_fixtures

#: Front-matter keys a shared skill may use. Anything else only works
#: in one tool, so it is reported by :func:`check_skill`.
FRONT_MATTER_KEYS = ("name", "description")

#: Short names for the checks :func:`check_skill` runs. Each names a
#: feature that only one tool supports, so a shared skill must not use
#: it.
TOOL_ONLY_FEATURES = (
    "front-matter key other than name and description",
    "slash command",
    "argument placeholder",
    "shell-injection prefix",
    "tool-specific variable",
    "command that is not a python3 -m fcp_pipeline command",
)

_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_FRONT_KEY_RE = re.compile(r"^([A-Za-z0-9_.-]+)\s*:\s*(.*)$")
_COMMAND_RE = re.compile(r"^python3\s+-m\s+fcp_pipeline(?:\s+(.*))?$")
_FENCE_RE = re.compile(r"^```(.*)$")
_COMMAND_TAGS = ("", "sh", "bash", "shell")
_SLASH_START_RE = re.compile(
    r"^\s*(?:[-*+>]\s*|\d+[.)]\s*)?/([A-Za-z0-9_-]+)(?![A-Za-z0-9_/-])")
_SLASH_TICKS_RE = re.compile(r"`([^`]*)`")
_SLASH_IN_TICKS_RE = re.compile(r"/([A-Za-z0-9_-]+)(?![A-Za-z0-9_/-])")
_PLACEHOLDER_RE = re.compile(r"\$ARGUMENTS|\$[1-9]\b")
_TOOL_VAR_RE = re.compile(r"\$[{]?CLAUDE_|\$[{]?CODEX_")
_TRIP_PLACEHOLDER = "<trip>"


class SkillError(Exception):
    """A skill file breaks a rule.

    The message always names the file, the 1-based line number (0 when
    no line applies) and the rule that was broken.
    """


@dataclass(frozen=True)
class Skill:
    """One parsed skill: its name, description, body and commands.

    ``commands`` holds each command line's arguments after
    ``python3 -m fcp_pipeline``, in document order, as argument lists.
    """

    name: str
    description: str
    body: str
    commands: list


def _unquote(value):
    """Strip one pair of surrounding double quotes from a value."""
    value = value.strip()
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def _iter_body_lines(lines):
    """Yield ``(lineno, in_command_block, text)`` for body lines.

    ``lines`` are the file's lines after the front matter; ``lineno``
    is the 1-based file line number. Fence marker lines themselves are
    skipped.
    """
    in_block = False
    command_block = False
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        fence = _FENCE_RE.match(stripped)
        if fence is not None:
            if not in_block:
                in_block = True
                command_block = fence.group(1).strip() in _COMMAND_TAGS
            else:
                in_block = False
                command_block = False
            continue
        yield (index + 5, in_block and command_block, raw)


def _parse_front_matter(path, lines):
    """Return ``(name, description)`` or raise :class:`SkillError`."""
    if not lines or lines[0].strip() != "---":
        raise SkillError(
            "%s:1: missing front matter: the first line must be '---'" % path)
    close = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            close = index
            break
    if close is None:
        raise SkillError(
            "%s:%d: broken front matter: no closing '---' line" % (path, len(lines)))
    seen = {}
    for offset in range(1, close):
        lineno = offset + 1
        match = _FRONT_KEY_RE.match(lines[offset].strip())
        if match is None:
            raise SkillError(
                "%s:%d: broken front matter: each line must be 'key: value'"
                % (path, lineno))
        seen.setdefault(match.group(1), []).append(
            (lineno, _unquote(match.group(2))))
    for key in FRONT_MATTER_KEYS:
        if key not in seen:
            raise SkillError(
                "%s:2: missing front matter: no '%s: ...' line" % (path, key))
    for key in seen:
        if key not in FRONT_MATTER_KEYS:
            raise SkillError(
                "%s:%d: tool-only front-matter key '%s': a shared skill "
                "allows only 'name' and 'description'" % (path, seen[key][0][0], key))
    if close != 3:
        raise SkillError(
            "%s:2: broken front matter: exactly two lines, 'name: ...' and "
            "'description: ...' in either order, then '---'" % path)
    name_line, name = seen["name"][0]
    desc_line, description = seen["description"][0]
    folder = Path(path).parent.name
    if name != folder:
        raise SkillError(
            "%s:%d: skill name '%s' must equal its folder name '%s'"
            % (path, name_line, name, folder))
    if not _NAME_RE.match(name) or len(name) > 64:
        raise SkillError(
            "%s:%d: skill name '%s' must match ^[a-z0-9]+(-[a-z0-9]+)*$ "
            "and be at most 64 characters" % (path, name_line, name))
    if not description.strip():
        raise SkillError(
            "%s:%d: skill description must be non-empty" % (path, desc_line))
    if len(description) > 500:
        raise SkillError(
            "%s:%d: skill description is %d characters, at most 500 allowed"
            % (path, desc_line, len(description)))
    return name, description


def parse_skill(path):
    """Parse ``path`` into a :class:`Skill`.

    Raises :class:`SkillError` naming the file, the line and the rule
    for any problem.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillError("%s:0: cannot read skill file: %s" % (path, exc))
    except UnicodeError as exc:
        raise SkillError("%s:0: skill file is not UTF-8: %s" % (path, exc))
    lines = text.split("\n")
    name, description = _parse_front_matter(str(path), lines)
    body_lines = lines[4:]
    body = "\n".join(body_lines)
    commands = []
    for lineno, in_command, raw in _iter_body_lines(body_lines):
        stripped = raw.strip()
        if not in_command or not stripped or stripped.startswith("#"):
            continue
        match = _COMMAND_RE.match(stripped)
        if match is None:
            continue
        try:
            commands.append(shlex.split(match.group(1) or ""))
        except ValueError as exc:
            raise SkillError(
                "%s:%d: command line cannot be split: %s" % (path, lineno, exc))
    return Skill(name=name, description=description, body=body,
                 commands=commands)


def check_skill(path):
    """Report tool-only features found in the skill at ``path``.

    Returns a list of messages, one per feature found; an empty list
    means the skill uses only shared features. A file that does not
    parse is reported as a single finding.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return ["%s:0: cannot read skill file: %s" % (path, exc)]
    except UnicodeError as exc:
        return ["%s:0: skill file is not UTF-8: %s" % (path, exc)]
    try:
        parse_skill(path)
    except SkillError as exc:
        return [str(exc)]
    findings = []
    body_lines = text.split("\n")[4:]
    for lineno, in_command, raw in _iter_body_lines(body_lines):
        stripped = raw.strip()
        if not stripped:
            continue
        if in_command and not stripped.startswith("#"):
            if _COMMAND_RE.match(stripped) is None:
                findings.append(
                    "%s:%d: tool-only command %r: a shared skill runs only "
                    "'python3 -m fcp_pipeline ...' lines" % (path, lineno, stripped))
                continue
            # A valid pipeline command line can still carry tool-only
            # markers, so fall through to the checks below instead of
            # skipping them.
        elif in_command:
            continue
        start = _SLASH_START_RE.match(raw)
        if start is not None:
            findings.append(
                "%s:%d: tool-only slash command '/%s': a shared skill "
                "uses plain steps instead" % (path, lineno, start.group(1)))
        for span in _SLASH_TICKS_RE.findall(raw):
            inner = _SLASH_IN_TICKS_RE.search(span)
            if inner is not None:
                findings.append(
                    "%s:%d: tool-only slash command '/%s': a shared skill "
                    "uses plain steps instead" % (path, lineno, inner.group(1)))
        placeholder = _PLACEHOLDER_RE.search(raw)
        if placeholder is not None:
            findings.append(
                "%s:%d: tool-only argument placeholder '%s': a shared skill "
                "spells its arguments out" % (path, lineno, placeholder.group(0)))
        if "!`" in raw:
            findings.append(
                "%s:%d: tool-only shell-injection prefix '!`': a shared skill "
                "runs only 'python3 -m fcp_pipeline ...' lines" % (path, lineno))
        tool_var = _TOOL_VAR_RE.search(raw)
        if tool_var is not None:
            findings.append(
                "%s:%d: tool-only variable '%s': a shared skill uses no "
                "tool variables" % (path, lineno, tool_var.group(0)))
    return findings


def check_commands(skill):
    """Report commands in ``skill`` that fail to parse or run.

    Each command is parsed with ``fcp_pipeline.cli.build_parser()``
    after ``<trip>`` is replaced by a folder name (a ``SystemExit`` is
    reported, naming the command), and ``python3 -m fcp_pipeline
    <step> --help`` is run with :data:`sys.executable` and must exit 0.
    Returns a list of messages; empty means every command is known.
    """
    findings = []
    parser = build_parser()
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(_repository_root())
    for args in skill.commands:
        shown = "python3 -m fcp_pipeline %s" % " ".join(args)
        replaced = [item.replace(_TRIP_PLACEHOLDER, "trip") for item in args]
        try:
            parsed = parser.parse_args(replaced)
        except SystemExit as exc:
            findings.append(
                "command '%s' does not parse (exit %s): unknown step or "
                "bad arguments" % (shown, exc.code))
            continue
        step = parsed.command
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "fcp_pipeline", step, "--help"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=env, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            findings.append(
                "command '%s' has no runnable step '%s': %s" % (shown, step, exc))
            continue
        if completed.returncode != 0:
            findings.append(
                "command '%s' step '%s --help' exits %d" %
                (shown, step, completed.returncode))
    return findings


def load_claude(home, name):
    """Load the skill ``name`` through the Claude Code installed link."""
    return parse_skill(Path(home) / ".claude" / "skills" / name / "SKILL.md")


def load_codex(home, repo, name):
    """Load the skill ``name`` through the Codex pointer and link.

    Requires a line in ``<repo>/AGENTS.md`` naming
    ``skills/<name>/SKILL.md`` (the literal ``skills/<name>/SKILL.md``
    pointer covers every skill), then parses both
    ``<repo>/skills/<name>/SKILL.md`` and
    ``<home>/.codex/skills/<name>/SKILL.md``; both copies must parse
    to the same skill, else :class:`SkillError`.
    """
    agents = Path(repo) / "AGENTS.md"
    try:
        pointer = agents.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillError("%s:0: cannot read Codex pointer: %s" % (agents, exc))
    wanted = "skills/%s/SKILL.md" % name
    if wanted not in pointer and "skills/<name>/SKILL.md" not in pointer:
        raise SkillError(
            "%s:0: Codex pointer names no skill: no line names '%s'"
            % (agents, wanted))
    repo_skill = parse_skill(Path(repo) / "skills" / name / "SKILL.md")
    home_skill = parse_skill(
        Path(home) / ".codex" / "skills" / name / "SKILL.md")
    if repo_skill != home_skill:
        raise SkillError(
            "%s:0: Codex copies differ: the repository copy and the "
            "installed copy under '<home>/.codex/skills/%s' parsed to "
            "different skills" % (repo, name))
    return repo_skill


def _repository_root():
    """Return the repository root holding this harness."""
    return Path(__file__).resolve().parent.parent


def dry_run(skill, trip, *, home):
    """Dry-run each command of ``skill`` on the synthetic ``trip``.

    Commands run in order with :data:`sys.executable` ``-m
    fcp_pipeline``, ``<trip>`` replaced by the trip folder, ``--dry-run``
    and ``--offline`` added when absent, the trip folder as the working
    folder, ``HOME`` set to ``home``, ``PYTHONDONTWRITEBYTECODE=1`` and
    ``PYTHONPATH`` the repository root. Exit 0 and exit 3 (an unbuilt
    step, "not built yet") are accepted; anything else raises
    :class:`SkillError` naming the command. Returns the step names in
    the order they ran.
    """
    trip = Path(trip)
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(_repository_root())
    steps = []
    for args in skill.commands:
        if not args:
            raise SkillError(
                "%s:0: empty command: 'python3 -m fcp_pipeline' names no step"
                % skill.name)
        command = [item.replace(_TRIP_PLACEHOLDER, str(trip)) for item in args]
        if "--dry-run" not in command:
            command.append("--dry-run")
        if "--offline" not in command:
            command.append("--offline")
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "fcp_pipeline"] + command,
                cwd=trip, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            raise SkillError(
                "%s:0: command 'python3 -m fcp_pipeline %s' did not run: %s"
                % (skill.name, " ".join(command), exc))
        if completed.returncode not in (0, 3):
            detail = completed.stderr.decode("utf-8", "replace").strip()
            raise SkillError(
                "%s:0: command 'python3 -m fcp_pipeline %s' exits %d: %s"
                % (skill.name, " ".join(command), completed.returncode, detail))
        steps.append(command[0])
    return steps


def build_synthetic_trip(dest):
    """Build a fresh synthetic trip folder at ``dest`` and return it.

    Copies ``tests/fixtures/trips/synthetic/`` (``project.toml`` and
    ``outline.md``) into ``dest``. When ``ffmpeg`` and ``ffprobe`` are
    on ``PATH``, its ``media/footage`` folder is made with
    :func:`tests.media_fixtures.make_card_tree`; otherwise that folder
    is created empty (the dry run reads no media).
    """
    dest = Path(dest)
    fixture = (_repository_root() / "tests" / "fixtures"
               / "trips" / "synthetic")
    dest.mkdir(parents=True, exist_ok=True)
    for leaf in ("project.toml", "outline.md"):
        shutil.copyfile(fixture / leaf, dest / leaf)
    footage = dest / "media" / "footage"
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        footage.parent.mkdir(parents=True, exist_ok=True)
        media_fixtures.make_card_tree(footage)
    else:
        footage.mkdir(parents=True, exist_ok=True)
    return dest


def _check_one(skill_path):
    """Check and dry-run one skill; return a one-line report and a pass."""
    name = skill_path.parent.name
    try:
        skill = parse_skill(skill_path)
    except SkillError as exc:
        return "FAIL %s: %s" % (name, exc), False
    problems = check_skill(skill_path) + check_commands(skill)
    if problems:
        return "FAIL %s: %s" % (name, "; ".join(problems)), False
    with tempfile.TemporaryDirectory(prefix="skill-dry-run-") as tmp:
        trip = build_synthetic_trip(Path(tmp) / "trip")
        home = Path(tmp) / "home"
        home.mkdir()
        try:
            dry_run(skill, trip, home=home)
        except SkillError as exc:
            return "FAIL %s: %s" % (name, exc), False
    return "ok %s" % name, True


def main(argv=None):
    """Check and dry-run every skill in ``<skills-folder>``.

    Prints one line per skill and exits 0 when every skill passes
    (including when there are none), 1 otherwise.
    """
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 1:
        print("Usage: python3 -m tests.skill_harness <skills-folder>",
              file=sys.stderr)
        return 2
    folder = Path(argv[0])
    if not folder.is_dir():
        print("not a skills folder: %s" % folder, file=sys.stderr)
        return 1
    try:
        skill_paths = sorted(folder.glob("*/SKILL.md"))
    except OSError as exc:
        print("cannot list skills folder %s: %s" % (folder, exc),
              file=sys.stderr)
        return 1
    if not skill_paths:
        print("no skills found")
        return 0
    failed = 0
    for skill_path in skill_paths:
        line, passed = _check_one(skill_path)
        print(line)
        if not passed:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
