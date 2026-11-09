"""Offline FCPXML check.

Reads one exported ``.fcpxml`` file, or the ``Info.fcpxml`` inside an
``.fcpxmld`` bundle, through the hardened reader in
:mod:`fcp_pipeline.fcpxml_read`, validates the bytes with ``xmllint``
against the DTD that ships inside the installed Final Cut Pro for the
file's own declared version (1.0 to 1.14), checks that every referenced
media file exists, lists the event and project names, and returns the
SHA-256 of the bytes that were read.

The check is offline and read-only: it never contacts, launches or
scripts Final Cut Pro or Motion, imports nothing, writes nothing, and
uses no network (``xmllint`` runs with ``--nonet`` and gets the
already-read bytes on stdin). When the DTD folder, the version's DTD
file, or ``xmllint`` is missing, the result is "cannot verify", never
"valid".
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from xml.parsers import expat

from fcp_pipeline.fcpxml_read import parse_xml_bytes, xml_bytes

VALID = "valid"
INVALID = "invalid"
REFUSED = "refused"
CANNOT_VERIFY = "cannot verify"

DTD_DIR_ENV = "FCP_DTD_DIR"
DEFAULT_DTD_DIR = (
    "/Applications/Final Cut Pro.app/Contents/Frameworks/"
    "Interchange.framework/Versions/A/Resources"
)

_XMLLINT_TIMEOUT_SECONDS = 60

_VERSION_RE = re.compile(r"\d+\.\d+")
_VALIDITY_RE = re.compile(r"^-:(\d+): element (\S+): validity error : ?(.*)$")


@dataclass
class Problem:
    """One thing the check found, with its element and line when known."""

    message: str
    element: str | None = None
    line: int | None = None


@dataclass
class CheckResult:
    """The outcome of :func:`check_fcpxml`."""

    path: str | Path
    status: str
    ok: bool
    sha256: str | None
    version: str | None
    dtd: Path | None
    events: list = field(default_factory=list)
    projects: list = field(default_factory=list)
    missing_media: list = field(default_factory=list)
    problems: list = field(default_factory=list)


def dtd_folder():
    """Return the folder holding the installed Final Cut DTDs."""
    override = os.environ.get(DTD_DIR_ENV)
    if override:
        return Path(override)
    return Path(DEFAULT_DTD_DIR)


def _error_line(exc):
    """Return the 1-based line number carried by a parse error, if any."""
    line = getattr(exc, "lineno", None)
    if isinstance(line, int):
        return line
    position = getattr(exc, "position", None)
    if isinstance(position, tuple) and position and isinstance(position[0], int):
        return position[0]
    return None


def _missing_media_entry(src):
    """Return the missing-media entry for one ``src`` value, or None.

    A ``file:`` URL with an empty or ``localhost`` host is decoded and
    its path must exist. Any other value (another scheme, a relative
    value, or a value whose path cannot be decoded) cannot be checked
    locally, so the raw value itself is reported.
    """
    try:
        split = urllib.parse.urlsplit(src)
    except ValueError:
        return src
    if split.scheme != "file" or split.netloc not in ("", "localhost"):
        return src
    try:
        decoded = urllib.parse.unquote(split.path)
    except Exception:
        return src
    if os.path.exists(decoded):
        return None
    return decoded


def _version_parts(version):
    """Split a well-formed ``digits.digits`` version into ints."""
    major, _, minor = version.partition(".")
    return int(major), int(minor)


def check_fcpxml(path, *, dtd_dir=None, dtd_path=None, check_media=True):
    """Check one ``.fcpxml`` file or ``.fcpxmld`` folder, read-only.

    ``dtd_dir`` None means :func:`dtd_folder`. ``dtd_path``, when given,
    names the one DTD file to validate against whatever the file's
    version. Returns a :class:`CheckResult`; never raises for anything
    about the file's content or the machine.
    """
    problems: list = []
    missing_media: list = []
    events: list = []
    projects: list = []
    version = None
    sha256 = None

    given = Path(path)
    target = given
    if given.suffix == ".fcpxmld" and given.is_dir():
        target = given / "Info.fcpxml"
    try:
        raw = xml_bytes(target)
    except ValueError as exc:
        problems.append(Problem(message=str(exc) or "unsupported input"))
        return CheckResult(
            path=path, status=REFUSED, ok=False, sha256=None,
            version=None, dtd=None, problems=problems,
        )
    except OSError as exc:
        problems.append(Problem(message="cannot read %s: %s" % (path, exc)))
        return CheckResult(
            path=path, status=INVALID, ok=False, sha256=None,
            version=None, dtd=None, problems=problems,
        )
    sha256 = hashlib.sha256(raw).hexdigest()

    try:
        root = parse_xml_bytes(raw)
    except ValueError as exc:
        problems.append(Problem(message=str(exc) or "refused input"))
        return CheckResult(
            path=path, status=REFUSED, ok=False, sha256=sha256,
            version=None, dtd=None, problems=problems,
        )
    except (expat.ExpatError, ET.ParseError) as exc:
        problems.append(
            Problem(message=str(exc) or "malformed XML",
                    line=_error_line(exc))
        )
        return CheckResult(
            path=path, status=INVALID, ok=False, sha256=sha256,
            version=None, dtd=None, problems=problems,
        )

    raw_version = root.get("version")
    version_valid = (
        isinstance(raw_version, str)
        and _VERSION_RE.fullmatch(raw_version) is not None
    )
    in_range = False
    if not version_valid:
        if raw_version is None:
            problems.append(
                Problem(message="missing version attribute on fcpxml root")
            )
        else:
            problems.append(
                Problem(
                    message="malformed version %r; expected digits.digits "
                    "like 1.14" % (raw_version,)
                )
            )
    else:
        version = raw_version
        major, minor = _version_parts(version)
        in_range = major == 1 and 0 <= minor <= 14
        if dtd_path is None and not in_range:
            problems.append(
                Problem(
                    message="unsupported FCPXML version %s; supported "
                    "versions are 1.0 to 1.14" % (version,)
                )
            )

    for element in root.iter("event"):
        name = element.get("name")
        if name is not None:
            events.append(name)
    for element in root.iter("project"):
        name = element.get("name")
        if name is not None:
            projects.append(name)

    if check_media:
        for element in root.iter():
            if element.tag not in ("media-rep", "asset"):
                continue
            src = element.get("src")
            if src is None:
                continue
            entry = _missing_media_entry(src)
            if entry is not None and entry not in missing_media:
                missing_media.append(entry)

    invalid = not version_valid or bool(missing_media)
    cannot_verify = dtd_path is None and version_valid and not in_range

    dtd = Path(dtd_path) if dtd_path is not None else None
    dtd_file = None
    if dtd is not None:
        if not dtd.is_file():
            problems.append(
                Problem(message="DTD file not found: %s" % (dtd,))
            )
            cannot_verify = True
        elif shutil.which("xmllint") is None:
            problems.append(
                Problem(
                    message="xmllint not on PATH; cannot validate %s" % (path,)
                )
            )
            cannot_verify = True
        else:
            dtd_file = dtd
    elif version_valid and in_range:
        folder = Path(dtd_dir) if dtd_dir is not None else dtd_folder()
        _, minor = _version_parts(version)
        dtd = folder / ("FCPXMLv1_%d.dtd" % minor)
        if not folder.is_dir():
            problems.append(
                Problem(message="DTD folder not found: %s" % (folder,))
            )
            cannot_verify = True
        elif not dtd.is_file():
            problems.append(
                Problem(message="DTD file not found: %s" % (dtd,))
            )
            cannot_verify = True
        elif shutil.which("xmllint") is None:
            problems.append(
                Problem(
                    message="xmllint not on PATH; cannot validate %s" % (path,)
                )
            )
            cannot_verify = True
        else:
            dtd_file = dtd

    if dtd_file is not None:
        xmllint = shutil.which("xmllint")
        try:
            completed = subprocess.run(
                [xmllint, "--noout", "--nonet", "--dtdvalid",
                 dtd_file.resolve().as_uri(), "-"],
                input=raw,
                capture_output=True,
                timeout=_XMLLINT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            problems.append(
                Problem(message="xmllint timed out validating %s" % (path,))
            )
            cannot_verify = True
        except OSError as exc:
            problems.append(
                Problem(message="cannot run xmllint: %s" % (exc,))
            )
            cannot_verify = True
        else:
            stderr = completed.stderr.decode("utf-8", "replace")
            if completed.returncode != 0:
                found = []
                for line in stderr.splitlines():
                    match = _VALIDITY_RE.match(line)
                    if match is not None:
                        lineno, element, message = match.groups()
                        found.append(
                            Problem(message=message, element=element,
                                    line=int(lineno))
                        )
                if found:
                    problems.extend(found)
                else:
                    output = stderr.strip()
                    if not output:
                        output = completed.stdout.decode(
                            "utf-8", "replace").strip()
                    if not output:
                        output = "xmllint exited %d validating %s" % (
                            completed.returncode, path)
                    problems.append(Problem(message=output))
                invalid = True

    if invalid:
        status = INVALID
    elif cannot_verify:
        status = CANNOT_VERIFY
    else:
        status = VALID
    return CheckResult(
        path=path, status=status, ok=(status == VALID), sha256=sha256,
        version=version, dtd=dtd, events=events, projects=projects,
        missing_media=missing_media, problems=problems,
    )


def _format_problem(problem):
    """Render one problem with its line and element when known."""
    bits = []
    if problem.line is not None:
        bits.append("line %s" % (problem.line,))
    if problem.element is not None:
        bits.append("element %s" % (problem.element,))
    bits.append(problem.message)
    return ": ".join(bits)


def main(argv=None):
    """Run the offline check on one file; return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="fcp_pipeline.bridge.check",
        description="Offline FCPXML check: validate one exported .fcpxml "
        "file or .fcpxmld folder against the installed Final Cut DTD, "
        "list missing media, events and projects, and print its SHA-256. "
        "Reads the one input file only; writes nothing.",
    )
    parser.add_argument(
        "path",
        help=".fcpxml file, or .fcpxmld folder whose Info.fcpxml is read",
    )
    args = parser.parse_args(argv)
    result = check_fcpxml(args.path)
    print(result.status)
    print("path: %s" % (result.path,))
    print("version: %s" % (result.version
                            if result.version is not None else "unknown"))
    print("dtd: %s" % (result.dtd if result.dtd is not None else "none"))
    print("sha256: %s" % (result.sha256
                           if result.sha256 is not None else "unknown"))
    print("events: %s" % (", ".join(result.events),))
    print("projects: %s" % (", ".join(result.projects),))
    if result.missing_media:
        print("missing media:")
        for entry in result.missing_media:
            print("  %s" % (entry,))
    else:
        print("missing media: none")
    if result.problems:
        print("problems:")
        for problem in result.problems:
            print("  %s" % (_format_problem(problem),))
    else:
        print("problems: none")
    if result.status == VALID:
        return 0
    if result.status == CANNOT_VERIFY:
        return 4
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
