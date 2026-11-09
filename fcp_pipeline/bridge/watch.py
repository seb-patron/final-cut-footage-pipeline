"""Watch one folder for a new exported FCPXML file or bundle.

``python3 -m fcp_pipeline.bridge.watch --folder <folder>`` polls the one
folder the editor exports XML into, waits for a new ``.fcpxml`` file or
``.fcpxmld`` bundle to appear and stop changing (for a bundle, until its
``Info.fcpxml`` exists and the bundle has stopped changing), runs the
offline check from :mod:`fcp_pipeline.bridge.check` on exactly the bytes
read through a no-follow open, and prints its path, its SHA-256
fingerprint, and the check's verdict.

Exports already in the folder when watching starts are ignored for the
whole run. Inside the watched folder the watcher only reads: it never
creates, changes, renames or deletes any file or folder there, and it
never writes any file outside Python bytecode caches
(``__pycache__/*.pyc``), suppressed with ``python3 -B``,
``PYTHONDONTWRITEBYTECODE=1`` or the ``sys.dont_write_bytecode`` flag set
below. It never contacts, launches or scripts Final Cut Pro or Motion,
and uses no network.
"""

import sys

# Suppress bytecode writes for the package imports below. This must run
# before importing any fcp_pipeline modules. Running the command with
# ``python3 -B`` or ``PYTHONDONTWRITEBYTECODE=1`` suppresses even this
# module's own cache entry as well.
sys.dont_write_bytecode = True

import argparse
import errno
import os
import stat
import time
from dataclasses import dataclass

from fcp_pipeline.bridge.check import (
    INVALID,
    REFUSED,
    VALID,
    CheckResult,
    Problem,
    check_fcpxml,
)

import fcp_pipeline.bridge.check as _check_module

_FILE_SUFFIX = ".fcpxml"
_BUNDLE_SUFFIX = ".fcpxmld"

_ABSENT = object()

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


class _Refused(Exception):
    """A candidate cannot be opened safely as a regular file."""


def _read_fd_all(fd):
    """Read every byte from an open fd."""
    chunks = []
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _read_candidate_bytes(folder, name, kind):
    """Return exactly the bytes of a stable candidate, no-follow.

    The candidate is opened with ``O_NOFOLLOW`` (and ``O_NONBLOCK``, then
    ``fstat`` ``S_ISREG``) relative to a directory fd for the watched
    folder, so a symlink swapped in after the stability check is refused
    instead of followed outside the folder. A bundle's ``Info.fcpxml``
    is opened the same way relative to the bundle directory fd.
    Raises ``_Refused`` for a symlink or non-regular file and
    ``FileNotFoundError`` when the candidate vanished.
    """
    if (not name or name in (".", "..") or "/" in name
            or "\x00" in name or name.startswith(".")):
        raise _Refused("refused %s: not a candidate" % (name,))
    try:
        dir_fd = os.open(folder, os.O_RDONLY | _O_DIRECTORY)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise FileNotFoundError(str(exc)) from exc
        raise _Refused("cannot open watched folder: %s" % (exc,)) from exc
    try:
        if kind == "file":
            try:
                fd = os.open(
                    name,
                    os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK,
                    dir_fd=dir_fd,
                )
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    raise FileNotFoundError(str(exc)) from exc
                raise _Refused(
                    "refused %s: %s" % (name, exc.strerror or exc)
                ) from exc
            try:
                opened = os.fstat(fd)
                if not stat.S_ISREG(opened.st_mode):
                    raise _Refused(
                        "refused %s: not a regular file" % (name,))
                return _read_fd_all(fd)
            finally:
                os.close(fd)
        elif kind == "bundle":
            try:
                bundle_fd = os.open(
                    name,
                    os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW,
                    dir_fd=dir_fd,
                )
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    raise FileNotFoundError(str(exc)) from exc
                raise _Refused(
                    "refused %s: %s" % (name, exc.strerror or exc)
                ) from exc
            try:
                bundle_stat = os.fstat(bundle_fd)
                if not stat.S_ISDIR(bundle_stat.st_mode):
                    raise _Refused(
                        "refused %s: not a directory" % (name,))
                try:
                    fd = os.open(
                        "Info.fcpxml",
                        os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK,
                        dir_fd=bundle_fd,
                    )
                except OSError as exc:
                    if exc.errno == errno.ENOENT:
                        raise FileNotFoundError(str(exc)) from exc
                    raise _Refused(
                        "refused %s/Info.fcpxml: %s"
                        % (name, exc.strerror or exc)
                    ) from exc
                try:
                    opened = os.fstat(fd)
                    if not stat.S_ISREG(opened.st_mode):
                        raise _Refused(
                            "refused %s/Info.fcpxml: not a regular file"
                            % (name,))
                    return _read_fd_all(fd)
                finally:
                    os.close(fd)
            finally:
                os.close(bundle_fd)
        else:
            raise _Refused("refused %s: unknown kind" % (name,))
    finally:
        os.close(dir_fd)


def _validate_secure_bytes(raw, display_path):
    """Validate exactly ``raw`` bytes through the offline check.

    Small adapter: the checker's file reader is pointed at the already
    opened bytes, so :func:`check_fcpxml` validates those bytes without
    re-opening the candidate path (which may now be a symlink outside
    the watched folder).
    """
    original = _check_module.xml_bytes
    _check_module.xml_bytes = lambda _target: raw
    try:
        return check_fcpxml(display_path)
    finally:
        _check_module.xml_bytes = original


def _refused_result(display_path, message):
    """Return a refused CheckResult that carries no file bytes."""
    return CheckResult(
        path=display_path, status=REFUSED, ok=False, sha256=None,
        version=None, dtd=None,
        problems=[Problem(message=message)],
    )


@dataclass
class WatchResult:
    """One new export found stable, with the check's answer for it."""

    path: str
    sha256: str | None
    check: CheckResult


def _classify(folder, name):
    """Return "file", "bundle", or None for one top-level entry name.

    Uses lstat, so a symlink is never a candidate.
    """
    if name.startswith("."):
        return None
    if not (name.endswith(_FILE_SUFFIX) or name.endswith(_BUNDLE_SUFFIX)):
        return None
    try:
        entry_stat = os.lstat(os.path.join(folder, name))
    except OSError:
        return None
    if name.endswith(_FILE_SUFFIX) and stat.S_ISREG(entry_stat.st_mode):
        return "file"
    if name.endswith(_BUNDLE_SUFFIX) and stat.S_ISDIR(entry_stat.st_mode):
        return "bundle"
    return None


def _file_signature(full):
    """Return (size, mtime_ns) for a file, None when empty, _ABSENT."""
    try:
        entry_stat = os.lstat(full)
    except OSError:
        return _ABSENT
    if not stat.S_ISREG(entry_stat.st_mode):
        return _ABSENT
    if entry_stat.st_size == 0:
        return None
    return (entry_stat.st_size, entry_stat.st_mtime_ns)


def _bundle_signature(full):
    """Return the sorted bundle listing, None when empty, _ABSENT.

    The listing holds (path relative to the bundle, kind, size,
    mtime_ns) for every entry under the bundle, walked without
    following symlinks. A bundle has no signature while its
    ``Info.fcpxml`` is missing or is not a regular file of more than
    zero bytes.
    """
    try:
        bundle_stat = os.lstat(full)
    except OSError:
        return _ABSENT
    if not stat.S_ISDIR(bundle_stat.st_mode):
        return _ABSENT
    try:
        info_stat = os.lstat(os.path.join(full, "Info.fcpxml"))
    except OSError:
        return None
    if not stat.S_ISREG(info_stat.st_mode) or info_stat.st_size == 0:
        return None
    found = []
    stack = [full]
    try:
        while stack:
            current = stack.pop()
            with os.scandir(current) as iterator:
                for entry in iterator:
                    entry_stat = os.lstat(entry.path)
                    relative = os.path.relpath(entry.path, full)
                    if stat.S_ISDIR(entry_stat.st_mode):
                        kind = "dir"
                        stack.append(entry.path)
                    elif stat.S_ISREG(entry_stat.st_mode):
                        kind = "file"
                    else:
                        kind = "other"
                    found.append((relative, kind, entry_stat.st_size,
                                  entry_stat.st_mtime_ns))
    except OSError:
        return _ABSENT
    found.sort()
    return found


def _signature(full, kind):
    """Return the current signature for a candidate, None, or _ABSENT."""
    if kind == "file":
        return _file_signature(full)
    return _bundle_signature(full)


def _top_names(folder):
    """Return the sorted top-level entry names, or [] when unlistable."""
    try:
        with os.scandir(folder) as iterator:
            return sorted(entry.name for entry in iterator)
    except OSError:
        return []


def watch_export(folder, *, timeout=600.0, interval=1.0, settle=2.0,
                 clock=None, sleep=None, check=None):
    """Wait for new exports in ``folder``; return a list of WatchResult.

    The folder is listed once at the start and every name in it is
    ignored for the whole run. Each poll lists the folder again (top
    level only). A candidate is complete when the same signature was
    seen at two or more consecutive polls and at least ``settle``
    seconds passed since the poll that first saw that signature. At the
    first poll with complete candidates, each one is checked once, in
    name order; a candidate whose signature changed during its check is
    dropped and starts over. Returns the standing results in name order,
    or [] when the deadline (start plus ``timeout``) is reached first.

    The default check opens each candidate with ``O_NOFOLLOW`` (and
    ``O_NONBLOCK``, then ``fstat`` ``S_ISREG``) relative to a directory
    fd for the watched folder and validates exactly those bytes, never
    re-opening the candidate path. A custom ``check`` callable (a test
    hook taking the candidate path) re-opens the path as before.
    """
    if clock is None:
        clock = time.monotonic
    if sleep is None:
        sleep = time.sleep
    secure = check is None
    if secure:
        check = check_fcpxml
    start = clock()
    deadline = start + timeout
    try:
        with os.scandir(folder) as iterator:
            start_set = {entry.name for entry in iterator}
    except OSError:
        start_set = set()
    tracked = {}
    while True:
        now = clock()
        complete = []
        seen = set()
        for name in _top_names(folder):
            if name in start_set:
                continue
            kind = _classify(folder, name)
            if kind is None:
                continue
            full = os.path.join(folder, name)
            signature = _signature(full, kind)
            if signature is _ABSENT:
                continue
            seen.add(name)
            previous = tracked.get(name)
            if signature is None:
                tracked[name] = (None, now)
            elif (previous is None or previous[0] is None
                    or previous[0] != signature):
                tracked[name] = (signature, now)
            elif now - previous[1] >= settle:
                complete.append((name, kind, full, signature))
        for name in list(tracked):
            if name not in seen:
                del tracked[name]
        if complete:
            complete.sort(key=lambda item: item[0])
            results = []
            for name, kind, full, signature in complete:
                if secure:
                    try:
                        raw = _read_candidate_bytes(folder, name, kind)
                    except FileNotFoundError:
                        tracked.pop(name, None)
                        continue
                    except _Refused as exc:
                        outcome = _refused_result(full, str(exc))
                        results.append(WatchResult(
                            path=full, sha256=None, check=outcome))
                        continue
                    outcome = _validate_secure_bytes(raw, full)
                    again = _signature(full, kind)
                    if again is _ABSENT:
                        try:
                            current = os.lstat(full)
                        except OSError:
                            tracked.pop(name, None)
                            continue
                        if kind == "file" and not stat.S_ISREG(
                                current.st_mode):
                            outcome = _refused_result(
                                full,
                                "refused %s: not a regular file" % (name,))
                            results.append(WatchResult(
                                path=full, sha256=None, check=outcome))
                            continue
                        if kind == "bundle" and not stat.S_ISDIR(
                                current.st_mode):
                            outcome = _refused_result(
                                full,
                                "refused %s: not a directory" % (name,))
                            results.append(WatchResult(
                                path=full, sha256=None, check=outcome))
                            continue
                        tracked.pop(name, None)
                        continue
                    if again != signature:
                        tracked[name] = (again, now)
                        continue
                    results.append(WatchResult(path=full,
                                               sha256=outcome.sha256,
                                               check=outcome))
                else:
                    outcome = check(full)
                    again = _signature(full, kind)
                    if again is _ABSENT:
                        tracked.pop(name, None)
                        continue
                    if again != signature:
                        tracked[name] = (again, now)
                        continue
                    results.append(WatchResult(path=full,
                                               sha256=outcome.sha256,
                                               check=outcome))
            if results:
                return results
        if now >= deadline:
            return []
        wait = deadline - now
        if wait > interval:
            wait = interval
        sleep(wait)


def _seconds(text):
    """Parse an argparse option value as seconds above zero."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("not a number: %r" % (text,))
    if not value > 0:
        raise argparse.ArgumentTypeError("must be above 0: %r" % (text,))
    return value


def _format_seconds(value):
    """Render a seconds value without a needless trailing fraction."""
    try:
        whole = int(value)
    except (OverflowError, ValueError):
        return str(value)
    if whole == value:
        return str(whole)
    return str(value)


def _format_problem(problem):
    """Render one problem with its line and element when known."""
    bits = []
    if problem.line is not None:
        bits.append("line %s" % (problem.line,))
    if problem.element is not None:
        bits.append("element %s" % (problem.element,))
    bits.append(problem.message)
    return ": ".join(bits)


def main(argv=None, *, clock=None, sleep=None):
    """Run the export watcher on one folder; return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="fcp_pipeline.bridge.watch",
        description="Watch one folder for a new exported .fcpxml file or "
        ".fcpxmld bundle, wait until it stops changing, run the offline "
        "check on exactly the no-follow bytes and print its path, SHA-256 "
        "and the check's verdict. Files already in the folder are "
        "ignored. Inside the folder it only reads; outside it writes at "
        "most Python bytecode caches unless -B is used.",
    )
    parser.add_argument(
        "--folder", required=True,
        help="the folder the editor exports XML into",
    )
    parser.add_argument(
        "--timeout", type=_seconds, default=600.0,
        help="seconds to wait for a new export (default: 600)",
    )
    parser.add_argument(
        "--interval", type=_seconds, default=1.0,
        help="seconds between polls (default: 1.0)",
    )
    parser.add_argument(
        "--settle", type=_seconds, default=2.0,
        help="seconds a candidate must stand still (default: 2.0)",
    )
    args = parser.parse_args(argv)
    if not os.path.isdir(args.folder):
        parser.error("not a directory: %s" % (args.folder,))
    results = watch_export(
        args.folder, timeout=args.timeout, interval=args.interval,
        settle=args.settle, clock=clock, sleep=sleep,
    )
    if not results:
        print("timed out: no new export in %s after %s seconds"
              % (args.folder, _format_seconds(args.timeout)))
        return 3
    for result in results:
        print("found: %s" % (result.path,))
        print("sha256: %s" % (result.sha256 if result.sha256 is not None
                               else "unknown"))
        print("check: %s" % (result.check.status,))
        for problem in result.check.problems:
            print("  %s" % (_format_problem(problem),))
    if all(item.check.status == VALID for item in results):
        return 0
    if any(item.check.status in (INVALID, REFUSED) for item in results):
        return 1
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
