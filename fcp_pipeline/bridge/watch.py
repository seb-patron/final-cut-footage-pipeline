"""Watch one folder for a new exported FCPXML file or bundle.

``python3 -m fcp_pipeline.bridge.watch --folder <folder>`` polls the one
folder the editor exports XML into, waits for a new ``.fcpxml`` file or
``.fcpxmld`` bundle to appear and stop changing (for a bundle, until its
``Info.fcpxml`` exists and the bundle has stopped changing), runs the
offline check from :mod:`fcp_pipeline.bridge.check` on it, and prints
its path, its SHA-256 fingerprint, and the check's verdict.

Exports already in the folder when watching starts are ignored for the
whole run. The watcher only reads: it never creates, changes, renames
or deletes any file or folder, never imports anything, never contacts,
launches or scripts Final Cut Pro or Motion, and uses no network.
"""

import argparse
import os
import stat
import time
from dataclasses import dataclass

from fcp_pipeline.bridge.check import (
    INVALID,
    REFUSED,
    VALID,
    CheckResult,
    check_fcpxml,
)

_FILE_SUFFIX = ".fcpxml"
_BUNDLE_SUFFIX = ".fcpxmld"

_ABSENT = object()


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
    """
    if clock is None:
        clock = time.monotonic
    if sleep is None:
        sleep = time.sleep
    if check is None:
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
        "check on it, and print its path, SHA-256 and the check's "
        "verdict. Files already in the folder are ignored. Reads only; "
        "writes nothing.",
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
