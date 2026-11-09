"""Tests for the export watcher in ``fcp_pipeline.bridge.watch``.

Every document is synthetic, with invented names, written into a
temporary folder: a small 1.14 document holding a library with one
invented event and no media references. The watcher is driven with a
fake clock and fake sleep: clock() returns a number that only
sleep(seconds) advances, and each sleep also runs the test's next
scripted write step, so writes land between polls with no real waiting.
Every test that uses the real check points FCP_DTD_DIR at a missing
folder, so the check answers "cannot verify" without xmllint, the same
on every machine. Only the command-line subprocess subTest sleeps for
real (about 0.3 s); the whole file sleeps for real well under 2 s.
"""

import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from fcp_pipeline.bridge import check as check_module
from fcp_pipeline.bridge.check import (
    CANNOT_VERIFY,
    VALID,
    CheckResult,
)
from fcp_pipeline.bridge.watch import WatchResult, main, watch_export

DOCUMENT = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <library>
    <event name="Invented Event" uid="INVENTED-EVENT-UID">
    </event>
  </library>
</fcpxml>
"""

MALFORMED = DOCUMENT.replace("</event>", "</library>")


class FakeClock:
    """A clock advanced only by sleep, running one write step per sleep."""

    def __init__(self):
        self.now = 0.0
        self.steps = []
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        if self.steps:
            self.steps.pop(0)()


def _snapshot(root):
    """Map each entry under ``root`` to its size and mtime."""
    root = Path(root)
    data = {}
    entries = [root] + sorted(root.rglob("*"))
    for entry in entries:
        try:
            entry_stat = entry.stat()
        except OSError:
            continue
        data[str(entry.relative_to(root))] = (
            entry_stat.st_size, entry_stat.st_mtime_ns)
    return data


def _append_byte(path):
    """Append one byte to ``path``, creating it when missing."""
    with open(path, "ab") as handle:
        handle.write(b"x")


def _missing_dtd(folder):
    """Return a DTD folder path that does not exist, inside ``folder``."""
    return os.path.join(folder, "no-such-dtd")


class TestWatch(unittest.TestCase):
    def test_new_file_returned_when_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = DOCUMENT.encode("utf-8")
            target = os.path.join(tmp, "new.fcpxml")
            fake = FakeClock()
            after_write = {}

            def write_step():
                Path(target).write_bytes(data)
                after_write["tree"] = _snapshot(tmp)

            fake.steps.append(write_step)
            with mock.patch.dict(os.environ,
                                 {"FCP_DTD_DIR": _missing_dtd(tmp)}):
                results = watch_export(
                    tmp, timeout=30, interval=0.5, settle=1.0,
                    clock=fake.clock, sleep=fake.sleep)
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertIsInstance(result, WatchResult)
            self.assertEqual(result.path, target)
            self.assertEqual(result.sha256,
                             hashlib.sha256(data).hexdigest())
            self.assertEqual(result.sha256, result.check.sha256)
            self.assertEqual(result.check.status, CANNOT_VERIFY)
            # The file is written during the first sleep, so the first
            # poll to see it runs at 0.5; the return must come at least
            # settle past that poll.
            self.assertGreaterEqual(fake.now - 0.5, 1.0)
            # The watcher only reads: the tree is unchanged since the
            # write step ran.
            self.assertEqual(_snapshot(tmp), after_write["tree"])

            with self.subTest("main-reports-cannot-verify"):
                other = os.path.join(tmp, "main-a.fcpxml")
                inner = FakeClock()
                inner.steps.append(
                    lambda: Path(other).write_bytes(data))
                out = io.StringIO()
                with mock.patch.dict(os.environ,
                                     {"FCP_DTD_DIR": _missing_dtd(tmp)}):
                    with redirect_stdout(out):
                        code = main(
                            ["--folder", tmp, "--timeout", "30",
                             "--interval", "0.5", "--settle", "1.0"],
                            clock=inner.clock, sleep=inner.sleep)
                self.assertEqual(code, 4)
                text = out.getvalue()
                self.assertIn("found: %s" % other, text)
                self.assertIn("sha256: %s"
                              % hashlib.sha256(data).hexdigest(), text)
                self.assertIn("check: cannot verify", text)

            with self.subTest("main-valid"):
                valid_path = os.path.join(tmp, "main-b.fcpxml")
                inner = FakeClock()
                inner.steps.append(
                    lambda: Path(valid_path).write_bytes(data))
                stub = CheckResult(
                    path=valid_path, status=VALID, ok=True,
                    sha256="0" * 64, version="1.14", dtd=None,
                    events=[], projects=[], missing_media=[],
                    problems=[])
                out = io.StringIO()
                with mock.patch(
                        "fcp_pipeline.bridge.watch.check_fcpxml",
                        return_value=stub):
                    with redirect_stdout(out):
                        code = main(
                            ["--folder", tmp, "--timeout", "30",
                             "--interval", "0.5", "--settle", "1.0"],
                            clock=inner.clock, sleep=inner.sleep)
                self.assertEqual(code, 0)
                self.assertIn("check: valid", out.getvalue())

            with self.subTest("main-invalid"):
                broken = os.path.join(tmp, "main-c.fcpxml")
                inner = FakeClock()
                inner.steps.append(
                    lambda: Path(broken).write_bytes(
                        MALFORMED.encode("utf-8")))
                out = io.StringIO()
                with mock.patch.dict(os.environ,
                                     {"FCP_DTD_DIR": _missing_dtd(tmp)}):
                    with redirect_stdout(out):
                        code = main(
                            ["--folder", tmp, "--timeout", "30",
                             "--interval", "0.5", "--settle", "1.0"],
                            clock=inner.clock, sleep=inner.sleep)
                self.assertEqual(code, 1)
                text = out.getvalue()
                self.assertIn("check: invalid", text)
                self.assertIn("line ", text)

    def test_partial_write_not_returned_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = DOCUMENT.encode("utf-8")
            cuts = [len(data) * part // 5 for part in range(6)]
            chunks = [data[cuts[part]:cuts[part + 1]] for part in range(5)]
            # The first chunk is a strict prefix, so it cannot stand
            # alone as a well-formed document.
            self.assertFalse(
                chunks[0].rstrip().endswith(b"</fcpxml>"))
            target = os.path.join(tmp, "growing.fcpxml")
            fake = FakeClock()
            write_times = []

            def make_append(chunk):
                def step():
                    with open(target, "ab") as handle:
                        handle.write(chunk)
                    write_times.append(fake.now)
                return step

            for chunk in chunks:
                fake.steps.append(make_append(chunk))
            seen_bytes = []

            def recording(path):
                seen_bytes.append(Path(path).read_bytes())
                return check_module.check_fcpxml(path)

            with mock.patch.dict(os.environ,
                                 {"FCP_DTD_DIR": _missing_dtd(tmp)}):
                results = watch_export(
                    tmp, timeout=60, interval=0.5, settle=1.0,
                    clock=fake.clock, sleep=fake.sleep, check=recording)
            self.assertEqual(len(results), 1)
            result = results[0]
            # The result comes back only after the last chunk stood for
            # settle, and every check call saw the whole document.
            self.assertGreaterEqual(fake.now - write_times[-1], 1.0)
            self.assertTrue(seen_bytes)
            for recorded in seen_bytes:
                self.assertEqual(recorded, data)
            self.assertEqual(result.sha256,
                             hashlib.sha256(data).hexdigest())
            self.assertEqual(result.check.status, CANNOT_VERIFY)

            with self.subTest("always-growing"):
                growing = os.path.join(tmp, "endless.fcpxml")
                inner = FakeClock()
                inner.steps.extend([lambda: _append_byte(growing)] * 30)
                calls = []

                def never(path):
                    calls.append(path)
                    return check_module.check_fcpxml(path)

                outcome = watch_export(
                    tmp, timeout=5, interval=0.5, settle=1.0,
                    clock=inner.clock, sleep=inner.sleep, check=never)
                self.assertEqual(outcome, [])
                self.assertEqual(calls, [])

            with self.subTest("write-during-check"):
                rewritten = os.path.join(tmp, "rewritten.fcpxml")
                inner = FakeClock()
                inner.steps.append(
                    lambda: Path(rewritten).write_bytes(data))
                calls = []

                def appending(path):
                    calls.append(path)
                    if len(calls) == 1:
                        with open(path, "ab") as handle:
                            handle.write(b"\n")
                    return check_module.check_fcpxml(path)

                with mock.patch.dict(os.environ,
                                     {"FCP_DTD_DIR": _missing_dtd(tmp)}):
                    outcome = watch_export(
                        tmp, timeout=60, interval=0.5, settle=1.0,
                        clock=inner.clock, sleep=inner.sleep,
                        check=appending)
                final = data + b"\n"
                self.assertEqual(len(outcome), 1)
                # The first result is dropped because a write landed
                # during its check; the second call sees the final
                # bytes once they are stable again.
                self.assertEqual(len(calls), 2)
                self.assertEqual(outcome[0].sha256,
                                 hashlib.sha256(final).hexdigest())
                self.assertEqual(Path(rewritten).read_bytes(), final)

    def test_bundle_waits_for_info_fcpxml(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = DOCUMENT.encode("utf-8")
            bundle = os.path.join(tmp, "new.fcpxmld")
            info_path = os.path.join(bundle, "Info.fcpxml")
            half = len(info) // 2
            fake = FakeClock()
            events = {}

            def make_bundle():
                os.mkdir(bundle)
                events["mkdir"] = fake.now

            def write_part():
                Path(info_path).write_bytes(info[:half])
                events["part"] = fake.now

            def finish_info():
                with open(info_path, "ab") as handle:
                    handle.write(info[half:])
                events["done"] = fake.now

            fake.steps.append(make_bundle)
            fake.steps.extend([lambda: None] * 5)
            fake.steps.append(write_part)
            fake.steps.append(finish_info)
            calls = []

            def recording(path):
                calls.append((path, fake.now))
                return check_module.check_fcpxml(path)

            with mock.patch.dict(os.environ,
                                 {"FCP_DTD_DIR": _missing_dtd(tmp)}):
                results = watch_export(
                    tmp, timeout=60, interval=0.5, settle=1.0,
                    clock=fake.clock, sleep=fake.sleep, check=recording)
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result.path, bundle)
            self.assertEqual(result.sha256,
                             hashlib.sha256(info).hexdigest())
            self.assertEqual(result.sha256, result.check.sha256)
            self.assertEqual(result.check.status, CANNOT_VERIFY)
            # Polls covering more than twice settle passed between the
            # empty bundle and the single check call on the folder.
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], bundle)
            self.assertGreater(calls[0][1] - events["mkdir"], 2.0)

            with self.subTest("empty-info"):
                hollow = os.path.join(tmp, "hollow.fcpxmld")
                hollow_info = os.path.join(hollow, "Info.fcpxml")
                inner = FakeClock()
                inner.steps.append(lambda: os.mkdir(hollow))
                inner.steps.append(
                    lambda: Path(hollow_info).write_bytes(b""))
                inner.steps.extend([lambda: None] * 12)
                calls = []

                def never(path):
                    calls.append(path)
                    return check_module.check_fcpxml(path)

                outcome = watch_export(
                    tmp, timeout=5, interval=0.5, settle=1.0,
                    clock=inner.clock, sleep=inner.sleep, check=never)
                self.assertEqual(outcome, [])
                self.assertEqual(calls, [])

            with self.subTest("empty-file"):
                empty = os.path.join(tmp, "empty.fcpxml")
                inner = FakeClock()
                inner.steps.append(
                    lambda: Path(empty).write_bytes(b""))
                inner.steps.extend([lambda: None] * 12)
                calls = []

                def never(path):
                    calls.append(path)
                    return check_module.check_fcpxml(path)

                outcome = watch_export(
                    tmp, timeout=5, interval=0.5, settle=1.0,
                    clock=inner.clock, sleep=inner.sleep, check=never)
                self.assertEqual(outcome, [])
                self.assertEqual(calls, [])

    def test_existing_files_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            with tempfile.TemporaryDirectory() as outer:
                data = DOCUMENT.encode("utf-8")
                old = os.path.join(tmp, "old.fcpxml")
                old_bundle = os.path.join(tmp, "old.fcpxmld")
                old_info = os.path.join(old_bundle, "Info.fcpxml")
                Path(old).write_bytes(data)
                os.mkdir(old_bundle)
                Path(old_info).write_bytes(data)
                Path(os.path.join(tmp, "notes.txt")).write_bytes(
                    b"invented notes")
                outside = os.path.join(outer, "outside.fcpxml")
                Path(outside).write_bytes(data)
                probe = os.path.join(tmp, "probe-symlink")
                try:
                    os.symlink(os.path.join(tmp, "notes.txt"), probe)
                except OSError as exc:
                    symlink_ok = False
                    symlink_reason = str(exc) or "os.symlink failed"
                else:
                    symlink_ok = True
                    symlink_reason = ""
                    os.remove(probe)

                def touch_old():
                    with open(old, "ab") as handle:
                        handle.write(b"\n")

                def rewrite_info():
                    Path(old_info).write_bytes(data)

                def delete_old():
                    os.remove(old)

                def recreate_old():
                    Path(old).write_bytes(data)

                def hidden():
                    Path(os.path.join(tmp, ".hidden.fcpxml")).write_bytes(
                        data)

                def other():
                    Path(os.path.join(tmp, "other.txt")).write_bytes(
                        b"invented")

                def sub():
                    folder = os.path.join(tmp, "sub")
                    os.mkdir(folder)
                    Path(os.path.join(folder, "inner.fcpxml")).write_bytes(
                        data)

                def link():
                    if symlink_ok:
                        os.symlink(outside,
                                   os.path.join(tmp, "link.fcpxml"))

                fake = FakeClock()
                fake.steps.extend([touch_old, rewrite_info, delete_old,
                                   recreate_old, hidden, other, sub, link])
                fake.steps.extend([lambda: None] * 6)
                calls = []

                def recording(path):
                    calls.append(path)
                    return check_module.check_fcpxml(path)

                results = watch_export(
                    tmp, timeout=6, interval=0.5, settle=1.0,
                    clock=fake.clock, sleep=fake.sleep, check=recording)
                self.assertEqual(results, [])
                self.assertEqual(calls, [])
                if not symlink_ok:
                    with self.subTest("symlink-candidate"):
                        self.skipTest("symlinks unavailable: %s"
                                      % symlink_reason)

    def test_timeout_returns_nothing(self):
        with tempfile.TemporaryDirectory() as parent:
            folder = os.path.join(parent, "exports")
            os.mkdir(folder)
            fake = FakeClock()
            with mock.patch.dict(
                    os.environ,
                    {"FCP_DTD_DIR": os.path.join(parent, "no-such-dtd")}):
                before_parent = _snapshot(parent)
                before_folder = _snapshot(folder)
                results = watch_export(
                    folder, timeout=5, interval=1,
                    clock=fake.clock, sleep=fake.sleep)
                after_parent = _snapshot(parent)
                after_folder = _snapshot(folder)
            self.assertEqual(results, [])
            self.assertTrue(fake.sleeps)
            for waited in fake.sleeps:
                self.assertLessEqual(waited, 1.0)
            self.assertLessEqual(abs(fake.now - 5.0), 1e-9)
            # The watcher only reads: neither the watched folder nor
            # its temporary parent changed.
            self.assertEqual(after_folder, before_folder)
            self.assertEqual(after_parent, before_parent)

            with self.subTest("main-timeout"):
                inner = FakeClock()
                out = io.StringIO()
                with redirect_stdout(out):
                    code = main(
                        ["--folder", folder, "--timeout", "5",
                         "--interval", "1"],
                        clock=inner.clock, sleep=inner.sleep)
                self.assertEqual(code, 3)
                lines = out.getvalue().splitlines()
                self.assertTrue(lines)
                self.assertTrue(lines[0].startswith("timed out:"))
                self.assertIn(folder, lines[0])

            with self.subTest("real-subprocess-timeout"):
                repo_root = Path(__file__).resolve().parent.parent
                live = os.path.join(parent, "live")
                os.mkdir(live)
                env = dict(os.environ)
                env["FCP_DTD_DIR"] = os.path.join(parent, "no-such-dtd")
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                completed = subprocess.run(
                    [sys.executable, "-m", "fcp_pipeline.bridge.watch",
                     "--folder", live, "--timeout", "0.3",
                     "--interval", "0.1"],
                    cwd=str(repo_root),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(completed.returncode, 3,
                                 completed.stderr)
                lines = completed.stdout.splitlines()
                self.assertTrue(lines, completed)
                self.assertTrue(lines[0].startswith("timed out:"),
                                completed)
                self.assertEqual(os.listdir(live), [])

    def test_command_line_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            afile = os.path.join(tmp, "afile.fcpxml")
            Path(afile).write_bytes(DOCUMENT.encode("utf-8"))
            missing = os.path.join(tmp, "no-such-folder")
            cases = (
                [],
                ["--folder", missing],
                ["--folder", afile],
                ["--folder", tmp, "--timeout", "0"],
                ["--folder", tmp, "--interval", "-1"],
                ["--folder", tmp, "--settle", "abc"],
            )
            before = _snapshot(tmp)
            for argv in cases:
                with self.subTest(argv=argv):
                    with redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as ctx:
                            main(argv)
                    self.assertEqual(ctx.exception.code, 2)
            self.assertEqual(_snapshot(tmp), before)


if __name__ == "__main__":
    unittest.main()
