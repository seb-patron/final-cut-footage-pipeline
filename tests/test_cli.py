"""Tests for the fcp_pipeline package skeleton CLI."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fcp_pipeline import cli

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_TRIP = REPO_ROOT / "tests" / "fixtures" / "trips" / "minimal"

MAIN_PATH_STEPS = [
    "scan",
    "pull",
    "transcribe",
    "tag",
    "photos",
    "memo",
    "organize",
    "patch",
    "export",
    "roughcut",
]


def snapshot_tree(root: Path) -> dict:
    """Map every file under root to its (size, mtime); names included."""
    found = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            found[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return found


def run_package(args, cwd):
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not existing else str(REPO_ROOT) + os.pathsep + existing
    return subprocess.run(
        [sys.executable, "-m", "fcp_pipeline", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


class TestCli(unittest.TestCase):
    def test_help_lists_steps_in_order(self):
        proc = run_package(["--help"], cwd=REPO_ROOT)
        self.assertEqual(proc.returncode, 0)
        lines = proc.stdout.splitlines()

        def first_line_index(word):
            for index, line in enumerate(lines):
                if word in line.strip().split():
                    return index
            raise AssertionError(f"{word} not listed in --help")

        positions = [first_line_index(step) for step in MAIN_PATH_STEPS]
        self.assertEqual(positions, sorted(positions))
        ingest_index = first_line_index("ingest")
        self.assertGreater(ingest_index, positions[-1])
        self.assertIn("optional", lines[ingest_index])

    def test_unbuilt_step_exits_3(self):
        with tempfile.TemporaryDirectory() as tmp:
            trip = Path(tmp) / "trip"
            shutil.copytree(FIXTURE_TRIP, trip)
            work = Path(tmp) / "work"
            work.mkdir()
            before_trip = snapshot_tree(trip)
            before_work = snapshot_tree(work)
            for extra in ([], ["--dry-run"]):
                with self.subTest(extra=extra):
                    proc = run_package(
                        ["scan", "--trip", str(trip), *extra], cwd=work
                    )
                    self.assertEqual(proc.returncode, 3)
                    self.assertIn("not built yet", proc.stderr)
            self.assertEqual(snapshot_tree(trip), before_trip)
            self.assertEqual(snapshot_tree(work), before_work)

    def test_run_never_runs_ingest(self):
        calls = []

        def fake_scan(trip, dry_run):
            calls.append(("scan", trip, dry_run))
            return 0

        def fake_ingest(trip, dry_run):
            calls.append(("ingest", trip, dry_run))
            return 0

        handlers = {"scan": fake_scan, "ingest": fake_ingest}
        original = cli.STEPS
        try:
            cli.STEPS = [
                cli.Step(step.name, step.main_path, handlers.get(step.name))
                for step in original
            ]
            with tempfile.TemporaryDirectory() as tmp:
                code = cli.main(["run", "--trip", tmp])
        finally:
            cli.STEPS = original
        self.assertEqual(code, 0)
        self.assertIn(("scan", tmp, False), calls)
        self.assertEqual([name for name, _, _ in calls], ["scan"])


if __name__ == "__main__":
    unittest.main()
