"""Tests for fcp_pipeline.safe_write, the pipeline's one write path."""

import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from fcp_pipeline.safe_write import (
    MAX_COMMITTED_BYTES,
    SafeWriteError,
    SafeWriter,
)
from fcp_pipeline.trip import load_trip


def snapshot_tree(root: Path):
    """Map each entry under root to (is_symlink, size, mtime_ns)."""
    out = {}
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            entry = Path(base) / name
            try:
                info = os.lstat(entry)
            except FileNotFoundError:
                continue
            out[str(entry.relative_to(root))] = (
                stat.S_ISLNK(info.st_mode),
                info.st_size,
                info.st_mtime_ns,
            )
    return out


class TestSafeWrite(unittest.TestCase):
    """Each project lives in a temporary folder with invented values."""

    def make_trip(self, tmp: Path):
        media = tmp / "media"
        footage = media / "footage"
        photos = media / "photos"
        assets = media / "assets"
        library = tmp / "libs" / "EditLib.fcpbundle"
        card = tmp / "cards" / "CARD-A"
        for folder in (footage, photos, assets, library, card):
            folder.mkdir(parents=True)
        (footage / "clip-a.mov").write_bytes(b"footage-bytes")
        (photos / "photo-a.jpg").write_bytes(b"photo-bytes")
        (assets / "asset-a.txt").write_bytes(b"asset-bytes")
        (library / "library.fcplibrary").write_bytes(b"library-bytes")
        project = tmp / "project"
        project.mkdir()
        (project / "project.toml").write_text(
            "title = \"Invented Survey\"\n"
            "[final_cut]\n"
            f"library = \"{library.as_posix()}\"\n"
            "events = [\"Invented Event\"]\n"
            "[media]\n"
            f"footage = [\"{footage.as_posix()}\"]\n"
            f"photos = [\"{photos.as_posix()}\"]\n"
            f"assets = \"{assets.as_posix()}\"\n"
            "[pipeline]\n"
            "timezone = \"UTC\"\n"
            "start_date = 2020-06-15\n",
            encoding="utf-8",
        )
        settings = load_trip(project)
        writer = SafeWriter(settings, cards=[card])
        places = {
            "tmp": tmp,
            "media": media,
            "footage": footage,
            "photos": photos,
            "assets": assets,
            "library": library,
            "card": card,
            "project": project,
            "settings": settings,
            "writer": writer,
        }
        return places

    def test_refuses_inside_footage(self):
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            target = places["footage"] / "fresh.txt"
            with self.assertRaises(SafeWriteError) as raised:
                places["writer"].write_new(target, b"nope")
            self.assertIn(str(target), str(raised.exception))
            self.assertFalse(os.path.lexists(target))
            self.assertEqual(
                sorted(p.name for p in places["footage"].iterdir()), ["clip-a.mov"]
            )
            self.assertEqual((places["footage"] / "clip-a.mov").read_bytes(), b"footage-bytes")

    def test_refuses_inside_final_cut_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            target = places["library"] / "Event" / "fresh.txt"
            before = snapshot_tree(places["library"])
            with self.assertRaises(SafeWriteError):
                places["writer"].write_new(target, b"nope")
            self.assertFalse(os.path.lexists(target))
            self.assertEqual(snapshot_tree(places["library"]), before)

    def test_refuses_inside_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            target = places["card"] / "DCIM" / "fresh.txt"
            before = snapshot_tree(places["card"])
            with self.assertRaises(SafeWriteError):
                places["writer"].write_new(target, b"nope")
            self.assertFalse(os.path.lexists(target))
            self.assertEqual(snapshot_tree(places["card"]), before)

    def test_refuses_inside_fcpbundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            writer = places["writer"]
            for bundle in ("Evt.fcpbundle", "UPPER.FCPBUNDLE"):
                with self.subTest(bundle=bundle):
                    bundle_dir = places["project"] / "edit" / bundle
                    bundle_dir.mkdir(parents=True)
                    target = bundle_dir / "fresh.txt"
                    with self.assertRaises(SafeWriteError) as raised:
                        writer.write_new(target, b"nope")
                    self.assertIn(bundle, str(raised.exception))
                    self.assertFalse(os.path.lexists(target))

    def test_refuses_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            pipeline = places["project"] / ".pipeline"
            pipeline.mkdir(parents=True)
            os.symlink(places["footage"], pipeline / "evil")
            before = snapshot_tree(places["footage"])
            with self.assertRaises(SafeWriteError):
                places["writer"].write_new(".pipeline/evil/fresh.txt", b"nope")
            self.assertFalse(os.path.lexists(places["footage"] / "fresh.txt"))
            self.assertEqual(snapshot_tree(places["footage"]), before)

    def test_numbered_outputs_never_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            edit = places["project"] / "edit"
            edit.mkdir(parents=True)
            (edit / "001-x.fcpxml").write_bytes(b"first-bytes")
            final = places["writer"].write_numbered("x.fcpxml", b"second-bytes")
            self.assertEqual(final.name, "002-x.fcpxml")
            self.assertEqual((edit / "001-x.fcpxml").read_bytes(), b"first-bytes")
            self.assertEqual(final.read_bytes(), b"second-bytes")

    def test_racing_writers_get_distinct_numbers(self):
        cases = (
            ("same name", "race.fcpxml", "race.fcpxml"),
            ("different names", "racer-a.fcpxml", "racer-b.fcpxml"),
        )
        for label, first_name, second_name in cases:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as tmp:
                    places = self.make_trip(Path(tmp))
                    barrier = threading.Barrier(2)
                    outcomes = {}

                    def worker(index, name, payload):
                        writer = SafeWriter(places["settings"], cards=[places["card"]])
                        try:
                            barrier.wait(timeout=15)
                            outcomes[index] = (
                                "ok",
                                writer.write_numbered(name, payload),
                                payload,
                            )
                        except Exception as exc:  # noqa: BLE001
                            outcomes[index] = ("error", exc, payload)

                    payloads = (b"racer-one-payload", b"racer-two-payload")
                    threads = [
                        threading.Thread(
                            target=worker, args=(0, first_name, payloads[0])
                        ),
                        threading.Thread(
                            target=worker, args=(1, second_name, payloads[1])
                        ),
                    ]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join(60)
                    for index in (0, 1):
                        self.assertEqual(outcomes[index][0], "ok", outcomes[index][1])
                    paths = [outcomes[0][1], outcomes[1][1]]
                    self.assertEqual(
                        sorted(p.name[:3] for p in paths), ["001", "002"]
                    )
                    for index in (0, 1):
                        self.assertEqual(paths[index].read_bytes(), payloads[index])

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            transcripts = places["project"] / "transcripts"
            transcripts.mkdir(parents=True)
            (transcripts / "a.txt").write_bytes(b"kept")
            before = snapshot_tree(places["tmp"])
            dry = SafeWriter(
                places["settings"], dry_run=True, cards=[places["card"]]
            )
            final = dry.write_new("transcripts/plan.txt", b"hello-dry")
            self.assertEqual(snapshot_tree(places["tmp"]), before)
            self.assertFalse(os.path.lexists(places["project"] / "transcripts" / "plan.txt"))
            self.assertEqual(len(dry.planned), 1)
            self.assertEqual(dry.planned[0].path, final)
            self.assertEqual(dry.planned[0].size, len(b"hello-dry"))
            self.assertEqual(dry.planned[0]["path"], final)
            self.assertEqual(dry.planned[0]["size"], len(b"hello-dry"))

    def test_dry_run_numbered_plans_successive_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            before = snapshot_tree(places["tmp"])
            dry = SafeWriter(
                places["settings"], dry_run=True, cards=[places["card"]]
            )
            first = dry.write_numbered("cut.fcpxml", b"one")
            second = dry.write_numbered("cut.fcpxml", b"two")
            self.assertEqual(first.name, "001-cut.fcpxml")
            self.assertEqual(second.name, "002-cut.fcpxml")
            self.assertEqual([r.path for r in dry.planned], [first, second])
            self.assertEqual(snapshot_tree(places["tmp"]), before)

    def test_refuses_outside_allowed_places(self):
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            writer = places["writer"]
            project = places["project"]
            refused = [
                "outline.md",
                "project.toml",
                "memo/voice.m4a",
                "memo/nested/note.json",
                ".scratch/x",
                ".",
                str(places["tmp"] / "outside" / "x.txt"),
                "transcripts/../../outside.txt",
            ]
            for target in refused:
                with self.subTest(target=target):
                    with self.assertRaises(SafeWriteError):
                        writer.write_new(target, b"nope")
            self.assertFalse(os.path.lexists(project / "outline.md"))
            self.assertFalse(os.path.lexists(project / ".scratch"))
            self.assertFalse(os.path.lexists(places["tmp"] / "outside"))
            writable = [
                ("memo/note.json", b"{}"),
                ("memo/outline-draft.md", "# draft"),
                ("transcripts/deep/nest/t.txt", b"words"),
                ("shots/a.txt", "text-data"),
                ("contact-sheets/a.txt", b"sheet"),
                (str(project / "shots" / "abs.txt"), b"absolute"),
            ]
            for target, data in writable:
                with self.subTest(target=target):
                    final = writer.write_new(target, data)
                    expected = data.encode("utf-8") if isinstance(data, str) else data
                    self.assertEqual(final.read_bytes(), expected)

    def test_existing_target_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            writer = places["writer"]
            transcripts = places["project"] / "transcripts"
            transcripts.mkdir(parents=True)
            kept = transcripts / "keep.txt"
            kept.write_bytes(b"original")
            with self.assertRaises(SafeWriteError):
                writer.write_new("transcripts/keep.txt", b"replacement")
            self.assertEqual(kept.read_bytes(), b"original")
            taken = transcripts / "subdir"
            taken.mkdir()
            with self.assertRaises(SafeWriteError):
                writer.write_new("transcripts/subdir", b"replacement")
            self.assertTrue(taken.is_dir())
            broken = transcripts / "broken"
            os.symlink(places["tmp"] / "missing-target", broken)
            with self.assertRaises(SafeWriteError):
                writer.write_new("transcripts/broken", b"replacement")
            self.assertTrue(os.path.islink(broken))

    def test_refuses_large_write_outside_pipeline(self):
        self.assertEqual(MAX_COMMITTED_BYTES, 5_000_000)
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            writer = places["writer"]
            big = b"x" * (MAX_COMMITTED_BYTES + 1)
            with self.assertRaises(SafeWriteError):
                writer.write_new("shots/big.bin", big)
            self.assertFalse(
                os.path.lexists(places["project"] / "shots" / "big.bin")
            )
            final = writer.write_new(".pipeline/big.bin", big)
            self.assertEqual(len(final.read_bytes()), MAX_COMMITTED_BYTES + 1)
            edge = writer.write_new("shots/edge.bin", b"y" * MAX_COMMITTED_BYTES)
            self.assertEqual(len(edge.read_bytes()), MAX_COMMITTED_BYTES)

    def test_refuses_every_media_and_final_cut_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            roots = [
                places["settings"].footage[0],
                places["settings"].photos[0],
                places["settings"].assets,
                places["settings"].library,
            ]
            for root in roots:
                for mode in ("normal", "dry-run"):
                    with self.subTest(root=root, mode=mode):
                        writer = (
                            SafeWriter(
                                places["settings"],
                                dry_run=True,
                                cards=[places["card"]],
                            )
                            if mode == "dry-run"
                            else places["writer"]
                        )
                        before = snapshot_tree(Path(root))
                        target = Path(root) / "probe-fresh.txt"
                        with self.assertRaises(SafeWriteError):
                            writer.write_new(target, b"nope")
                        self.assertFalse(os.path.lexists(target))
                        self.assertEqual(snapshot_tree(Path(root)), before)
                        if mode == "dry-run":
                            self.assertEqual(writer.planned, [])

    def test_numbered_refuses_when_pipeline_is_symlinked_into_footage(self):
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            project = places["project"]
            footage = places["footage"]
            os.symlink(footage, project / ".pipeline")
            before = snapshot_tree(footage)
            with self.assertRaises(SafeWriteError) as raised:
                places["writer"].write_numbered("cut.fcpxml", b"data")
            self.assertIn("refused lock file", str(raised.exception))
            self.assertFalse(os.path.lexists(footage / "safe-write.lock"))
            self.assertEqual(snapshot_tree(footage), before)
            self.assertFalse(os.path.lexists(project / "edit" / "001-cut.fcpxml"))

    def test_numbered_lock_swap_after_check_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            writer = places["writer"]
            project = places["project"]
            footage = places["footage"]
            before = snapshot_tree(footage)
            original = SafeWriter._check_lock_path
            calls = {"count": 0}

            def checking(self, lock):
                original(self, lock)
                calls["count"] += 1
                if calls["count"] == 2:
                    pipeline = project / ".pipeline"
                    if pipeline.is_symlink():
                        pipeline.unlink()
                    else:
                        os.rmdir(pipeline)
                    os.symlink(footage, pipeline)

            with mock.patch.object(
                SafeWriter, "_check_lock_path", autospec=True, side_effect=checking
            ):
                with self.assertRaises(SafeWriteError) as raised:
                    writer.write_numbered("cut.fcpxml", b"data")
            self.assertIn("nothing was written", str(raised.exception))
            self.assertEqual(calls["count"], 2)
            self.assertFalse(os.path.lexists(footage / "safe-write.lock"))
            self.assertEqual(snapshot_tree(footage), before)
            self.assertFalse(os.path.lexists(project / "edit" / "001-cut.fcpxml"))

    def test_numbered_lock_reverification_refuses_replaced_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            writer = places["writer"]
            project = places["project"]
            real_stat = os.stat

            def swapping_stat(path, *args, **kwargs):
                if path == ".pipeline" and kwargs.get("dir_fd") is not None:
                    pipeline = project / ".pipeline"
                    pipeline.rename(project / ".pipeline-orig")
                    pipeline.mkdir()
                return real_stat(path, *args, **kwargs)

            with mock.patch(
                "fcp_pipeline.safe_write.os.stat", side_effect=swapping_stat
            ):
                with self.assertRaises(SafeWriteError) as raised:
                    writer.write_numbered("cut.fcpxml", b"data")
            self.assertIn("changed while taking the number lock", str(raised.exception))
            self.assertEqual([entry.name for entry in (project / ".pipeline").iterdir()], [])
            self.assertFalse(os.path.lexists(project / "edit" / "001-cut.fcpxml"))

    def test_numbered_lock_failure_leaves_no_lock_file_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            writer = places["writer"]
            project = places["project"]
            pipeline = project / ".pipeline"
            with mock.patch(
                "fcp_pipeline.safe_write.fcntl.flock",
                side_effect=OSError("lock boom"),
            ):
                with self.assertRaises(SafeWriteError):
                    writer.write_numbered("cut.fcpxml", b"data")
            self.assertFalse(os.path.lexists(pipeline / "safe-write.lock"))
            self.assertEqual(
                [entry.name for entry in pipeline.iterdir()]
                if pipeline.exists()
                else [],
                [],
            )
            edit = project / "edit"
            self.assertEqual(list(edit.iterdir()) if edit.exists() else [], [])
            pipeline.mkdir(parents=True, exist_ok=True)
            kept = pipeline / "safe-write.lock"
            kept.write_bytes(b"kept")
            with mock.patch(
                "fcp_pipeline.safe_write.fcntl.flock",
                side_effect=OSError("lock boom"),
            ):
                with self.assertRaises(SafeWriteError):
                    writer.write_numbered("cut.fcpxml", b"data")
            self.assertEqual(kept.read_bytes(), b"kept")
            self.assertEqual([entry.name for entry in pipeline.iterdir()], ["safe-write.lock"])
            self.assertEqual(list(edit.iterdir()) if edit.exists() else [], [])

    def test_numbered_fails_closed_without_lock(self):
        cases = (
            ("flock OSError", "flock"),
            ("locking unavailable", "no-fcntl"),
        )
        for label, mode in cases:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as tmp:
                    places = self.make_trip(Path(tmp))
                    writer = places["writer"]
                    project = places["project"]
                    if mode == "flock":
                        patcher = mock.patch(
                            "fcp_pipeline.safe_write.fcntl.flock",
                            side_effect=OSError("lock boom"),
                        )
                    else:
                        patcher = mock.patch(
                            "fcp_pipeline.safe_write.fcntl", None
                        )
                    with patcher:
                        with self.assertRaises(SafeWriteError):
                            writer.write_numbered("cut.fcpxml", b"data")
                    edit = project / "edit"
                    self.assertEqual(
                        list(edit.iterdir()) if edit.exists() else [], []
                    )
                    if mode == "no-fcntl":
                        self.assertFalse(
                            os.path.lexists(
                                project / ".pipeline" / "safe-write.lock"
                            )
                        )

    def test_numbered_names_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            places = self.make_trip(Path(tmp))
            writer = places["writer"]
            for bad in ("", "a/b", ".", "..", "back\\slash"):
                with self.subTest(name=bad):
                    with self.assertRaises(SafeWriteError):
                        writer.write_numbered(bad, b"nope")
            edit = places["project"] / "edit"
            edit.mkdir(parents=True)
            (edit / "999-last.fcpxml").write_bytes(b"last")
            with self.assertRaises(SafeWriteError):
                writer.write_numbered("next.fcpxml", b"nope")
            self.assertEqual(
                sorted(p.name for p in edit.iterdir()), ["999-last.fcpxml"]
            )


if __name__ == "__main__":
    unittest.main()
