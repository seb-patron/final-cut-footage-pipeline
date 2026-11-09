"""Tests for the offline FCPXML check in ``fcp_pipeline.bridge.check``.

Every document is synthetic, with invented names, written into a
temporary folder. Media files are small placeholder files (a few bytes,
not real video), since only existence is checked. Tests that need the
installed DTDs use :func:`dtd_folder` and skip with a reason naming the
folder, the missing DTD file or xmllint when it is absent. Invented DTDs
are tiny hand-written files in temporary folders; Apple's DTD text is
never copied.
"""

import hashlib
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest import mock

from fcp_pipeline.bridge.check import (
    CANNOT_VERIFY,
    INVALID,
    REFUSED,
    VALID,
    check_fcpxml,
    dtd_folder,
)
from tests.golden import assert_dtd_valid

ONE_CLIP_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="{version}">
  <resources>
    <format id="r1" name="Invented Format" frameDuration="100/2500s" width="1280" height="720"/>
    <asset id="r2" name="Invented Clip" uid="INVENTED-ASSET-UID" format="r1" duration="12s">
      <media-rep kind="original-media" src="{media_src}"/>
    </asset>
  </resources>
  <library>
    <event name="Invented Event" uid="INVENTED-EVENT-UID">
      <asset-clip ref="r2" format="r1" name="Invented Clip" start="3600s" duration="12s"/>
      <project name="Invented Project" uid="INVENTED-PROJECT-UID">
        <sequence format="r1">
          <spine>
            <asset-clip ref="r2" format="r1" name="Invented Clip" start="3600s" duration="12s"/>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
"""

MINIMAL_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="{version}"/>
"""

INVENTED_EMPTY_DTD = (
    "<!ELEMENT fcpxml EMPTY>\n"
    '<!ATTLIST fcpxml version CDATA #FIXED "{version}">\n'
)


def write_one_clip(directory, version="1.14", media_src=None, name=None):
    """Write the one-clip document; return its path.

    When ``media_src`` is None, a placeholder media file is written next
    to it and its URI is used, so the document can validate.
    """
    directory = Path(directory)
    if media_src is None:
        media = directory / "invented-clip.mov"
        media.write_bytes(b"placeholder")
        media_src = media.resolve().as_uri()
    if name is None:
        name = "clip-%s.fcpxml" % version.replace(".", "-")
    doc = directory / name
    doc.write_text(
        ONE_CLIP_TEMPLATE.format(version=version, media_src=media_src),
        encoding="utf-8",
    )
    return doc


def require_installed_dtd(testcase, version="1.14"):
    """Return the installed DTD file, or skip naming what is missing."""
    folder = dtd_folder()
    minor = version.split(".")[1]
    dtd = folder / ("FCPXMLv1_%s.dtd" % minor)
    if not dtd.is_file():
        testcase.skipTest("installed DTD not found: %s" % (dtd,))
    if shutil.which("xmllint") is None:
        testcase.skipTest("xmllint not on PATH")
    return dtd


class TestCheck(unittest.TestCase):
    def test_valid_one_clip_passes(self):
        require_installed_dtd(self, "1.14")
        with tempfile.TemporaryDirectory() as tmp:
            doc = write_one_clip(tmp, version="1.14")
            result = check_fcpxml(str(doc))
            self.assertEqual(result.status, VALID)
            self.assertTrue(result.ok)
            self.assertEqual(result.path, str(doc))
            self.assertEqual(result.version, "1.14")
            self.assertEqual(result.events, ["Invented Event"])
            self.assertEqual(result.projects, ["Invented Project"])
            self.assertEqual(
                result.sha256,
                hashlib.sha256(Path(doc).read_bytes()).hexdigest(),
            )
            self.assertEqual(result.dtd.name, "FCPXMLv1_14.dtd")
            self.assertEqual(result.missing_media, [])
            self.assertEqual(result.problems, [])

    def test_broken_file_fails_with_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = write_one_clip(tmp, version="1.14")
            text = Path(doc).read_text(encoding="utf-8")
            bogus = text.replace(
                "</fcpxml>", "  <bogus-element/>\n</fcpxml>")
            bogus_path = Path(tmp) / "bogus.fcpxml"
            bogus_path.write_text(bogus, encoding="utf-8")
            bogus_line = bogus.splitlines().index("  <bogus-element/>") + 1
            malformed = (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                "<!DOCTYPE fcpxml>\n"
                '<fcpxml version="1.14">\n'
                "<library>\n"
                "</fcpxml>\n"
            )
            malformed_path = Path(tmp) / "malformed.fcpxml"
            malformed_path.write_text(malformed, encoding="utf-8")
            malformed_line = malformed.splitlines().index("</fcpxml>") + 1
            with self.subTest("bogus-element"):
                require_installed_dtd(self, "1.14")
                result = check_fcpxml(str(bogus_path))
                self.assertEqual(result.status, INVALID)
                self.assertFalse(result.ok)
                matches = [
                    problem for problem in result.problems
                    if problem.element == "bogus-element"
                ]
                self.assertTrue(matches, result.problems)
                self.assertIn(bogus_line,
                              [problem.line for problem in matches])
            with self.subTest("malformed"):
                with mock.patch(
                    "subprocess.run",
                    side_effect=AssertionError("subprocess must not run"),
                ):
                    result = check_fcpxml(str(malformed_path))
                self.assertEqual(result.status, INVALID)
                self.assertFalse(result.ok)
                self.assertTrue(
                    any(problem.line == malformed_line
                        for problem in result.problems),
                    result.problems,
                )

    def test_external_entity_refused(self):
        cases = ("system-entity", "parameter-entity", "system-doctype")
        for kind in cases:
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as tmp:
                    sentinel = "InventedSentinelSevenSeven"
                    sentinel_path = Path(tmp) / "sentinel.txt"
                    sentinel_path.write_text(sentinel, encoding="utf-8")
                    sentinel_uri = sentinel_path.resolve().as_uri()
                    missing_src = "file:///InventedNowhere/missing.mov"
                    if kind == "system-entity":
                        doctype = (
                            "<!DOCTYPE fcpxml [\n"
                            '  <!ENTITY invented SYSTEM "%s">\n'
                            "]>" % (sentinel_uri,)
                        )
                        body = (
                            "<library><event name=\"&invented;\">"
                            "</event></library>"
                        )
                    elif kind == "parameter-entity":
                        doctype = (
                            "<!DOCTYPE fcpxml [\n"
                            '  <!ENTITY % invented "ignored">\n'
                            "  %invented;\n"
                            "]>"
                        )
                        body = "<library/>"
                    else:
                        doctype = (
                            '<!DOCTYPE fcpxml SYSTEM '
                            '"http://example.invalid/x.dtd">'
                        )
                        body = "<library/>"
                    doc = (
                        '<?xml version="1.0" encoding="UTF-8"?>\n'
                        "%s\n"
                        '<fcpxml version="1.14">\n'
                        "  <resources>\n"
                        '    <asset id="r1">\n'
                        '      <media-rep kind="original-media" src="%s"/>\n'
                        "    </asset>\n"
                        "  </resources>\n"
                        "  %s\n</fcpxml>\n"
                    ) % (doctype, missing_src, body)
                    doc_path = Path(tmp) / ("refused-%s.fcpxml" % kind)
                    doc_path.write_text(doc, encoding="utf-8")
                    with mock.patch(
                        "subprocess.run",
                        side_effect=AssertionError(
                            "subprocess must not run"),
                    ), mock.patch(
                        "socket.socket",
                        side_effect=AssertionError("no network"),
                    ):
                        result = check_fcpxml(str(doc_path))
                    self.assertEqual(result.status, REFUSED)
                    self.assertFalse(result.ok)
                    self.assertEqual(result.missing_media, [])
                    self.assertTrue(result.problems, result)
                    fields = (
                        result.status, result.ok, result.sha256,
                        result.version, result.dtd, result.events,
                        result.projects, result.missing_media,
                        result.problems, result.path,
                    )
                    self.assertNotIn(sentinel, repr(fields))

    def test_missing_media_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "gone.mov"
            src = missing.resolve().as_uri()
            doc = write_one_clip(tmp, version="1.14", media_src=src)
            expected = urllib.parse.unquote(
                urllib.parse.urlsplit(src).path)
            with mock.patch.dict(
                os.environ,
                {"FCP_DTD_DIR": str(Path(tmp) / "no-such-dtd")},
            ):
                result = check_fcpxml(str(doc))
            self.assertEqual(result.status, INVALID)
            self.assertFalse(result.ok)
            self.assertEqual(result.missing_media, [expected])
            with self.subTest("check-media-off"):
                with mock.patch.dict(
                    os.environ,
                    {"FCP_DTD_DIR": str(Path(tmp) / "no-such-dtd")},
                ):
                    skipped = check_fcpxml(str(doc), check_media=False)
                self.assertEqual(skipped.missing_media, [])

    def test_uses_dtd_for_file_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            dtd_dir = Path(tmp) / "dtd"
            dtd_dir.mkdir()
            (dtd_dir / "FCPXMLv1_10.dtd").write_text(
                INVENTED_EMPTY_DTD.format(version="1.10"), encoding="utf-8")
            (dtd_dir / "FCPXMLv1_14.dtd").write_text(
                INVENTED_EMPTY_DTD.format(version="1.14"), encoding="utf-8")
            doc10 = Path(tmp) / "ten.fcpxml"
            doc10.write_text(
                MINIMAL_TEMPLATE.format(version="1.10"), encoding="utf-8")
            doc14 = Path(tmp) / "fourteen.fcpxml"
            doc14.write_text(
                MINIMAL_TEMPLATE.format(version="1.14"), encoding="utf-8")
            with self.subTest("uses-version-dtd"):
                if shutil.which("xmllint") is None:
                    self.skipTest("xmllint not on PATH")
                result10 = check_fcpxml(str(doc10), dtd_dir=str(dtd_dir))
                self.assertEqual(result10.status, VALID)
                self.assertTrue(result10.ok)
                self.assertEqual(result10.dtd, dtd_dir / "FCPXMLv1_10.dtd")
                result14 = check_fcpxml(str(doc14), dtd_dir=str(dtd_dir))
                self.assertEqual(result14.status, VALID)
                self.assertTrue(result14.ok)
                self.assertEqual(result14.dtd, dtd_dir / "FCPXMLv1_14.dtd")
            with self.subTest("no-fallback"):
                (dtd_dir / "FCPXMLv1_10.dtd").unlink()
                result = check_fcpxml(str(doc10), dtd_dir=str(dtd_dir))
                self.assertEqual(result.status, CANNOT_VERIFY)
                self.assertFalse(result.ok)
                self.assertTrue(
                    any("FCPXMLv1_10.dtd" in problem.message
                        for problem in result.problems),
                    result.problems,
                )
            with self.subTest("installed-1.10"):
                require_installed_dtd(self, "1.10")
                doc = write_one_clip(tmp, version="1.10")
                result = check_fcpxml(str(doc))
                self.assertEqual(result.status, VALID)
                self.assertEqual(result.dtd.name, "FCPXMLv1_10.dtd")

    def test_unknown_versions_never_raise_and_cannot_verify(self):
        big = "1." + "7" * 5000
        cases = (
            ("long-digits", big),
            ("non-numeric", "1.x"),
            ("out-of-range", "2.0"),
            ("extra-component", "1.14.1"),
            ("empty", ""),
            ("whitespace", " 1.14"),
            ("leading-zero", "01.14"),
        )
        for label, declared in cases:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as tmp:
                    doc = Path(tmp) / ("unknown-%s.fcpxml" % label)
                    doc.write_text(
                        MINIMAL_TEMPLATE.format(version=declared),
                        encoding="utf-8",
                    )
                    with mock.patch(
                        "subprocess.run",
                        side_effect=AssertionError(
                            "subprocess must not run"),
                    ):
                        result = check_fcpxml(str(doc))
                    self.assertEqual(result.status, CANNOT_VERIFY)
                    self.assertFalse(result.ok)
                    self.assertIsNone(result.version)
                    self.assertIsNone(result.dtd)
                    self.assertEqual(result.missing_media, [])
                    self.assertTrue(result.problems, result.problems)
                    self.assertTrue(
                        any("unsupported FCPXML version" in problem.message
                            for problem in result.problems),
                        result.problems,
                    )
        with self.subTest(label="missing"):
            with tempfile.TemporaryDirectory() as tmp:
                doc = Path(tmp) / "unknown-missing.fcpxml"
                doc.write_text(
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    "<!DOCTYPE fcpxml>\n"
                    "<fcpxml/>\n",
                    encoding="utf-8",
                )
                with mock.patch(
                    "subprocess.run",
                    side_effect=AssertionError("subprocess must not run"),
                ):
                    result = check_fcpxml(str(doc))
                self.assertEqual(result.status, CANNOT_VERIFY)
                self.assertFalse(result.ok)
                self.assertIsNone(result.version)
                self.assertIsNone(result.dtd)
                self.assertEqual(result.missing_media, [])
                self.assertTrue(
                    any("missing version attribute" in problem.message
                        for problem in result.problems),
                    result.problems,
                )

    def test_hash_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = (
                b'<?xml version="1.0" encoding="UTF-8"?>\n'
                b"<!DOCTYPE fcpxml>\n"
                b'<fcpxml version="1.14"/>\n'
            )
            first = Path(tmp) / "first.fcpxml"
            second = Path(tmp) / "second.fcpxml"
            first.write_bytes(payload)
            second.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(check_fcpxml(str(first)).sha256, expected)
            self.assertEqual(check_fcpxml(str(second)).sha256, expected)
            changed = payload.replace(b"1.14", b"1.13")
            third = Path(tmp) / "third.fcpxml"
            third.write_bytes(changed)
            self.assertNotEqual(
                check_fcpxml(str(third)).sha256, expected)
            bundle = Path(tmp) / "clip.fcpxmld"
            bundle.mkdir()
            (bundle / "Info.fcpxml").write_bytes(payload)
            self.assertEqual(
                check_fcpxml(str(bundle)).sha256, expected)

    def test_no_final_cut_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = str(Path(tmp) / "missing-dtd-folder")
            doc = Path(tmp) / "plain.fcpxml"
            doc.write_text(
                MINIMAL_TEMPLATE.format(version="1.14"), encoding="utf-8")
            with mock.patch.dict(os.environ, {"FCP_DTD_DIR": folder}):
                result = check_fcpxml(str(doc))
            self.assertEqual(result.status, CANNOT_VERIFY)
            self.assertFalse(result.ok)
            self.assertTrue(
                any(folder in problem.message
                    for problem in result.problems),
                result.problems,
            )

    def test_missing_xmllint(self):
        with tempfile.TemporaryDirectory() as tmp:
            dtd_dir = Path(tmp) / "dtd"
            dtd_dir.mkdir()
            (dtd_dir / "FCPXMLv1_14.dtd").write_text(
                INVENTED_EMPTY_DTD.format(version="1.14"), encoding="utf-8")
            doc = Path(tmp) / "plain.fcpxml"
            doc.write_text(
                MINIMAL_TEMPLATE.format(version="1.14"), encoding="utf-8")
            with mock.patch("shutil.which", return_value=None):
                result = check_fcpxml(str(doc), dtd_dir=str(dtd_dir))
            self.assertEqual(result.status, CANNOT_VERIFY)
            self.assertFalse(result.ok)
            self.assertTrue(
                any("xmllint" in problem.message
                    for problem in result.problems),
                result.problems,
            )

    def test_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = write_one_clip(root, version="1.14")
            refused = root / "refused.fcpxml"
            refused.write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<!DOCTYPE fcpxml SYSTEM "http://example.invalid/x.dtd">\n'
                '<fcpxml version="1.14"/>\n',
                encoding="utf-8",
            )
            missing = write_one_clip(
                root, version="1.14", name="missing.fcpxml",
                media_src="file:///InventedNowhere/missing.mov",
            )
            before = _snapshot_tree(root)
            check_fcpxml(str(valid))
            check_fcpxml(str(refused))
            check_fcpxml(str(missing))
            self.assertEqual(_snapshot_tree(root), before)

    def test_command_line_exit_codes(self):
        repo_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            missing_doc = write_one_clip(
                tmp, version="1.14",
                media_src="file:///InventedNowhere/missing.mov",
            )
            refused_doc = Path(tmp) / "refused.fcpxml"
            refused_doc.write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<!DOCTYPE fcpxml SYSTEM "http://example.invalid/x.dtd">\n'
                '<fcpxml version="1.14"/>\n',
                encoding="utf-8",
            )
            plain_doc = Path(tmp) / "plain.fcpxml"
            plain_doc.write_text(
                MINIMAL_TEMPLATE.format(version="1.14"), encoding="utf-8")
            folder = str(Path(tmp) / "missing-dtd-folder")
            env = dict(os.environ)
            env["FCP_DTD_DIR"] = folder
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            cases = (
                (missing_doc, 1, "invalid"),
                (refused_doc, 1, "refused"),
                (plain_doc, 4, "cannot verify"),
            )
            for doc, code, word in cases:
                with self.subTest(doc=doc.name, code=code):
                    completed = subprocess.run(
                        [sys.executable, "-m", "fcp_pipeline.bridge.check",
                         str(doc)],
                        cwd=str(repo_root),
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    self.assertEqual(completed.returncode, code,
                                     completed.stderr)
                    lines = completed.stdout.splitlines()
                    self.assertTrue(lines, completed)
                    self.assertEqual(lines[0], word)
                    if word == "cannot verify":
                        self.assertIn(folder, completed.stdout)

    def test_golden_helper_uses_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain_doc = Path(tmp) / "plain.fcpxml"
            plain_doc.write_text(
                MINIMAL_TEMPLATE.format(version="1.14"), encoding="utf-8")
            folder = str(Path(tmp) / "missing-dtd-folder")
            with mock.patch.dict(os.environ, {"FCP_DTD_DIR": folder}):
                with mock.patch.dict(os.environ):
                    os.environ.pop("FCP_DTD", None)
                    with self.assertRaises(unittest.SkipTest) as ctx:
                        assert_dtd_valid(str(plain_doc))
            self.assertIn(folder, str(ctx.exception))
            if shutil.which("xmllint") is None:
                self.skipTest("xmllint not on PATH")
            dtd_dir = Path(tmp) / "dtd"
            dtd_dir.mkdir()
            dtd_file = dtd_dir / "FCPXMLv1_14.dtd"
            dtd_file.write_text(
                "<!ELEMENT fcpxml (resources?)>\n"
                "<!ELEMENT resources (asset*)>\n"
                "<!ELEMENT asset (media-rep*)>\n"
                "<!ELEMENT media-rep EMPTY>\n"
                '<!ATTLIST fcpxml version CDATA #IMPLIED>\n'
                '<!ATTLIST asset id ID #IMPLIED>\n'
                '<!ATTLIST media-rep kind CDATA #IMPLIED '
                'src CDATA #IMPLIED>\n',
                encoding="utf-8",
            )
            accepted = Path(tmp) / "accepted.fcpxml"
            accepted.write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                "<!DOCTYPE fcpxml>\n"
                '<fcpxml version="1.14">\n'
                "  <resources>\n"
                '    <asset id="r1">\n'
                '      <media-rep kind="original-media" '
                'src="file:///InventedNowhere/missing.mov"/>\n'
                "    </asset>\n"
                "  </resources>\n</fcpxml>\n",
                encoding="utf-8",
            )
            rejected = Path(tmp) / "rejected.fcpxml"
            rejected.write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                "<!DOCTYPE fcpxml>\n"
                '<fcpxml version="1.14">\n'
                "  <bogus-element/>\n</fcpxml>\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"FCP_DTD_DIR": str(dtd_dir)}):
                with mock.patch.dict(os.environ):
                    os.environ.pop("FCP_DTD", None)
                    self.assertIsNone(assert_dtd_valid(str(accepted)))
                    with self.assertRaises(AssertionError) as ctx:
                        assert_dtd_valid(str(rejected))
                    self.assertIn(str(rejected), str(ctx.exception))
                with mock.patch.dict(
                    os.environ, {"FCP_DTD": str(dtd_file)},
                ):
                    self.assertIsNone(assert_dtd_valid(str(accepted)))
                    with self.assertRaises(AssertionError):
                        assert_dtd_valid(str(rejected))
                with mock.patch.dict(os.environ, {"FCP_DTD_DIR": str(dtd_dir)}):
                    with mock.patch.dict(os.environ):
                        os.environ.pop("FCP_DTD", None)
                        checked = check_fcpxml(str(rejected))
                bogus_line = rejected.read_text(
                    encoding="utf-8").splitlines().index(
                        "  <bogus-element/>") + 1
                self.assertEqual(checked.status, INVALID)
                self.assertTrue(
                    any(problem.element == "bogus-element"
                        and problem.line == bogus_line
                        for problem in checked.problems),
                    checked.problems,
                )


def _snapshot_tree(root):
    """Map each entry under ``root`` to its size, mtime and kind."""
    root = Path(root)
    data = {}
    entries = [root] + sorted(root.rglob("*"))
    for entry in entries:
        try:
            stat = entry.stat()
        except OSError:
            continue
        data[str(entry.relative_to(root))] = (
            stat.st_size, stat.st_mtime_ns, entry.is_dir())
    return data


if __name__ == "__main__":
    unittest.main()
