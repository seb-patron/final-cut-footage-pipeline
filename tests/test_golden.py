"""Tests for the golden FCPXML harness in tests/golden.py.

All snippets are hand-written synthetic documents with invented names.
Goldens are written only into a temporary golden_dir, never into
tests/golden/.
"""

import contextlib
import os
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

from tests.golden import (
    DEFAULT_DTD_PATH,
    assert_dtd_valid,
    assert_golden,
    canonicalize,
)

DOC_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <resources>
    <format id="{format_id}" name="FFVideoFormat720p" frameDuration="100/2500s" width="1280" height="720"/>
    <asset id="{asset_id}" name="Synthetic Clip" uid="{asset_uid}" format="{format_id}" duration="12s">
      <media-rep kind="original-media" src="{media_src}"/>
    </asset>
    <media id="{media_id}" uid="{media_uid}" name="Synthetic Media"/>
    <effect id="{effect_id}" uid="{effect_uid}" name="Synthetic Glow"/>
  </resources>
  <library>
    <event name="Synthetic Event" uid="{event_uid}">
      <project name="Synthetic Project" uid="{project_uid}">
        <sequence format="{format_id}">
          <spine>
            <asset-clip ref="{asset_id}" format="{format_id}" name="Synthetic Clip" start="3600s" duration="12s"/>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
"""

ORDER_A = (
    b'<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE fcpxml>'
    b'<fcpxml version="1.14"><library>'
    b'<event name="Synthetic Event" uid="EVENT-A">'
    b'<asset-clip ref="r9" format="r7" name="Synthetic Clip" start="3600s" duration="12s"/>'
    b"</event></library></fcpxml>"
)

ORDER_B = (
    b'<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE fcpxml>'
    b'<fcpxml version="1.14"><library>'
    b'<event uid="EVENT-A" name="Synthetic Event">'
    b'<asset-clip duration="12s" start="3600s" name="Synthetic Clip" format="r7" ref="r9"/>'
    b"</event></library></fcpxml>"
)


def make_doc(format_id, asset_id, media_id, effect_id, asset_uid,
             media_uid, event_uid, project_uid, effect_uid, media_src):
    return DOC_TEMPLATE.format(
        format_id=format_id, asset_id=asset_id, media_id=media_id,
        effect_id=effect_id, asset_uid=asset_uid, media_uid=media_uid,
        event_uid=event_uid, project_uid=project_uid, effect_uid=effect_uid,
        media_src=media_src,
    ).encode("utf-8")


def temp_media_uri(random_folder):
    return (Path(tempfile.gettempdir()) / random_folder / "media" / "a.mov").as_uri()


@contextlib.contextmanager
def without_golden_update():
    """Hide GOLDEN_UPDATE from the environment, then restore it."""
    with mock.patch.dict(os.environ):
        os.environ.pop("GOLDEN_UPDATE", None)
        yield


class TestGolden(unittest.TestCase):
    def test_ignores_attribute_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            golden_dir = Path(tmp) / "goldens"
            with mock.patch.dict(os.environ, {"GOLDEN_UPDATE": "1"}):
                assert_golden(ORDER_A, "order", golden_dir=golden_dir)
            with without_golden_update():
                self.assertIsNone(assert_golden(ORDER_A, "order", golden_dir=golden_dir))
                self.assertIsNone(assert_golden(ORDER_B, "order", golden_dir=golden_dir))

    def test_mismatch_shows_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            golden_dir = Path(tmp) / "goldens"
            with mock.patch.dict(os.environ, {"GOLDEN_UPDATE": "1"}):
                assert_golden(ORDER_A, "clip", golden_dir=golden_dir)
            changed = ORDER_A.replace(b'name="Synthetic Clip"', b'name="Changed Clip"')
            with without_golden_update():
                with self.assertRaises(AssertionError) as ctx:
                    assert_golden(changed, "clip", golden_dir=golden_dir)
        message = str(ctx.exception)
        self.assertIn("asset-clip", message)
        lines = message.splitlines()
        self.assertTrue(any(line.startswith("-") for line in lines), message)
        self.assertTrue(any(line.startswith("+") for line in lines), message)

    def test_leaf_whitespace_significant(self):
        with_space = canonicalize(b"<fcpxml><title> </title></fcpxml>")
        empty = canonicalize(b"<fcpxml><title/></fcpxml>")
        self.assertNotEqual(with_space, empty)
        self.assertEqual(
            canonicalize(b"<fcpxml> <title/> </fcpxml>"),
            canonicalize(b"<fcpxml><title/></fcpxml>"),
        )

    def test_malformed_golden_reports_assertion(self):
        with tempfile.TemporaryDirectory() as tmp:
            golden_dir = Path(tmp) / "goldens"
            with mock.patch.dict(os.environ, {"GOLDEN_UPDATE": "1"}):
                assert_golden(ORDER_A, "clip", golden_dir=golden_dir)
            golden_path = golden_dir / "clip.fcpxml"
            golden_path.write_text("<fcpxml><unclosed>", encoding="utf-8")
            with without_golden_update():
                with self.assertRaises(AssertionError) as ctx:
                    assert_golden(ORDER_A, "clip", golden_dir=golden_dir)
            message = str(ctx.exception)
            self.assertIn(str(golden_path), message)
            self.assertIn("@@", message)
            self.assertIsInstance(ctx.exception.__cause__, ET.ParseError)

    def test_undecodable_golden_reports_assertion(self):
        with tempfile.TemporaryDirectory() as tmp:
            golden_dir = Path(tmp) / "goldens"
            golden_dir.mkdir(parents=True)
            golden_path = golden_dir / "clip.fcpxml"
            golden_path.write_bytes(b"\xff")
            with without_golden_update():
                with self.assertRaises(AssertionError) as ctx:
                    assert_golden(ORDER_A, "clip", golden_dir=golden_dir)
            message = str(ctx.exception)
            self.assertIn(str(golden_path), message)
            self.assertIsInstance(ctx.exception.__cause__, UnicodeDecodeError)

    def test_dtd_skips_when_missing(self):
        missing = "/nonexistent/FCPXMLv1_14.dtd"
        with mock.patch.dict(os.environ, {"FCP_DTD": missing}):
            with self.assertRaises(unittest.SkipTest) as ctx:
                assert_dtd_valid("whatever.fcpxml")
        self.assertIn(missing, str(ctx.exception))

    def test_dtd_rejects_invalid(self):
        dtd = Path(os.environ.get("FCP_DTD", DEFAULT_DTD_PATH))
        if not dtd.is_file() or shutil.which("xmllint") is None:
            self.skipTest("Final Cut 1.14 DTD or xmllint unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "invalid.fcpxml"
            bad.write_bytes(
                b'<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE fcpxml>'
                b'<fcpxml version="1.14"><bogus-element/></fcpxml>'
            )
            with self.assertRaises(AssertionError):
                assert_dtd_valid(bad)
            good = Path(tmp) / "valid.fcpxml"
            good.write_bytes(
                b'<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE fcpxml>'
                b'<fcpxml version="1.14"/>'
            )
            self.assertIsNone(assert_dtd_valid(good))

    def test_ids_uids_and_temp_uris_normalized(self):
        shared = dict(effect_uid="com.synthetic.glow",
                      media_src=temp_media_uri("tmpAAA111"))
        doc_a = make_doc(format_id="r30", asset_id="r41", media_id="r42",
                         effect_id="r43", asset_uid="ASSET-AAA",
                         media_uid="MEDIA-AAA", event_uid="EVENT-AAA",
                         project_uid="PROJECT-AAA", **shared)
        doc_b = make_doc(format_id="r7", asset_id="r9", media_id="r12",
                         effect_id="r13", asset_uid="ASSET-BBB",
                         media_uid="MEDIA-BBB", event_uid="EVENT-BBB",
                         project_uid="PROJECT-BBB",
                         effect_uid="com.synthetic.glow",
                         media_src=temp_media_uri("tmpBBB222"))
        with tempfile.TemporaryDirectory() as tmp:
            golden_dir = Path(tmp) / "goldens"
            with mock.patch.dict(os.environ, {"GOLDEN_UPDATE": "1"}):
                assert_golden(doc_a, "norm", golden_dir=golden_dir)
            with without_golden_update():
                self.assertIsNone(assert_golden(doc_a, "norm", golden_dir=golden_dir))
                self.assertIsNone(assert_golden(doc_b, "norm", golden_dir=golden_dir))
                doc_c = make_doc(format_id="r7", asset_id="r9", media_id="r12",
                                 effect_id="r13", asset_uid="ASSET-BBB",
                                 media_uid="MEDIA-BBB", event_uid="EVENT-BBB",
                                 project_uid="PROJECT-BBB",
                                 effect_uid="com.synthetic.other",
                                 media_src=temp_media_uri("tmpBBB222"))
                with self.assertRaises(AssertionError):
                    assert_golden(doc_c, "norm", golden_dir=golden_dir)

    def test_update_writes_golden(self):
        with tempfile.TemporaryDirectory() as tmp:
            golden_dir = Path(tmp) / "goldens"
            with mock.patch.dict(os.environ, {"GOLDEN_UPDATE": "1"}):
                self.assertIsNone(assert_golden(ORDER_A, "fresh", golden_dir=golden_dir))
            written = golden_dir / "fresh.fcpxml"
            self.assertTrue(written.is_file())
            self.assertEqual(written.read_text(encoding="utf-8"), canonicalize(ORDER_A))
            with without_golden_update():
                self.assertIsNone(assert_golden(ORDER_A, "fresh", golden_dir=golden_dir))

    def test_rejects_name_that_escapes_golden_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ("../escape", "sub/dir", "", "has space", "dot.name"):
                with self.subTest(name=bad):
                    with self.assertRaises(ValueError):
                        assert_golden(ORDER_A, bad, golden_dir=Path(tmp))


if __name__ == "__main__":
    unittest.main()
