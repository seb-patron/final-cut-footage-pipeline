"""Tests for the hardened FCPXML reader in fcp_pipeline.fcpxml_read."""
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
import xml.etree.ElementTree as ET

from fcp_pipeline.fcpxml_read import parse_xml, parse_xml_bytes, same_annotation, xml_bytes


class TestFcpxmlRead(unittest.TestCase):
    def test_utf16_input_refused(self):
        raw = '<fcpxml version="1.13"/>'.encode('utf-16')
        with self.assertRaises(ValueError):
            parse_xml_bytes(raw)

    def test_entity_declaration_refused(self):
        raw = b'<!DOCTYPE fcpxml [<!ENTITY x "expanded">]><fcpxml>&x;</fcpxml>'
        with self.assertRaises(ValueError):
            parse_xml_bytes(raw)

    def test_doctype_with_subset_or_system_id_refused(self):
        with self.assertRaises(ValueError):
            parse_xml_bytes(b'<!DOCTYPE fcpxml [<!ENTITY x "e">]><fcpxml/>')
        with self.assertRaises(ValueError):
            parse_xml_bytes(b'<!DOCTYPE fcpxml SYSTEM "file:///tmp/example"><fcpxml/>')

    def test_bare_doctype_accepted_with_comments_kept(self):
        raw = b'<!DOCTYPE fcpxml>\n<fcpxml><!-- kept --><resources/></fcpxml>'
        tree = parse_xml_bytes(raw)
        self.assertEqual(tree.tag, 'fcpxml')
        comments = [node for node in tree.iter() if node.tag is ET.Comment]
        self.assertTrue(comments)
        self.assertIn('kept', comments[0].text)

    def test_non_fcpxml_root_refused(self):
        with self.assertRaises(ValueError):
            parse_xml_bytes(b'<project></project>')

    def test_fcpxmld_folder_with_second_file_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'export.fcpxmld'
            bundle.mkdir()
            info = bundle / 'Info.fcpxml'
            info.write_text('<fcpxml/>', encoding='utf-8')
            (bundle / 'sidecar.bin').write_bytes(b'auxiliary')
            with self.assertRaisesRegex(ValueError, 'sidecars'):
                xml_bytes(info)
            lone = Path(directory) / 'Info.fcpxml'
            lone.write_text('<fcpxml/>', encoding='utf-8')
            self.assertEqual(xml_bytes(lone), b'<fcpxml/>')

    def test_parse_xml_reads_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.fcpxml'
            path.write_text('<!DOCTYPE fcpxml><fcpxml/>', encoding='utf-8')
            self.assertEqual(parse_xml(path).tag, 'fcpxml')

    def test_same_annotation_keyword_in_list_and_rational_times(self):
        clip_start = Fraction(3600)
        clip_duration = Fraction(10)
        element = ET.fromstring(
            '<keyword value="Other, Scene - Water" start="108029029/30000s" duration="31031/30000s"/>')
        attrs = dict(start='108029029/30000s', duration='31031/30000s', value='Scene - Water')
        self.assertTrue(same_annotation(element, 'keyword', attrs, clip_start, clip_duration))
        # Equal times written as different fractions still match.
        element2 = ET.fromstring('<keyword value="Scene - Water" start="3600s" duration="1s"/>')
        attrs2 = dict(start='3600/1s', duration='60/60s', value='Scene - Water')
        self.assertTrue(same_annotation(element2, 'keyword', attrs2, clip_start, clip_duration))
        # Different label or range does not match.
        self.assertFalse(same_annotation(
            element2, 'keyword', dict(attrs2, value='Other'), clip_start, clip_duration))
        self.assertFalse(same_annotation(
            element2, 'keyword', dict(attrs2, start='3601s'), clip_start, clip_duration))
        # Implicit range defaults to the clip range.
        bare = ET.fromstring('<keyword value="Scene - Water"/>')
        whole = dict(start='3600s', duration='10s', value='Scene - Water')
        self.assertTrue(same_annotation(bare, 'keyword', whole, clip_start, clip_duration))

    def test_same_annotation_marker(self):
        clip_start = Fraction(3600)
        clip_duration = Fraction(12)
        element = ET.fromstring('<marker value="Note" start="3601s" duration="7s"/>')
        attrs = dict(start='3601s', duration='7s', value='Note')
        self.assertTrue(same_annotation(element, 'marker', attrs, clip_start, clip_duration))
        self.assertFalse(same_annotation(element, 'keyword', attrs, clip_start, clip_duration))

    def test_prototype_uses_package(self):
        import footage_keywords
        import fcp_pipeline.fcptime
        import fcp_pipeline.fcpxml_read
        self.assertIs(footage_keywords.seconds, fcp_pipeline.fcptime.seconds)
        self.assertIs(footage_keywords.parse_xml_bytes, fcp_pipeline.fcpxml_read.parse_xml_bytes)


if __name__ == '__main__':
    unittest.main()
