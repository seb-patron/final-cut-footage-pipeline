"""Replay sanitized native evidence; these tests do not launch Final Cut.

A passing recorded-loss test confirms a known limitation, NOT successful native
metadata preservation. See docs/native-validation.md before native imports.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from footage_keywords import bind, parse_xml_bytes, seconds
from tests.fixture_library import EVENT, FIXTURE, SOURCE_PLACEHOLDER, proposal_document, materialize

NATIVE = FIXTURE / 'native'
ANNOTATIONS = {'keyword', 'marker', 'rating', 'chapter-marker', 'analysis-marker'}
REEL = 'com.apple.proapps.studio.reel'


def read(name):
    return parse_xml_bytes((NATIVE / name).read_bytes())


def canonical(element):
    """Normalize rational spelling, XML indentation and annotation ordering only."""
    attrs = dict(element.attrib)
    for key in ('start', 'duration', 'offset', 'posterOffset', 'frameDuration', 'tcStart'):
        if key in attrs:
            attrs[key] = str(seconds(attrs[key]))
    text = element.text or ''
    if not text.strip():
        text = ''
    children = [canonical(child) for child in element]
    if element.tag == 'asset-clip':
        children = [canonical(child) for child in element if child.tag not in ANNOTATIONS]
        children += sorted([canonical(child) for child in element if child.tag in ANNOTATIONS], key=repr)
    return (element.tag, tuple(sorted(attrs.items())), text, tuple(children))


def clip(root):
    return root.find(f'./library/event[@name="{EVENT}"]/asset-clip')


class NativeSnapshotTests(unittest.TestCase):
    def test_native_keyword_roundtrip_preserves_baseline_with_only_expected_additions(self):
        baseline = read('baseline.fcpxml')
        roundtrip = read('roundtrip.fcpxml')
        expected = [('keyword', 'Added Synthetic Keyword', '3601', '7'),
                    ('marker', 'Added Synthetic Marker', '3610', '1')]
        for tag, label, start, duration in expected:
            node = clip(roundtrip).find(f'{tag}[@value="{label}"]')
            self.assertIsNotNone(node)
            self.assertEqual(seconds(node.get('start')), seconds(start))
            self.assertEqual(seconds(node.get('duration')), seconds(duration))
            clip(roundtrip).remove(node)
        event = roundtrip.find(f'./library/event[@name="{EVENT}"]')
        collection = event.find('keyword-collection[@name="Added Synthetic Keyword"]')
        self.assertIsNotNone(collection)
        event.remove(collection)
        self.assertEqual(canonical(baseline), canonical(roundtrip))

    def test_preparatory_input_changes_only_browser_clip_format(self):
        initial = read('initial-export.fcpxml')
        corrected = read('compatible-input.fcpxml')
        self.assertNotEqual(clip(initial).get('format'), clip(corrected).get('format'))
        clip(initial).set('format', clip(corrected).get('format'))
        self.assertEqual(canonical(initial), canonical(corrected))

    def test_known_native_replace_loss_of_reel_is_explicitly_recorded(self):
        """This test intentionally verifies an observed LOSS, not preservation."""
        imported = read('compatible-input.fcpxml')
        exported = read('baseline.fcpxml')
        path = f'./resources/asset/metadata/md[@key="{REEL}"]'
        reel = imported.find(path)
        self.assertIsNotNone(reel)
        self.assertEqual(reel.get('value'), 'TEST-REEL')
        self.assertIsNone(exported.find(path))
        imported.find('./resources/asset/metadata').remove(reel)
        self.assertEqual(canonical(imported), canonical(exported))

    def test_initial_native_format_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = (NATIVE / 'initial-export.fcpxml').read_text(encoding='utf-8')
            xml = directory / 'initial.fcpxml'
            xml.write_text(raw.replace(SOURCE_PLACEHOLDER, (FIXTURE / 'media/generated.mov').resolve().as_uri()), encoding='utf-8')
            proposal = directory / 'proposal.json'
            proposal.write_text(json.dumps(proposal_document()), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'clip/resource format mismatch'):
                bind(xml, proposal, EVENT, ['synthetic'])

    def test_sanitized_native_baseline_can_bind_without_weakening_format_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            xml = materialize(directory / 'session', seed='native-baseline')
            proposal = directory / 'session' / 'proposals.json'
            manifest = json.loads((directory / 'session' / 'manifest.json').read_text())
            self.assertEqual(manifest['seed_variant'], 'native-baseline')
            bound = bind(xml, proposal, EVENT, ['synthetic'])
            self.assertEqual(len(bound['proposals']), 2)
            self.assertTrue(all(not row['approved'] and not row['reviewed'] for row in bound['proposals']))

    def test_native_snapshots_match_report_hashes_and_have_no_private_references(self):
        report = json.loads((NATIVE / 'report.json').read_text(encoding='utf-8'))
        self.assertTrue(report['broad_metadata_preservation'].startswith('NOT ESTABLISHED'))
        for name, expected_hash in report['snapshots'].items():
            with self.subTest(name=name):
                raw = (NATIVE / name).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_hash)
                self.assertNotIn(b'/Users/', raw)
                self.assertNotIn(b'/Volumes/', raw)
                self.assertNotIn(b'<bookmark', raw)
                tree = parse_xml_bytes(raw)
                self.assertIsNone(tree.find('library').get('location'))
                self.assertEqual([node.get('src') for node in tree.findall('.//media-rep')], [SOURCE_PLACEHOLDER])


if __name__ == '__main__':
    unittest.main()
