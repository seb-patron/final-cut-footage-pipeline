"""Structural preservation tests, independent of Final Cut's import behavior."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

from footage_keywords import bind, parse_xml_bytes, patch
from tests.fixture_library import EVENT, FIXTURE, materialize, proposal_document, seed_bytes


def signature(element):
    """Compare all XML content/attributes and sibling order, including comments.

    Attribute order is not XML semantics. Whitespace is deliberately retained, so
    accidental note changes are not masked by a general text-normalization step.
    """
    tag = '#comment' if element.tag is ET.Comment else element.tag
    return (tag, tuple(sorted(element.attrib.items())), element.text, element.tail,
            tuple(signature(child) for child in element))


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.xml = self.directory / 'baseline.fcpxml'
        self.xml.write_bytes(seed_bytes())
        self.proposal = self.directory / 'proposal.json'
        self.output = self.directory / 'review.zip'
        self.document = proposal_document()

    def write_tree(self, tree):
        self.xml.write_bytes(ET.tostring(tree, encoding='utf-8'))

    def original_tree(self):
        return parse_xml_bytes(self.xml.read_bytes())

    @staticmethod
    def clip(tree):
        return tree.find(f'./library/event[@name="{EVENT}"]/asset-clip')

    def run_patch(self):
        self.proposal.write_text(json.dumps(self.document), encoding='utf-8')
        bound = bind(self.xml, self.proposal, EVENT, ['synthetic'])
        # Test-only explicit approval of known synthetic labels, after binding.
        for row in bound['proposals']:
            row.update(reviewed=True, approved=True)
        self.proposal.write_text(json.dumps(bound), encoding='utf-8')
        before = self.xml.read_bytes()
        audit = patch(self.xml, self.proposal, EVENT, self.output)
        self.assertEqual(self.xml.read_bytes(), before)
        with zipfile.ZipFile(self.output) as packet:
            self.assertEqual(set(packet.namelist()), {'candidate.fcpxml', 'audit.json'})
            self.assertEqual(json.loads(packet.read('audit.json')), audit)
            candidate_bytes = packet.read('candidate.fcpxml')
        return parse_xml_bytes(candidate_bytes), audit, candidate_bytes

    def assert_only_additions(self, before, after, audit):
        stripped = copy.deepcopy(after)
        clip = self.clip(stripped)
        for addition in audit['annotations']:
            matches = [node for node in clip if node.attrib == addition['attributes']]
            self.assertEqual(len(matches), 1)
            clip.remove(matches[0])
        self.assertEqual(signature(before), signature(stripped))

    def test_entire_document_changes_only_by_reviewed_additions(self):
        before = self.original_tree()
        candidate, audit, _ = self.run_patch()
        self.assertEqual(len(audit['annotations']), 2)
        self.assert_only_additions(before, candidate, audit)
        self.assertEqual(audit['native_import'], 'NOT RUN')
        self.assertEqual(audit['duplicate_behavior'], 'UNKNOWN')

    def test_favorite_and_reject_ranges_and_notes_survive_overlapping_keyword(self):
        before = [signature(e) for e in self.clip(self.original_tree()).findall('rating')]
        candidate, _, _ = self.run_patch()
        ratings = self.clip(candidate).findall('rating')
        self.assertEqual([signature(e) for e in ratings], before)
        self.assertEqual({e.get('value') for e in ratings}, {'favorite', 'reject'})
        self.assertTrue(all(e.get('note') for e in ratings))

    def test_regular_todo_completed_and_chapter_marker_notes_preserved(self):
        before = self.clip(self.original_tree())
        candidate, _, _ = self.run_patch()
        after = self.clip(candidate)
        for tag in ('marker', 'chapter-marker'):
            with self.subTest(tag=tag):
                old = before.findall(tag)
                self.assertEqual([signature(e) for e in after.findall(tag)[:len(old)]],
                                 [signature(e) for e in old])
        self.assertEqual([e.get('completed') for e in after.findall('marker')[:3]], [None, '0', '1'])

    def test_typed_array_and_resource_metadata_preserved_exactly(self):
        before = self.original_tree()
        candidate, _, _ = self.run_patch()
        for location in ('./resources/asset/metadata', f'./library/event[@name="{EVENT}"]/asset-clip/metadata'):
            with self.subTest(location=location):
                self.assertEqual(signature(before.find(location)), signature(candidate.find(location)))
        array = self.clip(candidate).find('metadata/md[@key="org.footage-keywords.fixture.array"]/array')
        self.assertEqual([e.text for e in array], ['first', '日本語'])

    def test_clip_names_notes_roles_channel_settings_and_gain_preserved(self):
        before = self.clip(self.original_tree())
        candidate, _, _ = self.run_patch()
        after = self.clip(candidate)
        self.assertEqual(before.attrib, after.attrib)
        for tag in ('note', 'adjust-volume', 'audio-channel-source', 'analysis-marker'):
            with self.subTest(tag=tag):
                self.assertEqual(signature(before.find(tag)), signature(after.find(tag)))

    def test_source_identity_format_and_timecode_preserved(self):
        before = self.original_tree()
        candidate, _, _ = self.run_patch()
        self.assertEqual(signature(before.find('resources')), signature(candidate.find('resources')))
        self.assertEqual(before.find('library/event').attrib, candidate.find('library/event').attrib)
        self.assertEqual(self.clip(candidate).get('start'), '3600s')
        self.assertEqual(self.clip(candidate).get('duration'), '12s')

    def test_unselected_event_and_timeline_are_untouched(self):
        before = self.original_tree()
        candidate, _, _ = self.run_patch()
        for location in ('./library/event[@name="Untouched Event"]', './/project'):
            with self.subTest(location=location):
                self.assertEqual(signature(before.find(location)), signature(candidate.find(location)))

    def test_collections_and_library_metadata_are_untouched(self):
        before = self.original_tree()
        candidate, _, _ = self.run_patch()
        self.assertEqual(before.find('library').attrib, candidate.find('library').attrib)
        for tag in ('keyword-collection', 'collection-folder', 'smart-collection'):
            with self.subTest(tag=tag):
                self.assertEqual([signature(e) for e in before.iter(tag)],
                                 [signature(e) for e in candidate.iter(tag)])

    def test_existing_keyword_names_ranges_and_notes_not_rewritten(self):
        before = self.clip(self.original_tree()).findall('keyword')
        candidate, _, _ = self.run_patch()
        self.assertEqual([signature(e) for e in self.clip(candidate).findall('keyword')[:len(before)]],
                         [signature(e) for e in before])

    def test_new_annotations_precede_audio_channel_sources_and_metadata(self):
        candidate, audit, _ = self.run_patch()
        children = list(self.clip(candidate))
        for addition in audit['annotations']:
            added = next(e for e in children if e.attrib == addition['attributes'])
            self.assertLess(children.index(added), children.index(self.clip(candidate).find('audio-channel-source')))
            self.assertLess(children.index(added), children.index(self.clip(candidate).find('metadata')))

    def test_comments_and_inline_bookmark_retained(self):
        tree = self.original_tree()
        ET.SubElement(tree.find('.//media-rep'), 'bookmark').text = 'U1lOVEhFVElDLU5PVC1BLVJFQUwtQk9PS01BUks='
        self.write_tree(tree)
        candidate, audit, _ = self.run_patch()
        self.assert_only_additions(tree, candidate, audit)
        self.assertTrue(any(e.tag is ET.Comment for e in self.clip(candidate)))
        self.assertEqual(candidate.find('.//bookmark').text, tree.find('.//bookmark').text)

    def test_same_keyword_range_dedupes_without_erasing_existing_note(self):
        self.document['proposals'][0].update(label='Existing Blue', start_seconds='1', end_seconds='4')
        before = self.original_tree()
        candidate, audit, _ = self.run_patch()
        self.assertEqual(len(audit['annotations']), 1)
        self.assertEqual(audit['annotations'][0]['proposal_id'], 'synthetic-marker')
        self.assert_only_additions(before, candidate, audit)

    def test_same_label_different_range_adds_without_merging_or_deleting(self):
        self.document['proposals'][0].update(label='Existing Blue', start_seconds='2', end_seconds='5')
        before = self.original_tree()
        candidate, audit, _ = self.run_patch()
        self.assertEqual(len(audit['annotations']), 2)
        self.assert_only_additions(before, candidate, audit)

    def test_same_range_different_label_does_not_rename_original(self):
        self.document['proposals'][0].update(start_seconds='1', end_seconds='4')
        before = self.original_tree()
        candidate, audit, _ = self.run_patch()
        self.assert_only_additions(before, candidate, audit)
        self.assertEqual(len(self.clip(candidate).findall('keyword')), 3)

    def test_duplicate_proposals_add_only_one_annotation(self):
        self.document['proposals'].append(dict(self.document['proposals'][0], id='duplicate'))
        before = self.original_tree()
        candidate, audit, _ = self.run_patch()
        self.assertEqual(len(audit['annotations']), 2)
        self.assert_only_additions(before, candidate, audit)

    def test_marker_dedupe_preserves_todo_completion_and_note(self):
        self.document['proposals'][1].update(label='Existing completed to-do', start_seconds='5', end_seconds='126/25')
        before = self.original_tree()
        candidate, audit, _ = self.run_patch()
        self.assertEqual(len(audit['annotations']), 1)
        self.assert_only_additions(before, candidate, audit)
        self.assertEqual(self.clip(candidate).find('marker[@completed="1"]').get('note'), 'Completed marker note')

    def test_reapplying_after_fresh_binding_is_noop_without_packet(self):
        candidate, _, raw = self.run_patch()
        self.xml = self.directory / 'candidate-baseline.fcpxml'
        self.xml.write_bytes(raw)
        self.output = self.directory / 'second.zip'
        with self.assertRaisesRegex(ValueError, 'no new reviewed'):
            self.run_patch()
        self.assertFalse(self.output.exists())
        self.assertEqual(signature(candidate), signature(self.original_tree()))

    def test_arbitrary_synthetic_annotation_batch_preserves_every_existing_node(self):
        rng = random.Random(2117)
        base = self.document['proposals'][0]
        self.document['proposals'] = []
        for index in range(80):
            lo = rng.randrange(0, 299)
            hi = rng.randrange(lo + 1, 301)
            self.document['proposals'].append(dict(base, id=f'batch-{index}',
                kind='keyword' if index % 2 else 'marker', label=f'Synthetic {index} & café',
                start_seconds=str(lo / 25), end_seconds=str(hi / 25)))
        before = self.original_tree()
        candidate, audit, _ = self.run_patch()
        self.assertEqual(len(audit['annotations']), 80)
        self.assert_only_additions(before, candidate, audit)

    def test_unsupported_complex_resources_reject_without_output_or_source_mutation(self):
        for tag, attributes in (('effect', {'id': 'r3', 'uid': 'synthetic-effect'}),
                                ('media', {'id': 'r3', 'uid': 'synthetic-media'}),
                                ('locator', {'id': 'r3', 'url': 'file:///synthetic'})):
            with self.subTest(tag=tag):
                self.xml.write_bytes(seed_bytes())
                tree = self.original_tree()
                ET.SubElement(tree.find('resources'), tag, attributes)
                self.write_tree(tree)
                original = self.xml.read_bytes()
                with self.assertRaisesRegex(ValueError, 'non-asset/format resources'):
                    self.run_patch()
                self.assertEqual(self.xml.read_bytes(), original)
                self.assertFalse(self.output.exists())

    def test_whole_clip_implicit_rating_is_not_expanded_or_replaced(self):
        tree = self.original_tree()
        rating = self.clip(tree).find('rating')
        rating.attrib = {'value': 'favorite', 'note': 'Whole clip, implicit range'}
        self.write_tree(tree)
        candidate, audit, _ = self.run_patch()
        self.assert_only_additions(tree, candidate, audit)
        self.assertEqual(self.clip(candidate).find('rating').attrib, rating.attrib)

    def test_fixture_and_candidate_validate_against_installed_official_dtd(self):
        dtd = Path(os.environ.get('FCPXML_DTD', '/Applications/Final Cut Pro.app/Contents/Frameworks/Interchange.framework/Versions/A/Resources/FCPXMLv1_14.dtd'))
        if not dtd.is_file() or not shutil.which('xmllint'):
            self.skipTest('official Final Cut 1.14 DTD/xmllint unavailable; not a native test')
        _, _, raw = self.run_patch()
        candidate = self.directory / 'candidate.fcpxml'
        candidate.write_bytes(raw)
        for path in (self.xml, candidate):
            result = subprocess.run(['xmllint', '--noout', '--nonet', '--dtdvalid', dtd.resolve().as_uri(), str(path)],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)


class FixtureLibraryTests(unittest.TestCase):
    def test_committed_generated_media_matches_manifest_sha256(self):
        manifest = json.loads((FIXTURE / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256((FIXTURE / manifest['source']).read_bytes()).hexdigest(),
                         manifest['source_sha256'])

    def test_materializer_is_exclusive_and_leaves_media_in_place(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'new-fixture'
            result = materialize(directory)
            self.assertEqual(result.read_bytes(), seed_bytes())
            self.assertEqual(set(p.name for p in directory.iterdir()), {'seed.fcpxml', 'proposals.json', 'manifest.json'})
            proposals = json.loads((directory / 'proposals.json').read_text())
            self.assertTrue(all(not p['approved'] and not p['reviewed'] for p in proposals['proposals']))
            original = result.read_bytes()
            with self.assertRaises(FileExistsError):
                materialize(directory)
            self.assertEqual(result.read_bytes(), original)

    def test_public_fixture_has_no_absolute_private_path_or_native_bookmark(self):
        raw = (FIXTURE / 'library.fcpxml').read_text(encoding='utf-8')
        self.assertNotIn('/Users/', raw)
        self.assertNotIn('/Volumes/', raw)
        self.assertNotIn('<bookmark', raw)
        self.assertEqual(raw.count('file:///SYNTHETIC_FIXTURE_ROOT/'), 1)


if __name__ == '__main__':
    unittest.main()
