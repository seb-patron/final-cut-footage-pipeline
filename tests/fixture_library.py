"""Materialize the committed synthetic FCPXML seed without copying any media.

Run: python3 -m tests.fixture_library --output /new/path
This prepares importable XML, not a native .fcpbundle, and never opens Final Cut.
"""
import argparse
import hashlib
import json
from pathlib import Path

FIXTURE = Path(__file__).parent / 'fixtures' / 'metadata-preservation'
EVENT = 'Metadata Preservation Fixture'
SOURCE_PLACEHOLDER = 'file:///SYNTHETIC_FIXTURE_ROOT/media/generated.mov'
SEEDS = {'rich': 'library.fcpxml', 'native-baseline': 'native/baseline.fcpxml'}


def seed_bytes(seed='rich'):
    """Resolve only the public synthetic fixture's explicit URI placeholder."""
    source = (FIXTURE / 'media' / 'generated.mov').resolve(strict=True)
    if seed not in SEEDS:
        raise ValueError('unknown synthetic seed variant')
    raw = (FIXTURE / SEEDS[seed]).read_text(encoding='utf-8')
    if raw.count(SOURCE_PLACEHOLDER) != 1:
        raise ValueError('expected exactly one synthetic source placeholder')
    # A file URI percent-encodes characters that could conflict with XML attributes.
    return raw.replace(SOURCE_PLACEHOLDER, source.as_uri()).encode('utf-8')


def proposal_document():
    return {
        'schema': 1,
        'timestamp_basis': 'seconds relative to source file; not timeline time',
        'assets': {'synthetic': {
            'source_uri': (FIXTURE / 'media' / 'generated.mov').resolve(strict=True).as_uri(),
            'duration_seconds': '12', 'frame_rate': '25',
            'filename': 'generated.mov',
        }},
        'proposals': [
            {'id': 'synthetic-keyword', 'asset_id': 'synthetic', 'kind': 'keyword',
             'start_seconds': '1', 'end_seconds': '8', 'label': 'Added Synthetic Keyword',
             'evidence': 'Synthetic regression range overlaps both existing rating ranges.',
             'reviewed': False, 'approved': False},
            {'id': 'synthetic-marker', 'asset_id': 'synthetic', 'kind': 'marker',
             'start_seconds': '10', 'end_seconds': '11', 'label': 'Added Synthetic Marker',
             'evidence': 'Synthetic regression marker, not a transcript or scene claim.',
             'reviewed': False, 'approved': False},
        ],
    }


def materialize(output, seed='rich'):
    output = Path(output)
    # Refuse existing destinations. Never overwrite or clean up user files.
    output.mkdir(parents=False, exist_ok=False)
    xml = seed_bytes(seed)
    (output / 'seed.fcpxml').write_bytes(xml)
    (output / 'proposals.json').write_text(
        json.dumps(proposal_document(), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    manifest = json.loads((FIXTURE / 'manifest.json').read_text(encoding='utf-8'))
    manifest['seed_variant'] = seed
    manifest['seed_sha256'] = hashlib.sha256(xml).hexdigest()
    manifest['media_sha256'] = hashlib.sha256((FIXTURE / 'media/generated.mov').read_bytes()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return output / 'seed.fcpxml'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', choices=SEEDS, default='rich',
                        help='rich XML coverage fixture or sanitized native-compatible baseline')
    args = parser.parse_args()
    print(materialize(args.output, args.seed))


if __name__ == '__main__':
    main()
