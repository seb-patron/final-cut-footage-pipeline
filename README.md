# Final Cut Footage Pipeline

A small **offline, review-gated prototype**, not production-ready automation. It reuses local Whisper results, samples images with FFmpeg and Apple Vision, and prepares narrow FCPXML annotations. It never imports XML, edits native databases, uploads footage, or approves its own scene/speech proposals.

**Native-import warning:** In the Final Cut Pro 12.4 synthetic test, a preparatory **Replace** import dropped an existing Reel value. The later keyword/marker round-trip preserved its baseline, but that baseline already lacked Reel. Broad metadata preservation and production replacement remain unproven. Read the [native validation report](docs/native-validation.md) before importing any candidate.

## What it does today

- Converts existing local Whisper-derived transcript CSVs into proposed time-ranged Final Cut Pro markers.
- Samples one requested frame with FFmpeg and Apple Vision to suggest visual keywords.
- Combines those suggestions, binds them to an exact FCPXML export and local source bytes, and prepares a review packet containing candidate FCPXML plus an audit.
- Requires explicit human review and approval before it includes an annotation. It never imports that packet into Final Cut Pro.

This is more than keywording, but it is not yet an end-to-end transcription or visual-analysis system. It does not currently run Whisper, analyze complete clips, identify speakers or locations, or provide a review UI. The command names remain `footage_keywords.py` and `sample_vision.py` for now.

## Requirements and setup

Python 3.11+ (tomllib; standard library only), macOS with Swift/Apple Vision, and FFmpeg. Existing local Whisper output is required for transcript proposals; this tool does not install or rerun Whisper. The input contract supports CSVs prepared from local Whisper output with word timestamps. No external service is needed. Apple Vision may require an authorized execution context with CoreVideo access; a restricted sandbox failed with `Failed to create CVPixelBufferPool`. Do not interpret that as an empty classification.

Clone the public repository, then run commands from its root:

```sh
git clone https://github.com/seb-patron/final-cut-footage-pipeline.git
cd final-cut-footage-pipeline
python3 footage_keywords.py --help
python3 -m unittest discover -v
```

No Python packages are required. Install FFmpeg separately only if using visual sampling. AI steps additionally need the optional openai and anthropic packages (`pip install openai anthropic`) and an API key from the environment or the Keychain; see [docs/ai-providers.md](docs/ai-providers.md). Local transcription additionally needs the optional mlx-whisper package (`pip install mlx-whisper`) on an Apple-silicon Mac; see [docs/steps/transcribe.md](docs/steps/transcribe.md). Final Cut Pro is required only for native export/import validation. Keep your own project inputs, source media, transcripts, proposals and review packets outside this repository.

## Input contract

Inventory CSV fields: asset_id, filename, library_relative_path, duration_seconds, frame_rate. Additional columns are retained. asset_id and source path must be unique. Root plus relative path defines source identity; filename alone is never a match. Proposal generation establishes path identity only. Before review, the required bind step captures the native export hash, native IDs and a full local source-content SHA-256 plus filesystem identity. Relocated/replaced/modified sources and changed export bytes fail patch validation and require rebinding and fresh review. This does not retroactively prove that an old Whisper transcript came from the currently hashed media; reviewers must verify playback against the current source.

Segment CSV fields: asset_id, filename, source_relative_path, start_seconds, end_seconds, text, review_flags. Times begin at zero in the source file, not camera or project timecode. Flags and invalid ranges are preserved for review. All proposed transcript markers begin unreviewed and unapproved, including apparently high-confidence speech. Background TV, guide audio, names and exact quotes need audition.

## Generate transcript proposals

```sh
python3 footage_keywords.py propose \
  --inventory /path/inventory.csv --segments /path/segments.csv \
  --root /path/source-root --output /path/proposals.json
```

## Local visual sampling

```sh
python3 sample_vision.py --inventory /path/inventory.csv \
  --root /path/source-root --asset-id ASSET-ID --seconds 6 \
  --output /path/new-sample-folder
```

This creates one 960-pixel-wide JPEG derivative and JSON results, not another video. It does not copy original media. Apple Vision returns generic labels and confidence scores, not reliable geography or identity. It may label a river as lake or suggest snow incorrectly. Confidence is not calibrated correctness. Only scores >=0.6 become candidate labels, still unreviewed and unapproved. A sampled frame is evidence only for that instant, not the entire clip. The one-frame proposal interval is an approximation around the requested extraction timestamp; inspect decoded timing before frame-critical use.

Combine visual suggestions with transcripts using `propose --visual /path/new-sample-folder/visual-proposals.json`. First bind the selected asset proposals to a fresh validation export as described below. Then review playback/images, correct labels and ranges, document the evidence, and explicitly set both `reviewed` and `approved` to true for selected rows. For technical testing, use a clearly synthetic validation label rather than an invented scene description. Keep requests such as exact location, speaker identity and camera settings out of unverified labels.

## Prepare XML only

1. Coordinate exclusive Final Cut UI use. Use an explicitly authorized, dedicated validation library, not production.
2. Import one existing original with **Leave files in place**; disable optimized/proxy media and unneeded analyses. Reuse its single event.
3. Export the event's current XML. UTF-8 only. The prototype permits a `.fcpxmld` bundle only when it contains exactly `Info.fcpxml`; extra files/directories and external bookmark references are rejected rather than silently lost. Every media-rep URI throughout the export, including unselected assets, must be an absolute local file URI; relative/remote references fail closed. Non-asset/format resources are unsupported. A library location, when present, must also be an absolute local file URI. Retiming/conform-rate, nested media and extra media representations are unsupported. Preserve the original export unchanged.
4. Bind an explicit small set of source asset IDs **before** review:

```sh
python3 footage_keywords.py bind --xml /path/export.fcpxmld/Info.fcpxml \
  --proposal /path/proposals.json --event 'Validation Event' \
  --asset-id ASSET-ID --output /path/bound-proposals.json
```

Repeat `--asset-id` to select additional assets. Binding keeps only selected proposals and resets every approval/review flag. It reads the source bytes locally to record full SHA-256 and filesystem identity, and records native asset/event IDs, clip/format attributes and the exact export hash. Missing native identity, ambiguous source mapping, unsupported structure or missing media fails closed. No media is copied or transmitted. Binding a large set can take time because the source files are hashed in full.

5. Review the bound proposals against current playback. Set reviewed/approved true only after review, then prepare a new atomic review packet:

```sh
python3 footage_keywords.py patch --xml /path/export.fcpxmld/Info.fcpxml \
  --proposal /path/reviewed-bound-proposals.json --event 'Validation Event' \
  --output /path/new-review-packet.zip
```

Only an exact original-media URI mapping to one direct browser asset-clip is supported. The tool hashes each approved source once per patch, rechecks source filesystem identity and export bytes immediately before publication, rechecks bound native/source identity, preserves unrelated supported XML elements semantically (including comments and inline bookmarks), and appends annotations. Formatting is not byte-preserved. Source origin is added and rational frame boundaries enclose the interval. **Any interval exceeding the browser clip or native source is rejected, never truncated.** Existing equivalent keyword ranges are detected using rational times and membership in comma-separated keyword values, regardless of notes or fraction spelling.

The ZIP contains `candidate.fcpxml` and `audit.json`. Both are staged, flushed, and published as one atomic file using an exclusive same-filesystem hard link. Existing outputs are never overwritten. If serialization or publication fails, no partial destination packet is exposed. This deliberately replaces the old standalone XML + separate audit interface. Extract the packet to a new review folder before examining/importing it. The packet itself is not an importable FCPXML bundle.

6. Validate the extracted XML against the official installed DTD matching its version:

```sh
xmllint --noout --dtdvalid 'file:///Applications/Final%20Cut%20Pro.app/Contents/Frameworks/Interchange.framework/Versions/A/Resources/FCPXMLv1_14.dtd' /path/review/candidate.fcpxml
```

7. Review the structural diff and audit. Approval flags are a local workflow, **not** tamper-proof signatures. Binding verifies export-file freshness against its recorded snapshot, **not** whether someone changed the live native library after export. Always coordinate exclusive UI access and obtain a new export if native state may have changed. The CLI has no live-library access or import command; no production import is authorized by this tool.
8. Only in the explicitly authorized validation target, import and inspect the matching-items dialog. **Replace is replacement, not an additive database update.** Keep Both may create duplicates and is not tested. Do not suppress warnings. Re-export and compare event/clip counts, source UID/URI, start/duration, annotations and existing metadata. Do not assume the experiment authorizes production replacement.

## Pipeline commands

The `fcp_pipeline` package is a skeleton: the steps exist as named stubs and do no work yet.

```sh
python3 -m fcp_pipeline --help
```

Main-path steps in order: scan, pull, transcribe, tag, photos, memo, organize, patch, export, roughcut; then ingest (optional); then run. Every step takes `--trip <folder>` (required) and `--dry-run`. An unbuilt step reads and writes nothing, prints `<step>: not built yet` to stderr and exits 3, with or without `--dry-run`. `run --trip <folder> [--dry-run]` calls the built main-path steps in registry order, skips the unbuilt ones, never runs ingest, and exits 3 with `run: no steps built yet` while no main-path step is built. The prototype commands (`footage_keywords.py`, `sample_vision.py`) keep working.

Each trip folder holds a `project.toml` following the editor's project layout contract (see `tests/fixtures/trips/minimal/project.toml`): top-level `title`, with `status`, `created`, `dates` and `youtube` accepted but unused; `[final_cut]` records `library` and `events`; `[media]` records `footage`, `photos` and `assets` (all recorded as given, never created or checked); the pipeline-owned `[pipeline]` table holds `timezone`, `start_date` (a TOML local date, `YYYY-MM-DD`), `clock_offsets` and `places`, and the optional `[pipeline.transcribe]` table holds the local transcription `model`. Pipeline keys at the top level are rejected, and unknown keys warn once.

## Tests

```sh
python3 -m unittest discover -v
python3 -m py_compile footage_keywords.py sample_vision.py
```

The test suite covers nonzero source origins, semantic deduplication, no truncation, XML encoding/entity rejection, approval reset, source/export staleness, retiming/sidecar rejection and atomic publication failures/races. Tests use synthetic data and temporary files, never personal footage.

The committed [synthetic test library seed](tests/fixtures/metadata-preservation/) includes a 12-second generated blue frame/test tone and FCPXML with favorite/rejected ratings, keyword ranges, regular/to-do/chapter markers, notes, roles, collections, custom metadata and an untouched timeline. It contains no real footage. Materialize an importable seed with:

```sh
python3 -m tests.fixture_library --seed native-baseline --output /absolute/new/fixture-session
```

This creates path-resolved XML and unreviewed proposals from the sanitized native baseline, not a native `.fcpbundle`. Use `--seed rich` for broader offline XML coverage. See the [preservation test plan](docs/preservation-tests.md) for exact coverage and the separate native round-trip protocol. The optional official-DTD test skips when a legally installed DTD or `xmllint` is unavailable.

Six tests replay the committed sanitized native snapshots, including the observed Reel loss. A passing replay test records that limitation; it does not establish safe native preservation or launch Final Cut in CI. The final native annotation test preserved existing ratings, markers, notes and other fields present in its baseline, with no duplicate browser clips, but it did not test Reel after the earlier loss.

Validate an exported candidate against the installed DTD and use a dedicated synthetic test library before working with real media. Do not treat these results as approval for production replacement. There is no broad format/version or native metadata-preservation guarantee.

## Privacy and publication

This is a clean source snapshot with independent history. No private repository history, real footage, project transcripts, native library databases or personal project evidence is distributed. Runtime proposal/audit files contain absolute source paths and may contain transcripts, so treat them as private. Do not post them publicly without reviewing and redacting their contents.

The prototype commands make no network requests. The pipeline package reaches the network only through its network gate: `offline = true` in `[pipeline]` or `--offline` refuses every call before connecting, and every call is logged to `.pipeline/network-log.tsv` with kinds and sizes, never content. AI steps need the optional openai and anthropic packages and an API key from the environment or the Keychain; see [docs/ai-providers.md](docs/ai-providers.md) and [docs/privacy.md](docs/privacy.md). FFmpeg and Apple Vision are invoked locally. There is no automatic import or live-library access. The tool's approval flags are a local review convention, not an access-control boundary.

## License

Original source code and original synthetic fixtures are available under the [MIT License](LICENSE). External tools and Apple frameworks retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Final Cut Pro, Apple Vision, FFmpeg, Whisper, models and Apple DTDs are not bundled.

## Roadmap: local analysis through reviewed Final Cut metadata

These are planned directions, not implemented features or guarantees:

1. **Preservation first:** isolate the observed native Reel loss; expand reproducible synthetic test libraries and native round-trip checks for ratings, markers, keywords, roles and other existing metadata; broader FCPXML/version coverage.
2. **Local transcription:** an explicit, resumable Whisper stage with model selection, timing, language/confidence signals, source fingerprints and review flags. Do not treat ASR as verified quotations or speaker identity.
3. **Visual analysis:** scene-aware multi-frame sampling, coverage reporting, local classifier/model adapters and evidence-linked labels. Avoid claiming whole-clip coverage from a single frame.
4. **Unified review:** combine speech and visual evidence on the same source timeline, support playback/audition and corrections, then approve only selected ranges.
5. **Safer Final Cut handoff:** clearer dry-run diffs, native-library freshness checks and more supported export structures. Any future import workflow must preserve explicit authorization and metadata checks.

The goal is a complete local footage-understanding and Final Cut metadata pipeline, built incrementally on the current proposal/review layer. Bundled sidecars, complex native clip types and production-scale batch processing remain unsupported or unvalidated.

Apple references: [XML workflow](https://support.apple.com/en-az/guide/final-cut-pro/verdbd66ae/mac), [FCPXML creation](https://developer.apple.com/documentation/professional-video-applications/creating-fcpxml-documents), [DTD caveats](https://developer.apple.com/documentation/professional-video-applications/document-type-definition).
