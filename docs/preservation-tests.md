# Metadata-preservation tests

## Scope and current evidence

`python3 -m unittest -v` runs 54 tests. On the development Mac all 54 pass, including validation of the committed FCPXML 1.14 fixture and the resulting candidate against Final Cut Pro's installed official 1.14 DTD. On machines without Final Cut's DTD or `xmllint`, that single DTD test is skipped explicitly. Set `FCPXML_DTD` to a local, legally obtained 1.14 DTD to run it elsewhere. The DTD is not distributed with this repository.

The new tests establish **offline XML preservation**, not native-library import safety. The principal assertion compares the entire parsed XML tree before and after removing only the additions recorded in the audit. It checks every element, attribute, text value, comment and sibling position. XML attribute order is ignored; note whitespace is not. The source XML bytes remain unchanged, and the media is read in place. The tool does not modify `.fcpbundle` databases.

The portable test library is `tests/fixtures/metadata-preservation/library.fcpxml` plus its generated 64 KB media. It is a source fixture for constructing a fresh native test library, **not a native `.fcpbundle` or a backup**. The committed URI is an explicit placeholder. Use the materializer below before import. Do not import the placeholder file directly.

**Native preservation roundtrip for this fixture: NOT RUN.** Do not infer native guarantees from passing unit tests or DTD validation. The initial seed can contain metadata that Final Cut omits or normalizes; seed-to-first-export differences are not patch regressions. Establish and inspect the native baseline first, then compare the post-patch export with that baseline.

## Coverage

The fixture and tests cover:

- Favorite and rejected ranges, including notes, names and overlap with a new keyword.
- Regular markers, pending/completed to-do markers, chapter markers, poster offsets and notes.
- Whole-clip and range keywords, comma-separated keyword values and existing keyword notes.
- Clip names and multiline Unicode notes, XML-special characters and comments.
- Source and event UIDs, resource/source timecode, duration, format and original-media URI.
- Clip/resource metadata, including custom fields, arrays, booleans, integers, floats and dates.
- Audio/video roles, channel layout and gain, analysis markers and synthetic inline bookmark bytes.
- Keyword collections, collection folders, smart collections and library attributes.
- An untouched event and timeline referencing the same generated source.
- Duplicate proposals, reapplication after fresh binding, different labels on the same range and the same label on different ranges.
- An 80-annotation deterministic mixed keyword/marker batch.
- Fail-closed behavior for complex resources, with the existing suite covering stale source/export binding, retiming, sidecars, ambiguous targets and atomic publication failures.

The synthetic inline-bookmark test checks XML transport only. It does not assert that arbitrary bookmark bytes resolve in Final Cut, and the committed native seed contains no bookmark.

## Duplicate and conflict semantics

- Existing ratings, markers, notes and metadata are never deliberately rewritten by the patcher.
- Equal keyword label + equal rational range is a duplicate, including a label inside a comma-separated keyword list. Existing notes survive unchanged.
- Equal marker value + range is treated as a duplicate even if the existing marker has a note or a completed/to-do flag. A proposal does not convert or reset that marker.
- Different ranges or different labels are additive. Overlaps are not merged or shortened. Ratings are not changed to accommodate annotations.
- A fresh bind resets approvals. If everything already exists, patching raises `no new reviewed, approved annotations` and creates no packet.
- Reimporting an FCPXML event into Final Cut is a separate, consequential operation. **Replace is replacement, not an additive database operation.** Keep Both can duplicate browser items. Neither behavior follows from the offline tests.

## Materialize the portable library

From the repository root:

```sh
python3 -m tests.fixture_library --output /absolute/new/fixture-session
```

The destination's parent must exist, and the destination must be new. The command produces `seed.fcpxml`, unreviewed `proposals.json` and a hash manifest. It references the committed generated media in place; it does not copy media, create a native library, open Final Cut, approve proposals or import XML. Keep this output outside Git: it contains paths local to your machine.

Inspect the blue image and quiet tone if playing the media. The source lasts 12 seconds at 25 fps with a 01:00:00:00 timecode origin. Its content is deliberately not realistic footage or a transcript.

## Exclusive native validation protocol

1. Coordinate one editor at a time. Stop if another editor or the user is using Final Cut. Never test inside a production library, copy a production library, consolidate source media or delete existing data.
2. Create a new, explicitly authorized synthetic-only validation library. Import the materialized seed into that library, using Leave Files in Place and no proxy/optimized media or background analysis. Stop on missing-media, unexpected-copy, unsupported-file or uncertain conflict dialogs.
3. Inspect both test events, the generated source, the untouched timeline, ratings/marker notes and collections. Export the test event to a fresh baseline. Record Final Cut version and export version, source symlink status, event/clip counts and native UIDs. Record which seed metadata survived initial import separately.
4. Bind the fixture's proposals to that fresh native baseline. Inspect the known synthetic ranges and explicitly review/approve only the two synthetic additions. Generate a new packet, extract it into a new folder and validate the candidate against the matching official DTD. Review its complete structural diff.
5. Import the candidate into the same isolated validation target only. Read the matching-items dialog. Record the chosen conflict action. Do not assume Replace is safe for production; do not use Keep Both merely to dismiss a warning.
6. Re-export into a new destination. Compare with the **native baseline**: event/clip counts, source UID/URI, source/clip time bounds, all preexisting ratings, keyword ranges, markers including notes/completion, metadata, roles, collections and the untouched timeline/event. Check both added annotations. Record canonicalized rational times, not decimal approximations.
7. Rebind/reapply the same proposals to the new export to verify the offline no-op gate. A second import is unnecessary for this case. Record failures as failures, and untested items as untested.
8. Keep native packages and raw exports local until audited for personal paths, bookmarks, hostnames, storage URLs, metadata and incidental caches. A synthetic source does not automatically make every native package byte public-safe. Do not publish a production package or production export.

## Unsupported or unproven cases

The CLI rejects retiming/conform-rate, external sidecar/bookmark references, bundled sidecars, nonlocal/relative media URIs, effect/media/locator resources, multiple original/proxy representations, missing/ambiguous native source identity and nested source shapes. These are rejection tests, not claims of preservation support. Complex compound, synchronized or multicam clips need their own support and fixtures.

Formatting, XML declaration spelling, DTD syntax, attribute ordering, CDATA spelling and processing instructions are not byte-preserved. In-root comments are covered; comments outside the root are not. The tests do not certify every metadata field recognized by every Final Cut version, app-internal metadata absent from FCPXML, camera sidecars, live-library freshness or concurrent editing. Custom metadata can survive XML serialization but still be discarded by Final Cut itself.
