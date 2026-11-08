# Native preservation result: Final Cut Pro 12.4

## Result and limitation

**Existing ratings, markers and keyword notes survived the final keyword/marker roundtrip. Broad native metadata preservation is not established.** During a preparatory replacement import, Final Cut dropped an existing resource Reel value even though the imported XML still contained it. Do not use this result as approval for production replacement.

This was an isolated synthetic-only library on Final Cut Pro **12.4**, exporting **FCPXML 1.14** with the **Extended** metadata view. The generated 12-second movie was referenced by a symlink, not copied. No optimized or proxy media was created. Production libraries were not edited. The validation library was closed afterward.

## What happened, in order

1. Imported the rich synthetic seed. Final Cut's first export retained favorite/reject ratings, marker types/notes, keywords, clip notes, roles, collections, the test timeline, Scene and Camera Angle. It omitted arbitrary custom metadata, Camera Name and Take from the seed. The resource Reel value `TEST-REEL` was present.
2. The source movie was interpreted as Rec.601, while the initial seed declared Rec.709. Its browser clip and resource therefore had different native format IDs. The patcher's existing format guard correctly rejected the export. The guard was not relaxed.
3. Prepared a new synthetic setup XML changing **only that browser clip's format reference** to its source resource's format. Imported it into the same synthetic library, choosing Replace at the matching-items dialog, leaving “Do not ask again” unchecked. The next export had the desired format, but **resource Reel `TEST-REEL` was now absent**. This is an observed native replacement loss, not an offline serializer loss. Its precise cause and generality remain unresolved.
4. Established that second native export as the keyword test baseline. Bound and explicitly approved two synthetic additions: a keyword at source seconds 1–8 and a marker at seconds 10–11. The candidate passed the installed official 1.14 DTD. Removing those two additions restored exact parsed-tree equality with the baseline before native import.
5. Imported the candidate into the synthetic target, again choosing Replace. Exported the complete library and compared it with the native baseline. All existing content in that baseline survived semantically. Both new annotations survived at the exact rational times. Final Cut created the expected keyword collection, reordered some annotation siblings and regenerated the media bookmark bytes.
6. Rebinding and reapplying the same two proposals against the resulting export produced the expected no-new-annotations error and no packet. No second import was needed.

The rich portable seed now declares the generated media's matching Rec.601 colorspace. That revised fresh-seed import has not been separately rerun. For a fixture matching the successfully bound native baseline, use the sanitized `native-baseline` variant below.

## Final roundtrip checks

The library had three events (including Final Cut's empty default event), two direct browser clips and one project before and after. No duplicate browser clips or projects appeared.

Preserved against the final native baseline:

- Favorite/reject range boundaries, names and notes, despite an overlapping added keyword.
- Existing full-clip/range keywords and their notes.
- Regular, pending/completed to-do and chapter markers; notes, completion state and poster offset.
- Clip name and multiline Unicode note.
- Source/event/project identity, media URI, source/clip start and duration.
- Audio roles, channel layout, gain and analysis marker.
- Scene, Camera Angle and other resource metadata actually present in that baseline.
- Existing collections/folders, the other event and the untouched timeline.

**Not a pass:** Reel was already missing before step 4. Custom metadata and seed fields absent from the first native export were not covered by the final native roundtrip. Their offline XML preservation remains tested, but native preservation cannot be promised.

## Committed evidence

The [native fixture folder](../tests/fixtures/metadata-preservation/native/) contains four sanitized snapshots and a machine-readable report:

- `initial-export.fcpxml`: first native export, including Reel and the mismatched browser format.
- `compatible-input.fcpxml`: imported setup XML; only the browser format changed.
- `baseline.fcpxml`: native export after setup replacement, with Reel missing.
- `roundtrip.fcpxml`: final native keyword/marker roundtrip.
- `report.json`: results, counts, limitations, sanitization and snapshot hashes.

Six regression tests replay this evidence. One explicitly asserts the **recorded Reel loss** so it cannot be silently relabeled as successful preservation. Passing replay tests do not run Final Cut and do not mean all metadata is safe.

Before publication, absolute source/library paths, embedded media bookmarks and media signatures were removed; native-created event/project IDs were consistently mapped to artificial fixture IDs; modification/ingest dates were fixed to year 2000; the empty default event was renamed. These are **sanitized XML evidence**, not untouched exports or native library backups.

The real `.fcpbundle` was **not published**: its lock information, native event database and original-media symlink contained local username/path information. No native database was edited to conceal it. The safe, portable FCPXML+generated-media fixture remains in the repository instead.

## Reproduce the supported baseline

```sh
python3 -m tests.fixture_library --seed native-baseline --output /absolute/new/session
```

This makes importable XML and unreviewed proposals pointing at the committed generated movie in place. Follow the [exclusive native validation protocol](preservation-tests.md#exclusive-native-validation-protocol). It still requires a fresh authorized test library, a fresh native export and review before patch/import. Keep native packages and materialized local paths out of Git.

The next investigation should isolate Reel handling with a minimal fresh library and separate no-change Replace control, format-only control and keyword-only case. That work must not run in production.
