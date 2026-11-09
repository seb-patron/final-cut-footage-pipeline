# Final Cut hand checks

These five samples are built by the check-sample generator for hand
checks inside Final Cut Pro. The generator only writes files into the
new output folder. It never opens Final Cut Pro and never scripts it.
For the export-based samples it reads the provided export file (or the
bundle's Info.fcpxml) but never modifies that export and never reads the
media files the export references. The editor does every import and every
check by hand and records what they see.

Samples built from a real event export stay on the editor's Mac. They
are never committed to the repository. Only the synthetic media-run
files and the committed sanitized fixture may be shared.

Build commands use placeholder folders. Pick a new empty folder outside
the repository checkout for each run. Media samples embed that output
folder as file:// URIs, so their XML bytes are identical only for runs
into the same output folder; to compare runs in different folders,
replace the output-folder URI with a placeholder first.

## Full capture sample (full.fcpxml)

What it holds: synthetic media from three made-up cameras (Check Camera
A, Check Camera B, Check Camera C), two clips per camera, three still
pictures, and one voiceover tone. The sample holds three events, one
per camera. Each event holds its camera's two clips with notes,
keywords, markers, to-do markers, chapter markers, favorites, and
rejects. The first event also holds the stills, two smart collections,
and one story project with clips from two cameras, a gap, and a still.

Build it:

    python3 -m tests.fcp_samples --output <new-folder>

This writes `<new-folder>/media/`, `<new-folder>/full.fcpxml`, and
`<new-folder>/vo-spine.fcpxml`.

Import it by hand: in Final Cut Pro choose File > Import > XML, select
`full.fcpxml`, and let Final Cut assign new identifiers on import.

Look at and record afterwards: each camera's clips and their notes,
keywords, markers, favorites, and rejects; the stills; the two smart
collections; the story project order (clip, gap, clip, still). Record
anything that fails to import or looks wrong.

## Voiceover spine sample (vo-spine.fcpxml)

What it holds: one event named Check voiceover holding one project.
The project spine holds only the twenty-second voiceover tone. Two
clips from two different cameras and one still sit connected above the
tone, inside its range, without overlapping. The tone carries two
chapter markers.

Build it: same command as the full sample above.

Import it by hand: in Final Cut Pro choose File > Import > XML, select
`vo-spine.fcpxml`, and let Final Cut assign new identifiers on import.

Look at and record afterwards: the tone on the spine with its full
twenty seconds; the two connected clips and the connected still above
it; the two chapter markers on the tone. Record anything that fails to
import or looks wrong.

## Replace control sample (replace-noop.fcpxml)

What it holds: the event export's bytes unchanged. It is the control
for the Replace check: importing it with Replace should change nothing.

Build it:

    python3 -m tests.fcp_samples --output <new-folder> --from-export <event-export.fcpxml>

An event export bundle works too: pass the folder ending in
`.fcpxmld` and the generator reads its `Info.fcpxml`. The run writes
only the three Replace and project-only samples: no media, no full or
voiceover samples.

Import it by hand: in Final Cut Pro choose File > Import > XML, select
`replace-noop.fcpxml`. When the matching-items dialog appears, choose
Replace first as the control step so later Replace runs have a baseline.

Look at and record afterwards: that the event, clips, notes, and
ratings match the export exactly, with nothing added or moved. Record
anything that differs.

## Replace annotations sample (replace-annotations.fcpxml)

What it holds: the same export with a small set of added annotations
on the first browser clip of each event that has one: one whole-clip
keyword, one shorter range keyword, one marker, one to-do marker, and
one favorite range, plus one added note line. No identifiers, sources,
or other attributes are changed.

Build it: same command as the control sample above.

Import it by hand: in Final Cut Pro choose File > Import > XML, select
`replace-annotations.fcpxml`. When the matching-items dialog appears,
choose Replace.

Look at and record afterwards: the five added items and the added note
line on each event's first clip; that existing ratings are untouched;
that nothing else moved. Record anything missing or changed beyond the
added items.

## Project-only sample (project-only.fcpxml)

What it holds: two top-level projects built from the export's own
browser clips, with no library and no events. Check rough cut plays
each browser clip back to back from zero. Check section selects plays
a shorter inner range of each clip back to back from zero. Every section
select is non-empty and strictly shorter than its source clip. When clips
use different frame rates, each section select is cut on its clip's own
frame grid and then aligned down to the first clip's sequence grid, and any
export where a clip is too short to leave a non-empty strictly-shorter
select on both grids is refused with a clear error instead. Exports
with any browser video clip of two source frames or fewer are refused
with a clear error instead. Resources carry over only the used assets
and formats.

Build it: same command as the control sample above.

Import it by hand: in Final Cut Pro choose File > Import > XML, select
`project-only.fcpxml`, and let Final Cut assign new identifiers on
import.

Look at and record afterwards: both projects and their clip order; that
rough cut durations match the whole-frame cuts and selects are strictly
shorter inner ranges; that playback order is back to back from zero. The
first clip of Check rough cut carries one added keyword and one added
marker; record that both are present. Record anything missing,
reordered, or wrongly timed.
