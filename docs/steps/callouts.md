# Callouts: turning your spoken marks into tags

While you shoot, you can talk to the pipeline: say one of four words on
its own and the step files it for you. Say **mark** and you get a marker
at that word. Say **keeper** and the take you just shot becomes a
favorite. Say **trash** and the take becomes a reject. Say **again** and
the take is marked superseded, so the next take is preferred.

Nothing is called by anything else yet; running it by hand looks like
this:

```
python3 -m fcp_pipeline.callouts --trip <folder>
```

`<folder>` is the trip folder holding `project.toml`.

## The four words and what each makes

- **Mark** stores a Marker tag called "Marker": a one-frame marker at
  the word itself.
- **Keeper** stores a Rating tag called "Favorite": a favorite range
  over the take the word ends.
- **Trash** stores a Rating tag called "Reject": a reject range over
  the take the word ends.
- **Again** stores a Take tag called "Superseded": the take it ends is
  marked superseded, so the next take is the one to prefer.

## What a take is

A take is the speech between callouts: everything you said since the
previous keeper, trash or again (or since the clip started), up to the
callout that ends it. A long silence splits takes too: a pause longer
than 2 seconds starts a new take, and `--gap` changes that length (see
below). When the pause before a keeper, trash or again word is longer
than the gap, the take ends at that pause and the callout stores
nothing: there is no take immediately before it. The callout word
itself is always left out of the range: the range covers your speech,
not the word you said to file it.

A take that ends at a pause or at the end of the clip, rather than at a
callout, gets no tag: there was no verdict on it.

## What counts as a callout

A word counts as a callout only when it stands alone: it is the first
word of its segment, or the pause before it is at least 0.4 seconds,
and it is the last word of its segment, or the pause after it is at
least 0.4 seconds. A callout word in the middle of a sentence ("I'll
mark it later") is ignored: with ordinary word spacing around it, it is
just speech.

In practice that means: say the word on its own, then pause. Say
"keeper", stop for a beat, and keep shooting. Mumble it mid-sentence
and the step rightly hears a sentence, not an instruction.

A callout after a long silence with no speech before it stores nothing,
and the step says so: the run prints a note like `no take before
keeper at 5` naming the word. There was no take to file.

## Whole frames

Times in the catalog are exact fractions, and every stored tag snaps
to the clip's whole frames: a marker starts at the frame the word falls
in and lasts exactly one frame, and a favorite, reject or superseded
range starts at the frame its take starts in and ends at the frame its
take ends in (never past the end of the clip). A take squeezed to zero
frames by snapping stores nothing.

## Which tags win

Every tag the step writes carries the source "callout". A locked tag,
or a tag from a higher source (editor, metadata, GPS, slate), is never
replaced, and neither is a tag already marked deleted: the callout tag
is skipped and the run prints a note like `Rating: kept slate tag`
naming the folder and the tag that was kept.
Running the step again replaces its own callout tags in place and never
adds duplicates.

## What it reads and what it never touches

The step reads only the transcripts already in the catalog: run
transcribe first, because it needs the word times, and it needs them
exact. It never opens a transcript or media file itself, never opens
the network, and never calls a speech model. It writes only catalog
tags through the catalog's own API, and it writes nothing to Final Cut:
carrying these markers, favorites and rejects into the edit is a later
step's job.

## The command, the gap, the dry run, and the exit codes

```
python3 -m fcp_pipeline.callouts --trip <folder> [--gap SECONDS] [--dry-run]
```

- `--gap SECONDS` sets the longest pause allowed inside one take
  (default 2). It must be a plain decimal number of seconds above 0
  (like `2` or `0.5`): fractions such as `1/2`, signs, exponents, and
  `inf` or `nan` are rejected.
- `--dry-run` reports what would be tagged and writes nothing: clips
  that would get tags are listed as `would-tag`, and no run is
  recorded.
- A finished run prints one `note: <path>: <text>` line per entry note
  (kept tags, callouts with no take), then a summary line such as
  `tagged 2, no-callouts 1, no-transcript 0, skipped 0` (a dry run puts
  `would-tag N, ` in front).

Exit codes:

- `0`: the run completed (clips with no callouts, no transcript or no
  frame rate do not fail it).
- `1`: the trip or the catalog could not be read.
- `2`: the command was used wrongly (the `--trip` folder is required)
  or `--gap` was not a plain decimal number above 0.
