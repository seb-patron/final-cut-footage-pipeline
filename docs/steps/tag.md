# Contact sheets: one picture per clip

Each clip gets one contact sheet: a single picture that shows what the
clip holds at a glance. A later step reads these pictures (and a person
can look at them too), so tagging never has to open the footage itself.
Nothing calls this step yet; running it by hand looks like this:

```
python3 -m fcp_pipeline.sheets --trip <folder>
```

`<folder>` is the trip folder holding `project.toml`.

## What a sheet looks like

A sheet is one JPEG of 1536 by 864 pixels, laid out as a 3 by 3 grid of
nine tiles. Each tile is 512 by 288 pixels and shows one frame of the
clip, scaled to fit without stretching (narrower frames get black bars
on the sides). The top-left corner of every tile carries a small stamp
with the frame's time in seconds, such as `31.0s`. Short clips get fewer
than nine frames; the empty cells stay black, and the sheet is still
1536 by 864.

## How the tile times are chosen

For a normal clip the step picks nine evenly spaced moments: the
centres of nine equal slices of the clip. It also asks `ffmpeg` to
report the clip's hard cuts (any change scoring above 0.4), and folds
those cut times in. When that makes more than nine moments, the crowded
ones are thinned out evenly until nine remain. Two examples for a
60-second clip:

- With no cuts, the tiles sit at 10/3, 10, 50/3, 70/3, 30, 110/3,
  130/3, 50 and 170/3 seconds (about 3.3, 10, 16.7, 23.3, 30, 36.7,
  43.3, 50 and 56.7 seconds).
- With a single cut 31 seconds in, the list is the same except the
  30-second tile moves to exactly 31 seconds, so the cut itself is on
  the sheet.

A clip under 3 seconds gets a single tile from its middle instead.

The exact tile times are stored as fractions of a second, never rounded
decimals, so the same clip always picks the same moments.

## Where the sheets go

Sheets live under `.pipeline/sheets/` inside the trip folder, next to
the catalog. They are working files and are not committed; the
`contact-sheets/` folder at the top level is reserved for summary sheets
a person asked for, which this step never writes.

Each sheet's file name starts with the clip's file name (with any
unusual characters replaced by `_`) followed by a short code for the
clip's current state, such as `harbor-morning-a1b2c3d4.jpg`. Next to
every sheet sits a manifest with the same name ending in `.json`. The
manifest holds only these fields:

- `version`: always `1`.
- `asset_id`: the clip's catalog id.
- `sheet`: the sheet's file name.
- `sheet_sha256`: checksum of the sheet bytes.
- `sheet_size`: `[1536, 864]`.
- `grid`: `[3, 3]`.
- `tile_size`: `[512, 288]`.
- `duration`: the exact clip duration as a fraction (for example `60/1`).
- `scene_count`: how many scene changes were detected (the times themselves are never stored).
- `truncated`: whether the stored tile list was capped (always `false` from this step; tile times are already thinned to nine).
- `tiles`: the drawn tile times with grid positions, at most nine entries of `{"index": n, "time": "N/D"}`.

Manifests stay far under the 5 MB per-file limit: only the drawn tile
times and the scene count are stored, and a manifest that would exceed
the limit raises instead of writing anything. The catalog itself has no
table for tile times yet, so the manifest next to each sheet is where
they live.

## Reruns skip what has not changed

The same clip always gives the same sheet, byte for byte. A clip whose
file is unchanged since its sheet was built is skipped without running
any tool. A changed clip gets new files under a new state code while
the old sheet and manifest stay where they are. A clip the scan could
not read yet is reported as stale and left alone until the scan runs
again.

## What the step needs

Only `ffmpeg` has to be installed. The time stamps are drawn by the
tool itself, so no special `ffmpeg` build and no font on the machine
are required. The duration of each clip comes from the catalog; the
clip itself is only ever opened for reading, and nothing beside the
footage is ever written.

## The command and its exit codes

The step prints one `failed: <path>: <reason>` or
`stale: <path>: <reason>` line per problem clip, then a summary line
such as `built 3, skipped 1, stale 0, failed 0`.

- `0`: the run completed (failed or stale clips do not fail it).
- `3`: `ffmpeg` is missing.
- `2`: the command was used wrongly (the `--trip` folder is required).
- `1`: the trip or catalog could not be read.

## Building sheets: nothing leaves the Mac

Building sheets uses no network and never opens, scripts, or launches
any editing program. It reads footage, writes only into `.pipeline/`,
and sends nothing anywhere.

## Tagging a clip from its sheet

Once a clip has a contact sheet, a second step reads that picture and
asks an AI service to describe the clip. This is the first step in the
pipeline that sends anything anywhere; everything before it runs fully
on your machine. Nothing calls this step yet; running it by hand looks
like this:

```
python3 -m fcp_pipeline.tag --trip <folder>
```

What the step does, per clip: it sends the clip's one contact-sheet
picture together with a short text (one line of facts about the clip,
its slate text, the first 300 words of its transcript, and the list of
allowed tags), reads back a structured answer, checks that answer, and
stores it in the catalog as tags. A clip whose file changed since the
scan is reported as stale; a clip with no sheet for its current state
is reported with no sheet; both are left alone until the earlier steps
run again.

Exactly what is sent: one contact-sheet picture, one line of facts
(camera, duration, day, place and capture time, leaving out whatever is
unknown), the clip's slate text, the first 300 words of its transcript,
and the list of allowed tags. What never is: footage, video,
full-resolution frames, file names and paths. Only the sheet image is
ever uploaded.

The provider and model come from `[pipeline.ai.tag]` in `project.toml`,
else the shared `[pipeline.ai]` values, OpenAI by default with Claude
selectable per step. No model id is built in: you copy a current id
from your provider's model list into the setting, for example:

| Step  | Provider | Model        |
|-------|----------|--------------|
| tag   | OpenAI   | (your model) |
| tag   | Claude   | (your model) |

The provider packages are optional and not bundled:

```sh
pip install openai anthropic
```

Keys come only from the environment or the macOS Keychain, as
described in [docs/ai-providers.md](../ai-providers.md).

The answer has five fields: a one-or-two-sentence description, the
framing (`wide`, `medium`, `close-up`, `detail`, `establishing`,
`aerial`, `pov`, `static` or `other`), time ranges with tags and a
confidence each, visible problems, and an optional hint for matching
the clip to the outline later.

What is stored where, all with source `ai` (the lowest precedence,
below slate, callout, metadata and editor tags): the framing as a
whole-clip `shot` tag, the description as the clip note, the problems
as a whole-clip to-do tag, the outline hint as a `check` tag for the
later outline step, and each range tag with its start and duration.
The `? ` prefix itself is never stored: a `check` status is what the
export step later renders as the `? ` keyword.

The cutoff is 0.75: a range at or above it is stored plain, below it
as a check tag that Final Cut will later show as `? <tag>`. The
`--cutoff` flag moves that line, for example `--cutoff 0.9`.

Every range is checked before anything is stored: ranges with a
confidence outside 0 to 1 are dropped, ranges outside the clip are
dropped, tags outside the trip vocabulary are dropped, and times are
snapped outwards to the nearest tile edges so a stored range always
covers what the model named. A worked example: on a 60-second clip the
model names 3.0 to 9.0 seconds, but the nearest tile times are 0 and
10 seconds, so the range is stored as 0 to 10 seconds.

The vocabulary is the trip's own words: place names from
`project.toml`, camera names from the catalog, existing catalog tags,
and an optional `--vocabulary` JSON file mapping extra folders to
values, for example:

```json
{"subject": ["harbor-seal", "fishing-boat"]}
```

A malformed answer is retried once, then the clip is listed as failed,
never silently skipped. Locked, deleted and higher-source tags are
never changed: an existing tag in the same slot keeps a human decision
safe. A rerun skips clips that already hold `ai` tags; `--retag` tags
them again. `--dry-run` reports what would be tagged without calling
or writing anything.

The command and its exit codes:

```
python3 -m fcp_pipeline.tag --trip <folder> [--cutoff N] [--retag] [--dry-run] [--vocabulary FILE] [--offline]
```

- `0`: the run completed (failed, stale and no-sheet clips do not fail it).
- `1`: the trip, catalog or vocabulary file could not be read.
- `2`: the command was used wrongly, or the model setting is missing.
- `3`: the switch is offline.

Offline mode refuses before anything is sent: with `offline = true`
in `[pipeline]` or `--offline` on the command, every call is refused
with one line on stderr and nothing leaves the process. Every call
that gets an answer appends one row to `.pipeline/ai-usage.tsv` with
the step, provider, model and token counts, never words or pixels; see
[docs/privacy.md](../privacy.md).

Batch mode, the side-by-side provider test and the pipeline command's
`tag` subcommand come later.
