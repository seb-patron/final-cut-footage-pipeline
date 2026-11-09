# Transcribe: turning your clips' speech into text

The transcribe step listens to every clip and writes down what it says.
It runs entirely on your Mac: the speech recognition happens locally,
no audio is ever uploaded, and the pipeline itself opens no network
connection while it works. Nothing calls this step yet; running it by
hand looks like this:

```
python3 -m fcp_pipeline.transcribe --trip <folder>
```

`<folder>` is the trip folder holding `project.toml`.

## What you need installed

Speech recognition comes from `mlx-whisper`, an optional package the
pipeline does not install for you:

```
pip install mlx-whisper
```

It works on Apple-silicon Macs only. You also need `ffmpeg` (which
provides the `ffprobe` tool the step uses to check which clips hold
audio). Without `mlx-whisper` the command stops and tells you how to
install it; without `ffprobe` it stops and names the missing tool.

The recognition model's weights are downloaded once, by
`mlx-whisper`'s own installer from the Hugging Face hub, the first
time it runs. The pipeline never opens that connection itself, so the
pipeline's network gate and offline switch do not cover that one
download. Once the weights are on your Mac, set `HF_HUB_OFFLINE=1` to
keep the machine fully offline for every later run.

## Which model it runs

The model is chosen in `project.toml`:

```
[pipeline.transcribe]
model = "mlx-community/whisper-small-mlx"
```

Without that table the step uses the default,
`mlx-community/whisper-large-v3-turbo`. Changing the model gives a
clip a fresh set of transcript files; the old ones stay where they
are.

## Why your gear and place words go in first

Before listening, the step tells the recognizer your gear and place
words: the camera names recorded in the catalog and the place names
from your `project.toml`, up to about 600 characters' worth. Names the
trip actually uses bias the recognizer toward hearing them correctly,
so an unusual camera or cove name comes out spelled the way you wrote
it. A prompt with no names in it is simply left out.

## Which clips are skipped

- Clips with no audio stream are skipped; there is nothing to hear.
- Clips already transcribed in their current state are skipped; the
  step checks the recorded file before listening again.
- Clips that changed since the scan are skipped with a note telling
  you to run the scan again, so half-new files are never transcribed
  against old facts.

Everything else is transcribed with word-level timings, and the words
land both in the catalog and in your `transcripts/` folder.

## The three files per clip

Each transcribed clip gets three files in `transcripts/`, written in
this order: a word-level `.json`, a `.srt` of timed cues, and a plain
`.txt` with one line per segment. A clip with no speech still gets all
three (empty subtitles and text, an empty segment list), so it is not
retried on the next run.

Each file is named from the clip's file name plus a short code for the
clip's current state: `<stem>-<8 characters>.<ext>`. A changed clip or
a changed model gets a new code, and therefore new files; the old
files are never overwritten or deleted. Your `transcripts/` folder is
meant for your project's own private repository, never this one.

## The command, the dry run, and the exit codes

```
python3 -m fcp_pipeline.transcribe --trip <folder> [--dry-run]
```

With `--dry-run` the step reports what it would transcribe and writes
nothing. A finished run prints one line per problem clip (`failed:` or
`stale:` with the reason) and then a summary line such as
`transcribed 2, skipped 1, no-audio 1, stale 0, failed 0`.

- `0`: the run completed (failed or stale clips do not fail it).
- `1`: the trip or catalog could not be read.
- `2`: the transcription package is missing, or the command was used
  wrongly (the `--trip` folder is required).
- `3`: `ffprobe` is missing.

## Reusing results you already have

If you already ran Whisper on this trip outside the pipeline and kept
the JSON result files, you can file them into the catalog instead of
listening to everything again:

```
python3 -m fcp_pipeline.transcribe --trip <folder> --import <results folder>
```

`<results folder>` is an ordinary folder holding one `.json` file per
clip. Each file keeps the shape Whisper wrote: an object with a
`segments` list (each segment has a `start`, an `end`, a `text`, and
optionally a `words` list), an optional `language`, and a top-level
`duration` in seconds. Batch result files that wrap that shape work
too: the clip length is then read from the asset's duration, or from
the last segment's end when the asset carries none. A result that
carries no duration anywhere is listed as unmatched, never guessed.

A result belongs to the clip whose file name it shares: `harbor-a.json`
goes with `harbor-a.mov` (the extension may be kept, as in
`harbor-b.mov.json`, and upper and lower case match), and the result's
length must agree with the clip's length to within one video frame.
Anything else is listed, one line per result, with the reason:

```
unmatched: no-such-clip.json: no clip named no-such-clip in the catalog
imported 0, skipped 0, unmatched 1, stale 0, failed 0
```

Clips that changed since the scan are left alone, exactly as with a
fresh transcription. An imported clip counts as done: running the plain
transcribe command later skips it, and its transcript rows are marked
as imported, since the model that first produced them is unknown.

The import never runs the speech model, never needs `mlx-whisper` or
`ffprobe`, and never touches the network. It only reads your results
folder and your footage, and it only writes the same three transcript
files and catalog rows a transcription would. With `--dry-run` it
reports what it would file (`would-import`) and writes nothing. The
import exits `0` when the run completed (listed results do not fail
it), `1` when the trip, the catalog, or the results folder could not
be read, and `2` when the command was used wrongly; there is no code
`3`, because the import never needs `ffprobe`.

## How a crash resumes

The catalog record for a clip is written only after its three files,
so a crash can never leave a clip looking finished without its files.
Run the command again and only the clips with no record for their
current state are transcribed; finished clips are not listened to
again and their files are not touched. If a crash left files behind
with no record, the next run quietly picks a `-2` name (then `-3`, and
so on) and leaves those leftovers alone, and the run after that
recognizes those renamed files as finished.

Cloud transcription of footage is not built: your clips stay on your
Mac, and this step has no setting that would send them anywhere.
