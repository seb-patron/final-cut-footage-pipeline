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
