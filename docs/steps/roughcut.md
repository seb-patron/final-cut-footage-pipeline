# Voiceover: turning your narration into word times

The voiceover step listens to the narration you recorded for the
video and writes down every word with its start and end time. It saves
one file per recording as `transcripts/<name>-<code>.json`: the
word-timed transcript, plus a reference to the recording (its path and
a fingerprint of its bytes) and a row in the catalog. The audio stays
where it is and is never copied, moved, or changed by this step.
Nothing calls this step yet; running it by hand looks like this:

```
python3 -m fcp_pipeline.vo transcribe --trip <folder>
```

`<folder>` is the trip folder holding `project.toml`.

## Where the recording comes from

You point the step at your narration in `project.toml`, with one path
or a list of paths:

```toml
[media]
voiceover = "voiceover/narration-take-2.wav"
```

or:

```toml
[media]
voiceover = ["voiceover/narration-part-one.wav", "voiceover/narration-part-two.wav"]
```

A relative path is read from inside the trip folder. The recording is
only opened to be read: once to fingerprint it, and once to listen to
it. Recordings that live inside the project's own output folders
(`transcripts/` or `.pipeline/`, even through a link) are refused,
because the step never writes next to what it reads.

## The default stays on your Mac

By default, or with this setting, the step transcribes locally on
your Mac with `mlx-whisper` and uploads nothing anywhere:

```toml
[pipeline.ai.vo_audio]
provider = "local"
```

Local transcription needs the optional `mlx-whisper` package, which
the pipeline does not install for you:

```
pip install mlx-whisper
```

It runs on Apple-silicon Macs. Without it, the recording's entry is
marked failed and the message tells you how to install it.

Before listening, the step feeds the recognizer your gear and place
words (the camera names in the catalog and the place names from your
`project.toml`), the same first prompt the footage transcription step
uses, so names from the trip are heard the way you wrote them.

## The hosted option sends your narration off the Mac

If you would rather use OpenAI's hosted audio transcription, say so
and name a model:

```toml
[pipeline.ai.vo_audio]
provider = "openai"
model = "<model-id>"
```

Replace `"<model-id>"` with a current model id from your provider's
model list; the model must be one that returns word times, otherwise
the step treats the answer as unusable and transcribes locally instead
(see below). The hosted call needs the optional package:

```
pip install openai
```

Keys come from the environment or the macOS Keychain, exactly as
described in [docs/ai-providers.md](ai-providers.md); they are never
read from the repo or written to any log.

Be plain about what this choice does: the hosted option uploads your
spoken narration off the Mac to OpenAI, using your own API credits. It
sends the whole recording's audio and nothing else: no footage, no
pictures, no other files. The default local choice sends nothing
anywhere. A recording over the hosted size limit is never split here:
it runs locally instead, and says so on the screen.

## Word times are required

Every saved word carries a start and an end. When the hosted answer
comes back without usable word times, the step transcribes locally
instead and says so on the screen as a `warning:` line. The screen
note names only the kind of failure, never keys, paths, or any of your
spoken words. The saved file then carries the local engine, exactly as
if you had asked for local in the first place. When even the local
answer has no word times, the recording's entry is marked failed with
the reason `no word times in the answer` and nothing is written: no
time is ever made up.

## Skips, changed recordings, and the saved file

A recording that is already done, recognized by a fingerprint
(SHA-256) of its bytes stored in an earlier `transcripts/` file, is
skipped without listening again. Change the recording and the next run
writes a second file next to the old one, with a new code from the new
fingerprint; existing files are never overwritten, changed, or
deleted. The saved file holds the transcript with per-word times
(seconds, rounded to 3 decimals), the recording's absolute path and
fingerprint, which engine and model were used, and the language. It
never holds audio, and the step writes nothing else: no `.srt`, no
`.txt`, nothing under `memo/`, and never `outline.md`.

## The command, the dry run, and the exit codes

```
python3 -m fcp_pipeline.vo transcribe --trip <folder> [--dry-run] [--offline]
```

With `--dry-run` the step reports what it would transcribe and writes
nothing. With `--offline` every network call is refused before
anything is sent, and the step uses the Mac instead. An offline run
says `offline: transcribed locally`. A finished run prints one
`warning:` line per note on the error output, one
`failed: <name>: <reason>` line per failed recording, and then a
summary line such as `transcribed 1, skipped 1, failed 0`.

- `0`: every recording was transcribed or skipped.
- `1`: a recording failed, or the trip, catalog, or safe write failed.
- `2`: the `[pipeline.ai.vo_audio]` setting names a missing model (the
  error names the key, for example
  `[pipeline.ai.vo_audio].model`).

Mapping the voiceover words onto shots, and the roughcut command that
will use them, come later; this step only writes down what was said.
