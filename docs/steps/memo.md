# Memo: turning your trip memo recording into text

The memo step listens to the voice memo you recorded about the trip
and writes down what you said, with times. It saves one file per
recording as `memo/<name>.json`: the transcript with segment times,
plus a reference to the recording (its path and a fingerprint of its
bytes). Nothing calls this step yet; running it by hand looks like
this:

```
python3 -m fcp_pipeline.memo transcribe --trip <folder>
```

`<folder>` is the trip folder holding `project.toml`.

## Where the recording comes from

You point the step at your memo in `project.toml`, with one path or a
list of paths:

```toml
[media]
memo = "/path/to/your/memo.m4a"
```

or:

```toml
[media]
memo = ["/path/to/memo-part-one.m4a", "/path/to/memo-part-two.m4a"]
```

A relative path is read from inside the trip folder. The audio stays
where it is, in Voice Memos or wherever you keep it: it is never
copied, moved, or changed by this step. Only `memo/<name>.json` files
are written, through the pipeline's safe write layer, which never
overwrites an existing file.

## The default stays on your Mac

By default, or with this setting, the step transcribes locally on your
Mac with `mlx-whisper` and uploads nothing anywhere:

```toml
[pipeline.ai.memo_audio]
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

## The hosted option sends your memo off the Mac

If you would rather use OpenAI's hosted audio transcription, say so
and name a model:

```toml
[pipeline.ai.memo_audio]
provider = "openai"
model = "<model-id>"
```

Replace `"<model-id>"` with a current model id from your provider's
model list; the model must be one that returns segment times,
otherwise the step treats the answer as unusable and transcribes
locally instead (see below). The hosted call needs the optional
package:

```
pip install openai
```

Keys come from the environment or the macOS Keychain, exactly as
described in [docs/ai-providers.md](ai-providers.md); they are never
read from the repo or written to any log.

Be plain about what this choice does: the hosted option uploads your
spoken memo off the Mac to OpenAI, using your own API credits. It
sends the whole recording's audio, in pieces if it is long, and
nothing else: no footage, no pictures, no other files. The default
local choice sends nothing anywhere.

## When it falls back to local

A hosted run drops back to local transcription, and says so on the
screen as a `warning:` line, when:

- the offline switch is on (`offline = true` in `[pipeline]`, or
  `--offline` on the command), in which case no call is made at all;
- the hosted call fails (for example a missing key or a failed
  upload);
- the answer comes back without segment times.

The screen note names only the kind of failure, never keys, paths, or
any of your spoken words. The saved file then carries the local
engine, exactly as if you had asked for local in the first place.

## Long recordings are split at silences

A recording over the hosted size limit (about 24 MB, under the
hosted service's own 25 MB ceiling) is split before anything is
uploaded: it is decoded once with `ffmpeg` and cut at silent gaps
into `chunk-001.wav`, `chunk-002.wav`, and so on. You need `ffmpeg`
installed for this; without it the step reports that it is missing and
falls back to local. The pieces live only in a temporary folder under
your system's temporary location, never inside your project, your
recording's folder, or any footage folder, and they are deleted when
the recording is done, whatever happens. Each piece is uploaded on its
own and the segment times are shifted back into place before the pieces
are joined; the saved file records how many pieces were uploaded.

## Skips, changed recordings, and the saved file

A recording that is already done, recognized by a fingerprint
(SHA-256) of its bytes stored in an earlier `memo/` file, is skipped
without listening again. Change the recording and the next run writes
a second file next to the old one (`<stem>-<8 characters>.json`); the
old file is left untouched. The saved file holds the transcript with
segment times (seconds, rounded to 3 decimals), the recording's
absolute path and fingerprint, which engine and model were used, the
language, and the piece count. It never holds audio.

## The command, the dry run, and the exit codes

```
python3 -m fcp_pipeline.memo transcribe --trip <folder> [--dry-run] [--offline]
```

With `--dry-run` the step reports what it would transcribe and writes
nothing. With `--offline` every network call is refused before
anything is sent. A finished run prints one `warning:` line per note
on the error output, one `failed: <name>: <reason>` line per failed
recording, and then a summary line such as
`transcribed 1, skipped 1, failed 0`.

- `0`: the run completed (failed recordings do not fail it).
- `1`: the trip or catalog could not be read, or `[media].memo` has
  the wrong type.
- `2`: the command was used wrongly, or the `[pipeline.ai.memo_audio]`
  setting names a missing model (the error names the key, for example
  `[pipeline.ai.memo_audio].model`).

Reading an outline out of the memo is a later step; this one only
writes down what was said.
