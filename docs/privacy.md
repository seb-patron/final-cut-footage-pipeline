# Privacy: network use, the offline switch, and the log

This pipeline is built to work without sending anything anywhere. This
page explains, in plain words, what could be sent and when, how to stop
it, what gets written down, and where the passwords live.

## What may be sent, and when

No step sends anything yet. The prototype commands (`footage_keywords.py`,
`sample_vision.py`) run fully on your machine: local transcript files,
local FFmpeg sampling, and local Apple Vision labels. Nothing leaves the
computer.

Later steps may ask an AI service (OpenAI by default, Claude selectable
per step) to read a frame or a transcript excerpt. When that happens, the
request goes through exactly one place: the network gate in
`fcp_pipeline/net.py`. No other module may contact the network. What goes
out is described only as kinds and sizes (for example "1 image, some
text"), never as a promise about which words or pixels. The gate refuses
anything it does not recognize rather than sending it quietly.

## The offline switch, both ways

There are two ways to go offline, and they add up:

- In the trip folder's `project.toml`, under the `[pipeline]` table:
  `offline = true`. Remove that line, or set it to `false`, to allow
  calls again.
- On any command: add `--offline`, for example
  `python3 -m fcp_pipeline scan --trip /path/to/trip --offline`.
  Leave the flag off to allow calls again.

When the switch is on, every call is refused with a clear message before
any connection is made, naming the step and the service and telling you
how to switch it back on. A command whose step tries a network call while
offline prints that message as a single line on stderr and exits 3, with
no traceback. Going offline never deletes or rewrites the
log; it just adds refusal lines.

## The log: what is written, and what never is

Every call, whether it was sent, failed, or refused while offline,
appends exactly one line to `.pipeline/network-log.tsv` inside the trip
folder. The file is created with a header line the first time it is
needed, and it is only ever appended to, never rewritten. The folder is
created if it is missing. The log is opened without following symlinks:
if `.pipeline` or the log file itself is a symlink, the write is refused
rather than followed, so a swapped link can never redirect the log
outside the trip folder.

Columns, in order:

1. `time`: when the call happened, in UTC (for example
   `2026-01-02T03:04:05Z`).
2. `step`: which pipeline step made the call (for example `scan`).
3. `service`: which AI service (`openai` or `anthropic`).
4. `model`: which model was requested.
5. `sent`: a short summary of kinds and sizes, never content (for
   example `1 image 412 KB, 1,840 characters`).
6. `outcome`: `sent`, `failed`, or `refused offline`.

An example line (fields separated by tabs):

```text
2026-01-02T03:04:05Z	scan	openai	gpt-4o-mini	1 image 412 KB, 1,840 characters	sent
```

Request text, response text, and passwords never go in the log. Only
counts and sizes do.

## Where the keys come from, and that they are never written down

AI services need a password called an API key. The gate reads it from
only two places, in this order:

1. The environment: `OPENAI_API_KEY` for OpenAI, `ANTHROPIC_API_KEY`
   for Claude. Setting a variable for one session looks like
   `export OPENAI_API_KEY=...` in your shell, and it vanishes when the
   session ends.
2. Otherwise, on a Mac only, the macOS Keychain item named
   `fcp-pipeline-openai` or `fcp-pipeline-anthropic`. To store one:

```sh
security add-generic-password -s fcp-pipeline-openai -a "$USER" -w
security add-generic-password -s fcp-pipeline-anthropic -a "$USER" -w
```

Each command prompts for the key without printing it.

A missing key stops the call with an error naming the variable and the
Keychain item, never the key itself. Keys are passed to the service's
client in memory only: they are never saved to a file, never added to
the log, and never included in an error message.
