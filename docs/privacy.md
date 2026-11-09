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
folder, except that a successful call whose log write is refused appends
no line and emits one warning instead. The file is created with a header line the first time it is
needed, and it is only ever appended to, never rewritten. The folder is
created if it is missing. The log is opened relative to directory file
descriptors with `O_NOFOLLOW` at every component under the trip folder
(openat-style via `dir_fd`), so if `.pipeline` or the log file itself is
a symlink, the write is refused rather than followed, and a swapped link
can never redirect the log outside the trip folder. The log file itself is opened with `O_NONBLOCK` so an existing FIFO can never block the open, and only a regular file is ever written: any other file type at the log path is closed and refused without writing anything. After the log file is
opened, the opened `.pipeline` directory is re-verified with `fstat` and
`fstatat` (same `st_dev` and `st_ino` as `root_fd/.pipeline`); if that
check fails, only the file this call just created is removed and nothing
is written. When a call succeeds but the log write is refused, the gate
still returns the call's result and emits one warning naming the log
problem (never request text or keys) instead of raising.

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

## Threat model: a single-user tool on your own Mac

This is a single-user tool on the editor's own Mac. The layers are meant
to stop mistakes and stale configuration, not a live adversary on the same
machine:

- symlinks anywhere in the path at open time: a `.pipeline` directory or
  log file swapped for a symlink is refused instead of followed;
- `".."` and absolute paths that would escape the trip folder: the log
  path is two fixed names (`.pipeline`, `network-log.tsv`) opened
  relative to the project-root directory fd, never by joining
  user-supplied paths;
- protected-root overlap: the log can only ever land inside the trip
  folder's own `.pipeline`, never outside it;
- overwrites: the log is append-only and never rewritten;
- numbering collisions: the log file is created with `O_EXCL` tracking so
  the code knows whether this call created it, instead of guessing from
  names or sizes.

The layers must also never delete or overwrite anything. The only removal
the gate ever does is deleting the log file this same call just created
when its post-open check fails; it never touches a pre-existing file.

A concurrent process that actively renames or swaps directories in the
middle of an operation is out of scope: no local check can close that
race completely. The gate still uses the strongest practical method: it
opens every path component from the project-root directory fd with
`O_NOFOLLOW` (openat-style via `dir_fd`), and after opening it
re-verifies with `fstat`/`fstatat` that the opened directory is still the
project's own `.pipeline` (same `st_dev` and `st_ino` as
`root_fd/.pipeline`). If that check fails, it removes only the file this
call just created and writes nothing else, and the refused write keeps
the call's result with one warning naming the log problem instead of
raising.

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
