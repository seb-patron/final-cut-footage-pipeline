# Safe writes

Every file the pipeline creates goes through one write path,
`fcp_pipeline/safe_write.py`. It exists so a pipeline step can never
overwrite your work, drop a file where it does not belong, or touch your
media or Final Cut libraries. There are no exceptions to the rules below.

## Where the pipeline may write

Inside your project folder, the pipeline may create new files only in
these places:

- `transcripts/`, `shots/`, `edit/` and `contact-sheets/` (at any depth),
- `.pipeline/` (the pipeline's own working area),
- `memo/<name>.json` (a note directly inside `memo/`), and
- `memo/outline-draft.md` (the draft outline).

Everything else is refused: `outline.md`, `project.toml`, `.scratch/`,
the project folder itself, any other file in `memo/`, and anything outside
the project folder.

## What it never touches

The pipeline never writes into your media folders, your Final Cut library,
any attached card, or any folder whose name ends in `.fcpbundle`. This
holds even when one of those places overlaps a writable folder, and even
when a shortcut (symlink) points into one. If a write would land at or
under any of them, it is refused and nothing is left behind: no file, no
temporary file, no new folder.

## No overwrites, and numbered `edit/` outputs

A safe write never replaces an existing file, folder or shortcut. Each
file is written to a temporary file first and then published in one atomic
step, so you never see a half-written file.

Files in `edit/` get the next free three-digit number: with
`edit/001-cut.fcpxml` present, the next output lands as
`edit/002-cut.fcpxml`. A number that is already used is never reused, two
pipeline runs writing at the same time never get the same number, and the
bytes of existing files never change. Numbering stops at 999.

Numbering takes an exclusive lock on `.pipeline/safe-write.lock` first.
That lock path is checked against every rule above before the lock file
is created or opened: it must resolve to inside the project's own
`.pipeline/` folder, so a `.pipeline/` shortcut pointing into media, a
Final Cut library, or anywhere else outside `.pipeline/` makes the write
a refusal that leaves nothing behind. The lock file itself is then
opened relative to the project folder rather than by a bare path: the
project folder, `.pipeline/`, and the lock file are each opened with
shortcuts refused instead of followed, and the opened folder is checked
to still be the project's own `.pipeline/` before anything is locked. So even if `.pipeline/` were
swapped for a shortcut after the checks, the open itself is refused and
nothing is created anywhere else. When the lock cannot be taken, or
exclusive locking is unavailable at all, the numbered write is refused
with an error and nothing is written: numbering never proceeds without
the lock. A failed lock attempt removes only the lock file that attempt
just created; a lock file that was already there is never removed. The lock is claimed atomically with `O_CREAT|O_EXCL`, so only the call whose exclusive open succeeded counts as the creator and a racing caller's lock is never removed. The lock is opened with `O_NONBLOCK` and refused unless it is a regular file, so a FIFO at the lock path fails promptly with an error instead of blocking forever.

## Threat model

This is a single-user tool running on the editor's own Mac. The layers
above stop mistakes and stale configuration, not an attacker: a
shortcut (symlink) anywhere in the path at open time, `..` or absolute
paths escaping the project, overlap with a protected media or Final Cut
root, overwriting an existing file, and two runs handing out the same
number. A refused write never deletes or overwrites anything you had:
the only thing ever removed is the lock file the same failed attempt
just created. A shortcut swapped in before the lock is opened is
refused, not followed. Only a swap racing the individual open calls
themselves — landing after the last check inside a single system call —
is out of scope; nothing above defends against that, and the docs do
not claim it does.

## Size cap

Files bigger than 5,000,000 bytes (5 MB) are refused everywhere except
under `.pipeline/`, because files outside `.pipeline/` are committed to
Git. Large working files belong under `.pipeline/`.

## Dry run

With dry-run mode on, nothing on disk is created, changed or removed. Each
write is still checked against every rule above, refused writes still
raise the same error, and accepted writes are recorded in a plan listing
the path each write would use and its size in bytes.
