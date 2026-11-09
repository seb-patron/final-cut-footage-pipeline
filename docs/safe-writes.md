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

## Size cap

Files bigger than 5,000,000 bytes (5 MB) are refused everywhere except
under `.pipeline/`, because files outside `.pipeline/` are committed to
Git. Large working files belong under `.pipeline/`.

## Dry run

With dry-run mode on, nothing on disk is created, changed or removed. Each
write is still checked against every rule above, refused writes still
raise the same error, and accepted writes are recorded in a plan listing
the path each write would use and its size in bytes.
