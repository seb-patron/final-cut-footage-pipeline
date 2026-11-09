# Catalog

The catalog is the pipeline's memory: one SQLite file that records what the
pipeline knows about your media, and nothing else. It lives at
`.pipeline/catalog.sqlite` inside your project folder. Like everything under
`.pipeline/`, it is working data the pipeline owns, not something you edit
or commit.

Nothing fills the catalog yet. A later scan step will read your media and
write rows into it, and the pipeline command will grow a catalog subcommand.
Until then, the catalog only opens, upgrades, checks and exports.

## What it holds

Five tables:

- **media**: one row per media file. A stable string id, the kind of file
  (clip, still, memo or voiceover), its path, the file's size, modification
  time and inode, the camera, the duration, frame rate and start timecode,
  when and how the capture time was learned, GPS coordinates, the shooting
  day, the place, and the Final Cut event and asset ids. Everything except
  the id, asset id, kind and path may be unknown.
- **tags**: labels on a media file, such as the day, place, shot or gear.
  Each tag has a folder (what kind of label it is), a value, an optional
  time range, where it came from, an optional confidence from 0 to 1, and a
  status. See "Tag sources" below.
- **transcripts**: one row per transcript file for a media file, with its
  path, checksum, language, engine and creation time.
- **runs**: one row per pipeline step that ran, with its start and finish
  times, whether it was a dry run, and its outcome.
- **exports**: one row per file a run produced, with its path, checksum and
  the time it was read.

Every recorded time is UTC, written like `2020-06-15T08:30:00Z`.

## Exact times and rates

Times and rates are exact fractions, never rounded decimals. A duration of
1001/1000 of a second and a frame rate of 30000/1001 stay exactly those
fractions from the moment they are stored to the moment they are read back.
A float such as 29.97 is refused before anything is written, and a stored
value that is not a fraction is reported as an error when read. When a time
or rate is unknown, the catalog holds an empty value, never a guess.

## What "unchanged" means

A media file counts as unchanged when the catalog's row still describes the
file on disk: the same path, and the same size, modification time and file
identity the row recorded. If the file is missing, was never recorded, or
any of those three values is unknown, it counts as changed. Checking this
only looks at the file's details; it never opens or reads the file itself.

## Tag sources

A tag records where it came from. From lowest to highest, the sources are:
ai, vision-text, callout, slate, gps, metadata, editor. A higher source
outranks a lower one.

A tag can be locked. A locked tag is a human decision the pipeline must not
lose: a new value from a lower source never replaces it. A value from the
same source or a higher one still replaces it. Anything that is not locked
is freely replaced by the next value for the same label and time range.

## Schema versions

The catalog stamps its layout with a schema version. The current version is
2. Opening an older file upgrades it to the current layout in one step and
keeps every row. Opening a file stamped with a newer version than the
pipeline knows is refused, and the file is left completely untouched: same
bytes, same modification time, no extra files. A file with tables but no
version stamp is not a catalog, and opening it is refused the same way. An
empty database simply gets the current layout.

## Where the file lives, and its checks

The database file lives only in your own project's `.pipeline/` folder. It
is opened only after these checks, and any failure refuses the open and
leaves nothing behind: no `.pipeline/` folder, no file.

- Your project folder must be an existing folder.
- `.pipeline` must not be a shortcut, and when it exists it must be a
  folder. When it is missing it is created with a single step, and only
  when a new catalog is actually wanted.
- An existing catalog file must be a regular file, not a shortcut, with
  exactly one link to it.
- The database path must sit clear of every protected place: your footage,
  photos and assets folders, your Final Cut library, any attached card
  volumes handed in with the open, and anything inside a Final Cut bundle
  folder (a folder whose name ends in `.fcpbundle`).

SQLite's own side file while writing lives next to the database inside
`.pipeline/`, temporary tables stay in memory, and the catalog writes
nowhere else.

## The assets.tsv export

The editor's `assets.tsv` template has 19 columns, and the catalog exports
exactly that header plus one row per media file:

```
python3 -m fcp_pipeline.catalog export --tsv assets --trip <folder>
```

This prints the table to standard output. With `--output <path>` it writes
the file through the pipeline's safe write layer instead, so it lands only
in an allowed place and never overwrites an existing file. The export
refuses to run when there is no catalog, and it never creates one.

Columns the catalog holds are copied over: the asset id, the path and its
file name, the duration as decimal seconds, the exact frame rate as a
numerator and denominator (30000 and 1001, never 29.97), the camera, and the
Final Cut asset id and event. Columns the catalog does not hold are left
empty, and so is every unknown value: in this template, empty means
unknown, never false or zero.

## What comes later

Filling the catalog (the scan step) and a catalog subcommand of the
pipeline command come later. The export above stays available as
`python3 -m fcp_pipeline.catalog export ...` until that wiring exists.
