# Offline check and export watcher

The offline check looks at one exported timeline file and tells you
whether it is sound, without ever opening Final Cut. The export
watcher sits on the folder you export into and hands you the next new
export once it has finished landing.

## What it does

Give it one exported `.fcpxml` file (or an `.fcpxmld` bundle, in which
case it reads the `Info.fcpxml` inside). It then:

- reads the file's bytes and prints a SHA-256 fingerprint of exactly
  those bytes, so you can tell whether the file changed;
- checks the file against the description of the format that ships
  inside the installed Final Cut, for the file's own version (versions
  1.0 to 1.14 each have their own description; any other version is
  reported without comparing);
- checks that every media file the timeline points to actually exists
  on disk (it never opens or plays them, it only looks for them);
- lists the names of the events and projects it finds.

## The four answers

The first line printed is always one of four words:

- `valid`: the file was read, every media file exists, and the
  installed description accepts it.
- `invalid`: something is wrong (the file cannot be read, it is
  malformed, a media file is missing, or the description rejects it).
  The lines after say what.
- `refused`: the file was set aside as unsafe before anything else
  happened, and nothing further was done with it.
- `cannot verify`: the check could not run to a verdict, because the
  file's version is not exactly one of 1.0 to 1.14, or because the
  folder with the descriptions, the description for the file's version,
  or the `xmllint` tool is missing. Any other declared version (empty,
  non-numeric, extra components, leading zeros, absurdly long digits,
  surrounding whitespace, or a missing attribute) is reported with a
  reason and never mapped to a description.

"Cannot verify" is not a pass. It means the answer is unknown, not
that the file is fine.

## Where the descriptions come from

The descriptions live inside the installed Final Cut application, in
the Interchange framework's Resources folder. Point the check at a
different folder with the `FCP_DTD_DIR` environment variable when you
need to. The check needs the `xmllint` command-line tool to do the
comparing. No description is ever copied into this repository; only the
installed copy is read.

## Running it

```
python3 -m fcp_pipeline.bridge.check <file>
```

The exit code is 0 when the file is valid, 1 when it is invalid or
refused, and 4 when it cannot be verified. Anything after the first
line names the path, version, description, fingerprint, events,
projects, missing media, and one line per problem found.

## What it never does

It never opens, launches or scripts Final Cut Pro or Motion. It never
imports anything, writes anything, or uses the network. It reads the
one file you name (and, through `xmllint`, the one description file
for its version when the version is one of 1.0 to 1.14), checks named
media paths for existence only, and reports.

Apple's descriptions are never copied into this repository. A `bridge
check` step of the pipeline command, and having the export step use
this check as its own proofreading, both come later.

## Watching the export folder for a new file

The tool cannot ask Final Cut to export anything, so you export the
timeline XML by hand into a folder. The watcher watches that one
folder and tells you when the next new export has landed.

Start the watcher on the folder first, then export under a new name:
files already in the folder when the watcher starts are ignored, even
when they are replaced later. So every export you want noticed needs a
name the folder has never held while the watcher was running.

A new file is reported only once it has stopped changing. For a
bundle, the watcher waits until the `Info.fcpxml` inside it is there
and the bundle has stopped changing. It then runs the offline check
described above on the new export and prints, in this order:

- `found:` followed by the path of the new file or bundle;
- `sha256:` followed by the fingerprint of exactly the bytes the check
  read (the `Info.fcpxml` bytes for a bundle), or `unknown` when the
  check could not read them;
- `check:` followed by the check's answer word, then one indented line
  per problem the check found, with its line and element when known.

## Running the watcher

```
python3 -m fcp_pipeline.bridge.watch --folder <folder> [--timeout SECONDS] [--interval SECONDS] [--settle SECONDS]
```

The options default to 600 seconds of waiting, polling every 1.0
seconds, and requiring a file to stand still for 2.0 seconds before it
counts as finished. Each option must be a number above 0.

The exit code is 0 when every new export found is valid, 1 when any is
invalid or refused, and 4 when the answer is cannot verify, the same
codes the check itself uses. When nothing new arrives in time, it
prints one line saying so and exits 3. A missing folder option, a
folder that does not exist or is not a folder, or a bad number is a
usage error with exit code 2.

## What the watcher writes (and never does)

It only looks inside the watched folder: it never creates, modifies,
renames or deletes any file or folder there, and no new files appear
there while it runs. Outside the watched folder it writes nothing
except Python bytecode caches (`__pycache__/*.pyc`) — in particular
nothing under `.pipeline/` either — and even those go away when run
with `python3 -B` or `PYTHONDONTWRITEBYTECODE=1` (the command also sets
`sys.dont_write_bytecode` before importing package modules). It never
opens, launches or scripts Final Cut Pro or Motion, and uses no network.
It reads the watched folder's top-level listing to spot new names, opens
each new `.fcpxml` file (or bundle's `Info.fcpxml`) with `O_NOFOLLOW`
(and `O_NONBLOCK`, then `fstat` `S_ISREG`) relative to a directory fd for
the watched folder, validates exactly those bytes through the offline
check without re-opening the candidate path, and otherwise only prints
to the screen.

A `bridge watch` step of the pipeline command, which will run this
watcher for you, comes later.
