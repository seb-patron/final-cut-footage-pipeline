# Offline FCPXML check

The offline check looks at one exported timeline file and tells you
whether it is sound, without ever opening Final Cut.

## What it does

Give it one exported `.fcpxml` file (or an `.fcpxmld` bundle, in which
case it reads the `Info.fcpxml` inside). It then:

- reads the file's bytes and prints a SHA-256 fingerprint of exactly
  those bytes, so you can tell whether the file changed;
- checks the file against the description of the format that ships
  inside the installed Final Cut, for the file's own version (versions
  1.0 to 1.14 each have their own description);
- checks that every media file the timeline points to actually exists
  on disk (it never opens or plays them, it only looks for them);
- lists the names of the events and projects it finds.

## The four answers

The first line printed is always one of four words:

- `valid`: the file was read, every media file exists, and the
  installed description accepts it.
- `invalid`: something is wrong (the file cannot be read, it is
  malformed, its version is bad, a media file is missing, or the
  description rejects it). The lines after say what.
- `refused`: the file was set aside as unsafe before anything else
  happened, and nothing further was done with it.
- `cannot verify`: the check could not run to a verdict, because the
  folder with the descriptions, the description for the file's version,
  or the `xmllint` tool is missing.

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
for its version), checks named media paths for existence only, and
reports.

Apple's descriptions are never copied into this repository. A `bridge
check` step of the pipeline command, and having the export step use
this check as its own proofreading, both come later.
