# Scan: reading your clips into the catalog

The scan reads every clip and writes down its facts in the catalog (the
project's memory, kept under `.pipeline/`). Until the scan runs, the
catalog knows nothing about your footage. Nothing calls the scan yet;
running it by hand looks like this:

```
python3 -m fcp_pipeline.scan --trip <folder>
```

`<folder>` is the trip folder holding `project.toml`.

## What the scan reads from each clip

For every clip it reads, read-only, with the `ffprobe` and `exiftool`
tools:

- how long it is and its exact frame rate,
- its start timecode: the first non-empty `timecode` tag found in the
  format tags, then every video stream's tags in order, then every data
  stream's tags in order (a QuickTime `tmcd` track is a data stream),
- the camera that shot it,
- its GPS position, when the camera recorded one,
- when it was captured, and how that time was learned (see below).

It also counts the audio streams and measures the pixel size, and
reports those two numbers in its own result, but it does not store them:
the catalog has no columns for them yet.

Everything lands in the catalog under your project's `.pipeline/`
folder. The scan never edits the shooting day, the place, or any Final
Cut information, and it writes no tags, transcripts, or exports. Those
come from later steps.

## Your footage is never changed

The scan only reads. It never opens a clip for writing, and it never
renames, moves, deletes, or touches a file in your footage folders —
including clips inside a Final Cut bundle folder. After each clip it
checks the file is exactly as it found it; if a file changed
mid-scan, the scan stops and names the file instead of recording a
half-read fact. The only thing the scan writes is the catalog itself,
inside `.pipeline/`.

## Exact frame rates, never rounded decimals

Frame rates stay exact fractions from the clip to the catalog. A clip
shot at NTSC speed is stored as 30000/1001, never as 29.97, and its
one-second duration is stored as 1001/1000 of a second. No rounded
decimal ever becomes a stored duration, rate, or timecode.

## How the capture time is chosen

Each clip gets one capture time, picked in this order — the first one
that exists wins:

1. The camera's own creation date, read as UTC. This is the trusted
   time, and it beats anything in the file name.
2. A drone-style timestamp in the file name, such as the
   `DJI_20200102100000_` in `DJI_20200102100000_0001_D.MP4`, read as UTC.
   The digits must be a real calendar date; a name with impossible
   values is ignored.
3. The file's own modification time, read as UTC. This one is flagged
   unverified: it tells you when the file last changed, which may not
   be when the shot was taken.

The catalog records which of the three each time came from. A rerun
never relabels a clip's day or place.

## Reruns skip what has not changed

A second scan of the same footage does not probe everything again. A
clip whose file is byte-for-byte the same as last time — same size,
same modification time, same file identity — and which already has a
recorded duration is skipped with no probing. Change a clip's bytes and
only that clip is probed again.

## A clip that cannot be read is kept, not dropped

A file the reader cannot open is recorded as unreadable and reported,
one line per file, and the run goes on with the rest — but only after
the scan re-checks the file is unchanged. A file whose size or
modification time changed while the probe was running stops the run
with exit 1 instead, even when the probe itself failed, so a half-read
fact is never recorded as just unreadable. The file keeps a
placeholder row with no duration, so a later scan retries it, and a
clip that already had a row keeps that row untouched. Unreadable files
never fail the run: the scan still finishes and its summary still
prints all three counts.

A missing tool is different: when `ffprobe` or `exiftool` is not
installed, the scan stops and says which tool is missing instead of
marking every file unreadable. A missing footage folder is a warning on
its own, not a failed run.

## The command and its exit codes

```
python3 -m fcp_pipeline.scan --trip <folder>
```

The scan prints one `unreadable: <path>: <reason>` line per unreadable
file, then a summary line such as `scanned 2, skipped 5, unreadable 1`.

- `0`: the run completed (unreadable files do not fail it).
- `3`: `ffprobe` or `exiftool` is missing.
- `2`: the command was used wrongly (the `--trip` folder is required).
- `1`: the trip or catalog could not be read, or a file changed
  mid-scan.

## Days, places and gear

Once the scan has recorded each clip's facts, the days-and-places step
works out when and where each clip was shot and files that away as
keywords in the catalog. Every clip and still captured on or after
the trip's start date gets a trip day; a clip captured before the trip
starts gets no day, and a row with no capture time is left alone. A
clip whose camera recorded a position inside one of the trip's places
gets that place, and every clip with a known camera also gets a gear
keyword naming that camera. A clip whose time came only from the file's
own modification time keeps its day keyword marked "check" so you can
confirm it by hand. Positions that fall outside every place are grouped
into small unnamed clusters per day (each point joins the first cluster
within 500 metres of it), for the later memo step to name. Nothing
calls this step yet; running it by hand looks like this:

```
python3 -m fcp_pipeline.days_places --trip <folder>
```

Day 1 is the trip's start date from `project.toml`. Before the day is
counted, the capture time is moved twice: first by the camera's clock
offset, an entry under `clock_offsets` named exactly like the camera
the scan recorded (a camera with no entry counts as no offset), and
then into the trip's time zone. A clip captured at 23:58 on the start
date from a camera with a `+00:03:10` clock offset belongs to Day 2,
because its corrected local time is 00:01:10 the next morning.

Places come from the `[[pipeline.places]]` tables in `project.toml`,
each with a name, a centre point and a radius in metres, for example:

```
[[pipeline.places]]
name = "Invented Cove"
lat = 12.5
lon = 45.25
radius_m = 500
```

A clip counts as inside a place when its recorded position is within
the radius, measured as a great-circle distance; when several places
overlap, the nearest centre wins. A clip with no position, or one
outside every radius, simply gets no place.

A rerun never deletes a keyword it wrote before: a place keyword from
an earlier run stays even when the position no longer matches, and
running the step twice leaves exactly one keyword of each kind per
clip. The step prints one line per unnamed cluster and a final count
such as `assigned 5, before-start 0, no-time 1, check 2`. Its exit codes
mirror the scan's: `0` when the run completed, `2` when the command was
used wrongly, and `1` when the trip, the catalog, a place entry or the
time zone could not be read. It writes nothing but the catalog (besides
SQLite's own side file inside `.pipeline/`).

## What comes later

The scan reports audio stream counts and pixel sizes but does not store
them. Assigning each clip its day and place is done by the step above.
Naming the unnamed clusters, and reading the Final Cut export's clips,
come in later steps.
