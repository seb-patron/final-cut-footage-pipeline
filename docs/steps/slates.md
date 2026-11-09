# Slates: reading the spoken slate at the start of a clip

Many clips open with a spoken slate, one sentence that says what the
clip holds, shaped like this:

```
Slate. Day 3, Moonlit Cove, B-roll, Portra 400 on the F3.
```

The slates step finds that sentence in the transcripts already in the
catalog and turns each recognized part into a keyword on the whole
clip. It reads transcripts only, so run the transcribe step first.
It listens to nothing, opens no footage, calls no model, and uses no
network.

Nothing calls this step yet; running it by hand looks like this:

```
python3 -m fcp_pipeline.slates --trip <folder>
```

`<folder>` is the trip folder holding `project.toml`.

## The keywords it makes

Each recognized part becomes a whole-clip keyword with the source
`slate`, covering the full clip wherever in the clip the slate was
spoken:

- Day (which trip day, such as `Day 3`)
- Place (where, such as `Moonlit Cove`)
- Shot (the kind of shot, such as `B-roll`)
- Stock (the film stock, such as `Portra 400`)
- Camera (the camera, such as `F3`)

The slate sentence itself is saved as well, as a note-like keyword in
the `Slate` folder, so the exact heard words stay on the clip.

## How places, stocks and cameras are matched

The step matches what it heard against the trip's own words: your
places from `project.toml`, the camera names recorded in the catalog,
and a built-in list of common film stocks. Shot words come from a
built-in list of shot kinds (`A-roll`, `B-roll`, `Interview`,
`Establishing`, `Insert`, `Pickup`, `Time-lapse`). Small mishearings
are forgiven, so a transcribed `Porter 400` still becomes `Portra
400`. Digits must always match exactly, so `Portra 400` is never
turned into `Portra 800`, and `F3` is never turned into `F2`.

You can add your own words with a `--vocabulary` file, a JSON object
with `place`, `stock`, `camera` and `shot` lists:

```json
{
  "place": ["Willow Quarry"],
  "stock": ["Portra 400"],
  "camera": ["F3"],
  "shot": ["B-roll"]
}
```

```
python3 -m fcp_pipeline.slates --trip <folder> --vocabulary words.json
```

The file only adds words; it never replaces the built-ins or your
`project.toml` places.

## What happens to words it cannot match

A part the step cannot match is kept as heard, marked with a `? `
check for you to confirm, and never dropped or guessed. An unknown
place such as `Willow Quarry` becomes a Place keyword with a check;
an unknown stock stays on the clip the same way. Only the first
unknown remark becomes a keyword; further ones are skipped with a
note. A slate word with nothing recognizable in it stores nothing at
all.

Keywords from metadata, GPS or the editor always win over a spoken
slate, and locked keywords are never touched: when the slate says
something different, the step keeps the existing keyword and leaves a
note saying which one it kept.

## The command, the dry run, and the exit codes

```
python3 -m fcp_pipeline.slates --trip <folder> [--vocabulary FILE] [--dry-run]
```

With `--dry-run` the step reports what it would tag and writes
nothing. A finished run prints one `note:` line per remark and then a
summary line such as `tagged 2, no-slate 1, no-transcript 0` (a dry
run puts `would-tag N, ` in front).

- `0`: the run completed (clips with no slate or no transcript do not
  fail it).
- `1`: the trip, the catalog or the vocabulary file could not be read
  or has the wrong shape.
- `2`: the command was used wrongly (the `--trip` folder is required).

The built-in film stocks list is generic product names only. Slates
held up to the camera on a phone screen are a later step; this one
reads spoken slates only.
