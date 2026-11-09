# Photo card pairing

A photo card is a generator clip: a styled frame from one of the
installed Film Photo Card templates. The card itself does not hold a
photo. The photo lives in the template's published drop zone, and that
binding is made by hand in Final Cut, never in a file. So every card in
this sample ships with a paired source still, and the editor binds each
photo by hand and then checks the result.

What the file carries for a card is everything around the photo: which
generator template it uses, the template reference, the caption text,
the timing on the spine, and a note naming its source still. What the
file cannot carry is the photo inside the drop zone. No file the
generator writes can bind it: the drop-zone binding exists only after a
person applies a clip to the zone in Final Cut.

## How a card and its still are written

Each card is one `video` generator clip on the project spine. It points
at an `effect` that names the installed Motion template:

    <effect id="r2" name="Film Photo Card - Xpan" uid="~/Generators.localized/Film Photo Cards/Film Photo Card - Xpan/Film Photo Card - Xpan.motn"/>

The card points back at that effect and carries its caption, a
show-or-hide switch, and a note:

    <video ref="r2" offset="0s" name="P001 PHOTO | check-photo-001" start="0s" duration="3s">
      <param name="Text" key="9999/10003/20000/20001/2/369" value="Check meadow light | Check film 01"/>
      <param name="Show camera/film" key="9999/29001/100/29002/2/100" value="1"/>
      <note>Photo card P001 shows source still S001 (check-photo-001).</note>
    </video>

The matching still sits in the event as an unplaced clip. It is not on
any spine and it is not connected to the card; it waits in the browser
until the editor applies it:

    <asset-clip ref="r6" name="S001 SOURCE | check-photo-001" start="0s" duration="3s"/>

## Numbering and the manifest

Cards and stills are numbered together. Card 1 is `P001 PHOTO`, its
still is `S001 SOURCE`, and both share the stem `check-photo-001`,
through card 4 and still 4. The still file in the media folder follows
the same number: `photo-card-still-001.jpg` through
`photo-card-still-004.jpg`.

`photo-card-manifest.json` lists the four pairs so the card file and
the list cannot disagree; both are built from the same plan. Each entry
holds `number` (1 to 4), `variant` (the template name), `card_name`,
`still_name`, `stem`, `still_file` (the file name only), `offset` and
`duration` (the same time strings written in the file), and `text` (the
caption). It holds no absolute path and no timestamp.

## The four template variants

The spine holds one card per installed variant, in this order:

- Film Photo Card - Xpan
- Film Photo Card - 2x3
- Film Photo Card - Xpan Vertical
- Film Photo Card - 3x2 Vertical

Each variant publishes exactly one drop zone called "Photo Drop Zone".
An empty drop zone shows the template's built-in placeholder picture,
and in some variants that placeholder is a real-looking photo. That
means a card that was never bound looks like a finished card with the
wrong picture. Seeing a photo on a card proves nothing; only the
by-hand step below makes the card show its own still.

## Bind each photo by hand

For each of the four cards, in order:

1. Select the card on the spine.
2. In the inspector, under Published Parameters, click the Photo Drop
   Zone thumbnail.
3. In the browser, pick the source still with the matching number
   (`S001` for card `P001`, and so on).
4. Click Apply Clip.
5. Re-select the card and confirm the source name shown matches the
   expected still name from the manifest.

## Check the midpoint render

Render each card's midpoint and confirm it shows its own photo, not the
template placeholder. A card whose midpoint still shows the placeholder
was never bound, even if it looked finished before the render.

This binding step is manual. The pipeline never launches Final Cut or
scripts it in any way, and nothing in the generated file binds a photo
on its own. The generator's output is a hand-check sample: files to
import, bind, look at, and record.
