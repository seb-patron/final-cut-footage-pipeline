# Organize: reading outline.md into sections

The organize step reads the outline you keep at the top of the project
folder (`outline.md`) and turns it into numbered sections. It only reads:
the pipeline never edits `outline.md`. Your wording, your order of lines,
and your blank lines stay exactly as you left them; the step only reports
what it found. Matching footage to sections comes in a later issue, not
here.

## Which headings become sections

Headings are lines that start with one to six `#` characters followed
by a space, a tab, or the end of the line (`## Cold open`). The step uses
your level-2 headings (`## ...`) as the sections. A level-1 heading
(`# ...`) at the top is treated as the outline's title and is not a
section.

If your outline has no level-2 headings at all, the level-1 headings
become the sections instead.

Deeper headings (`### ...`) are never sections. They stay inside the
section around them and are kept as notes (see below).

## Numbers and ids

Sections are numbered 1, 2, 3 in the order their headings appear, and each
section's label is its number plus its title, such as `1 Cold open`.

Each section also gets an id made from its title: lowercased, with every
run of characters that are not letters or digits turned into one hyphen
(`What I brought` becomes `what-i-brought`). A heading that starts with a
section number written as digits followed by `.` or `)` and a space drops
that number first, so `## 3. Cold open` still gives the title `Cold open`;
a title such as `10 things I packed` keeps its number, because its digits
are not followed by `.` or `)`.

The id comes from the words, not the position, so reordering the outline
keeps every id and changes only the numbers. Move `What I brought` from
second place to first and it is still `what-i-brought`, now labeled
`1 What I brought`.

## Wish lines and notes

A wish line is a bullet line: a line starting with `-`, `*`, or `+`
followed by a space, for example `- Wide shot of the doorway`. Indented
bullets count too. Every bullet between one section heading and the next
belongs to that section, in order.

Anything else in between that is not blank is kept as a note of that
section: a plain prose line, or a deeper heading such as `### Beat note`.
Blank lines are dropped.

## Errors

A malformed outline stops the step with an error that names the line, for
example:

`outline.md line 4: duplicate section id 'cold-open' (first used on line 1)`

The errors are:

- No sections at all (an empty outline, or text with no usable headings):
  reported at line 1.
- A bullet before the first section heading: reported at that bullet's
  line. Start the outline with a heading first.
- A section heading with an empty title (for instance a bare `##`):
  reported at that heading's line.
- Two section headings that give the same id: reported at the later
  heading's line, also naming the earlier line. Give each section a
  distinct title.
