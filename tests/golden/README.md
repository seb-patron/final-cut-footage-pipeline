# Golden FCPXML files

A golden file is the stored canonical form of an FCPXML document, kept as
`tests/golden/<name>.fcpxml`. `assert_golden` in `tests/golden.py`
canonicalizes a document and compares it with its golden, so an exporter
change that alters the XML shows up as a test failure with a diff.

## What canonicalize ignores

Two documents compare equal when they differ only in:

- Attribute order: attributes are written sorted by name.
- Whitespace: whitespace-only text and tails between elements are dropped
  and the output is re-indented two spaces per level. Whitespace-only text
  inside a childless (leaf) element is significant and kept exactly, as is
  any other text.
- Generated ids: every `id` attribute is renumbered `r1`, `r2`, ... in
  document order, and every IDREF attribute the 1.14 DTD declares
  (`format`, `ref`, `projectRef`, `tracking`, `dataLocator`) that names
  one of those ids is rewritten to match. An IDREF naming no id in the
  document is left as is.
- Generated uids: the `uid` attribute of `event`, `project`, `media`, and
  `asset` elements becomes `UID`. An effect's `uid` names its Motion
  template, so it is compared exactly.
- Temp-folder file URIs: in any attribute value that is a `file://` URI
  whose decoded path lies under the system temp folder, the temp folder
  plus the first path component below it (the random per-test folder) is
  replaced by `file:///GOLDEN_TMP` and the rest is kept, so
  `file:///private/var/folders/xx/T/tmpab12/media/a.mov` becomes
  `file:///GOLDEN_TMP/media/a.mov`.

The XML declaration and DOCTYPE are not compared. Element order, element
text, comments, and every other attribute value are compared.

## Refreshing a golden

Run the tests with `GOLDEN_UPDATE=1`:

```
GOLDEN_UPDATE=1 /opt/homebrew/bin/python3 -m unittest tests.test_golden -v
```

This rewrites the golden files instead of comparing. Review the diff
before committing: `git diff tests/golden/` must show only changes you
expect from the exporter edit. Never commit a refreshed golden whose
diff you have not read.

## DTD check

`assert_dtd_valid` validates a file against the installed Final Cut Pro
1.14 DTD with `xmllint --noout --nonet --dtdvalid`. The `FCP_DTD`
environment variable overrides the DTD path. When the DTD file or
`xmllint` is missing, the check raises `unittest.SkipTest` instead of
failing.

Apple's DTD is never committed to this repository. Only the installed
copy is read, by `xmllint`; do not copy it into the repo.
