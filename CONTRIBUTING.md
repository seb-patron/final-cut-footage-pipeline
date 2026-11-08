# Contributing

This is a conservative, review-gated prototype. Contributions should preserve that boundary: never silently approve proposals, overwrite existing packets, truncate ranges, mutate native library databases or import into Final Cut Pro automatically.

Run from the repository root:

```sh
python3 -m unittest discover -v
python3 -m py_compile footage_keywords.py sample_vision.py
```

Use temporary files and original synthetic inputs for tests. Add a regression test for each behavior change. Separate XML-level assertions from native Final Cut Pro validation, and record which one was actually run. Never commit real user media, transcripts, native path/bookmark data, credentials or private source history. Read-only XML comparisons are not proof of a native database round trip.

Before publishing test fixtures, inspect every file and metadata field. Native libraries can include machine-specific paths and bookmarks even when their media is synthetic. Prefer portable fixture generators and sanitized, reviewable XML; a native binary fixture needs a separate privacy audit.
