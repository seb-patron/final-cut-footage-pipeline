# Synthetic metadata-preservation library seed

This fixture consists of portable FCPXML and one original generated movie. It contains no captured footage, voice, personal project data, copied production library, third-party artwork or private history. All names, UIDs, metadata and dates are artificial. The repository's MIT license covers the fixture and its generated media.

`library.fcpxml` is a **portable source seed**, not a native Final Cut `.fcpbundle`. Its media URI is deliberately `file:///SYNTHETIC_FIXTURE_ROOT/media/generated.mov`. Run `python3 -m tests.fixture_library --output /new/session/path` from the repository root to produce an importable version that points at this media in place. Do not commit the materialized paths or raw native exports.

The generated movie is a solid blue frame and a quiet 440 Hz test tone, 640×360, 25 fps, 12 seconds, starting at timecode 01:00:00:00. SHA-256:

```
404fb6ddb91878fb5da3cad35105cfdc2c9ffd09d7e289718d34c2d659f5b4fc
```

Creation command (FFmpeg already installed; no download required):

```sh
ffmpeg -hide_banner -loglevel error -nostdin -n \
  -f lavfi -i 'color=c=0x365f91:s=640x360:r=25:d=12' \
  -f lavfi -i 'sine=frequency=440:sample_rate=48000:duration=12' \
  -map 0:v -map 1:a -c:v libx264 -preset veryfast -crf 30 \
  -pix_fmt yuv420p -c:a aac -b:a 32k -af volume=0.02 \
  -map_metadata -1 -fflags +bitexact -flags:v +bitexact -flags:a +bitexact \
  -metadata creation_time=2000-01-01T00:00:00Z \
  -metadata title='Generated metadata-preservation fixture' \
  -timecode 01:00:00:00 -movflags +faststart /new/path/generated.mov
```

Encoder versions can change output bytes. Preserve the committed hash when running tests; regenerated variants require intentional fixture updates and a new hash.

See [the preservation test plan](../../../docs/preservation-tests.md) for coverage, conflict semantics, unsupported features and the native validation protocol. The committed seed and its patched result pass the installed official FCPXML 1.14 DTD. Native keyword/marker preservation passed against a native baseline on Final Cut Pro 12.4, but a preparatory native Replace lost an existing Reel value. See [the native report](../../../docs/native-validation.md) and `native/report.json`. Broad metadata preservation is not established. Use `--seed native-baseline` to materialize the sanitized compatible baseline; the default rich seed supplies broader offline XML coverage. The rich seed now declares Rec.601 to match the generated source, but its revised fresh import has not been rerun.
