"""Check-sample generator for the Final Cut hand checks.

Running ``python3 -m tests.fcp_samples --output <new folder>`` writes
synthetic media from three made-up cameras plus the full and vo-spine
FCPXML samples. With ``--from-export <event export>`` it instead writes
the replace-noop, replace-annotations and project-only samples built on
that export's own event and asset uids.

Nothing here opens Final Cut. Importing this package runs no tool and
writes nothing. Only the standard library plus tests.media_fixtures is
used, so the module works on macOS and Linux with Python 3.11.
"""

import argparse
import copy
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

#: Prefix for every value added by replace-annotations.
ADDED_PREFIX = "Check Added"

#: Note line appended by replace-annotations.
ADDED_NOTE_LINE = "Check added note line."

_XML_HEADER = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n'

_UTC = timezone.utc

# Camera plans: invented names, rates, makes, models and 2020 dates.
_CAMERAS = (
    {
        "key": "a",
        "camera": "Check Camera A",
        "rate": "30000/1001",
        "make": "Alder",
        "model": "Alder Field A",
        "when1": datetime(2020, 2, 1, 10, 0, tzinfo=_UTC),
        "when2": datetime(2020, 2, 1, 11, 0, tzinfo=_UTC),
        "timecode1": None,
        "folder": "Check Camera A event",
        "words": ("Alder meadow", "Alder river", "Alder hill"),
    },
    {
        "key": "b",
        "camera": "Check Camera B",
        "rate": "25/1",
        "make": "Birch",
        "model": "Birch Field B",
        "when1": datetime(2020, 2, 2, 10, 0, tzinfo=_UTC),
        "when2": datetime(2020, 2, 2, 11, 0, tzinfo=_UTC),
        "timecode1": "01:00:00:00",
        "folder": "Check Camera B event",
        "words": ("Birch meadow", "Birch river", "Birch hill"),
    },
    {
        "key": "c",
        "camera": "Check Camera C",
        "rate": "24000/1001",
        "make": "Cedar",
        "model": "Cedar Field C",
        "when1": datetime(2020, 2, 3, 10, 0, tzinfo=_UTC),
        "when2": datetime(2020, 2, 3, 11, 0, tzinfo=_UTC),
        "timecode1": None,
        "folder": "Check Camera C event",
        "words": ("Cedar meadow", "Cedar river", "Cedar hill"),
    },
)

_STILL_SIZES = ((320, 180), (640, 360), (480, 320))
_STILL_NAMES = ("Check still one", "Check still two", "Check still three")
_STILL_DURATION = Fraction(10, 1)
_TONE_DURATION = Fraction(20, 1)
_CLIP_SIZE = (320, 180)


def _parse_time(text):
    """Parse an FCPXML time like "1001/30000s" or "3600s" to Fraction."""
    body = text.strip()
    if not body.endswith("s"):
        raise ValueError("bad time %r" % (text,))
    body = body[:-1]
    if "/" in body:
        num, den = body.split("/", 1)
        return Fraction(int(num), int(den))
    return Fraction(int(body), 1)


def _fmt_time(value):
    """Format a Fraction of seconds as a reduced rational time string."""
    value = Fraction(value)
    if value.denominator == 1:
        return "%ds" % value.numerator
    return "%d/%ds" % (value.numerator, value.denominator)


def _fit_duration(duration, seq_frame):
    """Largest whole number of seq frames within duration (at least one)."""
    frames = duration // seq_frame
    if frames < 1:
        frames = 1
    return frames * seq_frame


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _output_refusal(output_arg):
    """Return a refusal reason for the output path, or None when usable."""
    resolved = Path(output_arg).resolve()
    for part in resolved.parts:
        if part.lower().endswith(".fcpbundle"):
            return "path is inside an .fcpbundle folder"
    repo = _repo_root()
    if resolved == repo or repo in resolved.parents:
        return "path is at or inside the repository checkout"
    if os.path.lexists(output_arg):
        probe = Path(output_arg)
        if not probe.is_dir():
            return "path exists and is not an empty folder"
        try:
            if any(probe.iterdir()):
                return "path exists and is not an empty folder"
        except OSError:
            return "path exists and is not an empty folder"
    return None


def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _write_bytes_exclusive(path, data):
    with open(path, "xb") as handle:
        handle.write(data)


def _copy_file_exclusive(src, dest):
    with open(src, "rb") as incoming:
        with open(dest, "xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)


def _media_file_plan():
    """Return the ordered (filename, kind) plan for the media folder."""
    plan = []
    for camera in _CAMERAS:
        key = camera["key"]
        plan.append(("check-camera-%s-1.mov" % key, ("clip", camera, 5)))
        plan.append(("check-camera-%s-2.mov" % key, ("clip", camera, 6)))
    for index in range(len(_STILL_SIZES)):
        plan.append(("check-still-%d.jpg" % (index + 1), ("still", index)))
    plan.append(("check-voiceover.wav", ("audio",)))
    return plan


def _build_cached_media():
    """Build cached media and return {filename: cached Path} (runs ffmpeg)."""
    from tests import media_fixtures

    made = {}
    for filename, kind in _media_file_plan():
        if kind[0] == "clip":
            _, camera, seconds = kind
            if seconds == 5:
                when = camera["when1"]
                timecode = camera["timecode1"]
            else:
                when = camera["when2"]
                timecode = None
            cached = media_fixtures.make_clip(
                seconds=seconds,
                rate=camera["rate"],
                size=_CLIP_SIZE,
                start_timecode=timecode,
                creation_time=when,
                make=camera["make"],
                model=camera["model"],
                audio="tone",
                container="mov",
            )
        elif kind[0] == "still":
            _, index = kind
            cached = media_fixtures.make_still(size=_STILL_SIZES[index])
        else:
            cached = media_fixtures.make_audio(seconds=20, kind="tone")
        made[filename] = cached
    return made


def _clip_info(camera, seconds):
    """Return (frame_rate Fraction, frame_duration, start, duration, frames)."""
    from tests.media_fixtures import clip_duration, clip_frames

    rate = Fraction(camera["rate"])
    frames = clip_frames(seconds, camera["rate"])
    duration = clip_duration(seconds, camera["rate"])
    if camera["key"] == "b" and seconds == 5:
        start = Fraction(3600, 1)
    else:
        start = Fraction(0, 1)
    return rate, Fraction(1, 1) / rate, start, duration, frames


def _format_id_for_camera(camera):
    return {"a": "r1", "b": "r2", "c": "r3"}[camera["key"]]


def _asset_id_for_clip(camera, seconds):
    base = {"a": 0, "b": 2, "c": 4}[camera["key"]]
    return "r%d" % (4 + base + (0 if seconds == 5 else 1))


def _first_clip_annotations(clip_el, camera, start, duration, frames, fd,
                            whole_value, range_value):
    """Add the full annotation set to a first clip (u placement rule)."""
    count = frames
    unit = max(1, count // 8)
    note = ET.SubElement(clip_el, "note")
    note.text = "Check note for the first %s clip." % camera["make"]
    whole = ET.SubElement(clip_el, "keyword")
    whole.set("start", _fmt_time(start))
    whole.set("duration", _fmt_time(duration))
    whole.set("value", whole_value)
    ranged = ET.SubElement(clip_el, "keyword")
    ranged.set("start", _fmt_time(start + unit * fd))
    ranged.set("duration", _fmt_time(2 * unit * fd))
    ranged.set("value", range_value)
    marker = ET.SubElement(clip_el, "marker")
    marker.set("start", _fmt_time(start + 4 * unit * fd))
    marker.set("duration", _fmt_time(fd))
    marker.set("value", "%s marker" % whole_value.rsplit(" ", 1)[0])
    todo = ET.SubElement(clip_el, "marker")
    todo.set("start", _fmt_time(start + (4 * unit + 1) * fd))
    todo.set("duration", _fmt_time(fd))
    todo.set("value", "%s task" % whole_value.rsplit(" ", 1)[0])
    todo.set("completed", "0")
    chapter = ET.SubElement(clip_el, "chapter-marker")
    chapter.set("start", _fmt_time(start + 6 * unit * fd))
    chapter.set("duration", _fmt_time(fd))
    chapter.set("value", "%s chapter" % whole_value.rsplit(" ", 1)[0])
    favorite = ET.SubElement(clip_el, "rating")
    favorite.set("name", "%s favorite" % whole_value.rsplit(" ", 1)[0])
    favorite.set("start", _fmt_time(start + unit * fd))
    favorite.set("duration", _fmt_time(2 * unit * fd))
    favorite.set("value", "favorite")
    reject = ET.SubElement(clip_el, "rating")
    reject.set("name", "%s reject" % whole_value.rsplit(" ", 1)[0])
    reject.set("start", _fmt_time(start + 5 * unit * fd))
    reject.set("duration", _fmt_time(2 * unit * fd))
    reject.set("value", "reject")


def _build_full_and_vo_spine(media_uris):
    """Build (full_root, vo_root) for the copied media file URIs."""
    # Resources shared by both samples.
    full = ET.Element("fcpxml", {"version": "1.14"})
    full_res = ET.SubElement(full, "resources")
    for fid, camera in (("r1", _CAMERAS[0]), ("r2", _CAMERAS[1]),
                       ("r3", _CAMERAS[2])):
        _, fd, _, _, _ = _clip_info(camera, 5)
        ET.SubElement(full_res, "format", {
            "id": fid,
            "frameDuration": _fmt_time(fd),
            "width": "320",
            "height": "180",
        })
    clip_assets = {}
    for camera in _CAMERAS:
        for seconds in (5, 6):
            _, fd, start, duration, _ = _clip_info(camera, seconds)
            aid = _asset_id_for_clip(camera, seconds)
            if seconds == 5:
                label = "first"
            else:
                label = "second"
            asset = ET.SubElement(full_res, "asset", {
                "id": aid,
                "name": "%s %s clip" % (camera["camera"], label),
                "start": _fmt_time(start),
                "duration": _fmt_time(duration),
                "hasVideo": "1",
                "format": _format_id_for_camera(camera),
                "hasAudio": "1",
                "videoSources": "1",
                "audioSources": "1",
                "audioChannels": "1",
                "audioRate": "48000",
            })
            ET.SubElement(asset, "media-rep", {
                "kind": "original-media",
                "src": media_uris[camera["key"]][seconds],
            })
            clip_assets[(camera["key"], seconds)] = (aid, start, duration, fd)
    still_ids = ("r10", "r11", "r12")
    for aid, name, index in zip(still_ids, _STILL_NAMES, range(3)):
        asset = ET.SubElement(full_res, "asset", {
            "id": aid,
            "name": name,
            "start": _fmt_time(Fraction(0, 1)),
            "duration": _fmt_time(_STILL_DURATION),
            "hasVideo": "1",
        })
        ET.SubElement(asset, "media-rep", {
            "kind": "original-media",
            "src": media_uris["still"][index],
        })
    tone = ET.SubElement(full_res, "asset", {
        "id": "r13",
        "name": "Check voiceover tone",
        "start": _fmt_time(Fraction(0, 1)),
        "duration": _fmt_time(_TONE_DURATION),
        "hasVideo": "0",
        "hasAudio": "1",
        "audioSources": "1",
        "audioChannels": "1",
        "audioRate": "48000",
    })
    ET.SubElement(tone, "media-rep", {
        "kind": "original-media",
        "src": media_uris["tone"],
    })

    # Full sample: three events, project in the first.
    for pos, camera in enumerate(_CAMERAS):
        event = ET.SubElement(full, "event", {"name": camera["folder"]})
        for seconds in (5, 6):
            aid, start, duration, fd = clip_assets[(camera["key"], seconds)]
            if seconds == 5:
                label = "first"
            else:
                label = "second"
            browser = ET.SubElement(event, "asset-clip", {
                "ref": aid,
                "name": "%s %s clip" % (camera["camera"], label),
                "start": _fmt_time(start),
                "duration": _fmt_time(duration),
                "format": _format_id_for_camera(camera),
            })
            words = camera["words"]
            if seconds == 5:
                _first_clip_annotations(
                    browser, camera, start, duration,
                    duration // fd, fd, words[0], words[1])
            else:
                whole = ET.SubElement(browser, "keyword")
                whole.set("start", _fmt_time(start))
                whole.set("duration", _fmt_time(duration))
                whole.set("value", words[2])
        words = camera["words"]
        for value in (words[0], words[2]):
            collection = ET.SubElement(event, "keyword-collection")
            collection.set("name", value)
        folder = ET.SubElement(event, "collection-folder")
        folder.set("name", "Check folder %s" % camera["key"].upper())
        inner = ET.SubElement(folder, "keyword-collection")
        inner.set("name", words[1])
        if pos == 0:
            for index, aid in enumerate(still_ids):
                still_clip = ET.SubElement(event, "asset-clip", {
                    "ref": aid,
                    "name": _STILL_NAMES[index],
                    "start": _fmt_time(Fraction(0, 1)),
                    "duration": _fmt_time(_STILL_DURATION),
                })
            favs = ET.SubElement(event, "smart-collection",
                                 {"name": "Check favorites", "match": "all"})
            ET.SubElement(favs, "match-ratings", {"value": "favorites"})
            stills = ET.SubElement(event, "smart-collection",
                                   {"name": "Check stills", "match": "all"})
            ET.SubElement(stills, "match-media",
                          {"rule": "is", "type": "stills"})
            project = ET.SubElement(event, "project", {"name": "Check story"})
            seq_fd = Fraction(1001, 30000)
            aid_a, start_a, dur_a, _ = clip_assets[("a", 5)]
            aid_b, start_b, dur_b, fd_b = clip_assets[("b", 5)]
            spine_a = _fit_duration(dur_a, seq_fd)
            gap_dur = 30 * seq_fd
            spine_b = _fit_duration(dur_b, seq_fd)
            still_dur = 50 * seq_fd
            total = spine_a + gap_dur + spine_b + still_dur
            sequence = ET.SubElement(project, "sequence", {
                "format": "r1",
                "duration": _fmt_time(total),
                "tcStart": "0s",
                "tcFormat": "NDF",
                "audioLayout": "stereo",
                "audioRate": "48k",
            })
            spine = ET.SubElement(sequence, "spine")
            ET.SubElement(spine, "asset-clip", {
                "ref": aid_a,
                "offset": _fmt_time(Fraction(0, 1)),
                "name": "Check Camera A first clip",
                "start": _fmt_time(start_a),
                "duration": _fmt_time(spine_a),
            })
            ET.SubElement(spine, "gap", {
                "name": "Check gap",
                "offset": _fmt_time(spine_a),
                "duration": _fmt_time(gap_dur),
            })
            ET.SubElement(spine, "asset-clip", {
                "ref": aid_b,
                "offset": _fmt_time(spine_a + gap_dur),
                "name": "Check Camera B first clip",
                "start": _fmt_time(start_b),
                "duration": _fmt_time(spine_b),
            })
            ET.SubElement(spine, "asset-clip", {
                "ref": "r10",
                "offset": _fmt_time(spine_a + gap_dur + spine_b),
                "name": _STILL_NAMES[0],
                "start": _fmt_time(Fraction(0, 1)),
                "duration": _fmt_time(still_dur),
            })

    # Voiceover sample: one event holding only one project.
    voice = ET.Element("fcpxml", {"version": "1.14"})
    voice_res = ET.SubElement(voice, "resources")
    for element in full_res:
        voice_res.append(copy.deepcopy(element))
    event = ET.SubElement(voice, "event", {"name": "Check voiceover"})
    project = ET.SubElement(event, "project", {"name": "Check voiceover story"})
    seq_fd = Fraction(1, 25)
    sequence = ET.SubElement(project, "sequence", {
        "format": "r2",
        "duration": _fmt_time(_TONE_DURATION),
        "tcStart": "0s",
        "tcFormat": "NDF",
        "audioLayout": "stereo",
        "audioRate": "48k",
    })
    spine = ET.SubElement(sequence, "spine")
    bed = ET.SubElement(spine, "asset-clip", {
        "ref": "r13",
        "offset": _fmt_time(Fraction(0, 1)),
        "name": "Check voiceover tone",
        "start": _fmt_time(Fraction(0, 1)),
        "duration": _fmt_time(_TONE_DURATION),
    })
    aid_a, start_a, dur_a, _ = clip_assets[("a", 5)]
    aid_c, start_c, dur_c, _ = clip_assets[("c", 5)]
    ET.SubElement(bed, "asset-clip", {
        "ref": aid_a,
        "lane": "1",
        "offset": _fmt_time(Fraction(1, 1)),
        "name": "Check Camera A first clip",
        "start": _fmt_time(start_a),
        "duration": _fmt_time(_fit_duration(dur_a, seq_fd)),
    })
    ET.SubElement(bed, "asset-clip", {
        "ref": aid_c,
        "lane": "1",
        "offset": _fmt_time(Fraction(7, 1)),
        "name": "Check Camera C first clip",
        "start": _fmt_time(start_c),
        "duration": _fmt_time(_fit_duration(dur_c, seq_fd)),
    })
    ET.SubElement(bed, "asset-clip", {
        "ref": "r11",
        "lane": "1",
        "offset": _fmt_time(Fraction(13, 1)),
        "name": _STILL_NAMES[1],
        "start": _fmt_time(Fraction(0, 1)),
        "duration": _fmt_time(50 * seq_fd),
    })
    for at, title in ((Fraction(5, 1), "Check part one"),
                      (Fraction(10, 1), "Check part two")):
        chapter = ET.SubElement(bed, "chapter-marker")
        chapter.set("start", _fmt_time(at))
        chapter.set("duration", _fmt_time(seq_fd))
        chapter.set("value", title)
    return full, voice


def _export_maps(root):
    resources = root.find("resources")
    assets = {}
    formats = {}
    if resources is not None:
        for asset in resources.findall("asset"):
            if asset.get("id") is not None:
                assets[asset.get("id")] = asset
        for fmt in resources.findall("format"):
            if fmt.get("id") is not None:
                formats[fmt.get("id")] = fmt
    return assets, formats


def _export_events(root):
    found = root.findall("library/event")
    found.extend(root.findall("event"))
    return found


def _browser_video_clips(root):
    """List browser video clips in document order with resolved timing."""
    assets, formats = _export_maps(root)
    clips = []
    for event in _export_events(root):
        for clip in event.findall("asset-clip"):
            ref = clip.get("ref")
            asset = assets.get(ref)
            fmt_id = clip.get("format")
            if fmt_id is None and asset is not None:
                fmt_id = asset.get("format")
            fmt = formats.get(fmt_id) if fmt_id is not None else None
            if fmt is None or fmt.get("frameDuration") is None:
                continue
            try:
                frame_dur = _parse_time(fmt.get("frameDuration"))
            except ValueError:
                continue
            start_text = clip.get("start")
            if start_text is None and asset is not None:
                start_text = asset.get("start")
            dur_text = clip.get("duration")
            if dur_text is None and asset is not None:
                dur_text = asset.get("duration")
            if dur_text is None:
                continue
            try:
                start = _parse_time(start_text or "0s")
                duration = _parse_time(dur_text)
            except ValueError:
                continue
            clips.append({
                "event": event,
                "clip": clip,
                "asset": asset,
                "ref": ref,
                "name": clip.get("name") or "",
                "format_id": fmt_id,
                "frame_dur": frame_dur,
                "start": start,
                "duration": duration,
            })
    return clips


def _favorite_start(start, duration, frame_dur, ratings):
    total_frames = duration // frame_dur
    unit = max(1, total_frames // 8)
    spans = []
    for name, rstart, rend in ratings:
        spans.append((rstart, rend))
    for slot in range(total_frames - unit + 1):
        cand = start + slot * frame_dur
        cend = cand + unit * frame_dur
        if cend > start + duration:
            break
        if not any(cand < rend and rstart < cend for rstart, rend in spans):
            return cand, unit * frame_dur
    return start + unit * frame_dur, unit * frame_dur


def _annotate_first_clips(root):
    """Add the check annotations to each event's first browser video clip."""
    for event in _export_events(root):
        first = None
        first_info = None
        for info in _browser_video_clips(root):
            if info["event"] is event:
                first = info["clip"]
                first_info = info
                break
        if first is None:
            continue
        start = first_info["start"]
        duration = first_info["duration"]
        frame_dur = first_info["frame_dur"]
        total_frames = duration // frame_dur
        unit = max(1, total_frames // 8)
        ratings = []
        for rating in first.findall("rating"):
            text = rating.get("start")
            try:
                rstart = _parse_time(text) if text is not None else start
            except ValueError:
                rstart = start
            text = rating.get("duration")
            try:
                rend = rstart + _parse_time(text) if text is not None else start + duration
            except ValueError:
                rend = start + duration
            ratings.append((rating.get("name") or "", rstart, rend))
        fav_start, fav_dur = _favorite_start(start, duration, frame_dur, ratings)
        if 3 * unit <= total_frames:
            range_start = start + unit * frame_dur
            range_dur = 2 * unit * frame_dur
        elif total_frames >= 2:
            range_start = start
            range_dur = (total_frames - 1) * frame_dur
        else:
            range_start = start
            range_dur = frame_dur
        last = max(0, total_frames - 1)
        marker_at = start + min(4 * unit, last) * frame_dur
        todo_at = start + min(4 * unit + 1, last) * frame_dur
        whole = ET.Element("keyword", {
            "start": _fmt_time(start),
            "duration": _fmt_time(duration),
            "value": "%s keyword whole" % ADDED_PREFIX,
        })
        ranged = ET.Element("keyword", {
            "start": _fmt_time(range_start),
            "duration": _fmt_time(range_dur),
            "value": "%s keyword range" % ADDED_PREFIX,
        })
        marker = ET.Element("marker", {
            "start": _fmt_time(marker_at),
            "duration": _fmt_time(frame_dur),
            "value": "%s marker" % ADDED_PREFIX,
        })
        todo = ET.Element("marker", {
            "start": _fmt_time(todo_at),
            "duration": _fmt_time(frame_dur),
            "value": "%s task" % ADDED_PREFIX,
            "completed": "0",
        })
        favorite = ET.Element("rating", {
            "name": "%s favorite" % ADDED_PREFIX,
            "start": _fmt_time(fav_start),
            "duration": _fmt_time(fav_dur),
            "value": "favorite",
        })
        added = [whole, ranged, marker, todo, favorite]
        existing = first.find("note")
        if existing is not None:
            existing.text = "%s\n%s" % (existing.text or "", ADDED_NOTE_LINE)
        else:
            note = ET.Element("note")
            note.text = ADDED_NOTE_LINE
            first.insert(0, note)
        stop_tags = {"audio-channel-source", "filter-video",
                     "filter-video-mask", "filter-audio", "metadata"}
        at = None
        for pos, child in enumerate(list(first)):
            if child.tag in stop_tags:
                at = pos
                break
        if at is None:
            for element in added:
                first.append(element)
        else:
            for shift, element in enumerate(added):
                first.insert(at + shift, element)


def _select_duration(start, duration, clip_fd, seq_fd, rough_dur):
    """Return the section-select duration, or None when the clip is too short.

    The select start stays one clip frame inside the source, on the clip
    grid; the offset and duration are timeline values on the sequence
    grid, so the source end (start + duration) need not land on a clip
    frame. The result must stay non-empty and strictly shorter than the
    rough duration.
    """
    remainder = duration - 2 * clip_fd
    if remainder <= 0:
        return None
    fitted = (remainder // seq_fd) * seq_fd
    select = min(fitted, rough_dur - seq_fd)
    if select < seq_fd:
        return None
    # Invariant guard: select <= fitted <= remainder, so this never fires.
    if start + clip_fd + select > start + duration:
        return None
    return select


def _short_clip_reason(clips):
    """Return a refusal reason when a clip cannot yield a shorter select."""
    seq_fd = clips[0]["frame_dur"] if clips else None
    for info in clips:
        if info["duration"] <= 2 * info["frame_dur"]:
            return (
                "clip %r is two source frames or fewer "
                "(duration %s with frame duration %s); "
                "section selects need more than two source frames"
                % (info["name"], _fmt_time(info["duration"]),
                   _fmt_time(info["frame_dur"]))
            )
        rough_dur = _fit_duration(info["duration"], seq_fd)
        if _select_duration(info["start"], info["duration"],
                            info["frame_dur"], seq_fd, rough_dur) is None:
            return (
                "clip %r cannot fit a section select on the sequence grid "
                "(duration %s at frame duration %s with sequence frame "
                "duration %s); section selects need a non-empty range "
                "strictly shorter than the source with the start on the "
                "clip grid and the offset and duration on the sequence grid"
                % (info["name"], _fmt_time(info["duration"]),
                   _fmt_time(info["frame_dur"]), _fmt_time(seq_fd))
            )
    return None


def _build_project_only(root, clips):
    reason = _short_clip_reason(clips)
    if reason is not None:
        raise ValueError(reason)
    version = root.get("version") or "1.14"
    out = ET.Element("fcpxml", {"version": version})
    resources = ET.SubElement(out, "resources")
    wanted_assets = set()
    wanted_formats = set()
    for info in clips:
        wanted_assets.add(info["ref"])
        if info["format_id"] is not None:
            wanted_formats.add(info["format_id"])
    seq_format = clips[0]["format_id"]
    wanted_formats.add(seq_format)
    seq_fd = clips[0]["frame_dur"]
    old_resources = root.find("resources")
    if old_resources is not None:
        for child in old_resources:
            if child.tag == "asset" and child.get("id") in wanted_assets:
                resources.append(copy.deepcopy(child))
            elif child.tag == "format" and child.get("id") in wanted_formats:
                resources.append(copy.deepcopy(child))
    rough = ET.SubElement(out, "project", {"name": "Check rough cut"})
    offset = Fraction(0, 1)
    rough_durs = []
    for info in clips:
        rough_durs.append(_fit_duration(info["duration"], seq_fd))
    rough_total = sum(rough_durs, Fraction(0, 1))
    rough_seq = ET.SubElement(rough, "sequence", {
        "format": seq_format,
        "duration": _fmt_time(rough_total),
        "tcStart": "0s",
        "tcFormat": "NDF",
        "audioLayout": "stereo",
        "audioRate": "48k",
    })
    rough_spine = ET.SubElement(rough_seq, "spine")
    for pos, info in enumerate(clips):
        item = ET.SubElement(rough_spine, "asset-clip", {
            "ref": info["ref"],
            "offset": _fmt_time(offset),
            "name": info["name"],
            "start": _fmt_time(info["start"]),
            "duration": _fmt_time(rough_durs[pos]),
        })
        offset += rough_durs[pos]
        if pos == 0:
            total_frames = info["duration"] // info["frame_dur"]
            unit = max(1, total_frames // 8)
            last = max(0, total_frames - 1)
            if 3 * unit <= total_frames:
                kw_start = info["start"] + unit * info["frame_dur"]
                kw_dur = 2 * unit * info["frame_dur"]
            elif total_frames >= 2:
                kw_start = info["start"]
                kw_dur = (total_frames - 1) * info["frame_dur"]
            else:
                kw_start = info["start"]
                kw_dur = info["frame_dur"]
            ET.SubElement(item, "keyword", {
                "start": _fmt_time(kw_start),
                "duration": _fmt_time(kw_dur),
                "value": "%s rough keyword" % ADDED_PREFIX,
            })
            ET.SubElement(item, "marker", {
                "start": _fmt_time(info["start"] + min(4 * unit, last) * info["frame_dur"]),
                "duration": _fmt_time(info["frame_dur"]),
                "value": "%s rough marker" % ADDED_PREFIX,
            })
    selects = ET.SubElement(out, "project", {"name": "Check section selects"})
    offset = Fraction(0, 1)
    select_items = []
    for info, rough_dur in zip(clips, rough_durs):
        sub_start = info["start"] + info["frame_dur"]
        sub_dur = _select_duration(info["start"], info["duration"],
                                   info["frame_dur"], seq_fd, rough_dur)
        select_items.append((sub_start, sub_dur))
    select_total = sum((dur for _, dur in select_items), Fraction(0, 1))
    select_seq = ET.SubElement(selects, "sequence", {
        "format": seq_format,
        "duration": _fmt_time(select_total),
        "tcStart": "0s",
        "tcFormat": "NDF",
        "audioLayout": "stereo",
        "audioRate": "48k",
    })
    select_spine = ET.SubElement(select_seq, "spine")
    for info, (sub_start, sub_dur) in zip(clips, select_items):
        ET.SubElement(select_spine, "asset-clip", {
            "ref": info["ref"],
            "offset": _fmt_time(offset),
            "name": info["name"],
            "start": _fmt_time(sub_start),
            "duration": _fmt_time(sub_dur),
        })
        offset += sub_dur
    return out


def build_from_export(export_bytes):
    """Build the three export samples; runs no tool and writes nothing."""
    try:
        root = ET.fromstring(export_bytes)
    except ET.ParseError as exc:
        raise ValueError("export does not parse: %s" % exc) from exc
    if root.tag != "fcpxml":
        raise ValueError("export root is not fcpxml")
    clips = _browser_video_clips(root)
    if not clips:
        raise ValueError("export has no browser video clip")
    annotated = copy.deepcopy(root)
    _annotate_first_clips(annotated)
    project_only = _build_project_only(root, clips)
    return {
        "replace-noop": bytes(export_bytes),
        "replace-annotations": (
            _XML_HEADER + ET.tostring(annotated, encoding="unicode")).encode("utf-8"),
        "project-only": (
            _XML_HEADER + ET.tostring(project_only, encoding="unicode")).encode("utf-8"),
    }


def _read_export_bytes(path_arg):
    """Read the export file or bundle; return (bytes, refusal_reason)."""
    probe = Path(path_arg)
    if not os.path.lexists(path_arg):
        return None, "export path does not exist"
    if probe.is_dir():
        if not probe.name.lower().endswith(".fcpxmld"):
            return None, "export path is not an .fcpxml file"
        info = probe / "Info.fcpxml"
        if not info.is_file():
            return None, ".fcpxmld folder has no Info.fcpxml"
        try:
            return info.read_bytes(), None
        except OSError:
            return None, "export path cannot be read"
    if probe.suffix.lower() != ".fcpxml":
        return None, "export path is not an .fcpxml file"
    try:
        return probe.read_bytes(), None
    except OSError:
        return None, "export path cannot be read"


def _check_export_bytes(data):
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return "export XML does not parse"
    if root.tag != "fcpxml":
        return "export root is not fcpxml"
    clips = _browser_video_clips(root)
    if not clips:
        return "export has no browser video clip"
    return _short_clip_reason(clips)


def _run_media_mode(output):
    cached = _build_cached_media()
    media_dir = output / "media"
    media_dir.mkdir(parents=True, exist_ok=False)
    written = []
    media_uris = {}
    for filename, kind in _media_file_plan():
        dest = media_dir / filename
        _copy_file_exclusive(cached[filename], dest)
        written.append(dest)
        uri = dest.resolve().as_uri()
        if kind[0] == "clip":
            _, camera, seconds = kind
            media_uris.setdefault(camera["key"], {})[seconds] = uri
        elif kind[0] == "still":
            media_uris.setdefault("still", {})[kind[1]] = uri
        else:
            media_uris["tone"] = uri
    full_root, voice_root = _build_full_and_vo_spine(media_uris)
    full_path = output / "full.fcpxml"
    voice_path = output / "vo-spine.fcpxml"
    _write_bytes_exclusive(
        full_path, (_XML_HEADER + ET.tostring(full_root, encoding="unicode")).encode("utf-8"))
    _write_bytes_exclusive(
        voice_path, (_XML_HEADER + ET.tostring(voice_root, encoding="unicode")).encode("utf-8"))
    written.append(full_path)
    written.append(voice_path)
    return written


def _run_export_mode(output, export_bytes):
    samples = build_from_export(export_bytes)
    written = []
    for name in ("replace-noop", "replace-annotations", "project-only"):
        dest = output / ("%s.fcpxml" % name)
        _write_bytes_exclusive(dest, samples[name])
        written.append(dest)
    return written


def main(argv=None):
    """Entry point for ``python3 -m tests.fcp_samples``."""
    parser = argparse.ArgumentParser(
        description="Write Final Cut hand-check samples.")
    parser.add_argument("--output", required=True,
                        help="new empty folder to write the samples into")
    parser.add_argument("--from-export", default=None,
                        help="event export .fcpxml file or .fcpxmld folder")
    args = parser.parse_args(argv)
    output_arg = args.output
    refusal = _output_refusal(output_arg)
    if refusal is not None:
        print("refusing output %s: %s" % (output_arg, refusal), file=sys.stderr)
        return 2
    output = Path(output_arg)
    export_bytes = None
    if args.from_export is not None:
        export_bytes, reason = _read_export_bytes(args.from_export)
        if reason is not None:
            print("refusing export %s: %s" % (args.from_export, reason),
                  file=sys.stderr)
            return 2
        reason = _check_export_bytes(export_bytes)
        if reason is not None:
            print("refusing export %s: %s" % (args.from_export, reason),
                  file=sys.stderr)
            return 2
    else:
        if not _ffmpeg_available():
            print("refusing output %s: ffmpeg and ffprobe are required" % output_arg,
                  file=sys.stderr)
            return 2
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError:
        print("refusing output %s: cannot create output folder" % output_arg,
              file=sys.stderr)
        return 2
    if export_bytes is not None:
        try:
            written = _run_export_mode(output, export_bytes)
        except ValueError as exc:
            print("refusing export %s: %s" % (args.from_export, exc), file=sys.stderr)
            return 2
    else:
        written = _run_media_mode(output)
    for path in written:
        print(str(Path(path).resolve()))
    return 0
