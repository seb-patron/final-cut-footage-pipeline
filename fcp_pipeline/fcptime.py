"""Exact-fraction time math for the footage pipeline.

Moved unchanged in behavior from the ``footage_keywords`` prototype so both
prototype commands keep working exactly as before.
"""
import math
from fractions import Fraction


def seconds(value):
    value = str(value)
    return Fraction(value[:-1] if value.endswith('s') else value)


def xtime(value):
    v = Fraction(value)
    return f'{v.numerator}/{v.denominator}s'


def snap_range(start, end, frame_duration, origin=Fraction(0)):
    """Snap a source-relative range out to enclosing frame edges.

    ``lo = origin + floor(start / frame_duration) * frame_duration`` and
    ``hi = origin + ceil(end / frame_duration) * frame_duration`` as exact
    Fractions, never floats.
    """
    lo = origin + math.floor(start / frame_duration) * frame_duration
    hi = origin + math.ceil(end / frame_duration) * frame_duration
    return (lo, hi)


def check_source_range(start, end, duration):
    """Refuse out-of-bounds source ranges instead of truncating them."""
    if not 0 <= start < end <= duration:
        raise ValueError('out-of-bounds source range')


def check_frame_range(lo, hi, clip_start, clip_duration, origin, source_duration):
    """Refuse frame-aligned ranges past the clip or native source.

    Never shortens a range.
    """
    if not clip_start <= lo < hi <= clip_start + clip_duration:
        raise ValueError('frame-aligned range exceeds browser clip; revise and review')
    if not origin <= lo < hi <= origin + source_duration:
        raise ValueError('frame-aligned range exceeds native source')


def clip_timing(clip, resource):
    """Read timing from the clip and resource XML elements.

    ``origin`` is the resource start (default ``"0s"``), ``clip_start``
    defaults to the resource start, ``clip_duration`` defaults to the
    resource duration, and ``source_duration`` is the resource duration.
    This is the 1:00:00:00 start handling.
    """
    origin = seconds(resource.get('start', '0s'))
    clip_start = seconds(clip.get('start', resource.get('start', '0s')))
    clip_duration = seconds(clip.get('duration', resource.get('duration')))
    source_duration = seconds(resource.get('duration'))
    return (origin, clip_start, clip_duration, source_duration)
