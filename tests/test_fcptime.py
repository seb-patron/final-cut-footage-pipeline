"""Tests for the exact-fraction time math in fcp_pipeline.fcptime."""
import unittest
from fractions import Fraction

from fcp_pipeline.fcptime import (
    check_frame_range,
    check_source_range,
    clip_timing,
    seconds,
    snap_range,
    xtime,
)
import xml.etree.ElementTree as ET


class TestTime(unittest.TestCase):
    def test_snap_ntsc_bounds(self):
        fd = Fraction(1001, 30000)
        lo, hi = snap_range(Fraction(1), Fraction(2), fd)
        self.assertEqual(lo, Fraction(29 * 1001, 30000))
        self.assertEqual(hi, Fraction(60 * 1001, 30000))
        self.assertIsInstance(lo, Fraction)
        self.assertIsInstance(hi, Fraction)

    def test_range_past_clip_refused(self):
        fd = Fraction(1001, 30000)
        lo, hi = snap_range(Fraction(1), Fraction(2), fd)
        # Snapped range ends past a 1s browser clip: refuse, never shorten.
        with self.assertRaisesRegex(ValueError, 'frame-aligned range exceeds browser clip'):
            check_frame_range(lo, hi, Fraction(0), Fraction(1), Fraction(0), Fraction(10))
        # Inside the browser clip but past the native source: native error.
        lo2, hi2 = snap_range(Fraction(9), Fraction(10), fd)
        with self.assertRaisesRegex(ValueError, 'frame-aligned range exceeds native source'):
            check_frame_range(lo2, hi2, Fraction(0), Fraction(12), Fraction(0), Fraction(9))

    def test_hour_origin(self):
        fd = Fraction(1001, 30000)
        origin = Fraction(3600)
        lo, hi = snap_range(Fraction(1), Fraction(2), fd, origin)
        plain_lo, plain_hi = snap_range(Fraction(1), Fraction(2), fd)
        self.assertEqual(lo - plain_lo, 3600)
        self.assertEqual(hi - plain_hi, 3600)
        resource = ET.fromstring('<asset start="3600s" duration="10s"/>')
        clip = ET.fromstring('<asset-clip/>')
        got_origin, clip_start, clip_duration, source_duration = clip_timing(clip, resource)
        self.assertEqual(got_origin, Fraction(3600))
        self.assertEqual(clip_start, Fraction(3600))
        self.assertEqual(clip_duration, Fraction(10))
        self.assertEqual(source_duration, Fraction(10))

    def test_seconds_and_xtime(self):
        self.assertEqual(seconds('1001/30000s'), Fraction(1001, 30000))
        self.assertEqual(seconds('6s'), Fraction(6))
        self.assertEqual(seconds('6'), Fraction(6))
        for value in ('1001/30000s', '6s', '3600s', '108029029/30000s'):
            self.assertEqual(seconds(xtime(seconds(value))), seconds(value))
        self.assertRegex(xtime(Fraction(1001, 30000)), r'^[0-9]+/[0-9]+s$')

    def test_source_range_refused(self):
        duration = Fraction(10)
        with self.assertRaisesRegex(ValueError, 'out-of-bounds source range'):
            check_source_range(Fraction(2), Fraction(2), duration)
        with self.assertRaisesRegex(ValueError, 'out-of-bounds source range'):
            check_source_range(Fraction(3), Fraction(2), duration)
        with self.assertRaisesRegex(ValueError, 'out-of-bounds source range'):
            check_source_range(Fraction(-1), Fraction(2), duration)
        with self.assertRaisesRegex(ValueError, 'out-of-bounds source range'):
            check_source_range(Fraction(1), Fraction(11), duration)


if __name__ == '__main__':
    unittest.main()
