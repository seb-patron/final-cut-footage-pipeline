"""Tests for fcp_pipeline.trip trip settings loading."""

import tempfile
import unittest
import warnings
from pathlib import Path

from fcp_pipeline.trip import TripSettingsError, load_trip, parse_clock_offset

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_TRIP = REPO_ROOT / "tests" / "fixtures" / "trips" / "minimal"


def write_project_trip(parent: Path, text: str) -> Path:
    trip = parent / "trip"
    trip.mkdir()
    (trip / "project.toml").write_text(text, encoding="utf-8")
    return trip


class TestTrip(unittest.TestCase):
    def test_missing_timezone_names_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_project_trip(
                Path(tmp),
                'name = "Harborlight Survey"\nstart_date = 2020-06-15\n',
            )
            with self.assertRaises(TripSettingsError) as raised:
                load_trip(trip)
            self.assertIn("timezone", str(raised.exception))

    def test_clock_offset_parses_signed(self):
        self.assertEqual(parse_clock_offset("-00:03:10"), -190)

    def test_unknown_key_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_project_trip(
                Path(tmp),
                'name = "Harborlight Survey"\n'
                'timezone = "UTC"\n'
                "start_date = 2020-06-15\n"
                'harbor_note = "extra"\n',
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                settings = load_trip(trip)
            self.assertEqual(settings.name, "Harborlight Survey")
            self.assertEqual(len(caught), 1)
            self.assertTrue(issubclass(caught[0].category, UserWarning))
            self.assertIn("harbor_note", str(caught[0].message))

    def test_minimal_fixture_loads(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            settings = load_trip(FIXTURE_TRIP)
        self.assertEqual(caught, [])
        layout = [
            settings.project_toml,
            settings.outline,
            settings.memo,
            settings.transcripts,
            settings.shots,
            settings.edit,
            settings.contact_sheets,
            settings.pipeline,
            settings.scratch,
        ]
        for path in layout:
            with self.subTest(path=path):
                self.assertTrue(Path(path).is_relative_to(FIXTURE_TRIP))


if __name__ == "__main__":
    unittest.main()
