"""Tests for fcp_pipeline.trip trip settings loading."""

import datetime
import tempfile
import unittest
import warnings
from pathlib import Path

from fcp_pipeline.trip import TripSettingsError, load_trip, parse_clock_offset

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_TRIP = REPO_ROOT / "tests" / "fixtures" / "trips" / "minimal"


def contract_text(top="", pipeline_extra=""):
    """Contract-shaped project.toml; top-level lines go in ``top``."""
    return (
        'title = "Harborlight Survey"\n'
        f"{top}"
        "[final_cut]\n"
        'library = "/synthetic/libraries/Harborlight.fcpbundle"\n'
        'events = ["Harborlight Event"]\n'
        "[media]\n"
        'footage = ["/synthetic/media/harbor-clip-a.mov"]\n'
        'photos = ["/synthetic/media/harbor-photo-a.jpg"]\n'
        'assets = "/synthetic/media/assets"\n'
        "[pipeline]\n"
        'timezone = "UTC"\n'
        "start_date = 2020-06-15\n"
        f"{pipeline_extra}"
        "[pipeline.clock_offsets]\n"
        'harbor-cam-a = "-00:03:10"\n'
        "[[pipeline.places]]\n"
        'name = "Gray Gull Cove"\n'
        "lat = 0.5\n"
        "lon = -140.25\n"
        "radius_m = 500\n"
    )


def write_project_trip(parent: Path, text: str) -> Path:
    trip = parent / "trip"
    trip.mkdir()
    (trip / "project.toml").write_text(text, encoding="utf-8")
    return trip


class TestTrip(unittest.TestCase):
    def test_missing_pipeline_timezone_names_key(self):
        text = contract_text().replace('timezone = "UTC"\n', "")
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_project_trip(Path(tmp), text)
            with self.assertRaises(TripSettingsError) as raised:
                load_trip(trip)
            self.assertIn("[pipeline].timezone", str(raised.exception))

    def test_missing_pipeline_start_date_names_key(self):
        text = contract_text().replace("start_date = 2020-06-15\n", "")
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_project_trip(Path(tmp), text)
            with self.assertRaises(TripSettingsError) as raised:
                load_trip(trip)
            self.assertIn("[pipeline].start_date", str(raised.exception))

    def test_top_level_pipeline_keys_belong_under_pipeline(self):
        extras = {
            "timezone": 'timezone = "UTC"\n',
            "start_date": "start_date = 2020-06-15\n",
            "clock_offsets": "[clock_offsets]\nspare-cam = \"00:00:01\"\n",
            "places": '[[places]]\nname = "Spare Cove"\n',
        }
        for key, top in extras.items():
            with self.subTest(key=key):
                with tempfile.TemporaryDirectory() as tmp:
                    trip = write_project_trip(Path(tmp), contract_text(top=top))
                    with self.assertRaises(TripSettingsError) as raised:
                        load_trip(trip)
                    message = str(raised.exception)
                    self.assertIn(key, message)
                    self.assertIn("belongs under [pipeline]", message)

    def test_datetime_start_date_rejected(self):
        text = contract_text().replace(
            "start_date = 2020-06-15\n", "start_date = 2020-01-01T12:00:00\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_project_trip(Path(tmp), text)
            with self.assertRaises(TripSettingsError):
                load_trip(trip)

    def test_clock_offset_parses_signed(self):
        self.assertEqual(parse_clock_offset("-00:03:10"), -190)

    def test_unknown_top_level_key_warns_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_project_trip(
                Path(tmp), contract_text(top='harbor_note = "extra"\n')
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                settings = load_trip(trip)
            self.assertEqual(settings.title, "Harborlight Survey")
            self.assertEqual(len(caught), 1)
            self.assertTrue(issubclass(caught[0].category, UserWarning))
            self.assertIn("harbor_note", str(caught[0].message))

    def test_unknown_pipeline_key_warns_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_project_trip(
                Path(tmp), contract_text(pipeline_extra='harbor_note = "extra"\n')
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                settings = load_trip(trip)
            self.assertEqual(settings.title, "Harborlight Survey")
            self.assertEqual(len(caught), 1)
            self.assertTrue(issubclass(caught[0].category, UserWarning))
            self.assertIn("harbor_note", str(caught[0].message))

    def test_accepted_top_level_keys_silent(self):
        top = 'status = "active"\ncreated = 2020-01-02\n'
        extra_tables = (
            "[dates]\nwrap = 2020-06-20\n"
            "[youtube]\nplaylist = \"PLsynthetic\"\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_project_trip(
                Path(tmp), contract_text(top=top) + extra_tables
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                settings = load_trip(trip)
            self.assertEqual(caught, [])
            self.assertEqual(settings.title, "Harborlight Survey")

    def test_minimal_fixture_loads(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            settings = load_trip(FIXTURE_TRIP)
        self.assertEqual(caught, [])
        self.assertEqual(settings.title, "Harborlight Survey")
        self.assertEqual(settings.timezone, "UTC")
        self.assertEqual(settings.start_date, datetime.date(2020, 6, 15))
        self.assertEqual(settings.clock_offsets, {"harbor-cam-a": -190})
        self.assertEqual(len(settings.places), 2)
        self.assertEqual(
            settings.library, "/synthetic/libraries/Harborlight.fcpbundle"
        )
        self.assertEqual(settings.events, ["Harborlight Event"])
        self.assertEqual(settings.footage, ["/synthetic/media/harbor-clip-a.mov"])
        self.assertEqual(settings.photos, ["/synthetic/media/harbor-photo-a.jpg"])
        self.assertEqual(settings.assets, "/synthetic/media/assets")
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
