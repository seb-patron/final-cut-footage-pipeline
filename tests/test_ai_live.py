"""Opt-in live tests for the AI backends; skipped unless FCP_AI_LIVE=1.

Each test makes one tiny text-and-image call in a temporary project and
prints the usage row's tokens and cost. A backend's test runs only when
its model is set in FCP_AI_LIVE_OPENAI_MODEL or
FCP_AI_LIVE_ANTHROPIC_MODEL, and skips when its SDK or key is missing.
"""

import os
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

LIVE_ENV = "FCP_AI_LIVE"
OPENAI_MODEL_ENV = "FCP_AI_LIVE_OPENAI_MODEL"
ANTHROPIC_MODEL_ENV = "FCP_AI_LIVE_ANTHROPIC_MODEL"

CONTRACT_TOML = (
    'title = "Harborlight Survey"\n'
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
)

LIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "seen": {"type": "string"},
    },
    "required": ["seen"],
    "additionalProperties": False,
}


def tiny_png() -> bytes:
    """Build a few-pixel PNG in memory: a 2x2 gray image."""
    raw = b"".join(b"\x00\x80\x80\x80" for _ in range(2))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def live_enabled() -> bool:
    return os.environ.get(LIVE_ENV) == "1"


@unittest.skipUnless(live_enabled(), "live AI test: set FCP_AI_LIVE=1 to run")
class TestLive(unittest.TestCase):
    def _run_one(self, provider: str, model_env: str, sdk_name: str) -> None:
        from fcp_pipeline.ai import Image, ask
        from fcp_pipeline.net import _read_key
        from fcp_pipeline.trip import load_trip

        model = os.environ.get(model_env, "")
        if not model:
            self.skipTest(f"{model_env} is not set")
        try:
            __import__(sdk_name)
        except ImportError:
            self.skipTest(f"the {sdk_name!r} pip package is not installed")
        try:
            _read_key(provider)
        except RuntimeError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmp:
            trip = Path(tmp) / "trip"
            trip.mkdir()
            (trip / "project.toml").write_text(
                CONTRACT_TOML
                + "[pipeline.ai]\n"
                f'provider = "{provider}"\n'
                f'model = "{model}"\n',
                encoding="utf-8",
            )
            settings = load_trip(trip)
            result = ask(
                settings,
                "tag",
                "Name one thing visible in this tiny test image.",
                LIVE_SCHEMA,
                images=[Image(tiny_png(), "image/png")],
            )
            lines = (
                trip / ".pipeline" / "ai-usage.tsv"
            ).read_text(encoding="utf-8").splitlines()
            print(f"live {provider} answer keys: {sorted(result)}")
            print(f"live {provider} usage row: {lines[-1]}")

    def test_live_openai(self):
        self._run_one("openai", OPENAI_MODEL_ENV, "openai")

    def test_live_anthropic(self):
        self._run_one("anthropic", ANTHROPIC_MODEL_ENV, "anthropic")


if __name__ == "__main__":
    unittest.main()
