"""Contract tests: recorded API responses parse to the checked object.

Each fixture under tests/fixtures/ai/recorded/ holds invented content in
one API's real response shape, with token usage and an invented model
name. Every fixture parses through its backend's real ``parse_response``
into the same checked object, and the image cases' ``build_request``
carries the image in that API's format. No network, key, or SDK is used.
"""

import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fcp_pipeline import net
from fcp_pipeline.ai import (
    AnthropicBackend,
    Image,
    OpenAIBackend,
    validate,
)
from fcp_pipeline.net import NetworkGate, set_offline
from fcp_pipeline.trip import load_trip

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDED = REPO_ROOT / "tests" / "fixtures" / "ai" / "recorded"

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
    "[pipeline.ai]\n"
    'provider = "openai"\n'
    'model = "fake-contract-model"\n'
)

EXPECTED = {
    "title": "Harborlight dawn",
    "takes": 2,
    "rating": 4.5,
    "approved": True,
    "note": None,
    "labels": ["harbor", "dawn"],
    "clips": [{"id": "a1", "reason": "wide water"}],
}

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "takes": {"type": "integer", "minimum": 0},
        "rating": {"type": "number", "minimum": 0, "maximum": 5},
        "approved": {"type": "boolean"},
        "note": {"type": "null"},
        "labels": {"type": "array", "items": {"type": "string"}},
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "title",
        "takes",
        "rating",
        "approved",
        "note",
        "labels",
        "clips",
    ],
    "additionalProperties": False,
}


def load_fixture(name: str) -> dict:
    with open(RECORDED / name, encoding="utf-8") as handle:
        return json.load(handle)


class TestContract(unittest.TestCase):
    def setUp(self):
        self._offline = net._PROCESS_OFFLINE
        set_offline(False)
        self._saved_modules = {
            name: sys.modules.get(name) for name in ("openai", "anthropic")
        }
        self._patchers = []
        for target in ("socket.socket", "socket.create_connection"):
            patcher = mock.patch(
                target, side_effect=AssertionError(f"real network use: {target}")
            )
            patcher.start()
            self._patchers.append(patcher)

    def tearDown(self):
        for patcher in reversed(self._patchers):
            patcher.stop()
        self._patchers = []
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        set_offline(self._offline)

    def test_recorded_responses_parse(self):
        cases = (
            ("openai-text.json", "openai", 120, 45, False),
            ("openai-image.json", "openai", 340, 52, True),
            ("anthropic-text.json", "anthropic", 130, 48, False),
            ("anthropic-image.json", "anthropic", 355, 55, True),
        )
        with tempfile.TemporaryDirectory() as tmp:
            trip = Path(tmp) / "trip"
            trip.mkdir()
            (trip / "project.toml").write_text(CONTRACT_TOML, encoding="utf-8")
            settings = load_trip(trip)
            for name, backend_name, input_tokens, output_tokens, image in cases:
                with self.subTest(fixture=name):
                    raw = load_fixture(name)
                    if backend_name == "openai":
                        backend = OpenAIBackend(NetworkGate(settings, "tag"))
                    else:
                        backend = AnthropicBackend(
                            NetworkGate(settings, "tag")
                        )
                    reply = backend.parse_response(raw)
                    self.assertEqual(reply.data, EXPECTED)
                    self.assertEqual(reply.input_tokens, input_tokens)
                    self.assertEqual(reply.output_tokens, output_tokens)
                    self.assertEqual(validate(reply.data, SCHEMA), [])
                    if image or backend_name == "anthropic":
                        request = backend.build_request(
                            "fake-contract-model",
                            "Describe the harbor frame.",
                            (
                                [Image(b"\x89PNG-fake-bytes-4417", "image/png")]
                                if image
                                else []
                            ),
                            SCHEMA,
                            64,
                        )
                        if backend_name == "openai":
                            block = request["input"][0]["content"][1]
                            self.assertEqual(block["type"], "input_image")
                            self.assertTrue(
                                block["image_url"].startswith(
                                    "data:image/png;base64,"
                                )
                            )
                        else:
                            self.assertEqual(
                                request["output_config"]["format"]["type"],
                                "json_schema",
                            )
                            self.assertEqual(
                                request["output_config"]["format"]["schema"],
                                SCHEMA,
                            )
                            self.assertNotIn("tools", request)
                            self.assertNotIn("tool_choice", request)
                            if image:
                                block = request["messages"][0]["content"][0]
                                self.assertEqual(block["type"], "image")
                                self.assertEqual(
                                    block["source"]["media_type"], "image/png"
                                )
                                self.assertEqual(
                                    block["source"]["type"], "base64"
                                )


if __name__ == "__main__":
    unittest.main()
