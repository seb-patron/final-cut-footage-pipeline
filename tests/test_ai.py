"""Tests for fcp_pipeline.ai, the single interface for every AI step.

No test here makes a real network call, reads a real key, or needs an
installed SDK: socket functions fail when called, fake backends override
``create`` only (building and parsing stay real), and fake SDK modules
stand in for the real clients. Every project is a temporary folder with
an invented project.toml; fake keys are obviously fake; images are a few
invented bytes.
"""

import base64
import json
import os
import socket
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

from fcp_pipeline import net
from fcp_pipeline.ai import (
    MAX_IMAGE_BYTES,
    AiAnswerError,
    AiConfigError,
    AiSchemaError,
    AnthropicBackend,
    Image,
    OpenAIBackend,
    ask,
    step_config,
)
from fcp_pipeline.net import NetworkGate, OfflineError, set_offline
from fcp_pipeline.trip import load_trip

REPO_ROOT = Path(__file__).resolve().parents[1]

FAKE_OPENAI_KEY = "fake-test-openai-key-ALPHA-7741"
FAKE_ANTHROPIC_KEY = "fake-test-anthropic-key-BETA-7741"
FAKE_KEYCHAIN_KEY = "fake-test-keychain-key-GAMMA-7741"

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

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["title", "reason"],
    "additionalProperties": False,
}
GOOD_ANSWER = {"title": "Harborlight dawn", "reason": "wide water"}
INSTRUCTIONS = "Describe the harbor frame for the transcript index."


def write_project(parent: Path, ai_extra: str = "", offline=None) -> Path:
    """Write a minimal trip folder with an invented [pipeline.ai] table."""
    trip = parent / "trip"
    trip.mkdir(parents=True, exist_ok=True)
    offline_line = ""
    if offline is not None:
        offline_line = f"offline = {'true' if offline else 'false'}\n"
    (trip / "project.toml").write_text(
        CONTRACT_TOML + offline_line + ai_extra, encoding="utf-8"
    )
    return trip


def openai_raw(answer_text: str, input_tokens=1000, output_tokens=500) -> dict:
    """Recorded-style OpenAI Responses result carrying ``answer_text``."""
    return {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": answer_text}],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def anthropic_raw(answer: dict, input_tokens=1000, output_tokens=500) -> dict:
    """Recorded-style Anthropic Messages result carrying ``answer`` as text JSON."""
    return {
        "content": [
            {"type": "text", "text": json.dumps(answer)},
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def bad_openai_raw() -> dict:
    """OpenAI-shaped answer missing the required ``reason`` field."""
    return openai_raw(json.dumps({"title": "Harborlight dawn"}))


def bad_anthropic_raw() -> dict:
    """Anthropic-shaped answer missing the required ``reason`` field."""
    return anthropic_raw({"title": "Harborlight dawn"})


class FakeOpenAI(OpenAIBackend):
    """Real OpenAI building and parsing; only ``create`` is faked."""

    def __init__(self, gate, raws):
        super().__init__(gate)
        self.requests: list = []
        self.raws = list(raws)

    def create(self, request):
        self.requests.append(request)
        return self.raws.pop(0)


class FakeAnthropic(AnthropicBackend):
    """Real Anthropic building and parsing; only ``create`` is faked."""

    def __init__(self, gate, raws):
        super().__init__(gate)
        self.requests: list = []
        self.raws = list(raws)

    def create(self, request):
        self.requests.append(request)
        return self.raws.pop(0)


def install_fake_sdks(recorded):
    """Put fake openai/anthropic modules in sys.modules; return saved state."""
    saved = {name: sys.modules.get(name) for name in ("openai", "anthropic")}
    openai_module = types.ModuleType("openai")
    anthropic_module = types.ModuleType("anthropic")

    class FakeResponses:
        def create(self, **request):
            recorded["openai_requests"].append(request)
            return recorded["openai_raw"]

    class FakeOpenAIClient:
        def __init__(self, api_key=None):
            recorded["openai_keys"].append(api_key)
            self.responses = FakeResponses()

    class FakeMessages:
        def create(self, **request):
            recorded["anthropic_requests"].append(request)
            return recorded["anthropic_raw"]

    class FakeAnthropicClient:
        def __init__(self, api_key=None):
            recorded["anthropic_keys"].append(api_key)
            self.messages = FakeMessages()

    openai_module.OpenAI = FakeOpenAIClient
    anthropic_module.Anthropic = FakeAnthropicClient
    sys.modules["openai"] = openai_module
    sys.modules["anthropic"] = anthropic_module
    return saved


class TestAi(unittest.TestCase):
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

    def test_step_uses_its_provider_and_model(self):
        ai_extra = (
            "[pipeline.ai]\n"
            'provider = "openai"\n'
            'model = "fake-openai-model"\n'
            "[pipeline.ai.tag]\n"
            'provider = "anthropic"\n'
            'model = "fake-anthropic-model"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_project(Path(tmp), ai_extra))
            gate = NetworkGate(settings, "tag")
            fake_openai = FakeOpenAI(
                gate, [openai_raw(json.dumps(GOOD_ANSWER))]
            )
            fake_anthropic = FakeAnthropic(
                gate, [anthropic_raw(dict(GOOD_ANSWER))]
            )
            backends = {"openai": fake_openai, "anthropic": fake_anthropic}
            self.assertEqual(
                ask(settings, "tag", INSTRUCTIONS, ANSWER_SCHEMA,
                    backends=backends),
                GOOD_ANSWER,
            )
            self.assertEqual(len(fake_anthropic.requests), 1)
            self.assertEqual(
                fake_anthropic.requests[0]["model"], "fake-anthropic-model"
            )
            self.assertEqual(fake_openai.requests, [])
            for step in ("photos", "memo", "organize", "roughcut"):
                with self.subTest(step=step):
                    gate = NetworkGate(settings, step)
                    fake_o = FakeOpenAI(
                        gate, [openai_raw(json.dumps(GOOD_ANSWER))]
                    )
                    fake_a = FakeAnthropic(
                        gate, [anthropic_raw(dict(GOOD_ANSWER))]
                    )
                    result = ask(
                        settings, step, INSTRUCTIONS, ANSWER_SCHEMA,
                        backends={"openai": fake_o, "anthropic": fake_a},
                    )
                    self.assertEqual(result, GOOD_ANSWER)
                    self.assertEqual(len(fake_o.requests), 1)
                    self.assertEqual(
                        fake_o.requests[0]["model"], "fake-openai-model"
                    )
                    self.assertEqual(fake_a.requests, [])
        # No provider set anywhere: a step goes to OpenAI.
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(
                write_project(
                    Path(tmp),
                    "[pipeline.ai]\nmodel = \"fake-openai-model\"\n",
                )
            )
            gate = NetworkGate(settings, "photos")
            fake_o = FakeOpenAI(gate, [openai_raw(json.dumps(GOOD_ANSWER))])
            result = ask(
                settings, "photos", INSTRUCTIONS, ANSWER_SCHEMA,
                backends={"openai": fake_o},
            )
            self.assertEqual(result, GOOD_ANSWER)
            self.assertEqual(fake_o.requests[0]["model"], "fake-openai-model")

    def test_missing_model_names_key(self):
        cases = {
            "no model anywhere": "[pipeline.ai]\nprovider = \"openai\"\n",
            "step provider differs": (
                "[pipeline.ai]\n"
                'provider = "openai"\n'
                'model = "fake-openai-model"\n'
                "[pipeline.ai.tag]\n"
                'provider = "anthropic"\n'
            ),
            "placeholder at step": (
                "[pipeline.ai]\n"
                'provider = "openai"\n'
                'model = "fake-openai-model"\n'
                "[pipeline.ai.tag]\n"
                'model = "<model-id>"\n'
            ),
            "placeholder at default": (
                "[pipeline.ai]\n" 'model = "<model-id>"\n'
            ),
        }
        for name, ai_extra in cases.items():
            with self.subTest(case=name):
                with tempfile.TemporaryDirectory() as tmp:
                    settings = load_trip(write_project(Path(tmp), ai_extra))
                    gate = NetworkGate(settings, "tag")
                    fake_o = FakeOpenAI(
                        gate, [openai_raw(json.dumps(GOOD_ANSWER))]
                    )
                    fake_a = FakeAnthropic(
                        gate, [anthropic_raw(dict(GOOD_ANSWER))]
                    )
                    backends = {"openai": fake_o, "anthropic": fake_a}
                    with self.assertRaises(AiConfigError) as raised:
                        step_config(settings, "tag")
                    self.assertIn(
                        "[pipeline.ai.tag].model", str(raised.exception)
                    )
                    with self.assertRaises(AiConfigError) as raised_ask:
                        ask(settings, "tag", INSTRUCTIONS, ANSWER_SCHEMA,
                            backends=backends)
                    self.assertIn(
                        "[pipeline.ai.tag].model", str(raised_ask.exception)
                    )
                    self.assertEqual(fake_o.requests, [])
                    self.assertEqual(fake_a.requests, [])

    def test_no_model_ids_in_code(self):
        patterns = [
            r"gpt-[0-9]",
            r"gpt-4o",
            r"chatgpt-",
            r"['\"]o[134]['\"]",
            r"whisper-1",
            r"-transcribe",
            r"claude-(opus|sonnet|haiku|instant|[0-9])",
        ]
        package = REPO_ROOT / "fcp_pipeline"
        files = [
            path
            for path in package.rglob("*")
            if path.suffix in (".py", ".sql")
            and "__pycache__" not in path.parts
        ]
        self.assertTrue(files)
        for path in files:
            text = path.read_text(encoding="utf-8")
            for pattern in patterns:
                with self.subTest(path=str(path), pattern=pattern):
                    self.assertNotRegex(text, pattern)

    def test_same_schema_both_backends(self):
        good_text = json.dumps(GOOD_ANSWER)
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                ai_extra = (
                    "[pipeline.ai]\n"
                    'provider = "openai"\n'
                    'model = "fake-openai-model"\n'
                    "[pipeline.ai.tag]\n"
                    'provider = "anthropic"\n'
                    'model = "fake-anthropic-model"\n'
                )
                with tempfile.TemporaryDirectory() as tmp:
                    trip = write_project(Path(tmp), ai_extra)
                    settings = load_trip(trip)
                    step = "photos" if provider == "openai" else "tag"
                    # The same answer validates to the same dict either way.
                    gate = NetworkGate(settings, step)
                    if provider == "openai":
                        fake = FakeOpenAI(gate, [openai_raw(good_text)])
                    else:
                        fake = FakeAnthropic(gate, [anthropic_raw(GOOD_ANSWER)])
                    result = ask(
                        settings, step, INSTRUCTIONS, ANSWER_SCHEMA,
                        backends={provider: fake},
                    )
                    self.assertEqual(result, GOOD_ANSWER)
                    # A bad answer followed by a good one retries once:
                    # exactly two calls and two usage rows.
                    if provider == "openai":
                        retry = FakeOpenAI(
                            NetworkGate(settings, step),
                            [bad_openai_raw(), openai_raw(good_text)],
                        )
                    else:
                        retry = FakeAnthropic(
                            NetworkGate(settings, step),
                            [bad_anthropic_raw(), anthropic_raw(GOOD_ANSWER)],
                        )
                    result = ask(
                        settings, step, INSTRUCTIONS, ANSWER_SCHEMA,
                        backends={provider: retry},
                    )
                    self.assertEqual(result, GOOD_ANSWER)
                    self.assertEqual(len(retry.requests), 2)
                    usage_lines = (
                        trip / ".pipeline" / "ai-usage.tsv"
                    ).read_text(encoding="utf-8").splitlines()
                    # One row for the first answer plus two for the
                    # retried call, after the header.
                    self.assertEqual(len(usage_lines), 4)
                    self.assertEqual(
                        usage_lines[0],
                        "time\tstep\tprovider\tmodel\t"
                        "input_tokens\toutput_tokens\tcost_usd",
                    )
                    # Two bad answers raise, naming the missing field.
                    if provider == "openai":
                        failing = FakeOpenAI(
                            NetworkGate(settings, step),
                            [bad_openai_raw(), bad_openai_raw()],
                        )
                    else:
                        failing = FakeAnthropic(
                            NetworkGate(settings, step),
                            [bad_anthropic_raw(), bad_anthropic_raw()],
                        )
                    with self.assertRaises(AiAnswerError) as raised:
                        ask(
                            settings, step, INSTRUCTIONS, ANSWER_SCHEMA,
                            backends={provider: failing},
                        )
                    self.assertEqual(len(failing.requests), 2)
                    message = str(raised.exception)
                    self.assertIn(step, message)
                    self.assertIn(provider, message)
                    self.assertIn("reason", message)
                    self.assertNotIn("Harborlight", message)

    def test_image_input_both_backends(self):
        image_bytes = b"\x89PNG-fake-bytes-7162"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        picture = Image(image_bytes, "image/png")
        ai_extra = (
            "[pipeline.ai]\n"
            'provider = "openai"\n'
            'model = "fake-openai-model"\n'
            "[pipeline.ai.tag]\n"
            'provider = "anthropic"\n'
            'model = "fake-anthropic-model"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_project(Path(tmp), ai_extra)
            settings = load_trip(trip)
            gate = NetworkGate(settings, "photos")
            fake = FakeOpenAI(gate, [openai_raw(json.dumps(GOOD_ANSWER))])
            self.assertEqual(
                ask(settings, "photos", INSTRUCTIONS, ANSWER_SCHEMA,
                    images=[picture], backends={"openai": fake}),
                GOOD_ANSWER,
            )
            self.assertEqual(len(fake.requests), 1)
            content = fake.requests[0]["input"][0]["content"]
            self.assertEqual(
                content[0], {"type": "input_text", "text": INSTRUCTIONS}
            )
            self.assertEqual(
                content[1],
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{encoded}",
                },
            )
            gate = NetworkGate(settings, "tag")
            fake_a = FakeAnthropic(gate, [anthropic_raw(dict(GOOD_ANSWER))])
            self.assertEqual(
                ask(settings, "tag", INSTRUCTIONS, ANSWER_SCHEMA,
                    images=[picture], backends={"anthropic": fake_a}),
                GOOD_ANSWER,
            )
            self.assertEqual(len(fake_a.requests), 1)
            blocks = fake_a.requests[0]["messages"][0]["content"]
            self.assertEqual(
                blocks[0],
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": encoded,
                    },
                },
            )
            self.assertEqual(blocks[1], {"type": "text", "text": INSTRUCTIONS})
            log_text = (trip / ".pipeline" / "network-log.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn("image", log_text)
            self.assertIn("1 image", log_text)

    def test_key_from_env_or_keychain(self):
        for provider, env_var, service in (
            ("openai", "OPENAI_API_KEY", "fcp-pipeline-openai"),
            ("anthropic", "ANTHROPIC_API_KEY", "fcp-pipeline-anthropic"),
        ):
            with self.subTest(provider=provider):
                ai_extra = (
                    "[pipeline.ai]\n"
                    f'provider = "{provider}"\n'
                    'model = "fake-live-model"\n'
                )
                raw = (
                    openai_raw(json.dumps(GOOD_ANSWER))
                    if provider == "openai"
                    else anthropic_raw(dict(GOOD_ANSWER))
                )
                recorded = {
                    "openai_keys": [],
                    "anthropic_keys": [],
                    "openai_requests": [],
                    "anthropic_requests": [],
                    "openai_raw": raw,
                    "anthropic_raw": raw,
                }
                saved = install_fake_sdks(recorded)
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        trip = write_project(Path(tmp), ai_extra)
                        settings = load_trip(trip)
                        key = (
                            FAKE_OPENAI_KEY
                            if provider == "openai"
                            else FAKE_ANTHROPIC_KEY
                        )
                        seen_keys = (
                            recorded["openai_keys"]
                            if provider == "openai"
                            else recorded["anthropic_keys"]
                        )
                        # The environment comes first.
                        with mock.patch.dict(
                            os.environ, {env_var: key}, clear=False
                        ):
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                result = ask(
                                    settings, "photos", INSTRUCTIONS,
                                    ANSWER_SCHEMA,
                                )
                            self.assertEqual(result, GOOD_ANSWER)
                            self.assertEqual(seen_keys, [key])
                        # Otherwise the patched Keychain reader.
                        with mock.patch.dict(os.environ, clear=False):
                            os.environ.pop("OPENAI_API_KEY", None)
                            os.environ.pop("ANTHROPIC_API_KEY", None)
                            with mock.patch(
                                "fcp_pipeline.net._keychain_password",
                                return_value=FAKE_KEYCHAIN_KEY,
                            ):
                                with warnings.catch_warnings(
                                    record=True
                                ) as caught_keychain:
                                    warnings.simplefilter("always")
                                    result = ask(
                                        settings, "photos", INSTRUCTIONS,
                                        ANSWER_SCHEMA,
                                    )
                                self.assertEqual(result, GOOD_ANSWER)
                                self.assertEqual(
                                    seen_keys, [key, FAKE_KEYCHAIN_KEY]
                                )
                        # With neither, the error names both places.
                        with mock.patch.dict(os.environ, clear=False):
                            os.environ.pop("OPENAI_API_KEY", None)
                            os.environ.pop("ANTHROPIC_API_KEY", None)
                            with mock.patch(
                                "fcp_pipeline.net._keychain_password",
                                return_value=None,
                            ):
                                with self.assertRaises(
                                    RuntimeError
                                ) as raised:
                                    ask(
                                        settings, "photos", INSTRUCTIONS,
                                        ANSWER_SCHEMA,
                                    )
                        message = str(raised.exception)
                        self.assertIn(env_var, message)
                        self.assertIn(service, message)
                        self.assertNotIn(key, message)
                        self.assertNotIn(FAKE_KEYCHAIN_KEY, message)
                        for record in list(caught) + list(caught_keychain):
                            self.assertNotIn(key, str(record.message))
                        for path in trip.rglob("*"):
                            if path.is_file():
                                text = path.read_text(encoding="utf-8")
                                self.assertNotIn(key, text)
                                self.assertNotIn(FAKE_KEYCHAIN_KEY, text)
                        usage_text = (
                            trip / ".pipeline" / "ai-usage.tsv"
                        ).read_text(encoding="utf-8")
                        self.assertNotIn(key, usage_text)
                        self.assertNotIn(FAKE_KEYCHAIN_KEY, usage_text)
                finally:
                    for name, module in saved.items():
                        if module is None:
                            sys.modules.pop(name, None)
                        else:
                            sys.modules[name] = module

    def test_usage_row_per_call(self):
        ai_extra = (
            "[pipeline.ai]\n"
            'provider = "openai"\n'
            'model = "fake-openai-model"\n'
            '[pipeline.ai.prices."fake-openai-model"]\n'
            "input_per_million = 2.5\n"
            "output_per_million = 10\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_project(Path(tmp), ai_extra)
            settings = load_trip(trip)
            gate = NetworkGate(settings, "tag")
            fake = FakeOpenAI(
                gate,
                [
                    openai_raw(json.dumps(GOOD_ANSWER), 1000, 500),
                    openai_raw(json.dumps(GOOD_ANSWER), 1000, 500),
                ],
            )
            backends = {"openai": fake}
            ask(settings, "tag", INSTRUCTIONS, ANSWER_SCHEMA, backends=backends)
            ask(settings, "tag", INSTRUCTIONS, ANSWER_SCHEMA, backends=backends)
            lines = (
                trip / ".pipeline" / "ai-usage.tsv"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(
                lines[0],
                "time\tstep\tprovider\tmodel\t"
                "input_tokens\toutput_tokens\tcost_usd",
            )
            for line in lines[1:]:
                fields = line.split("\t")
                self.assertEqual(len(fields), 7)
                self.assertRegex(
                    fields[0], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
                )
                self.assertEqual(fields[1:6], [
                    "tag", "openai", "fake-openai-model", "1000", "500",
                ])
                # 1,000 tokens at 2.5 plus 500 at 10 dollars per
                # million is 0.007500.
                self.assertEqual(fields[6], "0.007500")
        # A model with no price entry leaves the cost cell empty.
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_project(
                Path(tmp),
                "[pipeline.ai]\nmodel = \"fake-unpriced-model\"\n",
            )
            settings = load_trip(trip)
            gate = NetworkGate(settings, "tag")
            fake = FakeOpenAI(
                gate, [openai_raw(json.dumps(GOOD_ANSWER), 1000, 500)]
            )
            ask(settings, "tag", INSTRUCTIONS, ANSWER_SCHEMA,
                backends={"openai": fake})
            lines = (
                trip / ".pipeline" / "ai-usage.tsv"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[1].split("\t")[6], "")

    def test_offline_refuses_before_backend(self):
        ai_extra = (
            "[pipeline.ai]\n"
            'provider = "openai"\n'
            'model = "fake-openai-model"\n'
        )

        class RecordingFake(FakeOpenAI):
            def __init__(self, gate):
                super().__init__(gate, [])
                self.builds: list = []

            def build_request(self, *args, **kwargs):
                self.builds.append((args, kwargs))
                return super().build_request(*args, **kwargs)

        recorded = {
            "openai_keys": [],
            "anthropic_keys": [],
            "openai_requests": [],
            "anthropic_requests": [],
            "openai_raw": openai_raw(json.dumps(GOOD_ANSWER)),
            "anthropic_raw": anthropic_raw(dict(GOOD_ANSWER)),
        }
        saved = install_fake_sdks(recorded)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                trip = write_project(Path(tmp), ai_extra, offline=True)
                settings = load_trip(trip)
                fake = RecordingFake(NetworkGate(settings, "tag"))
                with self.assertRaises(OfflineError):
                    ask(settings, "tag", INSTRUCTIONS, ANSWER_SCHEMA,
                        backends={"openai": fake})
                # No backend method ran, no client was built, no key read.
                self.assertEqual(fake.builds, [])
                self.assertEqual(fake.requests, [])
                self.assertEqual(recorded["openai_keys"], [])
                self.assertEqual(recorded["openai_requests"], [])
                self.assertFalse((trip / ".pipeline" / "ai-usage.tsv").exists())
                log_lines = (
                    trip / ".pipeline" / "network-log.tsv"
                ).read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(log_lines), 2)
                self.assertIn("refused offline", log_lines[1])
            # The process-wide switch refuses the same way.
            with tempfile.TemporaryDirectory() as tmp:
                with mock.patch.dict(os.environ, clear=False):
                    os.environ.pop("OPENAI_API_KEY", None)
                    os.environ.pop("ANTHROPIC_API_KEY", None)
                    trip = write_project(Path(tmp), ai_extra)
                    settings = load_trip(trip)
                    set_offline(True)
                    try:
                        with self.assertRaises(OfflineError):
                            ask(settings, "tag", INSTRUCTIONS, ANSWER_SCHEMA)
                    finally:
                        set_offline(False)
                self.assertFalse((trip / ".pipeline" / "ai-usage.tsv").exists())
                log_lines = (
                    trip / ".pipeline" / "network-log.tsv"
                ).read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(log_lines), 2)
                self.assertIn("refused offline", log_lines[1])
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_step_config_edges(self):
        ai_extra = (
            "[pipeline.ai]\n"
            'provider = "openai"\n'
            'model = "fake-openai-model"\n'
            "[pipeline.ai.memo_audio]\n"
            'provider = "local"\n'
            "[pipeline.ai.vo_audio]\n"
            'provider = "openai"\n'
            'model = "fake-audio-model"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_project(Path(tmp), ai_extra))
            with self.assertRaises(AiConfigError):
                step_config(settings, "bogus")
            with self.assertRaises(AiConfigError):
                ask(settings, "bogus", INSTRUCTIONS, ANSWER_SCHEMA)
            local = step_config(settings, "memo_audio")
            self.assertEqual(local.provider, "local")
            self.assertIsNone(local.model)
            voiced = step_config(settings, "vo_audio")
            self.assertEqual(
                (voiced.provider, voiced.model),
                ("openai", "fake-audio-model"),
            )
            # Audio steps resolve but are refused by ask: transcription
            # is a later step.
            gate = NetworkGate(settings, "vo_audio")
            fake = FakeOpenAI(gate, [openai_raw(json.dumps(GOOD_ANSWER))])
            with self.assertRaises(AiConfigError):
                ask(settings, "vo_audio", INSTRUCTIONS, ANSWER_SCHEMA,
                    backends={"openai": fake})
            with self.assertRaises(AiConfigError):
                ask(settings, "memo_audio", INSTRUCTIONS, ANSWER_SCHEMA,
                    backends={"openai": fake})
            self.assertEqual(fake.requests, [])
            with self.assertRaises(ValueError):
                Image(b"fake-bytes", "image/gif")
            with self.assertRaises(ValueError):
                Image(b"x" * (MAX_IMAGE_BYTES + 1), "image/png")

    def test_ai_settings_validate(self):
        from fcp_pipeline.trip import TripSettingsError

        with tempfile.TemporaryDirectory() as tmp:
            # A bad provider names its key.
            with self.assertRaises(TripSettingsError) as raised:
                load_trip(
                    write_project(
                        Path(tmp),
                        "[pipeline.ai.tag]\nprovider = \"local\"\n",
                    )
                )
            self.assertIn("[pipeline.ai.tag].provider", str(raised.exception))
            # An unknown key and an unknown step table each warn once.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                settings = load_trip(
                    write_project(
                        Path(tmp),
                        "[pipeline.ai]\n"
                        'provider = "openai"\n'
                        'model = "fake-openai-model"\n'
                        'spare_key = "extra"\n'
                        "[pipeline.ai.bogus]\n"
                        'model = "fake-bogus-model"\n',
                    )
                )
            self.assertEqual(settings.ai["model"], "fake-openai-model")
            self.assertEqual(len(caught), 2)
            self.assertTrue(
                all(issubclass(record.category, UserWarning) for record in caught)
            )
            warned = sorted(str(record.message) for record in caught)
            self.assertEqual(
                warned,
                [
                    "unknown project.toml key: [pipeline.ai].bogus",
                    "unknown project.toml key: [pipeline.ai].spare_key",
                ],
            )
            for message in warned:
                self.assertNotIn("provider", message)
                self.assertNotIn("model", message)
            # A top-level ai table belongs under [pipeline].
            with self.assertRaises(TripSettingsError) as raised_top:
                load_trip(
                    write_project(Path(tmp), "[ai]\nmodel = \"x\"\n")
                )
            self.assertIn("belongs under [pipeline]", str(raised_top.exception))

    def test_ai_prices_reject_bad_rates(self):
        from fcp_pipeline.trip import TripSettingsError

        cases = {
            "nan input": ("input_per_million", "nan"),
            "inf input": ("input_per_million", "inf"),
            "minus inf input": ("input_per_million", "-inf"),
            "nan output": ("output_per_million", "nan"),
            "inf output": ("output_per_million", "inf"),
            "minus inf output": ("output_per_million", "-inf"),
            "negative input": ("input_per_million", "-2.5"),
            "string output": ("output_per_million", '"lots"'),
            "boolean input": ("input_per_million", "true"),
        }
        for name, (rate_key, toml_value) in cases.items():
            with self.subTest(case=name):
                other = (
                    "output_per_million"
                    if rate_key == "input_per_million"
                    else "input_per_million"
                )
                with tempfile.TemporaryDirectory() as tmp:
                    with self.assertRaises(TripSettingsError) as raised:
                        load_trip(
                            write_project(
                                Path(tmp),
                                "[pipeline.ai]\n"
                                'provider = "openai"\n'
                                'model = "fake-openai-model"\n'
                                '[pipeline.ai.prices."fake-openai-model"]\n'
                                f"{rate_key} = {toml_value}\n"
                                f"{other} = 10\n",
                            )
                        )
                self.assertIn(rate_key, str(raised.exception))

    def test_usage_cost_empty_without_finite_price(self):
        from fcp_pipeline.ai.usage import cost_for_model

        for rates in (
            {"input_per_million": float("nan"), "output_per_million": 10},
            {"input_per_million": float("inf"), "output_per_million": 10},
            {"input_per_million": 10, "output_per_million": float("-inf")},
            {"input_per_million": True, "output_per_million": 10},
            {"input_per_million": "lots", "output_per_million": 10},
        ):
            with self.subTest(rates=rates):
                self.assertEqual(
                    cost_for_model({"prices": {"m": rates}}, "m", 1000, 500),
                    "",
                )

    def test_anthropic_text_parse_edges(self):
        ai_extra = (
            "[pipeline.ai]\n"
            'provider = "anthropic"\n'
            'model = "fake-anthropic-model"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_trip(write_project(Path(tmp), ai_extra))
            backend = AnthropicBackend(NetworkGate(settings, "tag"))
            for content in (
                [],
                [{"type": "text", "text": "   "}],
                [{"type": "text", "text": "not json {"}],
                [{"type": "tool_use", "name": "answer", "input": {}}],
            ):
                with self.subTest(content=content):
                    reply = backend.parse_response(
                        {
                            "content": content,
                            "usage": {"input_tokens": 7, "output_tokens": 3},
                        }
                    )
                    self.assertIsNone(reply.data)
                    self.assertEqual(
                        (reply.input_tokens, reply.output_tokens), (7, 3)
                    )

    def test_refused_usage_write_keeps_answer(self):
        ai_extra = (
            "[pipeline.ai]\n"
            'provider = "openai"\n'
            'model = "fake-openai-model"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_project(Path(tmp), ai_extra)
            settings = load_trip(trip)
            gate = NetworkGate(settings, "tag")
            fake = FakeOpenAI(gate, [openai_raw(json.dumps(GOOD_ANSWER))])
            capture = Path(tmp) / "capture.tsv"
            (trip / ".pipeline").mkdir()
            (trip / ".pipeline" / "ai-usage.tsv").symlink_to(capture)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = ask(settings, "tag", INSTRUCTIONS, ANSWER_SCHEMA,
                             backends={"openai": fake})
            self.assertEqual(result, GOOD_ANSWER)
            self.assertEqual(len(caught), 1)
            self.assertIn("AI usage log", str(caught[0].message))
            self.assertFalse(capture.exists())

    def test_schema_subset_enforced(self):
        nested = dict(ANSWER_SCHEMA["properties"]["title"])
        cases = {
            "pattern": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "pattern": "^H"},
                    "reason": dict(nested),
                },
                "required": ["title", "reason"],
                "additionalProperties": False,
            },
            "oneOf": {
                "type": "object",
                "properties": {
                    "title": dict(nested),
                    "reason": {"oneOf": [{"type": "string"}]},
                },
                "required": ["title", "reason"],
                "additionalProperties": False,
            },
            "missing additionalProperties": {
                "type": "object",
                "properties": {
                    "title": dict(nested),
                    "reason": dict(nested),
                },
                "required": ["title", "reason"],
            },
            "additionalProperties true": {
                "type": "object",
                "properties": {
                    "title": dict(nested),
                    "reason": dict(nested),
                },
                "required": ["title", "reason"],
                "additionalProperties": True,
            },
            "non-object root": {
                "type": "array",
                "items": {"type": "string"},
            },
        }
        ai_extra = (
            "[pipeline.ai]\n"
            'provider = "openai"\n'
            'model = "fake-openai-model"\n'
        )
        for name, schema in cases.items():
            with self.subTest(schema=name):
                with tempfile.TemporaryDirectory() as tmp:
                    settings = load_trip(write_project(Path(tmp), ai_extra))
                    gate = NetworkGate(settings, "tag")
                    fake = FakeOpenAI(
                        gate, [openai_raw(json.dumps(GOOD_ANSWER))]
                    )
                    with self.assertRaises(AiSchemaError):
                        ask(settings, "tag", INSTRUCTIONS, schema,
                            backends={"openai": fake})
                    self.assertEqual(fake.requests, [])


if __name__ == "__main__":
    unittest.main()

