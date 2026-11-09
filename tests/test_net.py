"""Tests for fcp_pipeline.net, the only module that may reach the network.

No test here makes a real network call or a real Keychain lookup: socket
functions and subprocess are patched in setUp, and fake ``openai`` and
``anthropic`` modules are put in ``sys.modules`` wherever a client is
built. Fake keys are obviously fake. The process-wide offline switch is
reset after each test.
"""

import ast
import os
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from fcp_pipeline import cli, net
from fcp_pipeline.net import NetworkGate, OfflineError, set_offline
from fcp_pipeline.trip import TripSettingsError, load_trip

REPO_ROOT = Path(__file__).resolve().parents[1]

FAKE_OPENAI_KEY = "fake-test-openai-key-ALPHA-7741"
FAKE_ANTHROPIC_KEY = "fake-test-anthropic-key-BETA-7741"
FAKE_KEYCHAIN_KEY = "fake-test-keychain-key-GAMMA-7741"

SOCKET_TARGETS = (
    "socket.socket.connect",
    "socket.socket.connect_ex",
    "socket.create_connection",
    "socket.getaddrinfo",
)


def write_trip(parent: Path, offline: bool | None = None) -> Path:
    """Write a minimal trip folder; ``offline`` adds the [pipeline] key."""
    trip = parent / "trip"
    trip.mkdir(parents=True, exist_ok=True)
    offline_line = ""
    if offline is not None:
        offline_line = f"offline = {'true' if offline else 'false'}\n"
    (trip / "project.toml").write_text(
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
        f"{offline_line}"
        "[pipeline.clock_offsets]\n"
        'harbor-cam-a = "-00:03:10"\n',
        encoding="utf-8",
    )
    return trip


def install_fake_sdks():
    """Put fake openai/anthropic modules in sys.modules; return saved state."""
    saved = {name: sys.modules.get(name) for name in ("openai", "anthropic")}
    openai_calls: dict = {}
    anthropic_calls: dict = {}

    openai_module = types.ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, api_key=None):
            openai_calls["api_key"] = api_key

    openai_module.OpenAI = FakeOpenAI

    anthropic_module = types.ModuleType("anthropic")

    class FakeAnthropic:
        def __init__(self, api_key=None):
            anthropic_calls["api_key"] = api_key

    anthropic_module.Anthropic = FakeAnthropic

    sys.modules["openai"] = openai_module
    sys.modules["anthropic"] = anthropic_module
    return saved, openai_calls, anthropic_calls


def remove_fake_sdks(saved) -> None:
    """Restore sys.modules to its state before install_fake_sdks."""
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def read_log_text(trip: Path) -> str:
    return (trip / ".pipeline" / "network-log.tsv").read_text(encoding="utf-8")


class TestNet(unittest.TestCase):
    def setUp(self):
        set_offline(False)
        self._patchers = []
        self.socket_mocks = {}
        for target in SOCKET_TARGETS:
            patcher = mock.patch(
                target, side_effect=AssertionError(f"real network use: {target}")
            )
            self.socket_mocks[target] = patcher.start()
            self._patchers.append(patcher)
        for target in ("subprocess.run", "subprocess.Popen"):
            patcher = mock.patch(
                target, side_effect=AssertionError(f"real subprocess use: {target}")
            )
            patcher.start()
            self._patchers.append(patcher)
        self._saved_sdks = None

    def tearDown(self):
        if self._saved_sdks is not None:
            remove_fake_sdks(self._saved_sdks)
            self._saved_sdks = None
        for patcher in reversed(self._patchers):
            patcher.stop()
        self._patchers = []
        set_offline(False)

    def assertNoSocketUse(self):
        for target, double in self.socket_mocks.items():
            with self.subTest(target=target):
                double.assert_not_called()

    def test_offline_refuses_before_connect(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Case 1: offline = true in project.toml.
            trip = write_trip(Path(tmp) / "toml-off", offline=True)
            settings = load_trip(trip)
            self.assertTrue(settings.offline)
            gate = NetworkGate(settings, "scan")
            self.assertTrue(gate.offline)
            called = []

            def request():
                called.append(True)
                return "result"

            with self.assertRaises(OfflineError) as raised:
                gate.call("openai", "test-model", [("text", 10)], request)
            self.assertIn("scan", str(raised.exception))
            self.assertIn("openai", str(raised.exception))
            self.assertIn("--offline", str(raised.exception))
            with self.assertRaises(OfflineError):
                gate.openai_client()
            with self.assertRaises(OfflineError):
                gate.anthropic_client()
            self.assertEqual(called, [])
            self.assertNoSocketUse()

            # Case 2: the process-wide switch, with project.toml offline absent.
            set_offline(False)
            trip2 = write_trip(Path(tmp) / "flag-off")
            gate2 = NetworkGate(load_trip(trip2), "scan")
            self.assertFalse(gate2.offline)
            set_offline(True)
            self.assertTrue(gate2.offline)
            called2 = []

            def request2():
                called2.append(True)
                return "result"

            with self.assertRaises(OfflineError) as raised2:
                gate2.call("anthropic", "test-model", [("text", 10)], request2)
            self.assertIn("scan", str(raised2.exception))
            self.assertIn("anthropic", str(raised2.exception))
            with self.assertRaises(OfflineError):
                gate2.openai_client()
            with self.assertRaises(OfflineError):
                gate2.anthropic_client()
            self.assertEqual(called2, [])
            self.assertNoSocketUse()

    def test_log_line_has_no_content(self):
        marker = "UNIQUE-MARKER-ZEBRA-HARBOR-9f8e7d6c5b4a"
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_trip(Path(tmp))
            gate = NetworkGate(load_trip(trip), "scan")
            result = gate.call(
                "openai",
                "test-model",
                [("image", 421888), ("text", len(marker))],
                lambda: marker,
            )
            self.assertEqual(result, marker)
            text = read_log_text(trip)
            lines = text.splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0], "time\tstep\tservice\tmodel\tsent\toutcome")
            self.assertIn("412 KB", lines[1])
            self.assertIn("characters", lines[1])
            self.assertIn("sent", lines[1].rsplit("\t", 1)[-1])
            self.assertNotIn(marker, text)
            self.assertNoSocketUse()

    def test_key_never_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_trip(Path(tmp))
            self._saved_sdks, openai_calls, _ = install_fake_sdks()
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": FAKE_OPENAI_KEY}):
                gate = NetworkGate(load_trip(trip), "scan")
                self.assertEqual(
                    gate.call("openai", "test-model", [("text", 5)], lambda: "ok"),
                    "ok",
                )
                with self.assertRaises(RuntimeError) as raised:
                    gate.call(
                        "openai",
                        "test-model",
                        [("text", 5)],
                        _failing_request,
                    )
                client = gate.openai_client()
                self.assertIsNotNone(client)
                self.assertEqual(openai_calls.get("api_key"), FAKE_OPENAI_KEY)
                log_text = read_log_text(trip)
                self.assertNotIn(FAKE_OPENAI_KEY, log_text)
                self.assertNotIn(FAKE_OPENAI_KEY, str(raised.exception))
                self.assertNotIn(FAKE_OPENAI_KEY, repr(gate))
                self.assertNotIn(FAKE_OPENAI_KEY, repr(client))
            self.assertNoSocketUse()

    def test_only_net_imports_sdks(self):
        importers = set()
        for path in REPO_ROOT.rglob("*.py"):
            if "tests" in path.relative_to(REPO_ROOT).parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        if root in ("openai", "anthropic"):
                            importers.add(path)
                elif isinstance(node, ast.ImportFrom):
                    root = (node.module or "").split(".")[0]
                    if root in ("openai", "anthropic"):
                        importers.add(path)
        self.assertEqual(importers, {REPO_ROOT / "fcp_pipeline" / "net.py"})

    def test_keychain_used_when_env_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_trip(Path(tmp))
            self._saved_sdks, openai_calls, _ = install_fake_sdks()
            with mock.patch.dict(os.environ):
                os.environ.pop("OPENAI_API_KEY", None)
                os.environ.pop("ANTHROPIC_API_KEY", None)
                with mock.patch.object(sys, "platform", "darwin"):
                    seen_argv = []

                    def fake_run(argv, **kwargs):
                        seen_argv.append(list(argv))
                        return subprocess.CompletedProcess(
                            argv, 0, stdout=FAKE_KEYCHAIN_KEY + "\n", stderr=""
                        )

                    with mock.patch("subprocess.run", side_effect=fake_run):
                        gate = NetworkGate(load_trip(trip), "scan")
                        client = gate.openai_client()
                        self.assertIsNotNone(client)
                        self.assertEqual(openai_calls.get("api_key"), FAKE_KEYCHAIN_KEY)
                        self.assertEqual(
                            seen_argv,
                            [
                                [
                                    "/usr/bin/security",
                                    "find-generic-password",
                                    "-s",
                                    "fcp-pipeline-openai",
                                    "-w",
                                ]
                            ],
                        )
                        gate.call("openai", "test-model", [("text", 7)], lambda: "ok")
                        self.assertNotIn(FAKE_KEYCHAIN_KEY, read_log_text(trip))
                        self.assertNotIn(FAKE_KEYCHAIN_KEY, repr(gate))
            self.assertNoSocketUse()

    def test_offline_flag_turns_gate_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_trip(Path(tmp))
            seen = {}

            def fake_scan(trip_arg, dry_run):
                gate = NetworkGate(load_trip(trip_arg), "scan")
                seen["offline"] = gate.offline
                return 0

            original = cli.STEPS
            try:
                cli.STEPS = [
                    cli.Step(
                        step.name,
                        step.main_path,
                        (fake_scan if step.name == "scan" else None),
                    )
                    for step in original
                ]
                code = cli.main(["scan", "--trip", str(trip), "--offline"])
            finally:
                cli.STEPS = original
            self.assertEqual(code, 0)
            self.assertTrue(seen.get("offline"))
            self.assertNoSocketUse()

    def test_missing_key_names_variable_and_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            trip = write_trip(Path(tmp))
            with mock.patch.dict(os.environ):
                os.environ.pop("OPENAI_API_KEY", None)
                os.environ.pop("ANTHROPIC_API_KEY", None)
                with mock.patch.object(sys, "platform", "linux"):
                    gate = NetworkGate(load_trip(trip), "scan")
                    with self.assertRaises(RuntimeError) as raised:
                        gate.openai_client()
                self.assertIn("OPENAI_API_KEY", str(raised.exception))
                self.assertIn("fcp-pipeline-openai", str(raised.exception))
            self.assertNoSocketUse()


def _failing_request():
    raise RuntimeError("synthetic boom")


if __name__ == "__main__":
    unittest.main()
