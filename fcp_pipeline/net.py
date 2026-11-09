"""One network gate for the Final Cut footage pipeline.

Every request the pipeline package makes goes through :class:`NetworkGate`:
it builds the OpenAI and Anthropic clients and runs each call. An offline
switch (``offline = true`` in the ``[pipeline]`` table of ``project.toml``,
or ``--offline`` on any command, which flips the process-wide switch with
:func:`set_offline`) refuses each call before any connection. Every call
appends one content-free line to ``<project>/.pipeline/network-log.tsv``.
API keys come from the environment or the macOS Keychain and are never
written to disk, logs or errors.

Only this module may import the ``openai`` or ``anthropic`` SDKs, and only
inside the client-builder methods, so the package imports and every test
runs without either SDK installed.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path

#: Environment variables read first for API keys.
OPENAI_ENV_VAR = "OPENAI_API_KEY"
ANTHROPIC_ENV_VAR = "ANTHROPIC_API_KEY"
#: macOS Keychain generic-password services used when the environment has no key.
OPENAI_KEYCHAIN_SERVICE = "fcp-pipeline-openai"
ANTHROPIC_KEYCHAIN_SERVICE = "fcp-pipeline-anthropic"
#: Keychain reader; only ever executed on macOS.
SECURITY_BIN = "/usr/bin/security"
#: Log file name inside ``<project>/.pipeline/``.
LOG_NAME = "network-log.tsv"
#: Header written once when the log file is created.
LOG_HEADER = "time\tstep\tservice\tmodel\tsent\toutcome\n"
#: Services the gate accepts.
VALID_SERVICES = ("openai", "anthropic")

_PROCESS_OFFLINE = False


class OfflineError(RuntimeError):
    """Raised when a network call is refused because the gate is offline."""


def set_offline(value: bool) -> None:
    """Turn the process-wide offline switch on (or back off)."""
    global _PROCESS_OFFLINE
    _PROCESS_OFFLINE = bool(value)


def is_process_offline() -> bool:
    """Return whether the process-wide offline switch is on."""
    return _PROCESS_OFFLINE


def _sanitize_field(text: str) -> str:
    """Strip tabs and newlines so one call always logs exactly one line."""
    return str(text).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _format_sent(sent) -> str:
    """Summarize ``sent`` as kinds and sizes, never content.

    ``sent`` is a sequence of ``(kind, size)`` pairs such as
    ``[("image", 421888), ("text", 1840)]``. The summary names only the
    kinds, counts and sizes, for example
    ``"1 image 412 KB, 1,840 characters"``.
    """
    try:
        items = list(sent)
    except TypeError:
        items = []
    if not items:
        return "nothing"
    counts: dict[str, list[int]] = {}
    for pair in items:
        kind, size = pair
        kind = str(kind)
        size = int(size)
        if size < 0:
            raise ValueError(f"bad sent size: {size!r}")
        entry = counts.get(kind)
        if entry is None:
            counts[kind] = [1, size]
        else:
            entry[0] += 1
            entry[1] += size
    parts = []
    for kind, (count, total) in counts.items():
        if kind == "image":
            noun = "image" if count == 1 else "images"
            parts.append(f"{count} {noun} {total // 1024} KB")
        elif kind == "text":
            noun = "character" if total == 1 else "characters"
            parts.append(f"{total:,} {noun}")
        else:
            parts.append(f"{count} {kind} {total:,} bytes")
    return ", ".join(parts)


def _keychain_password(service: str) -> str | None:
    """Read a generic password from the macOS Keychain, or None.

    Only runs on macOS; everywhere else there is no Keychain to read.
    Returns None when the item is missing instead of raising.
    """
    if sys.platform != "darwin":
        return None
    try:
        proc = subprocess.run(
            [SECURITY_BIN, "find-generic-password", "-s", service, "-w"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    value = (proc.stdout or "").strip()
    return value or None


def _read_key(service: str) -> str:
    """Return the API key for ``service`` without ever logging it.

    The environment comes first; otherwise, on macOS only, the Keychain
    generic password is read. A missing key raises an error naming the
    variable and the Keychain service, never a key value.
    """
    if service == "openai":
        env_var = OPENAI_ENV_VAR
        keychain_service = OPENAI_KEYCHAIN_SERVICE
    elif service == "anthropic":
        env_var = ANTHROPIC_ENV_VAR
        keychain_service = ANTHROPIC_KEYCHAIN_SERVICE
    else:
        raise ValueError(f"unknown service: {service!r}")
    value = os.environ.get(env_var)
    if value:
        return value
    value = _keychain_password(keychain_service)
    if value:
        return value
    raise RuntimeError(
        f"missing API key: set {env_var} or store it "
        f"in the Keychain service {keychain_service!r}"
    )


class NetworkGate:
    """The only path from the pipeline package to the network.

    Built from the :class:`TripSettings` that
    :func:`fcp_pipeline.trip.load_trip` returns and a step name, for
    example ``NetworkGate(settings, step)``. The gate is offline when
    ``settings.offline`` is true or when the process-wide switch (see
    :func:`set_offline`) is on.
    """

    def __init__(self, settings, step: str) -> None:
        self._settings = settings
        self._step = str(step)

    @property
    def step(self) -> str:
        """The pipeline step name this gate logs under."""
        return self._step

    @property
    def settings(self):
        """The trip settings this gate was built from."""
        return self._settings

    @property
    def offline(self) -> bool:
        """Whether this gate currently refuses every call."""
        try:
            configured = bool(getattr(self._settings, "offline", False))
        except Exception:
            configured = False
        return configured or _PROCESS_OFFLINE

    @property
    def is_offline(self) -> bool:
        """Alias for :attr:`offline`."""
        return self.offline

    def __repr__(self) -> str:
        # Never include a key: the gate does not store one.
        return f"NetworkGate(step={self._step!r}, offline={self.offline!r})"

    def _offline_message(self, service: str) -> str:
        return (
            f"network call refused: step {self._step!r} is offline "
            f"for service {service!r}; remove 'offline = true' from "
            "[pipeline] in project.toml and run without --offline "
            "to switch it back on"
        )

    def _log_targets(self) -> tuple[Path, Path]:
        """Return the (pipeline dir, log path) after a containment check.

        Raises before creating anything when ``.pipeline/`` or the log
        would resolve outside the project folder.
        """
        folder = Path(getattr(self._settings, "folder"))
        folder_resolved = folder.resolve()
        pipeline_dir = folder / ".pipeline"
        log_path = pipeline_dir / LOG_NAME
        pipeline_resolved = pipeline_dir.resolve()
        log_resolved = log_path.resolve()
        if not pipeline_resolved.is_relative_to(folder_resolved):
            raise RuntimeError(
                "refusing to write the network log outside the project folder"
            )
        if not log_resolved.is_relative_to(folder_resolved):
            raise RuntimeError(
                "refusing to write the network log outside the project folder"
            )
        return pipeline_dir, log_path

    def _append_log(self, service: str, model: str, summary: str, outcome: str) -> None:
        """Append one content-free line to ``.pipeline/network-log.tsv``."""
        pipeline_dir, log_path = self._log_targets()
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        # Re-check after creating: a swapped symlink must still refuse.
        self._log_targets()
        now = (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        line = (
            f"{now}\t{_sanitize_field(self._step)}"
            f"\t{_sanitize_field(service)}\t{_sanitize_field(model)}"
            f"\t{_sanitize_field(summary)}\t{_sanitize_field(outcome)}\n"
        )
        creating = not log_path.exists()
        with open(log_path, "a", encoding="utf-8") as handle:
            if creating:
                handle.write(LOG_HEADER)
            handle.write(line)

    def call(self, service: str, model: str, sent, request):
        """Run ``request()`` and return its result, logging one line.

        ``service`` is ``"openai"`` or ``"anthropic"``; ``sent`` describes
        what goes out as kinds and sizes, never the content; ``request``
        is a zero-argument callable doing the actual SDK call. When the
        gate is offline, raise :class:`OfflineError` before calling
        ``request``. Allowed, failed and refused calls all log exactly
        one line with outcome ``sent``, ``failed`` or ``refused offline``.
        """
        if service not in VALID_SERVICES:
            raise ValueError(f"unknown service: {service!r}")
        summary = _format_sent(sent)
        if self.offline:
            self._append_log(service, model, summary, "refused offline")
            raise OfflineError(self._offline_message(service))
        try:
            result = request()
        except Exception:
            self._append_log(service, model, summary, "failed")
            raise
        self._append_log(service, model, summary, "sent")
        return result

    def openai_client(self):
        """Build an ``openai.OpenAI`` client with the API key passed in.

        The SDK is imported here, so the package imports without it; a
        missing SDK raises an error naming the pip package. When the gate
        is offline, raise :class:`OfflineError` before importing the SDK
        or reading a key.
        """
        if self.offline:
            raise OfflineError(self._offline_message("openai"))
        key = _read_key("openai")
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "the 'openai' pip package is not installed; "
                "install it with 'pip install openai'"
            ) from exc
        return openai.OpenAI(api_key=key)

    def anthropic_client(self):
        """Build an ``anthropic.Anthropic`` client with the API key passed in.

        The SDK is imported here, so the package imports without it; a
        missing SDK raises an error naming the pip package. When the gate
        is offline, raise :class:`OfflineError` before importing the SDK
        or reading a key.
        """
        if self.offline:
            raise OfflineError(self._offline_message("anthropic"))
        key = _read_key("anthropic")
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "the 'anthropic' pip package is not installed; "
                "install it with 'pip install anthropic'"
            ) from exc
        return anthropic.Anthropic(api_key=key)
