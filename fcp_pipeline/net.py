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
import errno
import os
import stat
import subprocess
import sys

#: Environment variables read first for API keys.
OPENAI_ENV_VAR = "OPENAI_API_KEY"
ANTHROPIC_ENV_VAR = "ANTHROPIC_API_KEY"
#: macOS Keychain generic-password services used when the environment has no key.
OPENAI_KEYCHAIN_SERVICE = "fcp-pipeline-openai"
ANTHROPIC_KEYCHAIN_SERVICE = "fcp-pipeline-anthropic"
#: Keychain reader; only ever executed on macOS.
SECURITY_BIN = "/usr/bin/security"
#: Directory holding the log file inside ``<project>/``.
PIPELINE_DIR_NAME = ".pipeline"
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

    def _append_log(self, service: str, model: str, summary: str, outcome: str) -> None:
        """Append one content-free line to ``.pipeline/network-log.tsv``.

        The log is opened relative to directory file descriptors with
        ``O_NOFOLLOW`` at every component under the project folder, so a
        ``.pipeline`` directory or log file swapped for a symlink is
        refused instead of followed. The log file itself is opened with
        ``O_NONBLOCK`` (plus ``O_CLOEXEC``) so an existing FIFO never
        blocks the open, and the opened descriptor is then checked with
        ``fstat``: anything that is not a regular file is closed and
        refused, so the call never writes into a FIFO, socket, device
        or directory. ``O_NONBLOCK`` is cleared after that type check,
        before any write. After the log file is opened, the
        opened ``.pipeline`` directory is re-verified with ``fstat`` and
        ``fstatat`` (``os.fstat`` against ``os.stat`` with ``dir_fd`` and
        ``follow_symlinks=False``): it must still be the project's own
        ``.pipeline`` (same ``st_dev`` and ``st_ino`` as
        ``root_fd/.pipeline``). A rename or swap of ``.pipeline`` between
        the directory open and the file open therefore fails closed: only
        the file this call just created is removed, nothing is written,
        and a refusal is raised.
        """
        folder = os.fspath(getattr(self._settings, "folder"))
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
        refusal = RuntimeError(
            "refusing to write the network log outside the project folder"
        )
        try:
            base_fd = os.open(
                folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
        except FileNotFoundError:
            os.makedirs(folder, exist_ok=True)
            try:
                base_fd = os.open(
                    folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise refusal from None
                raise
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise refusal from None
            raise
        try:
            try:
                pipeline_fd = os.open(
                    PIPELINE_DIR_NAME,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=base_fd,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(PIPELINE_DIR_NAME, 0o777, dir_fd=base_fd)
                except FileExistsError:
                    pass
                try:
                    pipeline_fd = os.open(
                        PIPELINE_DIR_NAME,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=base_fd,
                    )
                except OSError as exc:
                    if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                        raise refusal from None
                    raise
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise refusal from None
                raise
            try:
                created = False
                open_flags = (
                    os.O_WRONLY
                    | os.O_APPEND
                    | os.O_NOFOLLOW
                    | os.O_NONBLOCK
                    | os.O_CLOEXEC
                )
                try:
                    log_fd = os.open(
                        LOG_NAME,
                        open_flags | os.O_CREAT | os.O_EXCL,
                        0o666,
                        dir_fd=pipeline_fd,
                    )
                    created = True
                except FileExistsError:
                    try:
                        log_fd = os.open(
                            LOG_NAME,
                            open_flags | os.O_CREAT,
                            0o666,
                            dir_fd=pipeline_fd,
                        )
                    except OSError as exc:
                        # ENXIO: the non-blocking open hit a FIFO with no
                        # reader; EOPNOTSUPP: it hit a socket. Either is a
                        # non-regular file the fstat check below would
                        # refuse, so fail closed here with the same refusal.
                        if exc.errno in (
                            errno.ELOOP,
                            errno.EISDIR,
                            errno.ENXIO,
                            errno.EOPNOTSUPP,
                        ):
                            raise refusal from None
                        raise
                except OSError as exc:
                    if exc.errno in (
                        errno.ELOOP,
                        errno.EISDIR,
                        errno.ENXIO,
                        errno.EOPNOTSUPP,
                    ):
                        raise refusal from None
                    raise
                try:
                    try:
                        log_mode = os.fstat(log_fd).st_mode
                    except OSError:
                        log_mode = 0
                    if not stat.S_ISREG(log_mode):
                        try:
                            os.close(log_fd)
                        except OSError:
                            pass
                        log_fd = None
                        if created:
                            try:
                                os.unlink(LOG_NAME, dir_fd=pipeline_fd)
                            except OSError:
                                pass
                        raise refusal from None
                    os.set_blocking(log_fd, True)
                    try:
                        current = os.stat(
                            PIPELINE_DIR_NAME,
                            dir_fd=base_fd,
                            follow_symlinks=False,
                        )
                    except OSError:
                        current = None
                    opened = os.fstat(pipeline_fd)
                    if (
                        current is None
                        or current.st_ino != opened.st_ino
                        or current.st_dev != opened.st_dev
                        or not stat.S_ISDIR(current.st_mode)
                    ):
                        try:
                            os.close(log_fd)
                        except OSError:
                            pass
                        log_fd = None
                        if created:
                            try:
                                os.unlink(LOG_NAME, dir_fd=pipeline_fd)
                            except OSError:
                                pass
                        raise refusal from None
                    is_new = created or os.fstat(log_fd).st_size == 0
                    handle = os.fdopen(log_fd, "a", encoding="utf-8")
                    log_fd = None
                    with handle:
                        handle.write(LOG_HEADER + line if is_new else line)
                finally:
                    if log_fd is not None:
                        try:
                            os.close(log_fd)
                        except OSError:
                            pass
                        log_fd = None
            finally:
                os.close(pipeline_fd)
        finally:
            os.close(base_fd)

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
