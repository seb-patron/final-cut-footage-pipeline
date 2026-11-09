"""The one safe write path for pipeline steps.

Every pipeline step writes through :class:`SafeWriter`, which creates new
files only inside the project's pipeline-owned folders, refuses every
protected path (media, card, Final Cut library and ``*.fcpbundle`` paths,
even through symlinks), never overwrites an existing entry, publishes each
file atomically with an exclusive hard link, numbers ``edit/`` outputs
without collisions across threads and processes, caps committed files at
:data:`MAX_COMMITTED_BYTES`, and only records plans in dry-run mode.

No method in this module writes under any ``[media]`` or ``[final_cut]``
path. There are no exceptions and no override.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fcp_pipeline.trip import TripSettings

try:  # fcntl exists on macOS and Linux.
    import fcntl
except ImportError:  # pragma: no cover - non-Unix platforms only.
    fcntl = None  # type: ignore[assignment]

#: Largest file allowed outside ``.pipeline/`` (those files are committed).
MAX_COMMITTED_BYTES = 5_000_000

#: Leading ``NNN-`` prefix that marks a numbered ``edit/`` output.
_NUMBERED_RE = re.compile(r"([0-9]{3})-")
#: Highest number ``write_numbered`` hands out.
_MAX_NUMBER = 999
#: Lock file inside ``.pipeline/`` that serialises numbering across processes.
_LOCK_FILENAME = "safe-write.lock"
#: Prefix for temporary files, which live in the target's own folder.
_TEMP_PREFIX = ".safe-write-"
#: Mode for published files (owner write, everyone read).
_FINAL_MODE = 0o644

__all__ = [
    "MAX_COMMITTED_BYTES",
    "PlannedWrite",
    "SafeWriteError",
    "SafeWriter",
]


class SafeWriteError(Exception):
    """Raised when a safe write is refused.

    The message names the path and the rule it broke. A refusal leaves
    nothing behind: no file, no temporary file and no new folder.
    """


class _TargetExists(SafeWriteError):
    """Internal: the exclusive publish link lost a race with another writer."""


@dataclass(frozen=True)
class PlannedWrite:
    """One dry-run record: the final path it would write and its byte size."""

    path: Path
    size: int

    def __getitem__(self, key: str):
        """Allow mapping-style access (``record["path"]``) as well."""
        if key == "path":
            return self.path
        if key == "size":
            return self.size
        raise KeyError(key)


class SafeWriter:
    """Writes new files only inside the project's pipeline-owned places.

    ``settings`` is the :class:`TripSettings` that
    :func:`fcp_pipeline.trip.load_trip` returns; ``cards`` is any card
    volumes a caller hands in to protect. With ``dry_run=True`` nothing on
    disk is created, changed or removed; each accepted write is recorded in
    :attr:`planned` instead.
    """

    def __init__(self, settings: TripSettings, *, dry_run: bool = False, cards=()) -> None:
        project_abs = Path(os.path.abspath(os.fspath(settings.folder)))
        self._project_abs = project_abs
        self._project = project_abs.resolve()
        self._transcripts = self._project / "transcripts"
        self._shots = self._project / "shots"
        self._edit = self._project / "edit"
        self._contact_sheets = self._project / "contact-sheets"
        self._pipeline = self._project / ".pipeline"
        self._memo = self._project / "memo"
        self._edit_abs = self._project_abs / "edit"
        self._dry_run = bool(dry_run)
        self._guard = threading.Lock()
        self.planned: list[PlannedWrite] = []
        protected: list[tuple[str, Path]] = []

        def add(label: str, value) -> None:
            text = os.fspath(value)
            if not text:
                return
            root = Path(text)
            if not root.is_absolute():
                root = project_abs / root
            protected.append((label, root.resolve()))

        for entry in settings.footage:
            add("[media] footage", entry)
        for entry in settings.photos:
            add("[media] photos", entry)
        add("[media] assets", settings.assets)
        add("[final_cut] library", settings.library)
        for card in cards:
            add("card", card)
        self._protected = tuple(protected)

    # -- checks ------------------------------------------------------

    def _is_allowed(self, resolved: Path) -> bool:
        """Whether a resolved target sits in a pipeline-owned place."""
        for folder in (
            self._transcripts,
            self._shots,
            self._edit,
            self._contact_sheets,
            self._pipeline,
        ):
            if resolved != folder and resolved.is_relative_to(folder):
                return True
        if resolved.parent == self._memo:
            leaf = resolved.name
            if leaf == "outline-draft.md":
                return True
            if leaf.endswith(".json") and len(leaf) > len(".json"):
                return True
        return False

    def _check_target(self, resolved: Path, shown: Path, size: int) -> None:
        """Refuse protected, disallowed or oversized targets.

        Runs before anything is created on disk, so a refusal leaves
        nothing behind.
        """
        for label, root in self._protected:
            if resolved == root or resolved.is_relative_to(root):
                raise SafeWriteError(
                    f"refused write to {shown}: inside protected {label} "
                    f"path {root} (the pipeline never writes there)"
                )
        for part in resolved.parts:
            if part.lower().endswith(".fcpbundle"):
                raise SafeWriteError(
                    f"refused write to {shown}: inside a *.fcpbundle bundle "
                    "(the pipeline never writes into Final Cut bundles)"
                )
        if not self._is_allowed(resolved):
            raise SafeWriteError(
                f"refused write to {shown}: outside the pipeline-owned folders "
                "(allowed: transcripts/, shots/, edit/, contact-sheets/, "
                ".pipeline/, memo/<name>.json, memo/outline-draft.md)"
            )
        if size > MAX_COMMITTED_BYTES and not resolved.is_relative_to(self._pipeline):
            raise SafeWriteError(
                f"refused write to {shown}: {size} bytes is over the "
                f"{MAX_COMMITTED_BYTES}-byte cap outside .pipeline/"
            )

    # -- helpers -----------------------------------------------------

    @staticmethod
    def _coerce_data(data) -> bytes:
        if isinstance(data, str):
            return data.encode("utf-8")
        return bytes(data)

    def _final_for(self, path) -> Path:
        """Absolute (unresolved) final path for a relative or absolute input."""
        text = os.fspath(path)
        candidate = Path(text)
        if candidate.is_absolute():
            return Path(os.path.normpath(text))
        return Path(os.path.normpath(os.path.join(os.fspath(self._project_abs), text)))

    @staticmethod
    def _publish(parent: Path, final: Path, blob: bytes) -> None:
        """Write ``blob`` to a temp file, then publish it via exclusive link.

        ``os.link`` fails when the target exists, so a racing writer can
        never replace a file and a reader never sees a partial one. The
        temporary file is removed on every path out.
        """
        fd, tmppath = tempfile.mkstemp(prefix=_TEMP_PREFIX, dir=os.fspath(parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmppath, _FINAL_MODE)
            try:
                os.link(tmppath, os.fspath(final))
            except FileExistsError:
                raise _TargetExists(
                    f"refused write to {final}: target already exists "
                    "(safe writes never overwrite)"
                ) from None
        finally:
            try:
                os.unlink(tmppath)
            except OSError:
                pass

    def _scan_max_number(self) -> int:
        """Highest ``NNN-`` prefix already used by any entry in ``edit/``."""
        try:
            entries = list(os.scandir(os.fspath(self._edit)))
        except FileNotFoundError:
            return 0
        top = 0
        for entry in entries:
            match = _NUMBERED_RE.match(entry.name)
            if match:
                value = int(match.group(1))
                if value > top:
                    top = value
        return top

    def _planned_max_number(self) -> int:
        """Highest ``edit/`` number already planned by this dry-run writer."""
        top = 0
        for record in self.planned:
            if record.path.parent == self._edit_abs:
                match = _NUMBERED_RE.match(record.path.name)
                if match:
                    value = int(match.group(1))
                    if value > top:
                        top = value
        return top

    def _check_lock_path(self, lock: Path) -> None:
        """Refuse a number-lock path that escapes the project's ``.pipeline/``.

        Runs before the lock file is created or opened, so a refusal
        leaves nothing behind. The resolved lock must sit directly inside
        the project's ``.pipeline/`` and must pass every target rule
        (protected media / Final Cut / card paths, ``*.fcpbundle`` parts
        and the pipeline-owned-folders check), exactly like a write target.
        """
        resolved = lock.resolve()
        if resolved.parent != self._pipeline:
            raise SafeWriteError(
                f"refused lock file at {lock}: resolved to {resolved}, "
                "which is outside the project's .pipeline/ "
                "(the number lock must sit inside .pipeline/)"
            )
        self._check_target(resolved, lock, 0)

    def _take_number_lock(self):
        """Exclusive flock on the ``.pipeline/`` lock file (mandatory).

        The lock serialises number choice across processes; the exclusive
        hard link stays the final guard as well. Raises
        :class:`SafeWriteError` without handing out a number when the lock
        path is unsafe or the lock cannot be taken: numbering never
        proceeds without the lock.
        """
        lock = self._pipeline / _LOCK_FILENAME
        self._check_lock_path(lock)
        if fcntl is None:
            raise SafeWriteError(
                "refused numbered output in edit/: exclusive locking is "
                f"unavailable, so no number can be handed out safely "
                f"(lock file {lock})"
            )
        try:
            self._pipeline.mkdir(parents=True, exist_ok=True)
            self._check_lock_path(lock)
            handle = open(os.fspath(lock), "ab")  # noqa: PTH123
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except SafeWriteError:
            raise
        except OSError as exc:
            try:
                handle.close()  # type: ignore[possibly-undefined]
            except (OSError, NameError):
                pass
            raise SafeWriteError(
                f"refused numbered output in edit/: could not lock {lock} "
                f"({exc}); nothing was written"
            ) from exc
        return handle

    @staticmethod
    def _release_number_lock(handle) -> None:
        if handle is None or fcntl is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            try:
                handle.close()
            except OSError:
                pass

    # -- public API --------------------------------------------------

    def write_new(self, path, data) -> Path:
        """Write a new file; refuse protected, disallowed, big or existing ones."""
        blob = self._coerce_data(data)
        size = len(blob)
        final = self._final_for(path)
        resolved = final.resolve()
        self._check_target(resolved, final, size)
        if os.path.lexists(final) or os.path.lexists(resolved):
            raise SafeWriteError(
                f"refused write to {final}: target already exists "
                "(safe writes never overwrite)"
            )
        if self._dry_run:
            with self._guard:
                for record in self.planned:
                    if record.path == final:
                        raise SafeWriteError(
                            f"refused write to {final}: target already exists "
                            "(safe writes never overwrite)"
                        )
                self.planned.append(PlannedWrite(final, size))
                return final
        parent = final.parent
        parent.mkdir(parents=True, exist_ok=True)
        # Re-resolve after creating parents: a swapped symlink must not
        # redirect the write out of the checked place.
        fresh = final.resolve()
        self._check_target(fresh, final, size)
        if os.path.lexists(final) or os.path.lexists(fresh):
            raise SafeWriteError(
                f"refused write to {final}: target already exists "
                "(safe writes never overwrite)"
            )
        self._publish(parent, final, blob)
        return final

    def write_numbered(self, name, data) -> Path:
        """Write ``edit/NNN-<name>`` with the next free number; never reuse one.

        The number lock is mandatory: when its path is unsafe or the lock
        cannot be taken, this raises :class:`SafeWriteError` and writes
        nothing.
        """
        if (
            not isinstance(name, str)
            or not name
            or name in (".", "..")
            or "/" in name
            or "\\" in name
            or "\x00" in name
        ):
            raise SafeWriteError(
                f"refused numbered output for name {name!r}: name must be a "
                "plain, non-empty file name"
            )
        blob = self._coerce_data(data)
        size = len(blob)
        # The number never affects the verdict, so a provisional target lets
        # every rule be checked before anything is created on disk.
        provisional = self._edit_abs / f"001-{name}"
        self._check_target(provisional.resolve(), provisional, size)
        if self._dry_run:
            with self._guard:
                self._check_lock_path(self._pipeline / _LOCK_FILENAME)
                number = max(self._scan_max_number(), self._planned_max_number()) + 1
                if number > _MAX_NUMBER:
                    raise SafeWriteError(
                        "refused numbered output in edit/: "
                        "numbers past 999 are refused"
                    )
                final = self._edit_abs / f"{number:03d}-{name}"
                self._check_target(final.resolve(), final, size)
                if os.path.lexists(final) or os.path.lexists(final.resolve()):
                    raise SafeWriteError(
                        f"refused write to {final}: target already exists "
                        "(safe writes never overwrite)"
                    )
                for record in self.planned:
                    if record.path == final:
                        raise SafeWriteError(
                            f"refused write to {final}: target already exists "
                            "(safe writes never overwrite)"
                        )
                self.planned.append(PlannedWrite(final, size))
                return final
        with self._guard:
            handle = self._take_number_lock()
            if handle is None:
                raise SafeWriteError(
                    "refused numbered output in edit/: exclusive number "
                    "lock was unavailable; nothing was written"
                )
            try:
                tried = self._scan_max_number()
                while tried < _MAX_NUMBER:
                    tried += 1
                    final = self._edit_abs / f"{tried:03d}-{name}"
                    resolved = final.resolve()
                    self._check_target(resolved, final, size)
                    if os.path.lexists(final) or os.path.lexists(resolved):
                        continue
                    parent = final.parent
                    parent.mkdir(parents=True, exist_ok=True)
                    self._check_target(final.resolve(), final, size)
                    try:
                        self._publish(parent, final, blob)
                    except _TargetExists:
                        continue
                    return final
            finally:
                self._release_number_lock(handle)
        raise SafeWriteError(
            "refused numbered output in edit/: numbers past 999 are refused"
        )
