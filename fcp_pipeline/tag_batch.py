"""Tag a whole trip in batch mode through the provider's batch API.

:func:`submit_batch` sends every tag-eligible clip's contact sheet to
the tag step's provider as batch jobs (OpenAI's Batch API or
Anthropic's Message Batches, chosen by the same provider setting the
one-clip tag uses) through :mod:`fcp_pipeline.ai.batch` and the
network gate, saving each batch id in a run record and in a small job
file under ``.pipeline/tag-batch/``. :func:`collect_batch` later
fetches the results keyed by clip, in any order, checks every answer
exactly as the one-clip tag does, and stores the tags. Collecting
twice changes nothing. Offline mode refuses before anything is
uploaded or downloaded. Only the contact-sheet image and the same
prompt text the one-clip tag sends are ever attached.
"""

from __future__ import annotations

import datetime
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from fcp_pipeline import ai
from fcp_pipeline.ai import batch as ai_batch
from fcp_pipeline import catalog as catalog_module
from fcp_pipeline import tag
from fcp_pipeline.ai.usage import append_usage_row
from fcp_pipeline.net import OfflineError
from fcp_pipeline.safe_write import SafeWriter, SafeWriteError
from fcp_pipeline.sheets import read_manifest

#: Most requests one provider batch job holds.
MAX_JOB_REQUESTS = 5000
#: Most serialized request JSON bytes one provider batch job holds.
MAX_JOB_BYTES = 100_000_000
#: Largest job or done file ever written, in bytes.
MAX_STATE_BYTES = 5_000_000
#: Folder holding the job files inside ``.pipeline/``.
STATE_DIR_NAME = "tag-batch"
#: Version written into and required back from every state file.
JOB_VERSION = 1


class TagBatchError(ValueError):
    """Raised when a batch job file cannot be read or written."""


@dataclass(frozen=True)
class JobRef:
    """One submitted provider batch job: provider, id and clip count."""

    provider: str
    batch_id: str
    count: int


@dataclass(frozen=True)
class SubmitResult:
    """What one batch submit did, in catalog id order plus new jobs."""

    entries: tuple = ()
    jobs: tuple = ()

    @property
    def submitted(self) -> int:
        """How many clips were sent as batch requests."""
        return sum(1 for e in self.entries if e.status == "submitted")

    @property
    def would_submit(self) -> int:
        """How many clips a dry run would send."""
        return sum(1 for e in self.entries if e.status == "would-submit")

    @property
    def skipped(self) -> int:
        """How many clips already held ai tags and were left alone."""
        return sum(1 for e in self.entries if e.status == "skipped")

    @property
    def pending(self) -> int:
        """How many clips already sit in an open batch job."""
        return sum(1 for e in self.entries if e.status == "pending")

    @property
    def stale(self) -> int:
        """How many rows no longer matched their file."""
        return sum(1 for e in self.entries if e.status == "stale")

    @property
    def no_sheet(self) -> int:
        """How many clips had no contact sheet for their current state."""
        return sum(1 for e in self.entries if e.status == "no-sheet")

    @property
    def failed(self) -> int:
        """How many clips could not be submitted."""
        return sum(1 for e in self.entries if e.status == "failed")

    @property
    def notes(self) -> tuple:
        """Job-level notes; submit uses per-entry notes only."""
        return ()

    @property
    def outcome(self) -> str:
        """The run-record line naming every new batch id."""
        if self.would_submit:
            return (
                f"would-submit {self.would_submit}, skipped {self.skipped}, "
                f"pending {self.pending}, stale {self.stale}, "
                f"no-sheet {self.no_sheet}, failed {self.failed}"
            )
        if self.jobs:
            provider = self.jobs[0].provider
            ids = ", ".join(job.batch_id for job in self.jobs)
            head = (
                f"submitted {self.submitted} in {len(self.jobs)} "
                f"batches ({provider}: {ids})"
            )
        else:
            head = f"submitted {self.submitted} in 0 batches"
        return (
            f"{head}, skipped {self.skipped}, pending {self.pending}, "
            f"stale {self.stale}, no-sheet {self.no_sheet}, "
            f"failed {self.failed}"
        )


@dataclass(frozen=True)
class CollectResult:
    """What one batch collect did, in job order plus job-level notes."""

    entries: tuple = ()
    notes: tuple = ()

    @property
    def tagged(self) -> int:
        """How many clips were tagged from batch answers."""
        return sum(1 for e in self.entries if e.status == "tagged")

    @property
    def failed(self) -> int:
        """How many clips could not be tagged."""
        return sum(1 for e in self.entries if e.status == "failed")

    @property
    def expired(self) -> int:
        """How many clips had no usable answer when the batch ended."""
        return sum(1 for e in self.entries if e.status == "expired")

    @property
    def stale(self) -> int:
        """How many clips changed since their batch was submitted."""
        return sum(1 for e in self.entries if e.status == "stale")

    @property
    def running(self) -> int:
        """How many clips sit in a batch that is still running."""
        return sum(1 for e in self.entries if e.status == "running")

    @property
    def jobs(self) -> tuple:
        """Collect prints no per-job batch lines."""
        return ()

    @property
    def outcome(self) -> str:
        """The run-record line, or ``"no open batch"`` with none open."""
        if not self.entries:
            return "no open batch"
        return (
            f"tagged {self.tagged}, failed {self.failed}, "
            f"expired {self.expired}, stale {self.stale}, "
            f"running {self.running}"
        )


def _utc_now() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _state_dir(settings) -> Path:
    """Return the ``.pipeline/tag-batch/`` folder for these settings."""
    return Path(os.fspath(getattr(settings, "folder"))) / ".pipeline" / STATE_DIR_NAME


def _check_cutoff(cutoff) -> None:
    """Reject a cutoff that is not a number from 0 to 1."""
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, (int, float))
        or not math.isfinite(cutoff)
        or not 0 <= cutoff <= 1
    ):
        raise tag.TagError(
            f"bad cutoff {cutoff!r}: must be a number from 0 to 1"
        )


def _next_job_number(settings) -> int:
    """Return the next free three-digit job number for these settings."""
    folder = _state_dir(settings)
    top = 0
    try:
        children = sorted(folder.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return 1
    except OSError:
        return 1
    for child in children:
        name = child.name
        if (
            len(name) == len("NNN.job.json")
            and name.endswith(".job.json")
            and name[:3].isdigit()
        ):
            top = max(top, int(name[:3]))
    return top + 1


def _read_state_file(state_dir: Path, name: str) -> dict:
    """Read and validate one state file, or raise :class:`TagBatchError`.

    Refuses symlinks, requires the resolved parent to equal the
    resolved state folder, refuses a file over :data:`MAX_STATE_BYTES`,
    and names the file when it is unreadable, unparsable or the wrong
    version. A valid state file is never guessed over.
    """
    path = state_dir / name
    resolved_state = state_dir.resolve()
    if os.path.islink(os.fspath(path)):
        raise TagBatchError(
            f"bad job file {name}: it is a symlink; refusing to follow it"
        )
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise TagBatchError(
            f"bad job file {name}: could not be read ({exc})"
        ) from None
    if resolved.parent != resolved_state:
        raise TagBatchError(
            f"bad job file {name}: it escapes the tag-batch folder"
        )
    try:
        size = os.stat(resolved).st_size
    except OSError as exc:
        raise TagBatchError(
            f"bad job file {name}: could not be read ({exc})"
        ) from None
    if size > MAX_STATE_BYTES:
        raise TagBatchError(
            f"bad job file {name}: {size} bytes is over the "
            f"{MAX_STATE_BYTES}-byte cap"
        )
    try:
        raw = resolved.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise TagBatchError(
            f"bad job file {name}: could not be read ({exc})"
        ) from None
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise TagBatchError(
            f"bad job file {name}: not JSON ({exc})"
        ) from None
    if not isinstance(document, dict) or document.get("version") != JOB_VERSION:
        raise TagBatchError(
            f"bad job file {name}: wrong version "
            "(expected version 1)"
        )
    return document


def _check_job_document(name: str, document: dict) -> dict:
    """Require a job file's keys and value types.

    Anything unreadable was already refused; a file whose values do
    not have the shapes this module writes raises
    :class:`TagBatchError` naming it instead of being guessed over.
    """
    for key in (
        "provider", "model", "batch_id", "submitted", "cutoff", "retag",
        "vocabulary", "clips",
    ):
        if key not in document:
            raise TagBatchError(
                f"bad job file {name}: missing {key!r}"
            )
    if document.get("provider") not in ("openai", "anthropic"):
        raise TagBatchError(
            f"bad job file {name}: 'provider' must be "
            '"openai" or "anthropic"'
        )
    for key in ("model", "batch_id", "submitted"):
        value = document.get(key)
        if not isinstance(value, str) or not value:
            raise TagBatchError(
                f"bad job file {name}: {key!r} must be a non-empty string"
            )
    if (
        isinstance(document.get("cutoff"), bool)
        or not isinstance(document.get("cutoff"), (int, float))
    ):
        raise TagBatchError(
            f"bad job file {name}: 'cutoff' must be a number"
        )
    if not isinstance(document.get("retag"), bool):
        raise TagBatchError(
            f"bad job file {name}: 'retag' must be true or false"
        )
    clips = document.get("clips")
    if not isinstance(clips, list):
        raise TagBatchError(f"bad job file {name}: 'clips' must be a list")
    for clip in clips:
        if not isinstance(clip, dict):
            raise TagBatchError(
                f"bad job file {name}: every clip must be an object"
            )
        for key in ("custom_id", "media_id", "asset_id", "sheet_sha256"):
            if key not in clip:
                raise TagBatchError(
                    f"bad job file {name}: a clip is missing {key!r}"
                )
        if not isinstance(clip.get("custom_id"), str) or not clip.get(
            "custom_id"
        ):
            raise TagBatchError(
                f"bad job file {name}: a clip's 'custom_id' must be "
                "a non-empty string"
            )
        if isinstance(clip.get("media_id"), bool) or not isinstance(
            clip.get("media_id"), int
        ):
            raise TagBatchError(
                f"bad job file {name}: a clip's 'media_id' must be an integer"
            )
        for key in ("asset_id", "sheet_sha256"):
            if not isinstance(clip.get(key), str):
                raise TagBatchError(
                    f"bad job file {name}: a clip's {key!r} must be a string"
                )
    vocabulary = document.get("vocabulary")
    if not isinstance(vocabulary, dict):
        raise TagBatchError(
            f"bad job file {name}: 'vocabulary' must be an object"
        )
    for folder, values in vocabulary.items():
        if not isinstance(folder, str) or not isinstance(values, (list, tuple)):
            raise TagBatchError(
                f"bad job file {name}: 'vocabulary' must map folders "
                "to arrays of strings"
            )
        for value in values:
            if not isinstance(value, str):
                raise TagBatchError(
                    f"bad job file {name}: 'vocabulary' must map folders "
                    "to arrays of strings"
                )
    return document


def _load_open_jobs(settings) -> list:
    """Return ``(number, job document)`` for jobs with no done file yet."""
    state_dir = _state_dir(settings)
    try:
        children = sorted(state_dir.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return []
    except OSError:
        return []
    numbers: list[int] = []
    for child in children:
        name = child.name
        if (
            len(name) == len("NNN.job.json")
            and name.endswith(".job.json")
            and name[:3].isdigit()
        ):
            numbers.append(int(name[:3]))
    open_jobs: list = []
    for number in sorted(numbers):
        name = f"{number:03d}.job.json"
        document = _check_job_document(
            name, _read_state_file(state_dir, name)
        )
        done_name = f"{number:03d}.done.json"
        if os.path.lexists(os.fspath(state_dir / done_name)):
            continue
        open_jobs.append((number, document))
    return open_jobs


def _document_bytes(document: dict) -> bytes:
    """Return the exact bytes one state file would hold."""
    return (json.dumps(document, indent=2) + "\n").encode("utf-8")


def _write_state_file(settings, name: str, document: dict) -> Path:
    """Write one state file new; refuse oversized or existing targets."""
    data = _document_bytes(document)
    if len(data) > MAX_STATE_BYTES:
        raise TagBatchError(
            f"refusing to write {name}: {len(data)} bytes is over the "
            f"{MAX_STATE_BYTES}-byte cap"
        )
    writer = SafeWriter(settings)
    try:
        return writer.write_new(
            Path(".pipeline") / STATE_DIR_NAME / name, data
        )
    except SafeWriteError as exc:
        raise TagBatchError(str(exc)) from None


def _job_document(provider: str, model: str, batch_id: str, submitted: str,
                  cutoff: float, retag: bool, vocabulary: dict,
                  clips: list) -> dict:
    """Build one ``<NNN>.job.json`` document (no answer or key text)."""
    return {
        "version": JOB_VERSION,
        "provider": provider,
        "model": model,
        "batch_id": batch_id,
        "submitted": submitted,
        "cutoff": cutoff,
        "retag": retag,
        "vocabulary": {
            folder: list(values)
            for folder, values in sorted(vocabulary.items())
        },
        "clips": list(clips),
    }


def _clip_rows(catalog) -> list:
    """Return the catalog's ``kind == "clip"`` rows in id order."""
    rows = [
        row
        for row in catalog.list_media()
        if getattr(row, "kind", None) == "clip"
    ]
    rows.sort(key=lambda row: row.id)
    return rows


def _submit_entries(catalog, settings, rows, shown, schema, open_batch,
                    retag: bool, dry_run: bool, backends) -> tuple:
    """Decide every row; build requests for the eligible ones.

    Returns ``(entries, eligible)`` where ``eligible`` holds one dict
    per sendable clip with its row, manifest, request and vocabulary
    note. With ``dry_run`` no sheet is read and no request is built.
    """
    entries: list = []
    eligible: list = []
    for row in rows:
        path = os.fspath(getattr(row, "path"))
        asset_id = getattr(row, "asset_id")
        if getattr(row, "duration", None) is None or not catalog.unchanged(
            path
        ):
            entries.append(
                tag.TagEntry(
                    path=path,
                    asset_id=asset_id,
                    status="stale",
                    reason="changed since the scan; run the scan again",
                )
            )
            continue
        manifest = read_manifest(settings, row)
        if manifest is None:
            entries.append(
                tag.TagEntry(
                    path=path,
                    asset_id=asset_id,
                    status="no-sheet",
                    reason=(
                        "no contact sheet for this clip's current state; "
                        "run the sheets step first"
                    ),
                )
            )
            continue
        if not retag and any(
            getattr(existing, "source", None) == "ai"
            for existing in catalog.list_tags(row.id)
        ):
            entries.append(
                tag.TagEntry(
                    path=path,
                    asset_id=asset_id,
                    status="skipped",
                    reason=(
                        "already has ai tags; "
                        "use --retag to tag it again"
                    ),
                )
            )
            continue
        batch_id = open_batch.get(row.id)
        if batch_id is not None:
            entries.append(
                tag.TagEntry(
                    path=path,
                    asset_id=asset_id,
                    status="pending",
                    reason=f"already in batch {batch_id}; run --collect",
                )
            )
            continue
        if dry_run:
            entries.append(
                tag.TagEntry(
                    path=path, asset_id=asset_id, status="would-submit"
                )
            )
            continue
        data, failure = tag._load_sheet_bytes(settings, manifest)
        if data is None:
            assert failure is not None
            entries.append(
                tag.TagEntry(
                    path=path,
                    asset_id=asset_id,
                    status="failed",
                    reason=failure,
                )
            )
            continue
        slate = tag.slate_text(catalog, row.id)
        transcript, transcript_note = tag.transcript_text(catalog, row.id)
        instructions = tag.build_instructions(
            tag.metadata_line(row),
            slate,
            transcript,
            shown,
            tag._manifest_tile_times(manifest),
        )
        image = ai.Image(data=data, media_type="image/jpeg")
        _, request = ai_batch.prepare(
            settings, tag.STEP, instructions, schema, [image],
            backends=backends,
        )
        notes = (transcript_note,) if transcript_note is not None else ()
        entries.append(
            tag.TagEntry(
                path=path,
                asset_id=asset_id,
                status="submitted",
                notes=notes,
            )
        )
        eligible.append(
            {
                "row": row,
                "custom_id": f"clip-{row.id}",
                "request": request,
                "sheet_sha256": manifest.get("sheet_sha256"),
                "notes": notes,
            }
        )
    return entries, eligible


def _split_jobs(eligible: list) -> list:
    """Group eligible clips, in order, into provider batch jobs.

    A job holds at most :data:`MAX_JOB_REQUESTS` requests and at most
    :data:`MAX_JOB_BYTES` bytes of serialized request JSON; a job that
    would overflow starts a new one. One trip usually makes one job.
    """
    jobs: list = []
    current: list = []
    current_bytes = 0
    for item in eligible:
        size = len(json.dumps(item["request"]).encode("utf-8"))
        if current and (
            len(current) >= MAX_JOB_REQUESTS
            or current_bytes + size > MAX_JOB_BYTES
        ):
            jobs.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += size
    if current:
        jobs.append(current)
    return jobs


def _preview_job_bytes(provider: str, model: str, groups: list,
                       cutoff: float, retag: bool, shown: dict) -> None:
    """Raise before any provider call when a job file would be too big."""
    for group in groups:
        clips = [
            {
                "custom_id": item["custom_id"],
                "media_id": item["row"].id,
                "asset_id": getattr(item["row"], "asset_id"),
                "sheet_sha256": item["sheet_sha256"],
            }
            for item in group
        ]
        document = _job_document(
            provider, model, "", _utc_now(), cutoff, retag, shown, clips,
        )
        data = _document_bytes(document)
        if len(data) > MAX_STATE_BYTES:
            raise TagBatchError(
                "refusing the batch submit: a job file would be "
                f"{len(data)} bytes, over the {MAX_STATE_BYTES}-byte cap"
            )


def submit_batch(catalog, settings, *, vocabulary=None,
                 cutoff=tag.DEFAULT_CUTOFF, retag: bool = False,
                 dry_run: bool = False, backends=None) -> SubmitResult:
    """Submit every tag-eligible clip's sheet as provider batch jobs.

    Visits the catalog's ``kind == "clip"`` rows in id order and decides
    each one with the same rules, in the same order and with the same
    reason texts as :func:`fcp_pipeline.tag.tag_catalog`. A dry run
    records nothing, writes nothing and calls nothing.
    """
    _check_cutoff(cutoff)
    if vocabulary is None:
        shown = tag.build_vocabulary(catalog, settings)
    else:
        shown = tag._check_vocabulary_mapping(vocabulary)
    schema = tag.answer_schema()
    rows = _clip_rows(catalog)
    open_jobs = _load_open_jobs(settings)
    pending: dict = {}
    for _, job in open_jobs:
        for clip in job.get("clips", []):
            pending[clip.get("media_id")] = job.get("batch_id")
    if dry_run:
        entries, _ = _submit_entries(
            catalog, settings, rows, shown, schema, pending, retag, True,
            None,
        )
        return SubmitResult(entries=tuple(entries), jobs=())
    run_id = catalog.start_run("tag-batch", dry_run=False)
    try:
        config = ai.step_config(settings, tag.STEP)
        entries, eligible = _submit_entries(
            catalog, settings, rows, shown, schema, pending, retag, False,
            backends,
        )
        groups = _split_jobs(eligible)
        _preview_job_bytes(
            config.provider, config.model, groups, float(cutoff), retag,
            shown,
        )
        jobs: list = []
        for group in groups:
            items = [
                (item["custom_id"], item["request"]) for item in group
            ]
            batch_id = ai_batch.submit(
                settings, tag.STEP, config.provider, config.model, items,
                backends=backends,
            )
            clips = [
                {
                    "custom_id": item["custom_id"],
                    "media_id": item["row"].id,
                    "asset_id": getattr(item["row"], "asset_id"),
                    "sheet_sha256": item["sheet_sha256"],
                }
                for item in group
            ]
            document = _job_document(
                config.provider, config.model, batch_id, _utc_now(),
                float(cutoff), retag, shown, clips,
            )
            number = _next_job_number(settings)
            try:
                _write_state_file(settings, f"{number:03d}.job.json", document)
            except TagBatchError as exc:
                raise TagBatchError(
                    f"batch {batch_id} from provider {config.provider} "
                    f"was accepted but its job file could not be written: "
                    f"{exc}"
                ) from None
            jobs.append(
                JobRef(config.provider, batch_id, len(group))
            )
        finished = SubmitResult(entries=tuple(entries), jobs=tuple(jobs))
        catalog.finish_run(run_id, finished.outcome)
        return finished
    except (ai.AiConfigError, OfflineError) as exc:
        outcome = f"stopped: {type(exc).__name__}"
        try:
            catalog.finish_run(run_id, outcome)
        except (catalog_module.CatalogError, OSError):
            pass
        raise
    except Exception as exc:
        try:
            catalog.finish_run(run_id, f"failed: {exc}")
        except (catalog_module.CatalogError, OSError):
            pass
        raise


def _collect_finished(catalog, settings, job, batch_id, items,
                      schema) -> tuple:
    """Apply one finished job's items; return ``(entries, done_clips)``.

    Items match the job's clips by ``custom_id`` in any order. Unknown
    or duplicated ids are never applied; each becomes one job-level
    note. Succeeded answers are checked exactly like one-clip answers
    and stored through the same records, so locked, deleted and
    non-``"ai"`` tags in a slot are never replaced. Every succeeded
    item appends one usage row.
    """
    provider = job.get("provider")
    model = job.get("model")
    job_ids = {
        clip.get("custom_id") for clip in job.get("clips", [])
    }
    counts: dict = {}
    for item in items:
        counts[item.custom_id] = counts.get(item.custom_id, 0) + 1
    usable: dict = {}
    notes: list = []
    duplicates: list = []
    for item in items:
        if item.custom_id not in job_ids:
            notes.append(
                f"ignoring result for {item.custom_id!r} in batch "
                f"{batch_id}: no such clip in the job"
            )
        elif counts[item.custom_id] > 1:
            if item.custom_id not in duplicates:
                duplicates.append(item.custom_id)
        elif item.custom_id not in usable:
            usable[item.custom_id] = item
    for custom_id in duplicates:
        notes.append(
            f"ignoring duplicate results for {custom_id!r} in batch "
            f"{batch_id}: none of them applied"
        )
    rows_by_id = {row.id: row for row in catalog.list_media()}
    vocabulary = {
        folder: tuple(values)
        for folder, values in (job.get("vocabulary") or {}).items()
    }
    cutoff = job.get("cutoff")
    entries: list = []
    done_clips: list = []
    for clip in job.get("clips", []):
        media_id = clip.get("media_id")
        asset_id = clip.get("asset_id")
        custom_id = clip.get("custom_id")
        row = rows_by_id.get(media_id)
        path = os.fspath(getattr(row, "path")) if row is not None else (
            asset_id if isinstance(asset_id, str) else f"media {media_id}"
        )

        def record(status: str, reason=None, stored: int = 0,
                   entry_notes: tuple = ()):
            entries.append(
                tag.TagEntry(
                    path=path, asset_id=asset_id, status=status,
                    reason=reason, stored=stored, notes=entry_notes,
                )
            )
            done_clips.append(
                {"asset_id": asset_id, "status": status, "reason": reason}
            )

        item = usable.get(custom_id)
        if item is None:
            record("expired", "no result before the batch ended")
            continue
        if item.status in ("expired", "canceled"):
            record("expired", item.error or item.status)
            continue
        if item.status == "errored":
            record("failed", item.error or "error")
            continue
        append_usage_row(settings, tag.STEP, provider, model, item.reply)
        duration = getattr(row, "duration", None) if row is not None else None
        if (
            row is None
            or duration is None
            or not catalog.unchanged(path)
        ):
            record("stale", "changed since the scan; run the scan again")
            continue
        manifest = read_manifest(settings, row)
        if manifest is None or manifest.get("sheet_sha256") != clip.get(
            "sheet_sha256"
        ):
            record(
                "stale",
                "contact sheet changed since the batch was submitted; "
                "run the sheets step again",
            )
            continue
        problems = ai.validate(item.reply.data, schema)
        if problems:
            record("failed", f"answer failed checks: {problems[0]}")
            continue
        try:
            kept, check_notes = tag.check_answer(
                item.reply.data, vocabulary, duration,
                tag._manifest_edges(manifest, duration),
            )
        except ai.AiAnswerError as exc:
            record("failed", str(exc))
            continue
        records: list = []
        for folder, value, confidence, start, span in kept:
            records.append(
                catalog_module.TagRecord(
                    media_id=row.id,
                    folder=folder,
                    value=value,
                    start=start,
                    duration=span,
                    source="ai",
                    confidence=confidence,
                    status="plain" if confidence >= cutoff else "check",
                )
            )
        records.extend(tag._whole_clip_records(item.reply.data, row.id))
        stored, store_notes = tag._store_records(catalog, row.id, records)
        record("tagged", None, stored, tuple(check_notes) + tuple(store_notes))
    return entries, notes, done_clips


def collect_batch(catalog, settings, *, backends=None) -> CollectResult:
    """Collect every open batch job's results, in job number order.

    With no open job nothing is called and the outcome says so. A
    running job stays open with its clips listed as running. Finished
    jobs have their results fetched once, checked like one-clip answers
    and stored, then their done file is written. A second collect after
    everything finished finds no open job and changes nothing.
    """
    open_jobs = _load_open_jobs(settings)
    schema = tag.answer_schema()
    run_id = catalog.start_run("tag-collect", dry_run=False)
    try:
        if not open_jobs:
            result = CollectResult(entries=(), notes=())
            catalog.finish_run(run_id, result.outcome)
            return result
        entries: list = []
        notes: list = []
        for number, job in open_jobs:
            provider = job.get("provider")
            model = job.get("model")
            batch_id = job.get("batch_id")
            state = ai_batch.status(
                settings, tag.STEP, provider, model, batch_id,
                backends=backends,
            )
            if state.state == "running":
                rows_by_id = {
                    row.id: row for row in catalog.list_media()
                }
                for clip in job.get("clips", []):
                    row = rows_by_id.get(clip.get("media_id"))
                    asset_id = clip.get("asset_id")
                    if row is not None:
                        path = os.fspath(getattr(row, "path"))
                    elif isinstance(asset_id, str):
                        path = asset_id
                    else:
                        path = f"media {clip.get('media_id')}"
                    entries.append(
                        tag.TagEntry(
                            path=path,
                            asset_id=asset_id,
                            status="running",
                            reason=(
                                f"batch {batch_id} is still running; "
                                "run --collect again later"
                            ),
                        )
                    )
                continue
            items = ai_batch.results(
                settings, tag.STEP, provider, model, batch_id,
                backends=backends,
            )
            job_entries, job_notes, done_clips = _collect_finished(
                catalog, settings, job, batch_id, items, schema,
            )
            entries.extend(job_entries)
            notes.extend(job_notes)
            document = {
                "version": JOB_VERSION,
                "batch_id": batch_id,
                "collected": _utc_now(),
                "state": state.state,
                "clips": done_clips,
            }
            _write_state_file(settings, f"{number:03d}.done.json", document)
        result = CollectResult(entries=tuple(entries), notes=tuple(notes))
        catalog.finish_run(run_id, result.outcome)
        return result
    except (ai.AiConfigError, OfflineError) as exc:
        outcome = f"stopped: {type(exc).__name__}"
        try:
            catalog.finish_run(run_id, outcome)
        except (catalog_module.CatalogError, OSError):
            pass
        raise
    except Exception as exc:
        try:
            catalog.finish_run(run_id, f"failed: {exc}")
        except (catalog_module.CatalogError, OSError):
            pass
        raise


def run(args, settings, catalog, extra):
    """Run ``--batch`` or ``--collect`` for the tag step's main hook."""
    if getattr(args, "collect", False):
        return collect_batch(catalog, settings)
    return submit_batch(
        catalog,
        settings,
        vocabulary=extra,
        cutoff=args.cutoff,
        retag=args.retag,
        dry_run=args.dry_run,
    )
