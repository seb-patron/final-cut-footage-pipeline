"""OpenAI backend for AI steps, using the Responses API.

Only plain dicts cross this module's boundary. The SDK itself is never
imported here: :meth:`create` builds the client through the network gate,
which refuses offline calls before any key is read, makes one call, and
converts the SDK result to a plain dict when it offers ``model_dump``.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Reply:
    """One parsed backend answer: JSON data plus token counts.

    ``data`` is the parsed JSON value, or None when the answer holds
    none (a refusal part, or text that is not JSON). Usage counts come
    from the response's own usage block, defaulting to zero.
    """

    data: Any
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class BatchStatus:
    """One provider batch job's state plus its total request count.

    ``state`` is one of ``"running"``, ``"done"``, ``"failed"``,
    ``"expired"`` or ``"cancelled"``.
    """

    state: str
    total: int = 0


@dataclass(frozen=True)
class BatchItem:
    """One clip's answer inside a finished provider batch.

    ``status`` is one of ``"succeeded"``, ``"errored"``, ``"expired"``
    or ``"canceled"``. ``reply`` is a :class:`Reply` (whose data may
    still be None) only for ``"succeeded"``; otherwise ``reply`` is None
    and ``error`` carries a code-only reason, never provider or answer
    text.
    """

    custom_id: str
    status: str
    reply: Reply | None = None
    error: str | None = None


def _get(obj, key: str, default=None):
    """Read ``key`` from a plain dict or a response object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _count(value) -> int:
    """Best-effort token count; anything unusable counts as zero."""
    if isinstance(value, bool):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number >= 0 else 0


def _reject_constant(value: str):
    """Reject ``NaN``/``Infinity`` constants; model answers must be finite."""
    raise ValueError(f"invalid JSON constant: {value}")


def _batch_status(raw) -> BatchStatus:
    """Map a provider batch object to a :class:`BatchStatus`.

    Handles both providers' shapes: Anthropic's ``processing_status``
    (``in_progress``/``canceling`` run, ``ended`` is done) and OpenAI's
    ``status`` (``validating``, ``in_progress``, ``finalizing`` and
    ``cancelling`` run; ``completed`` is done; ``failed``, ``expired``
    and ``cancelled`` keep their names). Anything else is failed.
    """
    if not isinstance(raw, dict):
        return BatchStatus("failed", 0)
    processing = raw.get("processing_status")
    if isinstance(processing, str):
        if processing in ("in_progress", "canceling"):
            state = "running"
        elif processing == "ended":
            state = "done"
        else:
            state = "failed"
    else:
        status = raw.get("status")
        if status in (
            "validating", "in_progress", "finalizing", "cancelling",
        ):
            state = "running"
        elif status == "completed":
            state = "done"
        elif status in ("failed", "expired", "cancelled"):
            state = status
        else:
            state = "failed"
    counts = raw.get("request_counts")
    total = 0
    if isinstance(counts, dict):
        value = counts.get("total")
        if isinstance(value, bool):
            total = 0
        elif isinstance(value, int):
            total = max(value, 0)
        else:
            for key in (
                "processing", "succeeded", "errored", "canceled",
                "expired", "completed", "failed",
            ):
                part = counts.get(key)
                if isinstance(part, int) and not isinstance(part, bool):
                    total += max(part, 0)
    return BatchStatus(state, total)


def _openai_batch_item(line: dict, parse_response) -> BatchItem | None:
    """Parse one OpenAI batch output line into a :class:`BatchItem`."""
    custom_id = line.get("custom_id")
    if not isinstance(custom_id, str) or not custom_id:
        return None
    error = line.get("error")
    response = line.get("response")
    if not isinstance(response, dict):
        response = {}
    status_code = response.get("status_code")
    error_code = error.get("code") if isinstance(error, dict) else None
    if isinstance(error_code, str) and error_code == "expired":
        return BatchItem(custom_id, "expired", None, "expired")
    if isinstance(error_code, str) and error_code in (
        "canceled", "cancelled",
    ):
        return BatchItem(custom_id, "canceled", None, "canceled")
    if isinstance(error, dict) or status_code != 200:
        if status_code is None:
            reason = (
                error_code
                if isinstance(error_code, str) and error_code
                else "request failed"
            )
        elif isinstance(error_code, str) and error_code:
            reason = f"status {status_code}: {error_code}"
        else:
            reason = f"status {status_code}"
        return BatchItem(custom_id, "errored", None, reason)
    body = response.get("body")
    if not isinstance(body, dict):
        body = {}
    return BatchItem(custom_id, "succeeded", parse_response(body), None)


def _anthropic_batch_item(entry: dict, parse_response) -> BatchItem | None:
    """Parse one Anthropic batch result entry into a :class:`BatchItem`."""
    custom_id = entry.get("custom_id")
    if not isinstance(custom_id, str) or not custom_id:
        return None
    result = entry.get("result")
    if not isinstance(result, dict):
        result = {}
    kind = result.get("type")
    if kind == "succeeded":
        message = result.get("message")
        if not isinstance(message, dict):
            message = {}
        return BatchItem(
            custom_id, "succeeded", parse_response(message), None
        )
    if kind == "errored":
        error = result.get("error")
        code = error.get("type") if isinstance(error, dict) else None
        reason = code if isinstance(code, str) and code else "error"
        return BatchItem(custom_id, "errored", None, reason)
    if kind == "expired":
        return BatchItem(custom_id, "expired", None, "expired")
    if kind == "canceled":
        return BatchItem(custom_id, "canceled", None, "canceled")
    return BatchItem(custom_id, "errored", None, "unknown result")


class OpenAIBackend:
    """One OpenAI Responses API call through a network gate."""

    def __init__(self, gate) -> None:
        self._gate = gate

    def build_request(
        self, model: str, instructions: str, images, schema: dict,
        max_output_tokens: int,
    ) -> dict:
        """Return the pure ``responses.create`` keyword arguments."""
        content: list[dict] = [{"type": "input_text", "text": instructions}]
        for image in images:
            encoded = base64.b64encode(image.data).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": (
                        f"data:{image.media_type};base64,{encoded}"
                    ),
                }
            )
        return {
            "model": model,
            "max_output_tokens": max_output_tokens,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": schema,
                    "strict": True,
                }
            },
        }

    def create(self, request: dict) -> dict:
        """Make one Responses call; the only method touching the SDK."""
        client = self._gate.openai_client()
        result = client.responses.create(**request)
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        return dict(result)

    def parse_response(self, raw) -> Reply:
        """Parse a plain Responses result into a :class:`Reply`."""
        output = _get(raw, "output") or []
        texts: list[str] = []
        refused = False
        for item in output:
            item_type = _get(item, "type")
            if item_type == "message":
                for part in _get(item, "content") or []:
                    part_type = _get(part, "type")
                    if part_type == "output_text":
                        texts.append(_get(part, "text") or "")
                    elif part_type == "refusal":
                        refused = True
            elif item_type == "refusal":
                refused = True
        usage = _get(raw, "usage") or {}
        input_tokens = _count(_get(usage, "input_tokens"))
        output_tokens = _count(_get(usage, "output_tokens"))
        if refused:
            return Reply(None, input_tokens, output_tokens)
        text = "".join(
            part for part in texts if isinstance(part, str)
        )
        if not text.strip():
            return Reply(None, input_tokens, output_tokens)
        try:
            data = json.loads(text, parse_constant=_reject_constant)
        except ValueError:
            data = None
        return Reply(data, input_tokens, output_tokens)

    def send(
        self, model: str, instructions: str, images, schema: dict,
        max_output_tokens: int,
    ) -> Reply:
        """Build, make, and parse one Responses call."""
        return self.parse_response(
            self.create(
                self.build_request(
                    model, instructions, images, schema, max_output_tokens
                )
            )
        )

    def build_batch_input(self, items) -> bytes:
        """Return the UTF-8 JSONL batch input for ``items``.

        ``items`` holds ``(custom_id, request)`` pairs where ``request``
        is exactly what :meth:`build_request` returns. One line per
        item, each naming the ``/v1/responses`` endpoint, so the batch
        run answers the same requests the one-clip call would send.
        """
        lines: list[str] = []
        for custom_id, request in items:
            lines.append(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": request,
                    }
                )
            )
        return ("\n".join(lines) + "\n" if lines else "").encode("utf-8")

    def create_batch(self, batch_input: bytes) -> str:
        """Submit ``batch_input`` as one batch; the only SDK caller here.

        The JSONL bytes are uploaded from memory (never from a file on
        disk) with purpose ``"batch"``, then one batch is created on
        ``/v1/responses`` with a 24 h window. Returns the new batch id.
        """
        client = self._gate.openai_client()
        uploaded = client.files.create(
            file=("tag-batch.jsonl", batch_input), purpose="batch"
        )
        if hasattr(uploaded, "model_dump"):
            uploaded = uploaded.model_dump()
        file_id = _get(uploaded, "id")
        batch = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        if hasattr(batch, "model_dump"):
            batch = batch.model_dump()
        return _get(batch, "id")

    def retrieve_batch(self, batch_id: str) -> dict:
        """Return one batch object as a plain dict; touches the SDK."""
        client = self._gate.openai_client()
        result = client.batches.retrieve(batch_id)
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        return dict(result)

    def parse_batch_status(self, raw) -> BatchStatus:
        """Map a provider batch object to a :class:`BatchStatus`.

        Handles both providers' shapes: Anthropic's
        ``processing_status`` (``in_progress``/``canceling`` run,
        ``ended`` is done) and OpenAI's ``status`` (``validating``,
        ``in_progress``, ``finalizing`` and ``cancelling`` run;
        ``completed`` is done; ``failed``, ``expired`` and
        ``cancelled`` keep their names). Anything else is failed.
        """
        return _batch_status(raw)

    def fetch_batch_results(self, batch_id: str, raw_status) -> dict:
        """Download a batch's result files; the only SDK caller here.

        Returns ``{"output_text": ..., "error_text": ...}``: the text
        of the batch's ``output_file_id`` file and of its
        ``error_file_id`` file, each missing id giving empty text.
        """
        client = self._gate.openai_client()
        status = raw_status if isinstance(raw_status, dict) else {}

        def _read(file_id) -> str:
            if not file_id:
                return ""
            result = client.files.content(file_id)
            text = getattr(result, "text", None)
            if isinstance(text, str):
                return text
            if isinstance(result, bytes):
                return result.decode("utf-8")
            if isinstance(result, str):
                return result
            return ""

        return {
            "output_text": _read(status.get("output_file_id")),
            "error_text": _read(status.get("error_file_id")),
        }

    def parse_batch_results(self, raw) -> list[BatchItem]:
        """Parse downloaded batch data into :class:`BatchItem` items.

        Accepts the dict :meth:`fetch_batch_results` returns, JSONL
        text or bytes (one OpenAI result per line), or a list of
        entries (Anthropic result entries, or already-parsed lines).
        A line that is not JSON, or that carries no ``custom_id``, is
        skipped. Reasons are code-only, never provider or answer text.
        """
        documents: list = []
        if isinstance(raw, dict) and (
            "output_text" in raw or "error_text" in raw
        ):
            output = raw.get("output_text") or ""
            errors = raw.get("error_text") or ""
            text = f"{output}\n{errors}" if output and errors else output + errors
            for line in str(text).splitlines():
                line = line.strip()
                if line:
                    documents.append(line)
        elif isinstance(raw, (bytes, bytearray)):
            for line in bytes(raw).decode("utf-8").splitlines():
                line = line.strip()
                if line:
                    documents.append(line)
        elif isinstance(raw, str):
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    documents.append(line)
        elif isinstance(raw, list):
            documents.extend(raw)
        else:
            return []
        items: list[BatchItem] = []
        for document in documents:
            if isinstance(document, str):
                try:
                    parsed = json.loads(document)
                except ValueError:
                    continue
            elif isinstance(document, dict):
                parsed = document
            else:
                continue
            if not isinstance(parsed, dict):
                continue
            if "result" in parsed:
                item = _anthropic_batch_item(parsed, self.parse_response)
            else:
                item = _openai_batch_item(parsed, self.parse_response)
            if item is not None:
                items.append(item)
        return items

    def _parse_openai_line(self, line: dict) -> BatchItem | None:
        """Parse one OpenAI batch output line into a :class:`BatchItem`."""
        return _openai_batch_item(line, self.parse_response)

    def _parse_anthropic_entry(self, entry: dict) -> BatchItem | None:
        """Parse one Anthropic batch result entry into a :class:`BatchItem`."""
        return _anthropic_batch_item(entry, self.parse_response)
