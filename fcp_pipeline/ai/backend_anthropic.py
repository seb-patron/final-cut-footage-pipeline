"""Anthropic backend for AI steps, using the Messages API.

Only plain dicts cross this module's boundary. The SDK itself is never
imported here: :meth:`create` builds the client through the network gate,
which refuses offline calls before any key is read, makes one call, and
converts the SDK result to a plain dict when it offers ``model_dump``.
The answer is requested as structured outputs: the step's JSON schema
rides in ``output_config`` and the JSON is read back from a text block.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from .backend_openai import (
    BatchItem,
    BatchStatus,
    Reply,
    _anthropic_batch_item,
    _batch_status,
    _count,
    _get,
    _openai_batch_item,
    _reject_constant,
)


class AnthropicBackend:
    """One Anthropic Messages API call through a network gate."""

    def __init__(self, gate) -> None:
        self._gate = gate

    def build_request(
        self, model: str, instructions: str, images, schema: dict,
        max_output_tokens: int,
    ) -> dict:
        """Return the pure ``messages.create`` keyword arguments."""
        content: list[dict] = []
        for image in images:
            encoded = base64.b64encode(image.data).decode("ascii")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image.media_type,
                        "data": encoded,
                    },
                }
            )
        content.append({"type": "text", "text": instructions})
        return {
            "model": model,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": content}],
            "output_config": {
                "format": {"type": "json_schema", "schema": schema}
            },
        }

    def create(self, request: dict) -> dict:
        """Make one Messages call; the only method touching the SDK."""
        client = self._gate.anthropic_client()
        result = client.messages.create(**request)
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        return dict(result)

    def parse_response(self, raw) -> Reply:
        """Parse a plain Messages result into a :class:`Reply`.

        The JSON answer is read from the response's text blocks; text
        that is not JSON, or no text at all, parses to no data, and the
        caller treats that like any other failed check.
        """
        content = _get(raw, "content") or []
        texts: list[str] = []
        for block in content:
            if _get(block, "type") == "text":
                text = _get(block, "text")
                if isinstance(text, str):
                    texts.append(text)
        usage = _get(raw, "usage") or {}
        input_tokens = _count(_get(usage, "input_tokens"))
        output_tokens = _count(_get(usage, "output_tokens"))
        text = "".join(texts)
        if not text.strip():
            return Reply(None, input_tokens, output_tokens)
        try:
            data: Any = json.loads(text, parse_constant=_reject_constant)
        except ValueError:
            data = None
        return Reply(data, input_tokens, output_tokens)

    def send(
        self, model: str, instructions: str, images, schema: dict,
        max_output_tokens: int,
    ) -> Reply:
        """Build, make, and parse one Messages call."""
        return self.parse_response(
            self.create(
                self.build_request(
                    model, instructions, images, schema, max_output_tokens
                )
            )
        )

    def build_batch_input(self, items) -> list:
        """Return the Message Batches input list for ``items``.

        ``items`` holds ``(custom_id, request)`` pairs where ``request``
        is exactly what :meth:`build_request` returns. One entry per
        item, each carrying the one-clip request as its ``params``, so
        the batch run answers the same requests the one-clip call would
        send.
        """
        return [
            {"custom_id": custom_id, "params": request}
            for custom_id, request in items
        ]

    def create_batch(self, batch_input: list) -> str:
        """Submit ``batch_input`` as one batch; the only SDK caller here.

        Makes one ``messages.batches.create`` call and returns the new
        batch id.
        """
        client = self._gate.anthropic_client()
        batch = client.messages.batches.create(requests=batch_input)
        if hasattr(batch, "model_dump"):
            batch = batch.model_dump()
        return _get(batch, "id")

    def retrieve_batch(self, batch_id: str) -> dict:
        """Return one batch object as a plain dict; touches the SDK."""
        client = self._gate.anthropic_client()
        result = client.messages.batches.retrieve(batch_id)
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        return dict(result)

    def parse_batch_status(self, raw) -> BatchStatus:
        """Map a provider batch object to a :class:`BatchStatus`.

        Shared with the OpenAI backend's mapping: Anthropic's
        ``processing_status`` (``in_progress``/``canceling`` run,
        ``ended`` is done) and OpenAI's ``status`` (``validating``,
        ``in_progress``, ``finalizing`` and ``cancelling`` run;
        ``completed`` is done; ``failed``, ``expired`` and
        ``cancelled`` keep their names). Anything else is failed.
        """
        return _batch_status(raw)

    def fetch_batch_results(self, batch_id: str, raw_status) -> list:
        """Download a batch's result entries; the only SDK caller here.

        Returns each entry of ``messages.batches.results(batch_id)``
        as a plain dict.
        """
        client = self._gate.anthropic_client()
        entries = []
        for entry in client.messages.batches.results(batch_id):
            if hasattr(entry, "model_dump"):
                entry = entry.model_dump()
            entries.append(
                dict(entry) if isinstance(entry, dict) else entry
            )
        return entries

    def parse_batch_results(self, raw) -> list[BatchItem]:
        """Parse downloaded batch data into :class:`BatchItem` items.

        Accepts the same shapes as the OpenAI backend's parsing (JSONL
        text, entry lists, or the OpenAI fetch dict), so the same data
        parses the same way on either backend.
        """
        documents: list = []
        if isinstance(raw, dict) and (
            "output_text" in raw or "error_text" in raw
        ):
            output = raw.get("output_text") or ""
            errors = raw.get("error_text") or ""
            text = (
                f"{output}\n{errors}" if output and errors else output + errors
            )
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
