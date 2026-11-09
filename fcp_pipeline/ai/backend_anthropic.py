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

from .backend_openai import Reply, _count, _get


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
            data: Any = json.loads(text)
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
