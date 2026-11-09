"""Anthropic backend for AI steps, using the Messages API.

Only plain dicts cross this module's boundary. The SDK itself is never
imported here: :meth:`create` builds the client through the network gate,
which refuses offline calls before any key is read, makes one call, and
converts the SDK result to a plain dict when it offers ``model_dump``.
The answer is forced into one ``answer`` tool whose input carries the
structured JSON.
"""

from __future__ import annotations

import base64
from typing import Any

from .backend_openai import Reply, _count, _get

#: One-line description attached to the forced ``answer`` tool.
TOOL_DESCRIPTION = "Return the answer as structured data matching the schema."


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
            "tools": [
                {
                    "name": "answer",
                    "description": TOOL_DESCRIPTION,
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "answer"},
        }

    def create(self, request: dict) -> dict:
        """Make one Messages call; the only method touching the SDK."""
        client = self._gate.anthropic_client()
        result = client.messages.create(**request)
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        return dict(result)

    def parse_response(self, raw) -> Reply:
        """Parse a plain Messages result into a :class:`Reply`."""
        content = _get(raw, "content") or []
        data: Any = None
        for block in content:
            if _get(block, "type") == "tool_use" and _get(block, "name") == "answer":
                data = _get(block, "input")
                break
        usage = _get(raw, "usage") or {}
        return Reply(
            data,
            _count(_get(usage, "input_tokens")),
            _count(_get(usage, "output_tokens")),
        )

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
