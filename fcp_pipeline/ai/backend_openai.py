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
