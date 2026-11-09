"""Provider batch jobs for AI steps, through the network gate.

This module is the batch counterpart of :func:`fcp_pipeline.ai.ask`:
:func:`prepare` builds one clip's provider request without sending
anything, while :func:`submit`, :func:`status` and :func:`results` run
the whole provider interaction as the zero-argument callable handed to
:func:`fcp_pipeline.net.NetworkGate.call`, so an offline gate refuses
before the backend builds a client, reads a key, builds the upload or
makes any call, and every call appends one network-log line. They take
the provider and model explicitly because a collect must use the job's
own provider and model, not whatever the settings say now. Nothing
here writes a file or records a usage row.
"""

from __future__ import annotations

import base64

from fcp_pipeline.net import NetworkGate

from . import (
    AnthropicBackend,
    Image,
    OpenAIBackend,
    StepConfig,
    check_schema,
    step_config,
)
from .backend_openai import BatchItem, BatchStatus

__all__ = [
    "BatchItem",
    "BatchStatus",
    "prepare",
    "results",
    "status",
    "submit",
]


def _backend_for(gate, provider: str, backends):
    """Pick the backend exactly as :func:`fcp_pipeline.ai.ask` does."""
    if backends is not None and provider in backends:
        return backends[provider]
    if provider == "openai":
        return OpenAIBackend(gate)
    return AnthropicBackend(gate)


def _check_pictures(pictures) -> tuple:
    """Check images the way :func:`fcp_pipeline.ai.ask` does."""
    checked = tuple(pictures or ())
    for picture in checked:
        if not isinstance(picture, Image):
            raise ValueError("images must be Image objects")
    return checked


def prepare(settings, step: str, instructions: str, schema: dict,
            images=(), *, backends=None) -> tuple[StepConfig, dict]:
    """Build one clip's provider request without sending anything.

    Resolves the step config first, so a missing or placeholder model
    raises :class:`AiConfigError` here, before any sheet is read for
    upload or anything is sent. Returns ``(config, request)`` where
    ``request`` is the backend's ``build_request`` dict.
    """
    config = step_config(settings, step)
    check_schema(schema)
    pictures = _check_pictures(images)
    gate = NetworkGate(settings, step)
    backend = _backend_for(gate, config.provider, backends)
    request = backend.build_request(
        config.model,
        instructions,
        pictures,
        schema,
        config.max_output_tokens,
    )
    return config, request


def _request_text_and_images(provider: str, request: dict) -> tuple:
    """Return ``(text characters, image sizes)`` of one built request."""
    text = 0
    sizes: list[int] = []
    if provider == "openai":
        content = request["input"][0]["content"]
        for part in content:
            kind = part.get("type")
            if kind == "input_text":
                text += len(part.get("text") or "")
            elif kind == "input_image":
                uri = part.get("image_url") or ""
                encoded = uri.split(",", 1)[1] if "," in uri else ""
                sizes.append(len(base64.b64decode(encoded)))
    else:
        blocks = request["messages"][0]["content"]
        for block in blocks:
            kind = block.get("type")
            if kind == "text":
                text += len(block.get("text") or "")
            elif kind == "image":
                source = block.get("source") or {}
                sizes.append(len(base64.b64decode(source.get("data") or "")))
    return text, sizes


def _sent_for_requests(provider: str, items) -> list:
    """Build the gate ``sent`` summary for ``submit``'s items."""
    total_text = 0
    sizes: list[int] = []
    for _, request in items:
        text, found = _request_text_and_images(provider, request)
        total_text += text
        sizes.extend(found)
    sent = [("text", total_text)]
    sent.extend(("image", size) for size in sizes)
    return sent


def submit(settings, step: str, provider: str, model: str, items, *,
           backends=None) -> str:
    """Submit ``items`` as one provider batch; return the batch id.

    ``items`` holds ``(custom_id, request)`` pairs where ``request`` is
    exactly what the backend's ``build_request`` returns. The upload is
    built and sent inside the gate call, so an offline gate raises
    :class:`OfflineError` before anything is built or sent.
    """
    gate = NetworkGate(settings, step)
    backend = _backend_for(gate, provider, backends)
    sent = _sent_for_requests(provider, items)

    def do_request():
        batch_input = backend.build_batch_input(items)
        return backend.create_batch(batch_input)

    return gate.call(provider, model, sent, do_request)


def status(settings, step: str, provider: str, model: str, batch_id: str, *,
           backends=None) -> BatchStatus:
    """Return one batch's :class:`BatchStatus` through the gate."""
    gate = NetworkGate(settings, step)
    backend = _backend_for(gate, provider, backends)

    def do_request():
        raw = backend.retrieve_batch(batch_id)
        return backend.parse_batch_status(raw)

    return gate.call(provider, model, [], do_request)


def results(settings, step: str, provider: str, model: str, batch_id: str, *,
            backends=None) -> list:
    """Download and parse one finished batch into :class:`BatchItem`."""
    gate = NetworkGate(settings, step)
    backend = _backend_for(gate, provider, backends)

    def do_request():
        raw_status = backend.retrieve_batch(batch_id)
        data = backend.fetch_batch_results(batch_id, raw_status)
        return backend.parse_batch_results(data)

    return gate.call(provider, model, [], do_request)
