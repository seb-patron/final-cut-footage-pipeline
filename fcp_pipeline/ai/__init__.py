"""One interface for every AI step.

:func:`ask` sends instructions, optional images, and a JSON schema to the
step's provider and model from ``[pipeline.ai]`` in ``project.toml`` and
returns a dict checked against the schema, with one retry. OpenAI is the
default provider; Anthropic is selectable per step. Every call runs
through :class:`fcp_pipeline.net.NetworkGate`, so an offline gate refuses
before any client is built, any key is read, or any backend method runs.
Model ids live only in settings, never in code: a missing or placeholder
model stops here with the key's name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fcp_pipeline.net import NetworkGate

from .backend_anthropic import AnthropicBackend
from .backend_openai import OpenAIBackend, Reply
from .schema import AiSchemaError, check_schema, validate
from .usage import append_usage_row

#: Provider used when neither the step nor ``[pipeline.ai]`` names one.
DEFAULT_PROVIDER = "openai"
#: Token limit used when neither the step nor ``[pipeline.ai]`` sets one.
DEFAULT_MAX_OUTPUT_TOKENS = 4096
#: Placeholder shown in docs and examples; leaving it in place stops the call.
PLACEHOLDER_MODEL = "<model-id>"
#: Largest image payload accepted, in bytes.
MAX_IMAGE_BYTES = 5_000_000
#: Image media types both providers accept.
IMAGE_MEDIA_TYPES = ("image/jpeg", "image/png")
#: Steps served by :func:`ask`.
AI_STEPS = ("tag", "photos", "memo", "organize", "roughcut")
#: Transcription steps; resolved by :func:`step_config` but refused by :func:`ask`.
AUDIO_STEPS = ("memo_audio", "vo_audio")
#: Providers an AI step may name.
STEP_PROVIDERS = ("openai", "anthropic")
#: Providers a transcription step may name; "local" needs no model or key.
AUDIO_PROVIDERS = ("openai", "local")


class AiConfigError(ValueError):
    """Raised when a step's provider or model setting is missing or bad."""


class AiAnswerError(RuntimeError):
    """Raised when a checked answer still fails validation after one retry."""


@dataclass(frozen=True)
class Image:
    """One image input: raw bytes plus its media type."""

    data: bytes
    media_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise ValueError("image data must be bytes")
        if self.media_type not in IMAGE_MEDIA_TYPES:
            raise ValueError(
                f"unsupported image media type: {self.media_type!r}"
            )
        if len(self.data) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"image is {len(self.data)} bytes, "
                f"over the {MAX_IMAGE_BYTES} byte limit"
            )


@dataclass(frozen=True)
class StepConfig:
    """Resolved provider, model, and token limit for one step."""

    step: str
    provider: str
    model: str | None
    max_output_tokens: int


def _audio_config(settings, step: str, table: dict, tokens: int) -> StepConfig:
    """Resolve a transcription step.

    It never inherits the ``[pipeline.ai]`` provider or model; the token
    limit falls back to the shared default.
    """
    provider = table.get("provider", "local")
    if provider not in AUDIO_PROVIDERS:
        raise AiConfigError(
            f"bad provider for step {step!r}: set "
            f"'[pipeline.ai.{step}].provider' to \"openai\" or \"local\""
        )
    if provider == "local":
        return StepConfig(step, provider, None, tokens)
    model = table.get("model")
    if not model or model == PLACEHOLDER_MODEL:
        raise AiConfigError(
            f"missing model for step {step!r}: set "
            f"'[pipeline.ai.{step}].model' in project.toml"
        )
    return StepConfig(step, provider, model, tokens)


def step_config(settings, step: str) -> StepConfig:
    """Resolve ``step`` to its provider, model, and token limit.

    A step uses its own table's values, else ``[pipeline.ai]``'s, else
    the defaults. A ``[pipeline.ai]`` model is only inherited when the
    step's provider equals the ``[pipeline.ai]`` provider, so an OpenAI
    model is never sent to Anthropic or the other way round. Audio steps
    resolve to ``"openai"`` or ``"local"`` and need a model only with
    ``"openai"``.
    """
    ai_table = getattr(settings, "ai", {}) or {}
    if step in AUDIO_STEPS:
        table = ai_table.get(step, {}) or {}
        tokens = table.get(
            "max_output_tokens",
            ai_table.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
        )
        return _audio_config(settings, step, table, tokens)
    if step not in AI_STEPS:
        raise AiConfigError(f"unknown AI step: {step!r}")
    table = ai_table.get(step, {}) or {}
    default_provider = ai_table.get("provider", DEFAULT_PROVIDER)
    provider = table.get("provider", default_provider)
    if provider not in STEP_PROVIDERS:
        raise AiConfigError(
            f"bad provider for step {step!r}: set "
            f"'[pipeline.ai.{step}].provider' to \"openai\" or \"anthropic\""
        )
    own_model = table.get("model")
    default_model = ai_table.get("model")
    if own_model is not None:
        # A placeholder left in the step's own slot never inherits the
        # default: the step key still needs a real model id.
        if not own_model or own_model == PLACEHOLDER_MODEL:
            raise AiConfigError(
                f"missing model for step {step!r}: set "
                f"'[pipeline.ai.{step}].model' in project.toml"
            )
        model = own_model
    elif (
        provider == default_provider
        and default_model
        and default_model != PLACEHOLDER_MODEL
    ):
        model = default_model
    else:
        raise AiConfigError(
            f"missing model for step {step!r}: set "
            f"'[pipeline.ai.{step}].model' in project.toml"
        )
    tokens = table.get(
        "max_output_tokens",
        ai_table.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
    )
    return StepConfig(step, provider, model, tokens)


def _problems(data: Any, schema: dict) -> list[str]:
    """Validate ``data``; a missing answer counts as one problem."""
    if data is None:
        return ["no answer data in response"]
    return validate(data, schema)


def ask(
    settings,
    step: str,
    instructions: str,
    schema: dict,
    *,
    images=(),
    backends=None,
) -> dict:
    """Send one step request and return the checked answer dict.

    ``backends`` maps provider names to backend objects and replaces the
    real ones in tests. Each attempt runs as
    ``gate.call(provider, model, sent, request)``, where ``sent`` is
    ``[("text", len(instructions))]`` plus one ``("image", size)`` pair
    per image and ``request`` is a zero-argument callable doing the
    backend call, so an offline gate refuses before any client is built,
    any key is read, or any backend method runs. An answer with problems
    is retried once with the same request; a second failure raises
    :class:`AiAnswerError` naming the step, the provider, and the first
    problem, never the answer text.
    """
    config = step_config(settings, step)
    if step in AUDIO_STEPS or config.provider == "local":
        raise AiConfigError(
            f"step {step!r} is transcription; ask() serves "
            f"{', '.join(AI_STEPS)} only"
        )
    check_schema(schema)
    pictures = tuple(images or ())
    for picture in pictures:
        if not isinstance(picture, Image):
            raise ValueError("images must be Image objects")
        if picture.media_type not in IMAGE_MEDIA_TYPES:
            raise ValueError(
                f"unsupported image media type: {picture.media_type!r}"
            )
        if len(picture.data) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"image is {len(picture.data)} bytes, "
                f"over the {MAX_IMAGE_BYTES} byte limit"
            )
    gate = NetworkGate(settings, step)
    if backends is not None and config.provider in backends:
        backend = backends[config.provider]
    elif config.provider == "openai":
        backend = OpenAIBackend(gate)
    else:
        backend = AnthropicBackend(gate)
    sent = [("text", len(instructions))]
    sent.extend(("image", len(picture.data)) for picture in pictures)

    def do_request():
        # Built inside the gate: offline refuses before any backend
        # method runs. The retry rebuilds the same request from the
        # same inputs.
        raw = backend.create(
            backend.build_request(
                config.model,
                instructions,
                pictures,
                schema,
                config.max_output_tokens,
            )
        )
        reply = backend.parse_response(raw)
        append_usage_row(settings, step, config.provider, config.model, reply)
        return reply

    reply = gate.call(config.provider, config.model, sent, do_request)
    problems = _problems(reply.data, schema)
    if problems:
        reply = gate.call(config.provider, config.model, sent, do_request)
        problems = _problems(reply.data, schema)
        if problems:
            raise AiAnswerError(
                f"AI answer for step {step!r} from provider "
                f"{config.provider!r} failed checks: {problems[0]}"
            )
    return reply.data


__all__ = [
    "AI_STEPS",
    "AUDIO_STEPS",
    "AUDIO_PROVIDERS",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_PROVIDER",
    "IMAGE_MEDIA_TYPES",
    "MAX_IMAGE_BYTES",
    "PLACEHOLDER_MODEL",
    "STEP_PROVIDERS",
    "AiAnswerError",
    "AiConfigError",
    "AiSchemaError",
    "AnthropicBackend",
    "Image",
    "OpenAIBackend",
    "Reply",
    "StepConfig",
    "ask",
    "check_schema",
    "step_config",
    "validate",
]
