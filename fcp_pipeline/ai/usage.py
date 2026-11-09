"""Per-call AI usage log: tokens and cost, never prompt or answer text.

Every backend call that returns an answer appends one row to
``<project>/.pipeline/ai-usage.tsv``: when the call happened, which step
and provider and model answered, both token counts, and the cost in US
dollars from ``[pipeline.ai.prices]``. A backend call that raises adds no
row; an answer that fails validation still got an answer, so retried calls
keep their rows even when the step finally raises. A refused usage write
keeps the answer and emits exactly one warning. The file is only ever
appended to, through the network log's hardened writer.
"""

from __future__ import annotations

import datetime
import math
import os
import warnings
from decimal import Decimal, ROUND_HALF_UP

from fcp_pipeline.net import _sanitize_field, append_pipeline_line

#: Usage file name inside ``<project>/.pipeline/``.
USAGE_NAME = "ai-usage.tsv"
#: Header written once when the usage file is created.
USAGE_HEADER = "time\tstep\tprovider\tmodel\tinput_tokens\toutput_tokens\tcost_usd\n"
#: Label naming the usage log in refusal messages.
USAGE_LABEL = "AI usage log"

#: Decimals kept in the cost column.
_COST_PLACES = Decimal("0.000001")


def _finite_rate(value):
    """Return a usable dollars-per-million rate, or None without a price.

    Anything that is not a finite number (including booleans) counts
    as no price, so the cost cell stays empty instead of failing.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return value


def cost_for_model(ai_table, model, input_tokens: int, output_tokens: int) -> str:
    """Return the call cost as a six-decimal string, or "" without a price.

    Rates come from ``[pipeline.ai.prices."<model>"]`` in dollars per
    million tokens; a model with no price entry has an empty cost cell,
    never zero.
    """
    prices = (ai_table or {}).get("prices", {}) or {}
    entry = prices.get(model)
    if not isinstance(entry, dict):
        return ""
    try:
        input_rate = entry["input_per_million"]
        output_rate = entry["output_per_million"]
    except KeyError:
        return ""
    input_rate = _finite_rate(input_rate)
    output_rate = _finite_rate(output_rate)
    if input_rate is None or output_rate is None:
        return ""
    cost = (
        Decimal(str(input_rate)) * int(input_tokens)
        + Decimal(str(output_rate)) * int(output_tokens)
    ) / Decimal(1_000_000)
    return str(cost.quantize(_COST_PLACES, rounding=ROUND_HALF_UP))


def append_usage_row(settings, step: str, provider: str, model, reply) -> None:
    """Append one usage row for a backend call that returned an answer.

    A refused write keeps the answer and emits exactly one warning naming
    the log problem, never prompt text, answer text, or keys.
    """
    folder = os.fspath(getattr(settings, "folder"))
    ai_table = getattr(settings, "ai", {}) or {}
    cost = cost_for_model(
        ai_table, model, reply.input_tokens, reply.output_tokens
    )
    now = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    line = (
        f"{now}\t{_sanitize_field(step)}"
        f"\t{_sanitize_field(provider)}\t{_sanitize_field(model)}"
        f"\t{int(reply.input_tokens)}\t{int(reply.output_tokens)}"
        f"\t{cost}\n"
    )
    try:
        append_pipeline_line(
            folder, USAGE_NAME, USAGE_HEADER, line, label=USAGE_LABEL
        )
    except Exception as exc:
        warnings.warn(str(exc), stacklevel=2)
