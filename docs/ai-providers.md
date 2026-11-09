# AI providers: one setting per step

Each AI step (tagging, photos, memo, organize, roughcut) is served by one
small interface. You pick the provider and model for every step in the
trip folder's `project.toml`, under `[pipeline.ai]`. No step calls an AI
service until its own issue wires it into a command; this page covers the
setting itself.

## The per-step setting

OpenAI is the default provider for every AI step. To give one step to
Claude instead, set `provider = "anthropic"` in that step's own table:

```toml
[pipeline.ai]
provider = "openai"
model = "<model-id>"
max_output_tokens = 4096

[pipeline.ai.tag]
provider = "anthropic"
model = "<model-id>"
```

With this example, the tag step asks Claude with its own model, while
photos, memo, organize, and roughcut ask OpenAI with the shared model. A
step uses its own table's values when they are set, else the shared
`[pipeline.ai]` values, else the defaults (OpenAI, 4096 output tokens).
The available step tables are `[pipeline.ai.tag]`,
`[pipeline.ai.photos]`, `[pipeline.ai.memo]`, `[pipeline.ai.organize]`,
and `[pipeline.ai.roughcut]`.

## Model ids are settings, never code

No model id appears in the code. Copy a current id from your provider's
model list into `project.toml`, replacing each `"<model-id>"`
placeholder. A step with no model, or with the placeholder still in
place, stops before anything is sent and names the key to set, for
example `[pipeline.ai.tag].model`.

## Transcription is separate

Spoken-audio steps do not use the interface above. `[pipeline.ai.memo_audio]`
and `[pipeline.ai.vo_audio]` take `provider = "openai"` or
`provider = "local"`, with `local` when unset; local always stays the
fallback and needs no model or key. They never inherit the shared
`[pipeline.ai]` provider or model:

```toml
[pipeline.ai.memo_audio]
provider = "local"

[pipeline.ai.vo_audio]
provider = "openai"
model = "<model-id>"
```

## Where keys come from

Keys come only from the environment (`OPENAI_API_KEY` for OpenAI,
`ANTHROPIC_API_KEY` for Claude) or, on a Mac, from the Keychain items
`fcp-pipeline-openai` and `fcp-pipeline-anthropic`. They are never read
from the repo, `project.toml`, or a log. See
[docs/privacy.md](privacy.md) for the full story.

## Offline refuses every call

With `offline = true` in `[pipeline]`, or `--offline` on a command,
every AI call is refused before any client is built, any key is read, or
anything is sent, and the network log records one `refused offline`
line. See [docs/privacy.md](privacy.md).

## The usage log and the price table

Every call that gets an answer appends one row to
`.pipeline/ai-usage.tsv` inside the trip folder: when the call happened
(UTC), the step, provider, model, both token counts, and the cost in US
dollars. The file never holds prompt text, answer text, image bytes, or
keys. Prices come from your own table, in dollars per million tokens:

```toml
[pipeline.ai.prices."<model-id>"]
input_per_million = 2.5
output_per_million = 10
```

A model with no price entry leaves the cost cell empty. A call that
raises adds no usage row. If the usage write itself is refused, the
answer is kept and exactly one warning is emitted instead.

## The schema check and its one retry

Every answer comes back checked against a strict JSON schema: objects
list all their properties and forbid the rest, so providers return
exactly the declared shape. An answer that fails the check is retried
once with the same request; a second failure raises an error naming the
step, the provider, and the first failed rule, never the answer text.

## Installing the optional SDKs

AI steps need the provider packages, which are not bundled:

```sh
pip install openai anthropic
```

Everything else in the pipeline still runs on the standard library.

## The opt-in live test

`tests/test_ai_live.py` is skipped unless `FCP_AI_LIVE=1`. With that set,
and a model chosen per backend, it makes one tiny text-and-image call in
a temporary project and prints the usage row's tokens and cost:

```sh
FCP_AI_LIVE=1 FCP_AI_LIVE_OPENAI_MODEL="<model-id>" \
  python3 -m unittest tests.test_ai_live -v
```

Replace the placeholder with a current model id. The Anthropic side
uses `FCP_AI_LIVE_ANTHROPIC_MODEL`.
