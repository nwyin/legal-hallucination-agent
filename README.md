# Legal Hallucination Agent

An LLM agent that verifies citations, quotes, and holdings in legal briefs. Given a brief, the agent iteratively searches for and checks each citation using web search and CourtListener, then produces a final verdict on whether each citation is hallucinated.

## How It Works

The agent uses **Bayesian Optimal Experimental Design (BOED)** to strategically select which citation to investigate next. It maintains a running list of citations found in the brief, tracking verification status (pending / verified / hallucinated) for each. The belief state is represented in natural language and updated after each observation, so the agent is less likely to forget previously checked citations.

**Available actions:**
- `THINK` — internal reasoning step
- `OPEN_WEB_SEARCH` — general web search
- `OPEN_COURTLISTENER_SEARCH` — search CourtListener case database
- `ACCESS_COURTLISTENER_OPINION` — fetch a full court opinion by ID
- `SEARCH_LOCAL_OPINION` — search within a previously fetched opinion
- `READ_DOCUMENT` — read a line-windowed slice of a fetched opinion
- `EDIT_SCRATCHPAD` — append, insert, replace, or clear the agent's working scratchpad
- `PROVIDE_FINAL_RESPONSE` — submit final verdict on whether citations are hallucinated

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create
the project environment from the committed lockfile:

```bash
uv sync --locked
```

The project currently targets Python 3.12.

Use `uv add <package>` and `uv remove <package>` to manage dependencies.
After editing `pyproject.toml` manually, run `uv lock` and `uv sync`.

Set the required key(s) as environment variables, especially:

- `OPENROUTER_API_KEY` (for all LLM calls)

## Langfuse tracing

Langfuse tracing is enabled when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`
are configured. Add these to `.env` or export them before starting the runner:

```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development
```

Use your project's region or self-hosted URL for `LANGFUSE_BASE_URL` (for example,
`https://us.cloud.langfuse.com` for US Cloud). The normal experiment commands
produce one `verify-legal-brief` agent trace per example, containing action
selection, belief updates, predictions, retrieval/tool results, and individual
OpenAI SDK generations with model, token usage, latency, and provider errors.
Dataset, example, method, and final precision/recall/F1 appear as metadata.
Results include `langfuse_trace_id` for correlation. The CLI flushes pending
observations in `finally`; callers embedding `run_single_example` should call
`benchmark_agent.tracing.flush_traces()` before exiting.

Brief text, model prompts/responses, and tool content are captured by default.
Configured environment credentials are masked in exported span attributes;
this does **not** anonymize legal text or detect arbitrary PII. Set
`LANGFUSE_CAPTURE_CONTENT=false` to redact inputs, outputs, metadata, and error
messages while retaining model usage and timing. Set `LANGFUSE_TRACING_ENABLED=false`
to disable tracing altogether. An explicit `OTEL_SDK_DISABLED=true` also disables
tracing and must be removed to see traces. Guardrails usage telemetry is disabled
separately so it does not require disabling all OpenTelemetry.

Optional SDK settings include `LANGFUSE_SAMPLE_RATE` (0–1) and `LANGFUSE_RELEASE`.
See Langfuse's [tracing best practices](https://langfuse.com/docs/observability/best-practices).

Verify the real SDK integration using recorded synthetic model responses:

```bash
# Offline: blocks network and checks exported span structure and token capture
uv run --locked python scripts/smoke/tracing.py

# Sends only synthetic test traces to the configured Langfuse project;
# model responses are replayed, with no model API calls or charges
uv run --locked python scripts/smoke/tracing.py --export
```

## Running Experiments

The entry point is `python -m benchmark_agent.run`, configured via YAML files in `configs/`.

```bash
# Run on all examples in a dataset
uv run --locked python -m benchmark_agent.run --config-name=legal_hallucination_checker_gpt

# Run a single example by filename
uv run --locked python -m benchmark_agent.run --config-name=legal_hallucination_checker_gpt \
  data.example_id=caryn-strickland-v-united-states_191787292.pdf

# Limit batch size
uv run --locked python -m benchmark_agent.run --config-name=legal_hallucination_checker_gpt \
  data.limit=5
```

**Common overrides:**
- `data.example_id=` — run a single example instead of the full dataset
- `data.limit=` — cap the number of examples in batch mode
- `agent.model.model_id=` / `agent.model.provider=` — change the LLM
- `environment.max_steps=` — max actions per episode (0 = non-agentic baseline: direct prediction without tool actions or belief updates)
- `test_run=true` — write outputs to `test/` subfolders

## Regression Checks

Both checks are offline: they make no API calls and block network access.

```bash
# Replay the recorded OpenRouter baseline end to end
uv run --locked python scripts/smoke/baseline.py replay --directory reference_data/openrouter_baseline

# Check the retrieval paths the replay does not cover (imports, action dispatch)
uv run --locked python scripts/smoke/offline_checks.py
```

Replay rewrites timestamped artifacts under `reference_data/`; discard those
changes rather than committing them over the reference data.

## Configuration

Two example configs are provided in `configs/`:

### `legal_hallucination_checker_gpt.yaml`
For OpenRouter models:

```yaml
agent:
  model:
    provider: "openrouter"
    model_id: "openai/gpt-4o"
    temperature: 0.8
    max_tokens: 10000
```

### `legal_hallucination_checker_gptoss.yaml`
Another example using a different OpenRouter model ID:

```yaml
agent:
  model:
    provider: "openrouter"
    model_id: "deepseek/deepseek-r1"
    temperature: 0.8
    max_tokens: 4096
```

Use any OpenRouter model ID you have access to.

### Key config options

| Key | Description |
|-----|-------------|
| `data.dataset_path` | Path to JSONL dataset |
| `data.output_path` | Path to write output JSONL |
| `data.id_field` | Field in JSONL used as example ID |
| `data.skip_completed` | Skip examples already in output file |
| `agent.model.provider` | LLM provider (`openrouter`) |
| `agent.thinking_enabled` | Enable extended thinking / chain-of-thought |
| `agent.open_web_search_enabled` | Enable web search tool |
| `agent.courtlistener_search_enabled` | Enable CourtListener search |
| `agent.courtlistener_opinion_access_enabled` | Enable full opinion retrieval |
| `environment.max_steps` | Max actions per episode (0 = non-agentic baseline: direct prediction without tool actions or belief updates) |
| `search.top_k` | Number of results returned per search |

## Dataset Format

The dataset is a JSONL file where each line is a JSON object representing one legal brief to check. The field used as the example ID is set via `data.id_field` (defaults to `filename`).

## Output Structure

```
outputs/
└── run_checker.log           # Main run log
    search_results.log        # Search queries and results

metrics/
└── legal_hallucination_checker/{provider}/{model_id}/{method}/
    └── {example_id}.json     # Per-episode metrics
```
