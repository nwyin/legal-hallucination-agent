# Legal Hallucination Agent

An LLM agent that verifies citations, quotes, and holdings in legal briefs. Given a brief, the agent iteratively searches for and checks each citation using web search and CourtListener, then produces a final verdict on whether each citation is hallucinated.

## Research Context and Attribution

This project is aligned with the methodology from the paper [Who Checks the Citations? Benchmarking Legal Hallucination Detection](https://arxiv.org/html/2606.21155v2) (Liu, Stammbach, Henderson, 26 Jun 2026, arXiv:2606.21155v2).

The paper introduces:
- A taxonomy of legal citation hallucination types from real court filing failures.
- The LePhantomCite dataset of 1,300 brief excerpts with injected hallucinations.
- Agentic and non-agentic benchmarking of models for citation verification using tools like CourtListener, showing strong gains from retrieval-guided verification but persistent weakness on subtle citation errors.

## Paper datasets used for benchmark runs

- **Primary benchmark dataset:** LePhantomCite (1,300 excerpts), introduced in the paper.
- Dataset composition:
  - 1,000 excerpts from federal appellate briefs with injected hallucinations.
  - 300 excerpts from LLM-generated holdings (Dahl et al., 2024), cross-verified in the paper.
- Public split names used by the dataset card: `aux_train` (910 rows) and `eval` (390 rows), corresponding to the paper’s 70/30 train/evaluation split.
- The paper’s benchmark evaluations are run on the `eval` split (390 examples).
- The paper also references a separate controlled study over 92 drafting prompts to measure hallucination rates across ChatGPT generations (not the LePhantomCite benchmark split).

## Pulling LePhantomCite and running the benchmark here

The dataset in this repo is expected in JSONL rows with:
- `text` (brief excerpt text)
- `list_hallucinations` (dict or list of hallucinated spans/types)
- `filename` (used as example id)

You can normalize from Hugging Face into the expected structure with:

```bash
uv add datasets

uv run --locked python - <<'PY'
import json
from datasets import load_dataset

dataset = load_dataset("ai-law-society-lab/Legal_Phantom_Citation", split="eval")

with open("data/LePhantomCite-eval.jsonl", "w", encoding="utf-8") as f:
    for row in dataset:
        f.write(
            json.dumps(
                {
                    "filename": row.get("filename"),
                    "text": row.get("text", ""),
                    "list_hallucinations": row.get("hallucinations", {}),
                    "list_hallucination_types": row.get("list_hallucination_types", []),
                    "citations_in_segment": row.get("citations_in_segment", []),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

print("Wrote data/LePhantomCite-eval.jsonl")
PY
```

Then run the benchmark:

```bash
uv run --locked python -m benchmark_agent.run \
  --config-name=legal_hallucination_checker_gpt \
  data.dataset_path=data/LePhantomCite-eval.jsonl \
  data.id_field=filename \
  +dataset=legal_phantom_eval \
  environment.max_steps=30
```

To run a single example:

```bash
uv run --locked python -m benchmark_agent.run \
  --config-name=legal_hallucination_checker_gpt \
  data.dataset_path=data/LePhantomCite-eval.jsonl \
  data.id_field=filename \
  data.example_id=<filename>
```

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

Set the required key(s) as environment variables:

- `OPENROUTER_API_KEY` (for all LLM calls)
- `SERPAPI_API_KEY` (for `OPEN_WEB_SEARCH`; required when `agent.open_web_search_enabled` is true)
- `COURTLISTENER_API_KEY` (optional; raises CourtListener rate limits)

## Langfuse tracing

Every experiment requires Langfuse credentials and complete capture. Configure `.env`
or export these variables before running:

```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development
```

Create a project API key pair in Langfuse's **Settings → API Keys**. Use your
project's regional or self-hosted URL; `LANGFUSE_HOST` is also accepted.
The runner checks authentication before making model calls. Missing credentials,
disabled tracing (`OTEL_SDK_DISABLED` / `LANGFUSE_TRACING_ENABLED`), content redaction
(`LANGFUSE_CAPTURE_CONTENT=false`), and sampling below 1 are rejected.

Each run has a UUID printed as `Langfuse run/session`. Find it in Langfuse Sessions
or filter observations by session ID. Each example produces a `verify-legal-brief`
trace; an `experiment-summary` trace in the same session lists example IDs and episode
trace IDs. Local metrics include `langfuse_trace_id` and `langfuse_run_id`.
Precision, recall, F1, and step count are numeric **scores** on episode traces;
aggregate matching counts, precision/recall/F1, example count, and error count are
scores on the summary trace (including single-example runs).

Capture includes exact model requests and full responses (including provider fields),
SDK generations with tokens and latency, selected actions, validation failures and
re-asks, beliefs, predictions, and errors. Retrieval child spans retain raw HTTP
response bodies before filtering, including full opinions and external search
results. Tool spans separately record structured results and `agent_observation`,
the exact rendered observation. Full opinions remain out of the model-facing snippet
observation. Deleting the runtime opinion cache does not delete telemetry.

Run metadata contains the resolved configuration, OpenRouter/model identifiers,
dataset file SHA-256 (the revision of the actual input bytes), per-example SHA-256
and ID, Git revision/dirty state, source hashes, Python, and installed dependency
versions. For embedded calls without a dataset file, the example hash identifies
input content. Credential fields and configured environment secret values are
redacted; this does not anonymize legal text. Only send benchmark inputs you intend
to store in the configured Langfuse project.

Both the CLI and embedded `run_single_example` calls flush after ended observations,
on success and failure. Capture/export failures raise an error; simultaneous execution
and export failures retain both exceptions. SDK background score/media failures and
OTLP export rejection also fail the run. Only episodes with a successful export
receive the local `langfuse_exported` marker used by `skip_completed`; older metrics
without that marker will be rerun. Server-side indexing can lag a successful
flush by minutes; remote readback is separate from the fast smoke command. See [Langfuse tracing best practices](https://langfuse.com/docs/observability/best-practices).

```bash
# Fast live OpenRouter + Langfuse smoke; writes a report and trace link
uv run --locked python scripts/smoke/live.py

# Direct-prediction path, or choose a different model
uv run --locked python scripts/smoke/live.py --steps 0
uv run --locked python scripts/smoke/live.py --model google/gemini-2.5-flash-lite
```

The live smoke uses one invented example, a two-step budget, 1,024 output tokens
per call, a maximum of eight model calls, 20-second request timeouts, and no SDK
retries. It explicitly limits actions to THINK, EDIT_SCRATCHPAD, and
PROVIDE_FINAL_RESPONSE. Its configuration records these restrictions. This is a
connectivity and trajectory smoke test, not a paper-accuracy benchmark; it does
not exercise retrieval tools.

For an unexplained, dramatic paper-result divergence, use the stored requests,
retrieval evidence, observations, and scores to investigate fidelity. Exhaustive
paper reproduction is not required to validate telemetry or the fast smoke test.

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
- `agent.model.model_id=` — change the LLM
- `environment.max_steps=` — max actions per episode (0 = non-agentic baseline: direct prediction without tool actions or belief updates)
- `test_run=true` — write outputs to `test/` subfolders

## Code style

Ruff is installed with the development dependencies by `uv sync --locked`.

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .

# Apply lint fixes and formatting
uv run --locked ruff check . --fix
uv run --locked ruff format .
```

The configuration targets Python 3.12. Long prompt strings, intentional en dashes,
and imports that must follow environment setup have documented exceptions in
`pyproject.toml`.

## Configuration

Two example configs are provided in `configs/`:

### `legal_hallucination_checker_gpt.yaml`
For OpenRouter models:

```yaml
agent:
  model:
    model_id: "openai/gpt-4o"
    temperature: 0.8
```

### `legal_hallucination_checker_gptoss.yaml`
Another example using a different OpenRouter model ID:

```yaml
agent:
  model:
    model_id: "deepseek/deepseek-r1"
    temperature: 0.8
```

Use any OpenRouter model ID you have access to. Set token budgets under
`agent.max_tokens.action_selection`, `agent.max_tokens.belief_update`, and
`agent.max_tokens.prediction`.

### Key config options

| Key | Description |
|-----|-------------|
| `data.dataset_path` | Path to JSONL dataset |
| `data.id_field` | Field in JSONL used as example ID |
| `data.skip_completed` | Skip examples already in output file |
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
├── {dataset}/{method}/batch_results.json
└── experiments/{dataset}/{model_id}/{method}/
    └── {example_id}_{timestamp}.log  # When logging.experiment_logs is enabled

metrics/
└── {dataset}/{model_id}/{method}_steps{max_steps}/
    └── {example_id}.json     # Per-episode metrics
```
