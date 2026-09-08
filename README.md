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

## Running Experiments

The entry point is `scripts/experiments/run_experiment.py`, configured via YAML files in `configs/`.

```bash
# Run on all examples in a dataset
uv run --locked python scripts/experiments/run_experiment.py --config-name=legal_hallucination_checker_gpt

# Run a single example by filename
uv run --locked python scripts/experiments/run_experiment.py --config-name=legal_hallucination_checker_gpt \
  data.example_id=caryn-strickland-v-united-states_191787292.pdf

# Limit batch size
uv run --locked python scripts/experiments/run_experiment.py --config-name=legal_hallucination_checker_gpt \
  data.limit=5
```

**Common overrides:**
- `data.example_id=` — run a single example instead of the full dataset
- `data.limit=` — cap the number of examples in batch mode
- `agent.model.model_id=` / `agent.model.provider=` — change the LLM
- `environment.max_steps=` — max actions per episode (0 = unlimited)
- `test_run=true` — write outputs to `test/` subfolders

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
| `environment.max_steps` | Max steps per episode (0 = unlimited) |
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
