# OpenRouter smoke baseline

This is a synthetic integration fixture, not a LePhantomCite accuracy result.
The input explicitly identifies an invented citation. It requires no CourtListener
or SerpAPI access. Both the non-agentic (0 steps) and agentic (3 steps) runs use
the existing `run_single_example` runner, agent, parser, environment, and recorder.

Both runs were captured and successfully replayed offline. Four API calls reported
a combined cost of $0.00731182. The agentic sequence was THINK, PROVIDE_FINAL_RESPONSE,
with one intervening belief update. All four effective requests used temperature 0
and seed 42; every requested call now carries `seed` as well. The full saved
artifacts were checked for configured API-key values; none were present.

Synthetic fixture F1 was 1 for both direct prediction and the agentic run.
These are regression reference values, not estimates of benchmark performance.

Re-recorded after removing Guardrails so that action-selection requests forward
`seed` (the Guardrails wrapper had dropped it). Live model output is not
deterministic, so the captured trajectory differs from the original capture.

## Replay after refactoring

```sh
.venv/bin/python scripts/smoke/baseline.py replay --directory reference_data/openrouter_baseline
```

Replay makes no API calls and blocks socket connections. It checks every outgoing
model request against the capture, consumes every response, and compares the full
serialized episode summary (including actions, observations, beliefs, predictions,
and scores). A moved metrics-file path is excluded; timestamped metrics files and
logs are retained for inspection but are not compared byte for byte.

If modules move, update imports and the runner adapter in the smoke script; keep
the captured fixture files unchanged. A new live capture requires a new directory.

## Contents

- `manifest.json`: starting commit, Python version, and source/lockfile hashes.
- `config.json`, `input.json`: exact settings and synthetic input.
- `steps_0.json`, `steps_3.json`: ordered original/effective SDK request arguments
  and complete SDK response objects, including usage and provider metadata.
- `record/`: execution log, per-episode metrics, serialized summaries, results,
  and two offline scoring cases (empty lists and substring matching).
- `replay/`: the same outputs generated from captured responses.

## Harness-specific settings and limits

- All effective requests use temperature 0 and seed 42; Kimi reasoning is disabled
  to keep the smoke run small. Live model responses are not guaranteed deterministic.
- The existing prediction code requests temperature 0.1. The harness overrides it
  at the SDK boundary, and records both versions; production code is unchanged.
- The harness allows only THINK, EDIT_SCRATCHPAD, and PROVIDE_FINAL_RESPONSE.
  The agent re-asks the model when it selects an action outside its action space;
  the harness additionally blocks any such action before execution, since the
  feature flags alone do not disable every CourtListener action.
- Captures cover only the paths actually taken. They do not test external search,
  citation lookup, opinion retrieval/reading, Hydra CLI loading, batch sharding,
  or every parsing fallback. An additional retrieval baseline would be needed to
  cover those integrations.
- API keys and authentication headers are never included in the capture.
