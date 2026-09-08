# Before-refactor smoke baseline

This is a synthetic integration fixture, not a LePhantomCite accuracy result.
The input explicitly identifies an invented citation. It requires no CourtListener
or SerpAPI access. Both the non-agentic (0 steps) and agentic (3 steps) runs use
the existing `run_single_example` runner, agent, parser, environment, and recorder.

Both runs were captured and successfully replayed offline. Six API calls reported
a combined cost of $0.012440098. The agentic sequence was EDIT_SCRATCHPAD, THINK,
PROVIDE_FINAL_RESPONSE, with two intervening belief updates. All six effective
requests used temperature 0. The full saved artifacts were checked for configured
API-key values; none were present.

Synthetic fixture F1 was 2/3 for direct prediction and 1 for the agentic run.
These are regression reference values, not estimates of benchmark performance.

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
  It blocks other actions before execution, since the current feature flags alone
  do not disable every CourtListener action.
- Captures cover only the paths actually taken. They do not test external search,
  citation lookup, opinion retrieval/reading, Hydra CLI loading, batch sharding,
  or every parsing fallback. An additional retrieval baseline would be needed to
  cover those integrations.
- API keys and authentication headers are never included in the capture.
