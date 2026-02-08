# `configs` — Modular YAML configuration

## Layout
- `run/` : top-level run YAMLs (choose modules + run metadata)
- `market_feed/` : feed configs
- `strategy/` : strategy params
- `execution/` : execution/slippage params
- `ui/` : UI/ZMQ publish settings
- `preprocess/` : preprocessing pipeline configs

## How merging works
A run YAML contains:

```yaml
run:
  name: demo
modules:
  market_feed: ../market_feed/synth_1m.yaml
  execution:   ../execution/paper_fixed_bps.yaml
  strategy:    ../strategy/toy_rebalance.yaml
  ui:          ../ui/qt_dashboard.yaml
```

`runner.config.Config.load()` loads the run file and deep-merges each referenced module YAML
into a single dictionary.

## Design choices (and why)
- **Separation by concern**: you can swap strategy without touching feed/execution configs.
- **Relative module paths**: configs remain portable when moved as a folder.
- **Deep merge**: override only what you need (defaults + small diffs).
