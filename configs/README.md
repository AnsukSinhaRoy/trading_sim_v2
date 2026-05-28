# `configs` — sparse-only YAML configuration

## Layout

- `run/`: top-level run YAMLs.
- `market_feed/`: feed configs.
- `strategy/`: sparse switching mean-variance parameters.
- `execution/`: paper execution, slippage, fee, and liquidity settings.
- `ui/`: ZMQ publish settings for the PyQt dashboard.
- `preprocess/`: data-preprocessing pipeline configs.

## How merging works

A run YAML references module YAMLs:

```yaml
run:
  name: demo_synth_sparse_switch_mv
modules:
  market_feed: ../market_feed/synth_1m.yaml
  execution:   ../execution/paper_fixed_bps.yaml
  strategy:    ../strategy/sparse_switch_mv.yaml
  ui:          ../ui/qt_dashboard.yaml
```

`runner.config.Config.load()` loads the run YAML, resolves the relative module
paths, and deep-merges the module files into one runtime config.

Only `strategy.type: sparse_switch_mv` is supported in this refactor.
