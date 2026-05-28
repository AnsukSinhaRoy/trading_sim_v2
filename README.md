# Levitate Trading Stack — sparse-only refactor

This repository is an event-driven portfolio simulator focused on one strategy
path: **Sparse Switching Mean-Variance**. The uploaded project already had many
experimental strategy branches; this refactor removes those branches and keeps the
simulator readable around the sparse optimizer, paper execution, market feeds,
analytics, and the latest PyQt dashboard.

The core design rule remains:

> Events are the source of truth. Runtime state, dashboard views, NAV series,
> fills, positions, and analytics are derived from the event stream.

## What is still included

- Synthetic, folder-based, matrix-store, and sanitized matrix-store market feeds.
- Whole-share paper execution with fixed-bps slippage and fees.
- Volume-aware liquidity constraints and rolling ADV position caps.
- Sparse switching mean-variance strategy with support selection, restricted
  simplex optimization, and telemetry-only hindsight-return regret.
- Real-time PyQt dashboard over ZMQ, preserving the latest uploaded UI features:
  NAV controls, chart-type controls, asset analyser, return distribution,
  online regret, frictions, positions, PnL, fills, and trades.
- Event logging and derived analytics utilities.
- Preprocessing utilities for cube-store data.

## What was removed

Removed strategy paths that are no longer part of the simulator objective:

- toy rebalance
- EMA long
- cross-sectional momentum variants
- RL agent
- sparse Sortino experiment
- old patch zip files, notebooks, caches, build artifacts, and stale UI modules

## Project layout

```text
common/        Event models and JSONL event logger
market_feed/   Synthetic, folder, matrix, and sanitized matrix feeds
execution/     Paper execution, slippage, and portfolio accounting
strategy/      Sparse switching mean-variance strategy only
runner/        CLI, config loading, engine loop, factories, rebalancing, telemetry
analytics/     Post-run derived artifacts and NAV-spike audit tools
preprocess/    Data preprocessing and anomaly repair utilities
ui/            PyQt dashboard and small UI support modules
configs/       Sparse-only run/config YAMLs
```

## Quick start

```bash
pip install -r requirements.txt
python -m runner configs/run/demo_synth.yaml
```

or install the CLI:

```bash
pip install -e .
levitate configs/run/demo_synth.yaml
```

For the NIFTY cube-store sparse run:

```bash
python -m runner configs/run/cube_demo_sparse_switch_mv.yaml
```

## PyQt dashboard

Install UI dependencies:

```bash
pip install -r requirements-ui.txt
```

Start the dashboard:

```bash
python ui/qt_dashboard.py --url tcp://127.0.0.1:5555
```

Then run an experiment. To use a different ZMQ port:

```bash
python ui/qt_dashboard.py --url tcp://127.0.0.1:5560
python -m runner configs/run/cube_demo_sparse_switch_mv.yaml --zmq-port 5560
```

The dashboard reads live `nav`, `fill`, and `learn` topics. If the backtest is
faster than Qt can draw, increase `ui.publish_every_ticks` in
`configs/ui/qt_dashboard.yaml`.

## Config files

The active run configs are:

```text
configs/run/demo_synth.yaml
configs/run/cube_demo_sparse_switch_mv.yaml
```

The active strategy config is:

```text
configs/strategy/sparse_switch_mv.yaml
```

## Tests

```bash
PYTHONPATH=. pytest -q
```

A small smoke run can be created by overriding the synthetic feed to a short
number of minutes; the default synthetic config is intentionally long because it
was inherited from the uploaded project.
