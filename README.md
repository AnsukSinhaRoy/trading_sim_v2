# Levitate Trading Stack (config-driven, modular)

This repo is a **config-driven trading simulation/backtest scaffold** built around a simple principle:

> **Events are the source of truth.** Everything else (NAV series, fills tables, positions, dashboards) is derived from the event log.

## High-level architecture

```
+-------------+     +------------+     +------------+     +-----------+
| market_feed | --> | strategy   | --> | execution  | --> | eventlog   |
+-------------+     +------------+     +------------+     +-----------+
                                                            |
                                                            v
                                                   +------------------+
                                                   | analytics + UI    |
                                                   +------------------+
```

- **market_feed** produces aligned multi-asset snapshots (e.g., 1-minute bars).
- **strategy** reads snapshots + positions and emits target orders.
- **execution** turns orders into fills (with slippage/fees) and updates positions.
- **eventlog** persists everything as append-only JSONL.
- **analytics** derives Parquet tables from the event stream.
- **UI** can subscribe to live ZMQ events for real-time monitoring.

Each package has its own README (see `*/README.md`) describing internal architecture and design rationale.

## Config-driven runs

You run using a **run YAML** that references module YAMLs:

- Example: `configs/run/demo_synth.yaml`
  - points to:
    - `configs/market_feed/synth_1m.yaml`
    - `configs/execution/paper_fixed_bps.yaml`
    - `configs/strategy/toy_rebalance.yaml`
    - `configs/ui/qt_dashboard.yaml`

The loader deep-merges these YAMLs into one config object.

## Quick start

```bash
pip install -r requirements.txt
python -m runner.run --config configs/run/demo_synth.yaml
```

This creates:

```
runs/<run_id>/
  events.jsonl
  derived/
    nav.parquet
    fills.parquet
    positions.parquet
    trades_open.parquet
    trades_closed.parquet
```

## One-command CLI: `levitate`

Install in editable mode so the `levitate` command is available:

```bash
pip install -e .
```

Run any experiment with:

```bash
levitate configs/run/demo_synth.yaml
```

Optional overrides:

```bash
levitate configs/run/demo_synth.yaml --name myrun --out-dir runs
```

You can also run via Python module (works without installing scripts):

```bash
python -m runner --config configs/run/demo_synth.yaml
# or
python -m runner configs/run/cube_demo.yaml
```

## Qt real-time dashboard (ZMQ)

The engine can publish live NAV + fills over ZMQ for a lightweight desktop dashboard.

### Install optional UI deps

Choose one:

```bash
pip install -r requirements-ui.txt
```

or:

```bash
pip install -e ".[ui]"
```

### Start the dashboard

```bash
python ui/qt_dashboard.py --url tcp://127.0.0.1:5555
```

### Run an experiment

```bash
python -m runner configs/run/cube_demo.yaml
```

### Tuning for fast backtests

If the run is very fast, the UI may look frozen because the engine produces ticks faster than Qt can draw.

Tune:

- `ui.publish_every_ticks` (in `configs/ui/qt_dashboard.yaml`)

## Data ingestion (folder of per-symbol 1m CSV)

1) Put per-symbol files into a folder (example):
   - `data/RELIANCE.csv`, `data/TCS.csv`, ...

2) Edit `configs/market_feed/folder_csv_1m.yaml`:
   - set `data_dir`
   - set `symbols` (or use autodiscover configs)
   - set `timestamp_col` / `price_col`
   - set `start` and `end` (small slice first)

3) Run:

```bash
levitate configs/run/demo_folder_csv.yaml
```

## Smoke tests

```bash
pytest -q
```


CLI override example:
```bash
python -m runner configs/run/cube_demo_ema_long.yaml --zmq-port 5560
python ui/qt_dashboard.py --url tcp://127.0.0.1:5560
```
