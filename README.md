# Stream Trading Stack (config-driven, modular)

This scaffold cleanly separates:
- **Market feed** (multi-asset snapshots)
- **Strategy** (signals/targets)
- **Execution** (integer shares, slippage/fees, fills)
- **Event log** (source of truth)
- **Analytics + UI** (derived from event log)

## Config-driven (separate YAMLs per module)

You run using a **run YAML** that references module YAMLs:

- `configs/run/demo_synth.yaml`
  - points to:
    - `configs/market_feed/synth_1m.yaml`
    - `configs/execution/paper_fixed_bps.yaml`
    - `configs/strategy/toy_rebalance.yaml`
    - `configs/ui/live_dashboard.yaml`

The loader merges these into one config object.

## Quick start
```bash
pip install -r requirements.txt
python -m runner.run --config configs/run/demo_synth.yaml
```

Creates:
```
runs/<run_id>/
  events.jsonl
  derived/
    nav.parquet (or nav.csv)
    fills.parquet
    positions.parquet
    trades_open.parquet
    trades_closed.parquet
```

## Live NAV dashboard
```bash
streamlit run ui/live_dashboard.py -- --run runs/<run_id>
```



## Qt real-time dashboard (ZMQ)

This repo can stream live NAV + fills over ZMQ for a lightweight desktop dashboard.

### 1) Start the Qt dashboard (in one CMD window)
```bash
python ui/qt_dashboard.py --url tcp://127.0.0.1:5555
```

### 2) Run an experiment (in another CMD window)
```bash
levitate configs/run/demo_synth.yaml
# or
python -m runner configs/run/demo_synth.yaml
```

### Tuning (important for FAST backtests)
If the run is very fast, the UI may *look* frozen because the engine produces ticks faster than Qt can draw.

You can tune:
- `ui.publish_every_ticks` (default 5): publish one NAV update every N ticks
- `market_feed.speed: realtime` (or a slower speed) to make it visibly live

See `configs/ui/live_dashboard.yaml` for these UI keys.

## Smoke tests
```bash
pytest -q
```


## One-command CLI: `levitate`

Install the repo in editable mode so the `levitate` command is available:

```bash
pip install -e .
```

Now run any experiment with:

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
python -m runner configs/run/demo_synth.yaml
```


You can also use the `--config` alias:

```bash
levitate --config configs/run/demo_synth.yaml
```


## Consuming real 1-minute data (CSV/Parquet folder)

1) Put per-symbol files into a folder (example):
   - `data/RELIANCE.csv`, `data/TCS.csv`, ...

2) Edit `configs/market_feed/folder_csv_1m.yaml`:
   - set `data_dir`
   - set `symbols`
   - set `timestamp_col` / `price_col`
   - set `start` and `end` (small slice first)

3) Run:

```bash
levitate configs/run/demo_folder_csv.yaml
```

4) Visualize:

```bash
streamlit run ui/live_dashboard.py -- --run runs/<run_id>
```


## Auto-detect symbols from your data folder (no manual symbols list)

If your folder contains files like `reliance_minute.csv`, `tcs_minute.csv`, etc,
you can omit `market_feed.symbols` completely.

Use:
- `configs/market_feed/folder_csv_1m_autodiscover.yaml`
- `configs/run/demo_folder_autodiscover.yaml`

Steps:
1) Edit `data_dir`, `start/end`, and columns (`timestamp_col`, `price_col`).
2) Run:
```bash
levitate configs/run/demo_folder_autodiscover.yaml
```

Optional universe filtering/ordering:
- set `market_feed.universe_file` to a JSON list (like your `universe_symbols.json`)
- set `market_feed.universe_mode: intersect` to use discovered ∩ universe


## Troubleshooting

### 1) Only one line in events.jsonl (initial snapshot)
If you see only:
`position_snapshot` at the start, it usually means the feed is still **preloading & aligning**
historical files (especially with hundreds of symbols), or it crashed early.

Check:
- `runs/<run_id>/run.log` (this repo writes detailed progress there)
- tail the events:
  - PowerShell: `Get-Content runs/<run_id>/events.jsonl -Wait`
  - CMD: `type runs\<run_id>\events.jsonl`

### 2) levitate command not found
Install the repo in your active venv:
```bash
pip install -e .
```
Then verify:
- CMD: `where levitate`
- PowerShell: `Get-Command levitate`

Fallback (always works):
```bash
python -m runner --config configs/run/demo_synth.yaml
```

### 3) Detach (run in background)
Use:
```bash
levitate configs/run/demo_folder_autodiscover.yaml --detach
```
Then check `./runs/<latest>/run.log`


## Why you may see only one `position_snapshot` at first

In folder-based CSV mode (`market_feed.type=folder_1m`), the feed preloads & aligns the selected
time window for all symbols before yielding the first tick. With large universes this can take time.

Check `runs/<run_id>/run.log` for progress like:
- `MarketFeed(folder_1m): loaded 25/486 symbols ...`

## Recommended: preprocess once -> matrix parquet store (fast startup)

1) Build the reusable store (one-time):
```bash
levitate configs/preprocess/nifty500_1m_store.yaml
```

This creates:
- `processed_data/nifty500/1m_long_store/` (intermediate, partitioned by date+symbol)
- `processed_data/nifty500/1m_matrix_store/` (fast feed, partitioned by date)

2) Point your experiment feed to the matrix store by editing:
`configs/market_feed/matrix_store_1m.yaml` -> `store_dir: ...`

3) Run experiments:
```bash
levitate configs/run/demo_matrix_store.yaml
```


## Cube store (OHLCV daily matrices)

Recommended for large universes (e.g., Nifty500). Preprocess once to create a daily parquet "cube":
- rows = minutes in a day
- columns = symbols
- files = one per field (open/high/low/close/volume)

Layout:
```
processed_data/nifty500/1m_cube_store/
  date=YYYY-MM-DD/
    open.parquet
    high.parquet
    low.parquet
    close.parquet
    volume.parquet
```

Build it:
```bash
python -m preprocess --config configs/preprocess/nifty500_1m_store.yaml
```

### Consuming the cube store in experiments (close-only for now)
The existing `matrix_store_1m` feed reads `close.parquet` from each `date=...` folder.
So you can point it at the cube store directory and keep strategies/execution unchanged.

Edit `configs/market_feed/matrix_store_1m.yaml`:
- set `store_dir` to `processed_data/nifty500/1m_cube_store`

Later we can extend the event schema to stream full OHLCV bars.
