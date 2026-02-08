# `market_feed` — Snapshot generators

## What lives here
- `base.py`: feed interface and shared utilities.
- `synthetic.py`: small synthetic 1m feed for demos/tests.
- `folder_1m.py`: reads per-symbol 1-minute CSVs from a folder.
- `matrix_store_1m.py`: fast feed over preprocessed “matrix store” (daily aligned matrices).

## Architecture
A feed yields a sequence of **`MarketSnapshot`** objects:

- timestamp (`ts`)
- per-symbol prices (and optionally OHLCV fields)
- enough metadata for strategies/execution to make deterministic decisions

## Design choices (and why)
- **Snapshots, not “raw bars”**: strategies and execution see a consistent view of the market.
- **Alignment up-front** (folder feed): simpler downstream logic (no per-symbol drift handling).
- **Matrix store option**: large universes (e.g., Nifty 500) start faster when data is already
  stored as daily aligned matrices.

## When to use which feed
- `synthetic`: quick sanity checks, CI, fast iteration.
- `folder_1m`: flexible when you have raw per-symbol CSVs.
- `matrix_store_1m`: fastest startup + simplest runtime when the universe is large.

## Extension points
- Add new feed types (Parquet folder, live broker feed, etc.) that still output `MarketSnapshot`.
