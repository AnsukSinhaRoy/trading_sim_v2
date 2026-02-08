# `preprocess` — One-time data preparation for large universes

## What lives here
- `build.py`: CLI pipeline that converts raw per-symbol CSVs into fast stores.
- `__main__.py`: enables `python -m preprocess ...`.

## Architecture (two-stage)
This module exists because “load 500 CSVs and align them” can dominate runtime.

### Stage A — Long store
- Reads raw CSVs in chunks
- Normalizes columns (`ts`, `symbol`, requested OHLCV fields)
- Writes partitioned Parquet:
  - `.../1m_long_store/date=YYYY-MM-DD/symbol=XYZ/*.parquet`

### Stage B — Cube store (daily matrices)
- For each date, pivots long data into matrices:
  - rows = minutes
  - cols = symbols
- Writes:
  - `.../1m_cube_store/date=YYYY-MM-DD/<field>.parquet`

Feeds like `matrix_store_1m` can read these matrices extremely fast.

## Design choices (and why)
- **Chunked CSV loading**: avoids blowing RAM on very large files.
- **Partitioned Parquet**: makes incremental rebuilds and date-slicing cheap.
- **Explicit fill rules per field**: e.g., volume zeros, price forward-fill.

## Extension points
- Add corporate-action adjustments, holidays, timezone normalization, or resampling.
