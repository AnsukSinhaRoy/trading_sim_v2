# `analytics` — Derived artifacts from the event log

## What lives here
- `build.py`: reads `events.jsonl` and writes derived tables (Parquet/CSV).

## Architecture
Analytics is intentionally separate from the engine:

1. Read event stream
2. Filter by `kind`
3. Build derived tables:
   - NAV series
   - fills
   - positions
   - trade open/close

## Design choices (and why)
- **Post-run derivation**: you can regenerate derived tables without rerunning a backtest.
- **Parquet-first with CSV fallback**: Parquet for speed/storage, CSV for compatibility.

## Extension points
- Add performance metrics (drawdown, turnover, Sharpe), per-symbol PnL decomposition, etc.
- Add report builders (HTML/PDF) as a separate layer.
