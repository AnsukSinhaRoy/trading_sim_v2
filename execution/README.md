# `execution` — Portfolio bookkeeping + fill simulation

## What lives here
- `portfolio.py`: cash + integer share positions + NAV calculations.
- `slippage.py`: simple pluggable slippage/fee models.
- `paper.py`: a “paper execution” simulator producing fills from market snapshots.

## Architecture
Execution converts **`OrderRequest`** → **`Fill`** given a snapshot price:

- Determine reference price (from snapshot)
- Apply slippage + fees
- Update portfolio state
- Emit fill/trade events (consumed by analytics/UI)

## Design choices (and why)
- **Deterministic**: same inputs → same fills → reproducible runs.
- **Integer shares**: keeps the bookkeeping realistic for equity-style backtests.
- **Pluggable slippage**: easy to switch from “fixed bps” to more realistic models later.

## Extension points
- Add latency, partial fills, queue position, participation caps, etc.
- Add instrument types (futures, options) with separate margining rules.
