# `strategy` — Order generation logic

## What lives here
- `toy_rebalance.py`: simple rebalance example strategy.
- `xs_mom_vol_target.py`: cross-sectional momentum + vol targeting example.
- `__init__.py`: exports strategy symbols.

## Architecture
A strategy is called once per tick with:
- `MarketSnapshot`
- current `PositionSnapshot` / portfolio state
- configuration parameters

It returns a list of `OrderRequest` objects (desired trades).

## Design choices (and why)
- **Small interface, easy testing**: strategies are pure-ish functions of (snapshot, state).
- **Config-driven instantiation**: strategy selection happens in the engine via `strategy.type`.
- **Explicit factory**: prevents “unknown strategy” bugs by failing loudly for unsupported types.

## Extension points
- Add more strategy modules and register them in the engine’s factory.
- Add risk overlays (max leverage, sector caps, drawdown stops) as composable wrappers.
