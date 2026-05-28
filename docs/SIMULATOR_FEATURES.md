# Simulator feature inventory — sparse-only refactor

This document describes the current uploaded-code refactor. It does **not** list
removed experimental branches.

## Core runtime

The simulator is event-driven. Market feeds produce `MarketSnapshot` events,
`PaperExecutionEngine` marks the portfolio and executes orders, `EventLogger`
writes JSONL events, and the PyQt dashboard subscribes to live ZMQ packets.

The runner is split by concern:

- `runner/engine.py`: orchestration loop only.
- `runner/factories.py`: feed, execution, and sparse-strategy construction.
- `runner/rebalancer.py`: target weights to executable order requests.
- `runner/liquidity.py`: participation and ADV caps.
- `runner/publisher.py`: ZMQ publication.
- `runner/telemetry.py`: strategy telemetry extraction.

## Market feeds

Supported feed types:

- `synthetic_1m`
- `folder_1m`
- `matrix_store_1m`
- `sanitized_matrix_store_1m`

The sanitized matrix feed protects the simulator from absurd minute-to-minute
jumps and invalid prices while preserving the event-driven interface.

## Execution model

The active execution model is paper execution with:

- initial cash
- whole-share constraints
- fixed-bps slippage
- fixed-bps fees
- missing-price handling
- position snapshots
- fill events

Liquidity constraints are optional and data-driven:

- max participation per minute bar
- max position as a fraction of rolling ADV
- separate buy/sell behavior for missing volume
- diagnostics published to the dashboard's Frictions tab

## Strategy scope

The only supported strategy is:

```yaml
strategy:
  type: sparse_switch_mv
```

`SparseSwitchMVStrategy` performs:

1. minute snapshot ingestion;
2. daily close aggregation;
3. return-matrix construction;
4. mean and covariance estimation;
5. sparse support selection;
6. restricted long-only simplex optimization;
7. target-weight publication for engine-side rebalancing;
8. optional telemetry-only hindsight-return regret.

The implementation is split across:

- `strategy/sparse_switch_mv.py`
- `strategy/sparse_mv/history.py`
- `strategy/sparse_mv/optimizer.py`
- `strategy/sparse_mv/regret.py`
- `strategy/sparse_mv/math_utils.py`

## Dashboard

The PyQt dashboard is still the latest uploaded dashboard, not the older modular
prototype. It includes:

- overview NAV cards
- NAV plot controls
- chart type/window controls
- positions table with row numbers
- fills table
- PnL table
- trades table and inspector
- online parameters/regret panels
- return distribution with VaR/CVaR/skewness/kurtosis
- frictions and liquidity diagnostics
- asset analyser

Reusable UI helpers live in:

- `ui/axis.py`
- `ui/listener.py`
- `ui/widgets.py`

## Analytics

`analytics/build.py` can derive tables from `events.jsonl`. `analytics/nav_spike_audit.py`
keeps the NAV-jump debugging path separate from the engine.

## Removed scope

The refactor removes old non-sparse strategy branches and stale artifacts so the
codebase matches the current simulator objective.
