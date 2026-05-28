# `runner` — config loader, engine loop, CLI

The runner package owns the runtime loop and the small factories that build feed,
execution, strategy, liquidity, and telemetry components.

## Files

- `config.py`: YAML loader and deep-merge logic.
- `run.py`: creates the run folder, logging, effective config, and event logger.
- `cli.py`: `levitate ...` command and preprocess/run dispatch.
- `engine.py`: compact event loop.
- `factories.py`: feed, execution, and sparse-strategy construction.
- `liquidity.py`: bar-volume and ADV-based execution constraints.
- `rebalancer.py`: target-weight to order conversion with diagnostics.
- `publisher.py`: best-effort ZMQ publisher for the dashboard.
- `telemetry.py`: strategy telemetry extraction for the `learn` topic.

## Event-loop flow

1. Build feed, paper execution, sparse strategy, and liquidity tracker.
2. Stream market snapshots.
3. Mark the paper portfolio to market.
4. Let the strategy update target weights.
5. Rebalance toward target weights with cash and liquidity constraints.
6. Execute orders, log events, and publish dashboard packets.

The strategy factory is intentionally sparse-only. Adding dynamic auto-discovery
again would make the code more clever but less readable, and it would contradict
the current simulator scope.
