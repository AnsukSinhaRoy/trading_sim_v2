# `runner` — Config loader, engine loop, CLI

## What lives here
- `config.py`: YAML loader + deep-merge for modular configs.
- `engine.py`: the simulation loop (feed → strategy → execution → events).
- `cli.py` / `run.py`: entrypoints (`levitate ...` and `python -m runner ...`).
- `logging_utils.py`: structured run logging into `runs/<run_id>/run.log`.

## Architecture
The engine is deliberately small and explicit:

1. Load a **run YAML** (e.g. `configs/run/demo_synth.yaml`)
2. Merge referenced module YAMLs into one config.
3. Instantiate:
   - market feed (snapshot generator)
   - strategy (order generator)
   - execution (fill simulator + portfolio bookkeeping)
4. For each tick:
   - read snapshot
   - ask strategy for orders
   - execute orders → fills
   - update portfolio → position snapshot
   - emit events to `events.jsonl`
   - (optional) publish live events over **ZMQ** for the Qt dashboard

## Design choices (and why)
- **Deep-merged modular YAMLs**: keeps configs readable and prevents “one giant config file”.
- **Event-sourced loop**: makes runs replayable and derived artifacts regeneratable.
- **ZMQ PUB/SUB for UI**: decouples UI from engine runtime and avoids tight coupling to disk.

## Key runtime knobs
- `ui.publish_every_ticks`: reduces UI pressure in fast backtests by publishing less frequently.
- Feed speed controls (for “real-time-ish” playback vs max-throughput backtests).

## Extension points
- Add new strategies by dropping a module/package under `strategy/` and setting `strategy.type` in YAML.
  No engine edits are required; the loader resolves strategies by convention (or `module:Class` for explicit paths).
- Add new execution models (e.g., latency, queueing, partial fills).