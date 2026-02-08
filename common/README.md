# `common` — Event model + event log

## What lives here
- `events.py` defines the **event types** exchanged between components:
  - `MarketSnapshot` (what the feed produces each tick)
  - `OrderRequest` (what strategies request)
  - `Fill` + `PositionSnapshot` and trade open/close events (what execution/engine emits)
- `eventlog.py` implements a tiny **append-only JSONL event store**.

## Architecture
The repository is intentionally **event-sourced**:

1. Engine emits typed events (fills, position snapshots, trade open/close).
2. `EventLogger` appends them to `runs/<run_id>/events.jsonl`.
3. Analytics and UIs treat the log as the single source of truth.

## Design choices (and why)
- **JSONL (one JSON per line)**: easy to stream, tail, diff, and recover after crashes.
- **Loose schema at the edges**: events are plain dicts on disk, but are created from
  strongly-typed structures in code. This keeps the runtime flexible while keeping the core
  logic readable.
- **No in-place mutations**: append-only logs make debugging and reproducibility much easier.

## Extension points
- Add new `kind` values for new instruments, risk events, or diagnostics.
- Keep new events backward-compatible: prefer adding fields rather than changing meaning.
