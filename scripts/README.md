# `scripts` — Small helpers

## Contents
- `debug_zmq.py`: quick subscriber to inspect ZMQ messages (useful when debugging UI sync).
- `levitate.py`: thin wrapper / entrypoint convenience (CLI is also exposed via `pyproject.toml`).
- `run_demo.sh`: simple demo runner for Unix-like shells.

## Design choices (and why)
Scripts stay small and optional: the core API is in `runner`, and the canonical entrypoint is
`levitate` / `python -m runner`.
