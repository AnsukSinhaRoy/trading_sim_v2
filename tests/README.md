# `tests` — Smoke checks

This repo keeps tests lightweight (fast feedback for a scaffold project).

## What’s tested
- basic config loading
- a short demo run produces an event log
- derived artifacts build without crashing

## Design choices (and why)
- **Smoke tests** catch wiring/packaging regressions early.
- Heavy performance/profitability assertions are intentionally out of scope here; those belong
  in strategy-specific research notebooks or dedicated evaluation harnesses.
