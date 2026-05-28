# Refactor report

This refactor was applied directly to the uploaded `trading_sim_v2.zip` from the
current chat.

## Main goal

Keep the simulator behavior centered on the sparse optimization path while making
the code easier to read, navigate, and maintain.

## Structural changes

### Runner

The old `runner/engine.py` mixed factories, ZMQ publishing, liquidity controls,
rebalancing, telemetry, and the event loop. It is now split into:

- `runner/engine.py` — event loop only
- `runner/factories.py` — feed, execution, and sparse-strategy construction
- `runner/liquidity.py` — participation/ADV constraints and frictions payloads
- `runner/rebalancer.py` — target weights to orders
- `runner/publisher.py` — ZMQ publisher
- `runner/telemetry.py` — strategy telemetry extraction

Backward-compatible imports for `_rebalance`, `LiquidityConstraints`, and
`LiquidityTracker` are still exposed from `runner.engine` because existing tests
and debugging code import them from there.

### Sparse strategy

The sparse strategy is no longer one 800-line file. It is split into:

- `strategy/sparse_switch_mv.py` — dataclass, public API, and telemetry assembly
- `strategy/sparse_mv/history.py` — aggregated daily history and return matrix
- `strategy/sparse_mv/optimizer.py` — support selection and restricted optimizer
- `strategy/sparse_mv/regret.py` — hindsight-return regret telemetry
- `strategy/sparse_mv/math_utils.py` — capped-simplex projection and numeric helpers

### UI

The latest uploaded PyQt dashboard behavior was preserved. I did not replace it
with the older smaller modular dashboard because that would lose current UI
features.

Small reusable pieces were extracted from `ui/qt_dashboard.py` into:

- `ui/axis.py`
- `ui/listener.py`
- `ui/widgets.py`

The dashboard still supports direct execution:

```bash
python ui/qt_dashboard.py --url tcp://127.0.0.1:5555
```

and module execution:

```bash
python -m ui.qt_dashboard --url tcp://127.0.0.1:5555
```

## Removed scope

Removed non-sparse strategies and stale artifacts:

- RL agent strategy package
- EMA strategy
- toy rebalance strategy
- cross-sectional momentum strategies
- sparse Sortino experiment
- obsolete strategy configs and run configs
- old patch zip files
- notebooks, caches, `.git`, egg-info, debug artifacts, and stale UI modules

## Supported strategy

Only this is supported now:

```yaml
strategy:
  type: sparse_switch_mv
```

The strategy factory intentionally rejects old strategy names. That is less
flexible, but cleaner and aligned with the requested scope.

## Validation

Executed from project root:

```bash
PYTHONPATH=. pytest -q
```

Result:

```text
7 passed, 4 skipped
```

I also ran a short synthetic smoke simulation with a temporary small config. It
completed successfully.

## Important note

The default `configs/market_feed/synth_1m.yaml` still contains the long inherited
synthetic run length from the uploaded project. I did not silently shorten that
file because that would change the uploaded config behavior. Use a temporary or
custom run config for quick smoke runs.
