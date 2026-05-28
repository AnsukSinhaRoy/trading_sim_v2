# `strategy` — sparse allocation only

The strategy package now contains one live strategy:

```text
strategy/sparse_switch_mv.py
strategy/sparse_mv/
```

`SparseSwitchMVStrategy` consumes minute snapshots, aggregates them into daily
closes, estimates mean/covariance from the aggregated return history, selects a
sparse support, solves a restricted long-only allocation on that support, and
exposes target weights to the engine.

## Internal modules

- `sparse_switch_mv.py`: dataclass, public strategy API, and telemetry assembly.
- `sparse_mv/history.py`: daily aggregation and return-matrix construction.
- `sparse_mv/optimizer.py`: support selection and restricted simplex optimizer.
- `sparse_mv/regret.py`: telemetry-only hindsight-return regret diagnostics.
- `sparse_mv/math_utils.py`: finite-value helpers, capped-simplex projection,
  and covariance condition-number helper.

## Supported config

```yaml
strategy:
  type: sparse_switch_mv
```

The runner intentionally rejects old strategy names. That is deliberate: keeping
unused strategies in the factory made the code harder to understand and easier to
break.
