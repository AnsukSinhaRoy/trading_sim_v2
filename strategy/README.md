# `strategy` — Order generation logic

## What lives here
- `toy_rebalance.py`: simple rebalance example strategy.
- `xs_mom_vol_target.py`: cross-sectional momentum + vol targeting example.
- `xs_mom_vol_ema_stop.py`: momentum + vol targeting with EMA trend gate + trailing stoploss (lower churn).
- `sparse_sortino_optimizer.py`: sparse risk/return optimizer (example).
- `rl_agent/`: PPO-based long-only RL allocator (weekly-ish rebalance).

## How strategies are resolved (no factory edits needed)

The engine instantiates a strategy based on:

```yaml
strategy:
  type: <stype>
  # ...params forwarded to the strategy class constructor...
```

Where `<stype>` is resolved using these rules (in order):

1) **Explicit module + class** (always works):

```yaml
strategy:
  type: some.module.path:SomeStrategyClass
```

2) **Convention-based** (recommended):

- If `type: foo_bar`, the engine will try these import locations:
  - `strategy.foo_bar`
  - `strategy.foo_bar.strategy`
  - `strategy.foo_bar.agent`
  - (plus a few other common submodules)

- It will look for a class named like:
  - `FooBarStrategy`
  - with short acronyms preserved, e.g. `rl_agent` → `RLAgentStrategy`, `xs_mom` → `XSMomStrategy`.

- If the exact class name isn't found, and the module contains **exactly one** class whose name ends with `Strategy`,
  the engine will use that as a fallback.

## Adding a new strategy

To add `my_new_idea` without touching `runner/engine.py`:

### Option A — single file
Create:

```
strategy/my_new_idea.py
```

with:

```python
from dataclasses import dataclass
from common.events import MarketSnapshot, PositionSnapshot, OrderRequest

@dataclass
class MyNewIdeaStrategy:
    # params...

    def on_snapshot(self, snap: MarketSnapshot, portfolio: PositionSnapshot):
        return [OrderRequest(...)]
```

and set:

```yaml
strategy:
  type: my_new_idea
```

### Option B — package (folder)
Create:

```
strategy/my_new_idea/agent.py
```

with `MyNewIdeaStrategy` (or any `*Strategy` class), then set:

```yaml
strategy:
  type: my_new_idea
```

## Design choices (and why)
- **Config-driven instantiation**: you can add new strategies by dropping code under `strategy/` and pointing YAML at it.
- **Sane conventions + explicit escape hatch**: conventions keep configs simple; `module:Class` handles edge cases.
- **Fail loud**: if resolution fails, the error lists which modules were attempted and what class name was expected.
