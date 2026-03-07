# rl_agent — PPO long-only allocator (weekly-ish rebalance)

This strategy is a **serious RL** implementation designed for *long-term*, *long-only* portfolio allocation.

## What it does

At each rebalance (default ~weekly, configurable), it:

1. Builds per-symbol features from short + long history:
   - short/long momentum, short/long volatility
   - short max-drawdown proxy
   - short/long correlation-to-market proxy (market = cross-sectional mean return)

2. Uses a **2-stage stochastic policy** (Actor-Critic) trained with **PPO**:
   - Stage A: selects K assets **without replacement** (sequential categorical sampling)
   - Stage B: allocates weights among selected assets using a **Dirichlet** distribution
   - Output weights are **long-only**, sum(weights) <= leverage.

3. Optimizes a **stability-aware** objective:
   - log-return
   - minus turnover cost penalty
   - minus volatility penalty
   - minus drawdown penalty

4. Applies a **stoploss overlay** (optional) to reduce tail risk:
   - if price falls below `entry_price * (1 - stoploss_pct)`, it exits and blocks re-entry for `stop_cooldown_minutes`.

## Engine integration (important)

The engine reads `strat._last_target_weights` after calling `on_snapshot()`.
This strategy sets `_last_target_weights` only when it wants to trade (rebalance / stoploss).

## Configuration

See `configs/strategy/rl_agent.yaml`.

Key knobs:
- `rebalance_every_minutes`: set to ~week for long-term trading
- `max_assets`: portfolio sparsity
- `stoploss_pct`: tail risk control
- `tc_penalty`, `vol_penalty`, `dd_penalty`: stability bias
- PPO params: `lr`, `clip_ratio`, `ent_coef`, `update_every`, etc.

## Notes / expectations

With weekly actions, learning is slower — you want **many weeks** of data for good results.
For faster iteration you can temporarily reduce `rebalance_every_minutes` (e.g. daily) and then scale back.
