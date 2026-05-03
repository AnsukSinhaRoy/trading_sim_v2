# RL upgrade notes

## What was fixed

- The asset-selection policy was not actually learning because the probability path used by PPO was wrapped in `no_grad`.
- The runner was passing a stale portfolio snapshot into the strategy, which weakens online learning because the agent sees outdated holdings/NAV.
- Checkpoints were previously tied to the run directory by default, which makes cross-run learning awkward. The agent now supports persistent experiment-level checkpoints.

## What was upgraded

- Deeper policy network with pooled market context.
- Richer price-derived features (22 dims).
- Persistent checkpoint layout:
  - `latest.pt` = full training state
  - `policy_latest.pt` = model weights only
  - `meta.json` = readable training status
- Optional benchmark/downside/concentration reward shaping.
- Reproducibility improvements in `runner.run` seeding Python, NumPy, and Torch.

## How to keep learning across runs on your PC

In `configs/strategy/rl_agent.yaml`, keep the same value for one of these:

```yaml
checkpoint_experiment: nifty500_main
```

or

```yaml
checkpoint_dir: "D:/trading_models/nifty500_main"
```

Run the same config again and the agent will resume from the last saved checkpoint.

## What to watch after each run

Open the checkpoint folder and inspect `meta.json`. You should see:

- `decision_steps`
- `learner_updates`
- `last_reward`

Those numbers should continue from the previous run instead of resetting to zero.
