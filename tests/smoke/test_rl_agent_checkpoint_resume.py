from __future__ import annotations

from pathlib import Path

from strategy.rl_agent.agent import RLAgentStrategy


def test_rl_agent_checkpoint_resume(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "persistent_model"

    strat1 = RLAgentStrategy(
        device="cpu",
        checkpoint_enabled=True,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_every_steps=1,
        checkpoint_keep_last=2,
        min_history=5,
        lookback_short=5,
        lookback_long=10,
        corr_short=5,
        corr_long=10,
        recent_window=3,
        fast_window=5,
        slope_window=5,
        zscore_window=5,
    )
    strat1._decision_steps = 11
    strat1._last_reward = 0.42
    strat1._save_checkpoint()

    assert (ckpt_dir / "latest.pt").exists()
    assert (ckpt_dir / "policy_latest.pt").exists()
    assert (ckpt_dir / "meta.json").exists()

    strat2 = RLAgentStrategy(
        device="cpu",
        checkpoint_enabled=True,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_every_steps=1,
        checkpoint_keep_last=2,
        min_history=5,
        lookback_short=5,
        lookback_long=10,
        corr_short=5,
        corr_long=10,
        recent_window=3,
        fast_window=5,
        slope_window=5,
        zscore_window=5,
    )

    assert strat2._checkpoint_loaded is True
    assert strat2._decision_steps == 11
    assert abs(strat2._last_reward - 0.42) < 1e-9
