from __future__ import annotations

import torch

from strategy.rl_agent.buffer import Transition
from strategy.rl_agent.learner import Learner, PPOConfig
from strategy.rl_agent.policy import PolicyNet


def test_selection_head_receives_gradient_and_updates() -> None:
    torch.manual_seed(7)

    policy = PolicyNet(in_dim=4, hidden=32, depth=2, dropout=0.0)
    torch.set_num_threads(1)
    learner = Learner(
        policy,
        lr=1e-2,
        device="cpu",
        cfg=PPOConfig(train_epochs=1, batch_size=2, ent_coef=0.0),
    )

    X = torch.tensor(
        [
            [2.0, 1.0, 0.0, 1.0],
            [-2.0, -1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    idx = torch.tensor([0], dtype=torch.long)
    w = torch.tensor([1.0], dtype=torch.float32)

    with torch.no_grad():
        old_logp, _, _ = learner._logp_and_entropy(X, idx, w, temperature=1.0)

    transitions = [
        Transition(
            X=X,
            idx=idx,
            w=w,
            logp=float(old_logp.item()),
            value=0.0,
            reward=float(1.0 + 0.1 * step),
            done=False,
        )
        for step in range(2)
    ]
    transitions[-1].done = True

    before_w = policy.sel_head.weight.detach().clone()
    before_b = policy.sel_head.bias.detach().clone()

    stats = learner.ppo_update(transitions, temperature=1.0)

    after_w = policy.sel_head.weight.detach().clone()
    after_b = policy.sel_head.bias.detach().clone()

    assert stats["updates"] >= 1
    assert not torch.allclose(before_w, after_w), "selection head weights did not update"
    assert not torch.allclose(before_b, after_b), "selection head bias did not update"
