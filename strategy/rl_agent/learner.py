from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .buffer import Transition
from .policy import _masked_softmax, logprob_without_replacement


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    train_epochs: int = 6
    batch_size: int = 16
    grad_clip: float = 1.0


class Learner:
    def __init__(self, policy: nn.Module, lr: float = 3e-4, device: str = "cpu", cfg: PPOConfig | None = None):
        self.policy = policy.to(device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=float(lr))
        self.device = device
        self.cfg = cfg or PPOConfig()

        self.updates = 0

    def state_dict(self):
        return {
            "policy": self.policy.state_dict(),
            "opt": self.opt.state_dict(),
            "updates": int(self.updates),
            "cfg": self.cfg.__dict__,
        }

    def load_state_dict(self, sd):
        self.policy.load_state_dict(sd.get("policy", {}))
        if "opt" in sd:
            try:
                self.opt.load_state_dict(sd["opt"])
            except Exception:
                # optimizer state can be version-dependent; ignore if incompatible
                pass
        self.updates = int(sd.get("updates", 0))

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages + returns. values is [T+1] with bootstrap value at the end."""
        T = rewards.shape[0]
        adv = torch.zeros(T, device=rewards.device)
        last_gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * values[t + 1] * mask - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * last_gae
            adv[t] = last_gae
        ret = adv + values[:-1]
        return adv, ret

    def _logp_and_entropy(self, X: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, temperature: float = 1.0, mask: torch.Tensor | None = None):
        logits, alpha, value = self.policy(X)  # logits [n], alpha [n], value []
        if mask is None:
            mask = torch.ones_like(logits, dtype=torch.bool)
        probs = _masked_softmax(logits, mask, temperature=temperature)
        logp_sel, ent_sel = logprob_without_replacement(probs, idx)

        # Dirichlet logp over selected weights
        alpha_sel = alpha.gather(0, idx)
        dist = torch.distributions.Dirichlet(alpha_sel)
        logp_w = dist.log_prob(w)
        ent_w = dist.entropy()

        logp = logp_sel + logp_w
        ent = ent_sel + ent_w
        return logp, ent, value

    def ppo_update(
        self,
        transitions: List[Transition],
        temperature: float = 1.0,
        masks: List[torch.Tensor] | None = None,
    ):
        if not transitions:
            return

        # Prepare tensors
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        old_logp = torch.tensor([t.logp for t in transitions], dtype=torch.float32, device=self.device)
        old_value = torch.tensor([t.value for t in transitions], dtype=torch.float32, device=self.device)
        dones = torch.tensor([1.0 if t.done else 0.0 for t in transitions], dtype=torch.float32, device=self.device)

        # Bootstrap value for last state (use last stored value as a cheap approx)
        values = torch.cat([old_value, old_value[-1:].clone()], dim=0)

        adv, ret = self._compute_gae(rewards, values, dones)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(transitions)
        idx_order = torch.arange(n, device=self.device)

        for _ in range(int(self.cfg.train_epochs)):
            perm = idx_order[torch.randperm(n)]
            for start in range(0, n, int(self.cfg.batch_size)):
                mb = perm[start : start + int(self.cfg.batch_size)]

                new_logps = []
                ents = []
                new_vals = []

                for j, t_i in enumerate(mb.tolist()):
                    tr = transitions[t_i]
                    X = tr.X.to(self.device)
                    a_idx = tr.idx.to(self.device)
                    w = tr.w.to(self.device)
                    mask = None
                    if masks is not None:
                        mask = masks[t_i].to(self.device)

                    lp, ent, val = self._logp_and_entropy(X, a_idx, w, temperature=temperature, mask=mask)
                    new_logps.append(lp)
                    ents.append(ent)
                    new_vals.append(val)

                new_logp_t = torch.stack(new_logps)
                ent_t = torch.stack(ents)
                new_val_t = torch.stack(new_vals).squeeze(-1)

                ratio = torch.exp(new_logp_t - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_val_t, ret[mb]) * self.cfg.vf_coef
                entropy_bonus = ent_t.mean() * self.cfg.ent_coef

                loss = policy_loss + value_loss - entropy_bonus

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float(self.cfg.grad_clip))
                self.opt.step()

        self.updates += 1
