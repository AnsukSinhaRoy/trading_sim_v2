from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """Per-asset actor with pooled critic.

    Input: X [n, f]
    Outputs:
      - selection logits [n]
      - dirichlet concentration alpha [n]
      - value scalar
    """

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.sel_head = nn.Linear(hidden, 1)
        self.alpha_head = nn.Linear(hidden, 1)

        self.v_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits, alpha, value)."""
        h = self.embed(X)                    # [n, hidden]
        logits = self.sel_head(h).squeeze(-1)  # [n]
        alpha_raw = self.alpha_head(h).squeeze(-1)  # [n]
        alpha = F.softplus(alpha_raw) + 1.0         # >=1 for reasonable exploration

        pooled = h.mean(dim=0, keepdim=True)         # [1, hidden]
        value = self.v_head(pooled).squeeze(-1)      # []
        return logits, alpha, value


@torch.no_grad()
def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """mask: True means allowed, False means blocked."""
    temp = max(1e-6, float(temperature))
    x = logits / temp
    x = x.clone()
    x[~mask] = -1e9
    probs = torch.softmax(x, dim=0)
    probs = probs * mask.float()
    s = probs.sum()
    if s.item() <= 0:
        # if everything is masked, fall back to uniform over all
        probs = torch.ones_like(probs) / probs.numel()
    else:
        probs = probs / s
    return probs


@torch.no_grad()
def sample_without_replacement(probs: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sequential categorical sampling without replacement.

    Returns:
      idx: [k]
      logp: scalar tensor
      entropy: scalar tensor (approx: sum of per-step categorical entropies)
    """
    eps = 1e-12
    probs = probs.clone()
    idxs = []
    logp = torch.tensor(0.0, device=probs.device)
    ent = torch.tensor(0.0, device=probs.device)

    for _ in range(int(k)):
        probs = probs / (probs.sum() + eps)
        # categorical entropy
        ent = ent + (-(probs * (probs + eps).log()).sum())

        i = torch.multinomial(probs, num_samples=1, replacement=False).item()
        idxs.append(i)
        logp = logp + (probs[i] + eps).log()
        probs[i] = 0.0

    return torch.tensor(idxs, device=probs.device, dtype=torch.long), logp, ent


def logprob_without_replacement(probs: torch.Tensor, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute exact logprob for the same sequential without-replacement process used in sampling."""
    eps = 1e-12
    probs = probs.clone()
    logp = torch.tensor(0.0, device=probs.device)
    ent = torch.tensor(0.0, device=probs.device)

    for i in idx.tolist():
        probs = probs / (probs.sum() + eps)
        ent = ent + (-(probs * (probs + eps).log()).sum())
        logp = logp + (probs[i] + eps).log()
        probs[i] = 0.0

    return logp, ent
