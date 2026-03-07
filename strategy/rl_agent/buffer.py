from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class Transition:
    """One decision step transition stored for PPO."""

    # State features: [n, feat] (CPU)
    X: torch.Tensor

    # Action (two-stage):
    #   idx: selected asset indices into X, shape [k] (CPU, long)
    #   w: weights over selected assets, shape [k] (CPU, float, sum to 1)
    idx: torch.Tensor
    w: torch.Tensor

    # Behavior policy stats
    logp: float
    value: float

    # Reward from holding period (scalar)
    reward: float

    # Whether this is terminal for GAE bootstrapping
    done: bool = False


class RolloutBuffer:
    def __init__(self):
        self.data: List[Transition] = []

    def __len__(self) -> int:
        return len(self.data)

    def clear(self):
        self.data.clear()

    def add(self, t: Transition):
        # Ensure CPU storage
        t.X = t.X.detach().cpu()
        t.idx = t.idx.detach().cpu().long()
        t.w = t.w.detach().cpu().float()
        self.data.append(t)

    def get(self) -> List[Transition]:
        return list(self.data)

    def mark_last_done(self):
        if self.data:
            self.data[-1].done = True

    def state_dict(self):
        # light serialization (tensors + scalars)
        out = {
            "data": [
                {
                    "X": t.X,
                    "idx": t.idx,
                    "w": t.w,
                    "logp": float(t.logp),
                    "value": float(t.value),
                    "reward": float(t.reward),
                    "done": bool(t.done),
                }
                for t in self.data
            ]
        }
        return out

    def load_state_dict(self, sd):
        self.data = []
        for d in sd.get("data", []):
            self.data.append(
                Transition(
                    X=d["X"],
                    idx=d["idx"],
                    w=d["w"],
                    logp=float(d["logp"]),
                    value=float(d["value"]),
                    reward=float(d["reward"]),
                    done=bool(d.get("done", False)),
                )
            )
