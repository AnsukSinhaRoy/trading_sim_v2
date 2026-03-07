from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass
class CheckpointConfig:
    enabled: bool = True
    dir: str | None = None
    save_every_steps: int = 10
    keep_last: int = 5


class CheckpointManager:
    def __init__(self, cfg: CheckpointConfig, run_dir: str | None = None):
        self.cfg = cfg
        self.run_dir = run_dir

        if not cfg.enabled:
            self.dir = None
            return

        if cfg.dir:
            base = Path(cfg.dir)
        elif run_dir:
            base = Path(run_dir) / "checkpoints" / "rl_agent"
        else:
            base = Path("checkpoints") / "rl_agent"

        self.dir = base
        self.dir.mkdir(parents=True, exist_ok=True)

    def latest_path(self) -> Optional[Path]:
        if not self.dir:
            return None
        p = self.dir / "latest.pt"
        return p if p.exists() else None

    def save(self, payload: dict[str, Any], step: int):
        if not self.dir:
            return
        # atomic-ish save
        tmp = self.dir / "_tmp_latest.pt"
        torch.save(payload, tmp)
        tmp.replace(self.dir / "latest.pt")

        # numbered snapshot
        snap = self.dir / f"step_{int(step):08d}.pt"
        try:
            torch.save(payload, snap)
        except Exception:
            pass

        # GC old snapshots
        keep = int(self.cfg.keep_last)
        if keep > 0:
            snaps = sorted(self.dir.glob("step_*.pt"))
            if len(snaps) > keep:
                for p in snaps[: len(snaps) - keep]:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass

    def load(self) -> Optional[dict[str, Any]]:
        if not self.dir:
            return None
        p = self.latest_path()
        if not p:
            return None
        try:
            return torch.load(p, map_location="cpu")
        except Exception:
            return None
