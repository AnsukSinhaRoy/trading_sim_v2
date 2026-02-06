from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path
from datetime import datetime
import yaml

def _load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

@dataclass
class Config:
    raw: Dict[str, Any]
    base_dir: Path

    @staticmethod
    def load(run_yaml: str) -> "Config":
        run_path = Path(run_yaml)
        base_dir = run_path.parent
        cfg = _load_yaml(run_path)

        modules = cfg.get("modules", {}) or {}
        merged = dict(cfg)
        for _, rel_path in modules.items():
            mod_path = (base_dir / rel_path).resolve()
            merged = _deep_merge(merged, _load_yaml(mod_path))

        return Config(raw=merged, base_dir=base_dir)

    def get(self, *keys, default=None):
        cur: Any = self.raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)
