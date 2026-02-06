from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Iterable
from pydantic import BaseModel

class EventLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "events.jsonl"
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def append(self, event) -> None:
        if isinstance(event, BaseModel):
            payload = event.model_dump()
        else:
            payload = event

        def default(o):
            if isinstance(o, datetime):
                return o.isoformat()
            raise TypeError(f"Not JSON serializable: {type(o)}")

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=default) + "\n")

    def append_many(self, events: Iterable) -> None:
        for e in events:
            self.append(e)

    @staticmethod
    def read(path: Path) -> list[dict]:
        out: list[dict] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
