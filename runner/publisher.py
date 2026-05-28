from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import zmq


class ZmqPublisher:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, snd_hwm: int = 10000):
        self.host = host
        self.port = int(port)
        self.ctx = zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, int(snd_hwm))
        self.socket.bind(f"tcp://{self.host}:{self.port}")

    async def publish(self, topic: str, data: dict) -> None:
        """Best-effort publish (never blocks the engine loop)."""
        try:
            self.socket.send_multipart(
                [topic.encode("utf-8"), json.dumps(data, default=self._json_default).encode("utf-8")],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            # Drop if subscriber is slow / queue is full.
            return
        except Exception as e:
            # Keep the engine alive even if the dashboard is not running.
            print(f"ZMQ Publish Error: {e}")

    @staticmethod
    def _json_default(o: Any):
        """Make common python objects JSON serializable (esp. datetime in events)."""
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, uuid.UUID):
            return str(o)
        if isinstance(o, Path):
            return str(o)
        # pydantic BaseModel or similar
        if hasattr(o, "model_dump"):
            try:
                return o.model_dump(mode="json")
            except TypeError:
                return o.model_dump()
        # last-resort fallback (keeps engine alive)
        return str(o)

    def close(self) -> None:
        try:
            self.socket.close(0)
        except Exception:
            pass
