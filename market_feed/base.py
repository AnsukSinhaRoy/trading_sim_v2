from __future__ import annotations
from typing import Protocol, AsyncIterator
from common.events import MarketSnapshot

class MarketFeed(Protocol):
    async def stream(self) -> AsyncIterator[MarketSnapshot]:
        ...
