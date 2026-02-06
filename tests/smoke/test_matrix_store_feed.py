from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
import asyncio

from market_feed.matrix_store_1m import MatrixStoreMinuteFeed

def test_matrix_store_streams(tmp_path: Path):
    # Create a single date partition with wide data
    ddir = tmp_path / "date=2024-01-01"
    ddir.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range("2024-01-01 09:15:00", periods=3, freq="1min")
    df = pd.DataFrame({"ts": ts, "AAA": [1.0, 2.0, 3.0], "BBB": [10.0, 11.0, 12.0]})
    df.to_parquet(ddir / "close.parquet", index=False)

    feed = MatrixStoreMinuteFeed(
        store_dir=str(tmp_path),
        start=datetime(2024,1,1,9,15,0),
        end=datetime(2024,1,1,9,17,0),
        symbols=None,
        speed="fast",
    )

    async def collect():
        out = []
        async for s in feed.stream():
            out.append(s)
        return out

    snaps = asyncio.run(collect())
    assert len(snaps) == 3
    assert "AAA" in snaps[0].prices and "BBB" in snaps[0].prices
