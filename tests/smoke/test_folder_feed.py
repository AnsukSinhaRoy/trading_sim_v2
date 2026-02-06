from __future__ import annotations
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

from market_feed.folder_1m import FolderMinuteFeed

def test_folder_feed_streams_all_symbols(tmp_path: Path):
    # Create two tiny symbol CSVs with 3 minutes
    start = datetime(2024, 1, 1, 9, 15, 0)
    idx = [start + timedelta(minutes=i) for i in range(3)]
    for sym, base in [("AAA", 100.0), ("BBB", 200.0)]:
        df = pd.DataFrame({"datetime": idx, "close": [base, base+1, base+2]})
        df.to_csv(tmp_path / f"{sym}.csv", index=False)

    feed = FolderMinuteFeed(
        data_dir=str(tmp_path),
        symbols=["AAA","BBB"],
        start=start,
        end=start + timedelta(minutes=2),
        fmt="csv",
        file_pattern="{symbol}.{ext}",
        timestamp_col="datetime",
        price_col="close",
        freq="1min",
        fill="ffill",
        speed="fast",
    )

    async def collect():
        out = []
        async for snap in feed.stream():
            out.append(snap)
        return out

    import asyncio
    snaps = asyncio.run(collect())
    assert len(snaps) == 3
    assert all(("AAA" in s.prices and "BBB" in s.prices) for s in snaps)
