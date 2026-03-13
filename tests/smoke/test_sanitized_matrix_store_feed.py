from __future__ import annotations

from pathlib import Path
from datetime import datetime
import asyncio
import pandas as pd
import pytest

from market_feed.sanitized_matrix_store_1m import SanitizedMatrixStoreMinuteFeed


def test_sanitized_feed_rejects_absurd_minute_jump(tmp_path: Path):
    pytest.importorskip('pyarrow')
    ddir = tmp_path / 'date=2024-01-01'
    ddir.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range('2024-01-01 09:15:00', periods=3, freq='1min')
    df = pd.DataFrame({'ts': ts, 'AAA': [100.0, 40.0, 101.0]})
    df.to_parquet(ddir / 'close.parquet', index=False)

    feed = SanitizedMatrixStoreMinuteFeed(
        store_dir=str(tmp_path),
        start=datetime(2024, 1, 1, 9, 15),
        end=datetime(2024, 1, 1, 9, 17),
        speed='fast',
        max_abs_return=0.35,
    )

    async def collect():
        out = []
        async for snap in feed.stream():
            out.append(snap)
        return out

    snaps = asyncio.run(collect())
    assert snaps[0].prices['AAA'] == 100.0
    assert 'AAA' not in snaps[1].prices
    assert snaps[2].prices['AAA'] == 101.0


def test_sanitized_feed_rejects_zero_prices(tmp_path: Path):
    pytest.importorskip('pyarrow')
    ddir = tmp_path / 'date=2024-01-01'
    ddir.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range('2024-01-01 09:15:00', periods=2, freq='1min')
    df = pd.DataFrame({'ts': ts, 'AAA': [100.0, 0.0]})
    df.to_parquet(ddir / 'close.parquet', index=False)

    feed = SanitizedMatrixStoreMinuteFeed(
        store_dir=str(tmp_path),
        start=datetime(2024, 1, 1, 9, 15),
        end=datetime(2024, 1, 1, 9, 16),
        speed='fast',
        min_price=1.0,
    )

    async def collect():
        out = []
        async for snap in feed.stream():
            out.append(snap)
        return out

    snaps = asyncio.run(collect())
    assert snaps[0].prices['AAA'] == 100.0
    assert 'AAA' not in snaps[1].prices
