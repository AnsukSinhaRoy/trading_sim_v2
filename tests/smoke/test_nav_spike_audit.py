from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import pytest

from analytics.nav_spike_audit import audit_nav_spikes


def test_nav_spike_audit_identifies_main_symbol(tmp_path: Path):
    pytest.importorskip('pyarrow')
    run_dir = tmp_path / 'run'
    run_dir.mkdir(parents=True, exist_ok=True)
    events = [
        {'kind': 'position_snapshot', 'ts': '2024-01-01T09:15:00', 'cash': 0.0, 'positions': {'AAA': 10, 'BBB': 5}, 'mtm_prices': {}, 'nav': 2000.0},
        {'kind': 'position_snapshot', 'ts': '2024-01-01T09:16:00', 'cash': 0.0, 'positions': {'AAA': 10, 'BBB': 5}, 'mtm_prices': {}, 'nav': 1450.0},
    ]
    with (run_dir / 'events.jsonl').open('w', encoding='utf-8') as f:
        for row in events:
            f.write(json.dumps(row) + '\n')

    store = tmp_path / 'store' / 'date=2024-01-01'
    store.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range('2024-01-01 09:15:00', periods=2, freq='1min')
    df = pd.DataFrame({'ts': ts, 'AAA': [100.0, 50.0], 'BBB': [200.0, 190.0]})
    df.to_parquet(store / 'close.parquet', index=False)

    result = audit_nav_spikes(run_dir=run_dir, store_dir=tmp_path / 'store', pct_nav_change=0.10, top_k_symbols=2)
    assert len(result.spikes) == 1
    assert result.spikes.iloc[0]['top_symbol'] == 'AAA'
