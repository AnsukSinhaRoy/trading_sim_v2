from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from preprocess.intraday_anomaly_repair import repair_intraday_anomalies


def test_intraday_repair_scales_persistent_suffix(tmp_path: Path):
    pytest.importorskip('pyarrow')
    inp = tmp_path / 'inp'
    out = tmp_path / 'out'
    ddir = inp / 'date=2024-01-01'
    ddir.mkdir(parents=True, exist_ok=True)

    ts = pd.date_range('2024-01-01 09:15:00', periods=60, freq='1min')
    prices = [100.0] * 30 + [50.0] * 30
    df = pd.DataFrame({'ts': ts, 'AAA': prices})
    df.to_parquet(ddir / 'close.parquet', index=False)

    summary = repair_intraday_anomalies(
        input_store=inp,
        output_store=out,
        symbols=['AAA'],
        lookback_bars=10,
        lookahead_bars=10,
        min_valid_window=5,
        min_jump_abs_return=0.30,
        factor_tolerance=0.15,
    )

    repaired = pd.read_parquet(out / 'date=2024-01-01' / 'close.parquet')
    assert summary.anomalies_found >= 1
    assert repaired['AAA'].iloc[-1] == pytest.approx(100.0)
