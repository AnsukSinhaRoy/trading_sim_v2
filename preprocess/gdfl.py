from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from runner.config import Config, parse_dt

log = logging.getLogger("levitate")

RETRYABLE_DIAGNOSTICS = (
    "Authentication request received. Try request data in next moment.",
    "Welcome!",
)

FATAL_DIAGNOSTICS = {
    "Access Denied. Key not found.": "API key not found",
    "Access Denied. Key blocked.": "API key blocked",
    "Access Denied. Key unknown or empty.": "API key unknown or empty",
    "Key Expired.": "API key expired",
    "IP address not allowed.": "Current IP is not allow-listed for this key",
    "Function not enabled.": "Requested function is not enabled for this key",
    "Data for requested exchange is disabled.": "Requested exchange is not enabled for this key",
    "Reached instrument limitation.": "The key has hit its instrument limitation",
    "Bandwidth per hour is limited.": "The key has hit its hourly bandwidth limit",
}


class GDFLError(RuntimeError):
    pass


@dataclass
class GDFLRestClient:
    base_url: str
    access_key: str
    timeout_sec: int = 60
    retry_attempts: int = 5
    poll_attempts: int = 20
    poll_delay_sec: float = 1.0
    user_agent: str = "levitate-trading-stack/0.1 gdfl-ingestor"

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/") + "/"
        if not self.access_key:
            raise ValueError("GDFL access key is empty")

    def _request_text(self, function_name: str, params: Dict[str, Any]) -> str:
        query = {"accessKey": self.access_key}
        for k, v in params.items():
            if v is None:
                continue
            if isinstance(v, bool):
                query[k] = "true" if v else "false"
            else:
                query[k] = str(v)
        url = f"{self.base_url}{function_name}/?{urlencode(query, doseq=False)}"
        req = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(req, timeout=self.timeout_sec) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def _request_json(self, function_name: str, params: Dict[str, Any]) -> Any:
        last_text = ""
        for poll_idx in range(max(1, self.poll_attempts)):
            for _ in range(max(1, self.retry_attempts)):
                try:
                    text = self._request_text(function_name, params).strip()
                    last_text = text
                    break
                except Exception as exc:
                    last_text = str(exc)
                    time.sleep(min(5.0, self.poll_delay_sec))
            else:
                raise GDFLError(f"HTTP request failed for {function_name}: {last_text}")

            if not text:
                time.sleep(self.poll_delay_sec)
                continue

            if text.startswith("{") or text.startswith("["):
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    raise GDFLError(f"Invalid JSON from {function_name}: {exc}: {text[:200]}") from exc

            if any(text.startswith(msg) for msg in RETRYABLE_DIAGNOSTICS):
                time.sleep(self.poll_delay_sec)
                continue

            for msg, reason in FATAL_DIAGNOSTICS.items():
                if text.startswith(msg):
                    raise GDFLError(f"{reason} while calling {function_name}: {text}")

            if text.startswith("Data unavailable."):
                raise GDFLError(f"Server returned data unavailable for {function_name}: {text}")

            raise GDFLError(f"Unexpected non-JSON response from {function_name}: {text[:300]}")

        raise GDFLError(
            f"Timed out waiting for JSON payload from {function_name}. Last response: {last_text[:300]}"
        )

    def get_limitation(self) -> Dict[str, Any]:
        payload = self._request_json("GetLimitation", {})
        if not isinstance(payload, dict):
            raise GDFLError(f"Unexpected GetLimitation payload type: {type(payload)!r}")
        return payload

    def get_instruments(
        self,
        *,
        exchange: str,
        series: Optional[str] = None,
        only_active: bool = True,
        detailed_info: bool = True,
        show_dummy_isin: bool = False,
        show_etf: bool = False,
        show_interoperable: bool = False,
    ) -> List[Dict[str, Any]]:
        payload = self._request_json(
            "GetInstruments",
            {
                "exchange": exchange,
                "series": series,
                "onlyActive": only_active,
                "detailedInfo": detailed_info,
                "showDummyISIN": show_dummy_isin,
                "showETF": show_etf,
                "showInterOperable": show_interoperable,
            },
        )
        return _extract_records(payload, keys=("INSTRUMENTS", "Instruments", "Result", "result"))

    def get_history(
        self,
        *,
        function_name: str,
        exchange: str,
        instrument_identifier: str,
        periodicity: str,
        period: int,
        from_epoch: int,
        to_epoch: int,
        max_records: int = 0,
        adjust_splits: bool = True,
        is_short_identifier: bool = False,
        user_tag: Optional[str] = None,
    ) -> Any:
        return self._request_json(
            function_name,
            {
                "exchange": exchange,
                "instrumentIdentifier": instrument_identifier,
                "periodicity": periodicity,
                "period": period,
                "from": from_epoch,
                "to": to_epoch,
                "max": max_records,
                "AdjustSplits": adjust_splits,
                "isShortIdentifier": is_short_identifier,
                "userTag": user_tag,
            },
        )


def _extract_records(payload: Any, *, keys: Sequence[str]) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in keys:
            if key in payload:
                val = payload[key]
                if isinstance(val, list):
                    return [x for x in val if isinstance(x, dict)]
        if all(not isinstance(v, (dict, list)) for v in payload.values()):
            return [payload]
    raise GDFLError(f"Unable to extract records from payload: {str(payload)[:300]}")


def _time_bounds(cfg: Config) -> Tuple[datetime, datetime]:
    start_dt = cfg.get("preprocess", "start", default=None)
    end_dt = cfg.get("preprocess", "end", default=None)
    if not start_dt or not end_dt:
        raise ValueError("preprocess.start and preprocess.end are required for GDFL ingestion")
    start = parse_dt(start_dt)
    end = parse_dt(end_dt)
    if end < start:
        raise ValueError("preprocess.end must be >= preprocess.start")
    return start, end


def _resolve_secret(cfg: Config, section: str, key: str, env_key: str) -> Optional[str]:
    env_name = cfg.get("preprocess", section, f"{key}_env", default=env_key)
    if env_name:
        value = os.getenv(str(env_name))
        if value:
            return value
    return cfg.get("preprocess", section, key, default=None)


def _symbol_transform(symbol: str, mode: str) -> str:
    mode = (mode or "lower").lower()
    if mode == "lower":
        return symbol.lower()
    if mode == "upper":
        return symbol.upper()
    if mode == "asis":
        return symbol
    if mode == "safe_lower":
        out = symbol.lower().strip()
        for bad, good in ((" ", "_"), ("/", "_"), ("\\", "_"), ("&", "and"), ("-", "_"), (".", "_")):
            out = out.replace(bad, good)
        while "__" in out:
            out = out.replace("__", "_")
        return out
    raise ValueError(f"Unsupported gdfl.symbol_transform={mode}")


def _choose_engine_symbol(inst: Dict[str, Any], *, symbol_field: str, symbol_transform: str) -> str:
    raw = (
        inst.get(symbol_field)
        or inst.get(symbol_field.upper())
        or inst.get("TRADESYMBOL")
        or inst.get("IDENTIFIER")
        or inst.get("Identifier")
    )
    if not raw:
        raise ValueError(f"Instrument missing {symbol_field}/TRADESYMBOL/IDENTIFIER: {inst}")
    return _symbol_transform(str(raw).strip(), symbol_transform)


def _instrument_identifier(inst: Dict[str, Any], identifier_field: str) -> str:
    raw = (
        inst.get(identifier_field)
        or inst.get(identifier_field.upper())
        or inst.get("IDENTIFIER")
        or inst.get("Identifier")
        or inst.get("TRADESYMBOL")
    )
    if not raw:
        raise ValueError(f"Instrument missing identifier field {identifier_field}: {inst}")
    return str(raw).strip()


def _select_instruments(
    instruments: Sequence[Dict[str, Any]],
    *,
    universe: Optional[Sequence[str]],
    symbol_field: str,
    symbol_transform: str,
    identifier_field: str,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    universe_u = {str(x).upper() for x in universe} if universe else None

    for inst in instruments:
        try:
            engine_symbol = _choose_engine_symbol(inst, symbol_field=symbol_field, symbol_transform=symbol_transform)
            identifier = _instrument_identifier(inst, identifier_field)
            trade_symbol = str(inst.get("TRADESYMBOL") or "").strip()
        except Exception:
            continue

        if universe_u is not None:
            candidates = {engine_symbol.upper(), identifier.upper(), trade_symbol.upper()}
            if not (candidates & universe_u):
                continue

        copied = dict(inst)
        copied["_engine_symbol"] = engine_symbol
        copied["_gdfl_identifier"] = identifier
        selected.append(copied)

    selected.sort(key=lambda x: x["_engine_symbol"])
    return selected


def _infer_chunk_days(cfg: Config, limitation: Dict[str, Any]) -> int:
    configured = cfg.get("preprocess", "gdfl", "chunk_days", default=None)
    if configured:
        return max(1, int(configured))
    hist_lim = limitation.get("HistoryLimitation") if isinstance(limitation, dict) else None
    if isinstance(hist_lim, dict):
        val = hist_lim.get("MaxIntraday")
        try:
            if val is not None:
                v = int(val)
                if v > 0:
                    return v
        except Exception:
            pass
    return 5


def _window_ranges(start: datetime, end: datetime, chunk_days: int) -> Iterable[Tuple[datetime, datetime]]:
    cur = start
    step = timedelta(days=chunk_days)
    while cur <= end:
        win_end = min(end, cur + step - timedelta(seconds=1))
        yield cur, win_end
        cur = win_end + timedelta(seconds=1)


def _extract_history_rows(payload: Any) -> List[Dict[str, Any]]:
    return _extract_records(payload, keys=("Result", "result", "HISTORY", "History"))


def normalize_gdfl_history(
    payload: Any,
    *,
    symbol: str,
    timezone_str: str,
    market_start: dtime,
    market_end: dtime,
    drop_nonpositive_bars: bool = True,
) -> pd.DataFrame:
    rows = _extract_history_rows(payload)
    if not rows:
        return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows)
    cols_lower = {str(c).lower(): c for c in df.columns}

    def get_col(*names: str) -> Optional[str]:
        for name in names:
            if name.lower() in cols_lower:
                return cols_lower[name.lower()]
        return None

    ts_col = get_col("LastTradeTime", "lasttradetime", "timestamp")
    open_col = get_col("Open")
    high_col = get_col("High")
    low_col = get_col("Low")
    close_col = get_col("Close")
    vol_col = get_col("TradedQty", "TotalQty", "Volume", "LastTradeQty")

    if not ts_col or not close_col:
        raise GDFLError(f"GetHistory payload missing mandatory columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["ts"] = pd.to_datetime(pd.to_numeric(df[ts_col], errors="coerce"), unit="s", utc=True)
    out["ts"] = out["ts"].dt.tz_convert(timezone_str).dt.tz_localize(None)
    out["symbol"] = symbol

    for out_col, src_col in {
        "open": open_col,
        "high": high_col,
        "low": low_col,
        "close": close_col,
    }.items():
        out[out_col] = pd.to_numeric(df[src_col], errors="coerce") if src_col else pd.NA

    out["volume"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0) if vol_col else 0.0

    out = out.dropna(subset=["ts", "close"]).sort_values("ts")
    out = out.drop_duplicates(subset=["ts"], keep="last")
    if out.empty:
        return out

    tvals = out["ts"].dt.time
    out = out.loc[(tvals >= market_start) & (tvals <= market_end)]

    if drop_nonpositive_bars and not out.empty:
        price_cols = [c for c in ["open", "high", "low", "close"] if c in out.columns]
        max_price = out[price_cols].max(axis=1, skipna=True)
        out = out.loc[max_price > 0]
        out = out.loc[out["close"] > 0]

    return out.reset_index(drop=True)


def _date_strings(start: datetime, end: datetime) -> List[str]:
    out: List[str] = []
    cur = start.date()
    while cur <= end.date():
        if cur.weekday() < 5:
            out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def _existing_symbol_day(long_store: Path, symbol: str, date_str: str) -> bool:
    return (long_store / f"date={date_str}" / f"symbol={symbol}" / "data.parquet").exists()


def _write_symbol_day(long_store: Path, symbol: str, date_str: str, df: pd.DataFrame) -> None:
    part_dir = long_store / f"date={date_str}" / f"symbol={symbol}"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / "data.parquet"
    df.to_parquet(part_path, index=False)


def _frame_to_daily_parts(df: pd.DataFrame, fields: Sequence[str]) -> Iterable[Tuple[str, pd.DataFrame]]:
    if df.empty:
        return []
    work = df.copy()
    work["date"] = work["ts"].dt.date.astype(str)
    out: List[Tuple[str, pd.DataFrame]] = []
    for date_str, part in work.groupby("date", sort=True):
        keep_cols = ["ts", "symbol"] + [f for f in fields if f in part.columns]
        out.append((date_str, part[keep_cols].sort_values("ts").reset_index(drop=True)))
    return out


def _validate_allowed_function(limitation: Dict[str, Any], function_name: str) -> None:
    allowed = limitation.get("AllowedFunctions")
    if not isinstance(allowed, list):
        return
    enabled_by_name = {
        str(item.get("FunctionName", "")): bool(item.get("IsEnabled", False))
        for item in allowed
        if isinstance(item, dict)
    }
    if function_name in enabled_by_name and not enabled_by_name[function_name]:
        raise GDFLError(f"{function_name} is not enabled for this GDFL key according to GetLimitation")


def build_long_store_from_gdfl(cfg: Config, run_dir: Path, fields: Sequence[str]) -> Path:
    out_dir = Path(cfg.get("preprocess", "out_dir", default="processed_data"))
    dataset = cfg.get("preprocess", "dataset_name", default="dataset")
    long_store = out_dir / dataset / "1m_long_store"
    long_store.mkdir(parents=True, exist_ok=True)

    start, end = _time_bounds(cfg)
    market_start_raw = cfg.get("preprocess", "market_start", default="09:15:00")
    market_end_raw = cfg.get("preprocess", "market_end", default="15:30:00")
    market_start = datetime.strptime(str(market_start_raw), "%H:%M:%S").time()
    market_end = datetime.strptime(str(market_end_raw), "%H:%M:%S").time()

    base_url = _resolve_secret(cfg, "gdfl", "base_url", "GDFL_BASE_URL")
    access_key = _resolve_secret(cfg, "gdfl", "access_key", "GDFL_ACCESS_KEY")
    if not base_url:
        raise ValueError("GDFL base URL is missing. Set preprocess.gdfl.base_url or env var from preprocess.gdfl.base_url_env")
    if not access_key:
        raise ValueError("GDFL access key is missing. Set preprocess.gdfl.access_key or env var from preprocess.gdfl.access_key_env")

    gdfl_exchange = str(cfg.get("preprocess", "gdfl", "exchange", default="NSE"))
    gdfl_series = cfg.get("preprocess", "gdfl", "series", default="EQ")
    history_function = str(cfg.get("preprocess", "gdfl", "history_function", default="GetHistoryAfterMarket"))
    periodicity = str(cfg.get("preprocess", "gdfl", "periodicity", default="MINUTE"))
    period = int(cfg.get("preprocess", "gdfl", "period", default=1))
    is_short_identifier = bool(cfg.get("preprocess", "gdfl", "is_short_identifier", default=False))
    adjust_splits = bool(cfg.get("preprocess", "gdfl", "adjust_splits", default=True))
    max_records = int(cfg.get("preprocess", "gdfl", "max_records", default=0))
    request_pause_sec = float(cfg.get("preprocess", "gdfl", "request_pause_sec", default=0.25))
    skip_existing = bool(cfg.get("preprocess", "gdfl", "skip_existing", default=True))
    timezone_str = str(cfg.get("preprocess", "gdfl", "timezone", default="Asia/Kolkata"))
    symbol_field = str(cfg.get("preprocess", "gdfl", "symbol_field", default="IDENTIFIER"))
    symbol_transform = str(cfg.get("preprocess", "gdfl", "symbol_transform", default="lower"))
    identifier_field = str(cfg.get("preprocess", "gdfl", "instrument_identifier_field", default="IDENTIFIER"))
    drop_nonpositive_bars = bool(cfg.get("preprocess", "gdfl", "drop_nonpositive_bars", default=True))

    client = GDFLRestClient(
        base_url=base_url,
        access_key=access_key,
        timeout_sec=int(cfg.get("preprocess", "gdfl", "timeout_sec", default=60)),
        retry_attempts=int(cfg.get("preprocess", "gdfl", "retry_attempts", default=5)),
        poll_attempts=int(cfg.get("preprocess", "gdfl", "poll_attempts", default=20)),
        poll_delay_sec=float(cfg.get("preprocess", "gdfl", "poll_delay_sec", default=1.0)),
    )

    limitation = client.get_limitation()
    _validate_allowed_function(limitation, history_function)
    chunk_days = _infer_chunk_days(cfg, limitation)

    instruments = client.get_instruments(
        exchange=gdfl_exchange,
        series=gdfl_series,
        only_active=bool(cfg.get("preprocess", "gdfl", "only_active", default=True)),
        detailed_info=bool(cfg.get("preprocess", "gdfl", "detailed_instruments", default=True)),
        show_dummy_isin=bool(cfg.get("preprocess", "gdfl", "show_dummy_isin", default=False)),
        show_etf=bool(cfg.get("preprocess", "gdfl", "show_etf", default=False)),
        show_interoperable=bool(cfg.get("preprocess", "gdfl", "show_interoperable", default=False)),
    )

    from preprocess.build import _load_universe  # local import to avoid circular at module load

    universe = _load_universe(cfg.get("preprocess", "universe_file", default=None))
    selected = _select_instruments(
        instruments,
        universe=universe,
        symbol_field=symbol_field,
        symbol_transform=symbol_transform,
        identifier_field=identifier_field,
    )

    if not selected:
        raise GDFLError("No instruments selected from GDFL. Check exchange/series/universe_file/symbol field settings.")

    catalogue_df = pd.DataFrame(selected)
    catalogue_df.to_csv(run_dir / "gdfl_selected_instruments.csv", index=False)
    (run_dir / "gdfl_limitation.json").write_text(json.dumps(limitation, indent=2), encoding="utf-8")

    failures: Dict[str, str] = {}
    downloaded_days = 0
    downloaded_symbols = 0

    log.info(
        "GDFL ingest: selected %d instruments (exchange=%s series=%s function=%s chunk_days=%d)",
        len(selected), gdfl_exchange, gdfl_series, history_function, chunk_days,
    )

    for idx, inst in enumerate(selected, start=1):
        engine_symbol = inst["_engine_symbol"]
        gdfl_identifier = inst["_gdfl_identifier"]
        downloaded_symbols += 1
        log.info("GDFL ingest: %d/%d symbol=%s identifier=%s", idx, len(selected), engine_symbol, gdfl_identifier)

        try:
            for win_start, win_end in _window_ranges(start, end, chunk_days):
                date_strs = _date_strings(win_start, win_end)
                if skip_existing and date_strs and all(_existing_symbol_day(long_store, engine_symbol, d) for d in date_strs):
                    continue

                payload = client.get_history(
                    function_name=history_function,
                    exchange=gdfl_exchange,
                    instrument_identifier=gdfl_identifier,
                    periodicity=periodicity,
                    period=period,
                    from_epoch=int(win_start.timestamp()),
                    to_epoch=int(win_end.timestamp()),
                    max_records=max_records,
                    adjust_splits=adjust_splits,
                    is_short_identifier=is_short_identifier,
                    user_tag=f"{engine_symbol}:{win_start.date()}:{win_end.date()}",
                )
                frame = normalize_gdfl_history(
                    payload,
                    symbol=engine_symbol,
                    timezone_str=timezone_str,
                    market_start=market_start,
                    market_end=market_end,
                    drop_nonpositive_bars=drop_nonpositive_bars,
                )
                if frame.empty:
                    time.sleep(request_pause_sec)
                    continue

                frame = frame.loc[(frame["ts"] >= start) & (frame["ts"] <= end)].reset_index(drop=True)
                for date_str, part in _frame_to_daily_parts(frame, fields):
                    if skip_existing and _existing_symbol_day(long_store, engine_symbol, date_str):
                        continue
                    _write_symbol_day(long_store, engine_symbol, date_str, part)
                    downloaded_days += 1

                time.sleep(request_pause_sec)
        except Exception as exc:
            failures[engine_symbol] = str(exc)
            log.exception("GDFL ingest failed for symbol=%s (%s)", engine_symbol, exc)

    manifest = {
        "source": "gdfl",
        "long_store_dir": str(long_store),
        "gdfl_exchange": gdfl_exchange,
        "gdfl_series": gdfl_series,
        "history_function": history_function,
        "periodicity": periodicity,
        "period": period,
        "chunk_days": chunk_days,
        "timezone": timezone_str,
        "symbols_selected": len(selected),
        "symbols_attempted": downloaded_symbols,
        "daily_partitions_written": downloaded_days,
        "fields": list(fields),
        "failures": failures,
        "start": start.isoformat(sep=" "),
        "end": end.isoformat(sep=" "),
    }
    (run_dir / "preprocess_long_store_metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("GDFL ingest complete -> %s", long_store)
    return long_store
