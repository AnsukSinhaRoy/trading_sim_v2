from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import uuid, math, csv

from runner.config import Config, parse_dt
from common.events import MarketSnapshot, OrderRequest, PositionSnapshot
from common.eventlog import EventLogger

from market_feed.synthetic import SyntheticMinuteFeed
from market_feed.folder_1m import FolderMinuteFeed
from market_feed.matrix_store_1m import MatrixStoreMinuteFeed

from execution.paper import PaperExecutionEngine
from strategy.toy_rebalance import ToyRebalanceStrategy
from analytics.build import build_derived_from_events

import importlib

def _make_feed(cfg: Config):
    ftype = cfg.get("market_feed","type")

    if ftype == "synthetic_1m":
        return SyntheticMinuteFeed(
            symbols=cfg.get("market_feed","symbols"),
            start=parse_dt(cfg.get("market_feed","start")),
            minutes=int(cfg.get("market_feed","minutes")),
            init_prices=cfg.get("market_feed","init_prices", default={}),
            vol_bps=float(cfg.get("market_feed","vol_bps", default=15)),
            drift_bps=float(cfg.get("market_feed","drift_bps", default=0)),
            speed=cfg.get("market_feed","speed", default="fast"),
        )

    if ftype == "folder_1m":
        start = parse_dt(cfg.get("market_feed","start"))
        end = cfg.get("market_feed","end", default=None)
        minutes = cfg.get("market_feed","minutes", default=None)
        if end is None and minutes is None:
            raise ValueError("folder_1m requires either market_feed.end or market_feed.minutes")
        if end is None and minutes is not None:
            from datetime import timedelta
            end_dt = start + timedelta(minutes=int(minutes))
        else:
            end_dt = parse_dt(end)

        fmt = cfg.get("market_feed","format", default="csv")
        symbols = cfg.get("market_feed","symbols", default=None)  # may be None for autodiscover

        return FolderMinuteFeed(
            data_dir=cfg.get("market_feed","data_dir"),
            symbols=symbols,
            start=start,
            end=end_dt,
            fmt=fmt,
            file_pattern=cfg.get("market_feed","file_pattern", default="{symbol}_minute.{ext}"),
            timestamp_col=cfg.get("market_feed","timestamp_col", default="date"),
            price_col=cfg.get("market_feed","price_col", default="close"),
            freq=cfg.get("market_feed","freq", default="1min"),
            fill=cfg.get("market_feed","fill", default="ffill"),
            speed=cfg.get("market_feed","speed", default="fast"),
            discover_symbols=bool(cfg.get("market_feed","discover_symbols", default=True)),
            discover_recursive=bool(cfg.get("market_feed","discover_recursive", default=True)),
            universe_file=cfg.get("market_feed","universe_file", default=None),
            universe_mode=cfg.get("market_feed","universe_mode", default="intersect"),
            progress_every=int(cfg.get("market_feed","progress_every", default=25)),
        )

    if ftype == "matrix_store_1m":
        start = parse_dt(cfg.get("market_feed","start"))
        end = cfg.get("market_feed","end", default=None)
        minutes = cfg.get("market_feed","minutes", default=None)
        if end is None and minutes is None:
            raise ValueError("matrix_store_1m requires either market_feed.end or market_feed.minutes")
        if end is None and minutes is not None:
            from datetime import timedelta
            end_dt = start + timedelta(minutes=int(minutes))
        else:
            end_dt = parse_dt(end)

        return MatrixStoreMinuteFeed(
            store_dir=cfg.get("market_feed","store_dir"),
            start=start,
            end=end_dt,
            symbols=cfg.get("market_feed","symbols", default=None),
            speed=cfg.get("market_feed","speed", default="fast"),
        )

    raise ValueError(f"Unsupported market_feed.type: {ftype}")

def _make_execution(cfg: Config):
    return PaperExecutionEngine(
        initial_cash=float(cfg.get("execution","initial_cash", default=1_000_000)),
        slippage_bps=float(cfg.get("execution","slippage","bps", default=0)),
        fees_bps=float(cfg.get("execution","fees","bps", default=0)),
    )

def _make_strategy(cfg):
    stype = cfg.get("strategy", "type")
    params = dict(cfg.raw.get("strategy", {}))
    params.pop("type", None)

    if ":" in stype:
        mod_name, cls_name = stype.split(":", 1)
    else:
        mod_name = f"strategy.{stype}"
        cls_name = "".join(p.capitalize() for p in stype.split("_")) + "Strategy"

    mod = importlib.import_module(mod_name)
    if not hasattr(mod, cls_name):
        raise ValueError(f"Strategy class not found: {mod_name}:{cls_name}")

    cls = getattr(mod, cls_name)
    return cls(**params)

def _rebalance(ts, target_w: Dict[str, float], snap: MarketSnapshot, port: PositionSnapshot) -> List[OrderRequest]:
    prices = snap.prices
    nav = float(port.nav)
    cash = float(port.cash)
    cur_pos = dict(port.positions)
    desired_notional = {sym: float(w) * nav for sym, w in target_w.items()}
    orders: List[OrderRequest] = []

    for sym, qty in cur_pos.items():
        px = float(prices.get(sym, 0.0))
        if px <= 0:
            continue
        cur_val = qty * px
        tgt_val = desired_notional.get(sym, 0.0)
        if cur_val > tgt_val + 1e-9:
            desired_qty = int(math.floor(tgt_val / px))
            sell_qty = int(qty - desired_qty)
            if sell_qty > 0:
                orders.append(OrderRequest(ts=ts, order_id=str(uuid.uuid4()), symbol=sym, side="SELL", qty=sell_qty))

    remaining = cash
    for sym, w in sorted(target_w.items(), key=lambda kv: kv[1], reverse=True):
        px = float(prices.get(sym, 0.0))
        if px <= 0:
            continue
        qty = int(cur_pos.get(sym, 0))
        cur_val = qty * px
        tgt_val = desired_notional.get(sym, 0.0)
        if cur_val + 1e-9 >= tgt_val:
            continue
        buy_val = tgt_val - cur_val
        buy_qty = int(math.floor(buy_val / px))
        buy_qty = min(buy_qty, int(math.floor(remaining / px)))
        if buy_qty > 0:
            orders.append(OrderRequest(ts=ts, order_id=str(uuid.uuid4()), symbol=sym, side="BUY", qty=buy_qty))
            remaining -= buy_qty * px

    return orders

async def run_stream(cfg: Config, run_dir: Path, logger: EventLogger, logger_obj=None) -> None:
    log = logger_obj
    if log:
        log.info("Building market feed...")
    feed = _make_feed(cfg)
    if log:
        log.info("Building execution engine...")
    exe = _make_execution(cfg)
    if log:
        log.info("Building strategy...")
    strat = _make_strategy(cfg)

    start = parse_dt(cfg.get("market_feed","start"))
    port = exe.snapshot(start)
    logger.append(port)

    # OPTIMIZATION: Open a lightweight CSV for real-time NAV dashboarding
    nav_csv_path = run_dir / "nav.csv"
    nav_file = open(nav_csv_path, "w", newline="")
    nav_writer = csv.writer(nav_file)
    nav_writer.writerow(["ts", "nav", "cash"])
    nav_writer.writerow([start.isoformat(), port.nav, port.cash])
    nav_file.flush()

    if log:
        log.info("Initial snapshot written (NAV=%.2f, cash=%.2f)", port.nav, port.cash)

    tick_count = 0
    try:
        async for snap in feed.stream():
            tick_count += 1
            exe.update_market(snap)

            # OPTIMIZATION: Do NOT log the market snapshot. It is redundant and massive.
            # logger.append(snap) 

            strat.on_snapshot(snap, port)
            target_w = getattr(strat, "_last_target_weights", None)

            if isinstance(target_w, dict) and target_w:
                orders = _rebalance(snap.ts, target_w, snap, port)
                if orders:
                    logger.append_many(orders)
                    for e in exe.place_orders(snap.ts, orders):
                        logger.append(e)

            port = exe.snapshot(snap.ts)
            
            # Log position snapshot (now lightweight) to JSONL for audit
            logger.append(port)

            # Log NAV to CSV for fast Dashboard access
            nav_writer.writerow([snap.ts.isoformat(), port.nav, port.cash])
            # Flush every few ticks to allow dashboard to see updates without killing I/O
            if tick_count % 10 == 0:
                nav_file.flush()

            if log and (tick_count % int(cfg.get("run","progress_every_ticks", default=250)) == 0):
                log.info("Progress: ticks=%d | ts=%s | NAV=%.2f | cash=%.2f | n_syms=%d",
                        tick_count, snap.ts.isoformat(), port.nav, port.cash, len(snap.prices))
    finally:
        nav_file.close()

    if log:
        log.info("Streaming finished. Total ticks=%d. Building derived outputs...", tick_count)
    build_derived_from_events(run_dir)
    if log:
        log.info("Derived outputs built: %s", (Path(run_dir) / "derived"))