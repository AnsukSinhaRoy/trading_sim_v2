from __future__ import annotations

import asyncio
import csv
import math
from pathlib import Path
from typing import List

from analytics.build import build_derived_from_events  # kept for the post-run analytics hook
from common.eventlog import EventLogger
from common.events import MarketSnapshot, OrderRequest, PositionSnapshot
from runner.config import Config, parse_dt
from runner.factories import _make_execution, _make_feed, _make_strategy
from runner.liquidity import (
    LiquidityConstraints,
    LiquidityTracker,
    _accumulate_rebalance_diag,
    _frictions_payload_for_ui,
    _make_liquidity_constraints,
)
from runner.publisher import ZmqPublisher
from runner.rebalancer import _rebalance, _rebalance_with_diagnostics
from runner.telemetry import _extract_strategy_telemetry, _safe_float_for_ui


async def _produce_market_data(feed, queue: asyncio.Queue):
    """Reads from the historical feed and pushes snapshots to the queue."""
    async for snap in feed.stream():
        await queue.put(snap)
    # Signal end of stream
    await queue.put(None)


async def run_stream(cfg: Config, run_dir: Path, logger: EventLogger, logger_obj=None) -> None:
    log = logger_obj
    
    # 1. Initialize Components
    if log: log.info("Building engine components...")
    feed = _make_feed(cfg)
    exe = _make_execution(cfg)
    strat = _make_strategy(cfg, run_dir=run_dir)
    liquidity_constraints = _make_liquidity_constraints(cfg)
    liquidity_tracker = LiquidityTracker(liquidity_constraints.adv_lookback_days) if liquidity_constraints.enabled else None
    if log and liquidity_constraints.enabled:
        log.info(
            "Liquidity controls enabled: max_bar_participation=%.3f, "
            "max_position_adv_participation=%.3f, adv_lookback_days=%d, min_adv_history_days=%d",
            liquidity_constraints.max_bar_participation,
            liquidity_constraints.max_position_adv_participation,
            liquidity_constraints.adv_lookback_days,
            liquidity_constraints.min_adv_history_days,
        )

    # Friction/liquidity diagnostics are UI telemetry only. They do not change
    # order generation, execution prices, or portfolio accounting.
    cumulative_rebalance_diag: dict = {}
    last_rebalance_diag: dict = {}
    
    # Initialize ZMQ Publisher (for Qt Dashboard)
    zmq_host = str(cfg.get("ui", "zmq_host", default="127.0.0.1"))
    zmq_port = int(cfg.get("ui", "zmq_port", default=5555))
    publish_every_ticks = int(cfg.get("ui", "publish_every_ticks", default=1))

    pub = ZmqPublisher(host=zmq_host, port=zmq_port)
    if log: log.info("ZMQ Publisher bound to tcp://%s:%d (publish_every_ticks=%d)", zmq_host, zmq_port, publish_every_ticks)

    # 2. Setup State & Logging
    start = parse_dt(cfg.get("market_feed","start"))
    port = exe.snapshot(start)
    logger.append(port)

    # Give SUB clients a brief moment to connect so they don't miss the first tick.
    await asyncio.sleep(0.2)
    # Send a fully-populated NAV packet so the dashboard can render immediately.
    _positions0 = dict(port.positions)
    _pos_values0 = {}
    for _sym, _qty in _positions0.items():
        _px = float(port.mtm_prices.get(_sym, 0.0))
        _pos_values0[_sym] = float(_qty) * _px

    await pub.publish("nav", {
        "ts": port.ts.isoformat(),
        "nav": float(port.nav),
        "cash": float(port.cash),
        "positions": _positions0,
        "pos_values": _pos_values0,
        "visible_symbols": [],
        "prices": {},
        "target_weights": {},
        "frictions": _frictions_payload_for_ui(
            cfg, liquidity_constraints, liquidity_tracker, cumulative_rebalance_diag, last_rebalance_diag
        ),
    })
    await pub.publish("learn", {
        "ts": port.ts.isoformat(),
        "tick": 0,
        "strategy": strat.__class__.__name__,
        "mode": "bootstrap",
        "status": "waiting for first market snapshot",
        "scalars": {},
        "lists": {},
        "weights": {"target": {}},
        "latest_update": {},
    })

    # Lightweight CSV logging (backup for Dashboard)
    nav_csv_path = run_dir / "nav.csv"
    nav_file = open(nav_csv_path, "w", newline="")
    nav_writer = csv.writer(nav_file)
    nav_writer.writerow(["ts", "nav", "cash"])
    nav_writer.writerow([start.isoformat(), port.nav, port.cash])
    nav_file.flush()

    if log: log.info("Engine started. Initial NAV=%.2f", port.nav)

    # 3. The Event Loop Setup
    queue = asyncio.Queue(maxsize=1000) # Buffer to prevent memory explosion
    
    # Start the "Producer" (Market Feed)
    producer_task = asyncio.create_task(_produce_market_data(feed, queue))

    tick_count = 0
    warned_missing_volume = False
    try:
        while True:
            # Wait for next event
            event = await queue.get()

            # Sentinel Check (End of Stream)
            if event is None:
                break

            # --- Dispatch Logic ---

            if isinstance(event, MarketSnapshot):
                tick_count += 1
                
                # 1. Update Execution State (Mark-to-Market)
                exe.update_market(event)
                if liquidity_tracker is not None:
                    liquidity_tracker.update(event.ts, getattr(event, "volumes", {}))
                    if (
                        log
                        and liquidity_constraints.enabled
                        and liquidity_constraints.require_volume_for_buys
                        and not warned_missing_volume
                        and tick_count >= 1000
                        and liquidity_tracker.seen_volume_snapshots == 0
                    ):
                        warned_missing_volume = True
                        log.warning(
                            "Liquidity controls are enabled, but the market feed has emitted no volume data "
                            "after %d ticks. New buys will be blocked while require_volume_for_buys=true. "
                            "Use a cube store with volume.parquet, set market_feed.volume_store_dir to the "
                            "original cube store, or rebuild/repair the cube store preserving volume.parquet.",
                            tick_count,
                        )

                # 2. Strategy Logic
                strat.on_snapshot(event, port)
                
                # 3. Generate Orders
                #
                # Historical behavior restored intentionally: once a strategy
                # has published non-empty target weights, the engine keeps
                # aligning the actual paper portfolio to that target on every
                # market tick. This enables continuous target tracking until
                # the strategy clears or replaces `_last_target_weights`.
                orders: List[OrderRequest] = []
                target_w = getattr(strat, "_last_target_weights", None)
                if isinstance(target_w, dict) and target_w:
                    orders, last_rebalance_diag = _rebalance_with_diagnostics(
                        event.ts,
                        target_w,
                        event,
                        port,
                        liquidity=liquidity_constraints,
                        liquidity_tracker=liquidity_tracker,
                    )
                    _accumulate_rebalance_diag(cumulative_rebalance_diag, last_rebalance_diag)
                    
                    if orders:
                        logger.append_many(orders)
                        
                        # 4. Route Orders (Simulation)
                        # In Paper Trading, this returns Fills immediately.
                        execution_events = exe.place_orders(event.ts, orders)

                        # NOTE: PaperExecutionEngine emits plain dicts (via .model_dump()).
                        # We log everything and only special-case fills + the latest portfolio snapshot.
                        for e_evt in execution_events:
                            logger.append(e_evt)

                            if isinstance(e_evt, dict):
                                kind = e_evt.get("kind")

                                # BROADCAST FILL TO UI
                                if kind == "fill":
                                    await pub.publish("fill", e_evt)

                                # Keep `port` as a PositionSnapshot model for downstream code
                                elif kind == "position_snapshot":
                                    port = PositionSnapshot(**e_evt)
                
                # 5. Portfolio Snapshot (always refresh `port` for correctness)
                #
                # Even though PaperExecutionEngine returns a PositionSnapshot event, we also
                # refresh from the engine here to keep `port` consistent (and to simplify UI publishing).
                refreshed = exe.snapshot(event.ts)
                if not orders:
                    # Only append to logs when no orders were placed (avoids duplicate snapshots).
                    logger.append(refreshed)
                port = refreshed

                # 6. Dashboard & Monitor Updates
                nav_writer.writerow([event.ts.isoformat(), port.nav, port.cash])
                if tick_count % 10 == 0:
                    nav_file.flush()
                # BROADCAST NAV TO UI (throttled to avoid UI backlog in fast backtests)
                if tick_count % publish_every_ticks == 0:
                    # Include positions so the dashboard can render the Symbol/Qty/Value table.
                    positions = dict(port.positions)
                    # Use the execution engine's carried-forward last prices instead of the
                    # raw sparse minute snapshot. This prevents the UI table from showing
                    # temporary zero values when a held symbol is absent for a minute.
                    pos_values = {}
                    for sym, qty in positions.items():
                        px = float(exe.last_prices.get(sym, 0.0))
                        pos_values[sym] = float(qty) * px

                    visible_symbols = sorted(str(sym) for sym in (event.prices or {}).keys())
                    prices_for_ui = {
                        str(sym): float(px)
                        for sym, px in (event.prices or {}).items()
                        if math.isfinite(_safe_float_for_ui(px, 0.0)) and _safe_float_for_ui(px, 0.0) > 0.0
                    }
                    target_weights_for_ui = {
                        str(sym): _safe_float_for_ui(w)
                        for sym, w in (target_w or {}).items()
                        if abs(_safe_float_for_ui(w)) > 1e-12
                    } if isinstance(target_w, dict) else {}

                    await pub.publish("nav", {
                        "ts": event.ts.isoformat(),
                        "nav": float(port.nav),
                        "cash": float(port.cash),
                        "positions": positions,
                        "pos_values": pos_values,
                        "visible_symbols": visible_symbols,
                        "prices": prices_for_ui,
                        "target_weights": target_weights_for_ui,
                        "frictions": _frictions_payload_for_ui(
                            cfg, liquidity_constraints, liquidity_tracker, cumulative_rebalance_diag, last_rebalance_diag
                        ),
                    })
                    await pub.publish("learn", _extract_strategy_telemetry(strat, event, port, tick_count))
                if log and (tick_count % int(cfg.get("run","progress_every_ticks", default=250)) == 0):
                    log.info("Ticks=%d | %s | NAV=%.2f", tick_count, event.ts, port.nav)

            # In Live Trading, you would handle asynchronous fills here:
            # elif isinstance(event, FillEvent):
            #     port.apply_fill(event)
            #     await pub.publish("fill", event.model_dump())

            queue.task_done()

    finally:
        # Cleanup
        producer_task.cancel()
        nav_file.close()
        pub.close()

    if log: log.info("Run finished. Ticks=%d", tick_count)


__all__ = [
    "run_stream",
    "LiquidityConstraints",
    "LiquidityTracker",
    "_rebalance",
    "_rebalance_with_diagnostics",
]
