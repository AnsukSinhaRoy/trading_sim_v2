from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Dict, List, Any, TYPE_CHECKING, Optional
from datetime import datetime, date
from decimal import Decimal
from collections import defaultdict, deque
from dataclasses import dataclass
import uuid, math, csv, json
import importlib

# ZMQ for Real-Time Dashboard (Qt)
#
# NOTE (Windows): zmq.asyncio + ProactorEventLoop can be unreliable on some setups.
# We publish using a regular ZMQ socket in non-blocking mode for maximum reliability.
import zmq

# --- Project imports (runtime + Pylance type checking) ---
from runner.config import Config, parse_dt
from common.eventlog import EventLogger
from common.events import MarketSnapshot, OrderRequest, PositionSnapshot

if TYPE_CHECKING:
    # Only needed for type checkers / IDE intellisense.
    from common.events import FillEvent, OrderAck, TradeOpen, TradeClose
from execution.paper import PaperExecutionEngine
from market_feed import SyntheticMinuteFeed, FolderMinuteFeed, MatrixStoreMinuteFeed
from market_feed.sanitized_matrix_store_1m import SanitizedMatrixStoreMinuteFeed
from analytics.build import build_derived_from_events

# --- ZMQ Publisher Class ---
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


# --- Factories ---

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
        symbols = cfg.get("market_feed","symbols", default=None)

        return FolderMinuteFeed(
            data_dir=cfg.get("market_feed","data_dir"),
            symbols=symbols,
            start=start,
            end=end_dt,
            fmt=fmt,
            file_pattern=cfg.get("market_feed","file_pattern", default="{symbol}_minute.{ext}"),
            timestamp_col=cfg.get("market_feed","timestamp_col", default="date"),
            price_col=cfg.get("market_feed","price_col", default="close"),
            volume_col=cfg.get("market_feed","volume_col", default="volume"),
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
            volume_store_dir=cfg.get("market_feed", "volume_store_dir", default=None),
        )

    if ftype == "sanitized_matrix_store_1m":
        start = parse_dt(cfg.get("market_feed","start"))
        end = cfg.get("market_feed","end", default=None)
        minutes = cfg.get("market_feed","minutes", default=None)
        if end is None and minutes is None:
            raise ValueError("sanitized_matrix_store_1m requires either market_feed.end or market_feed.minutes")
        if end is None and minutes is not None:
            from datetime import timedelta
            end_dt = start + timedelta(minutes=int(minutes))
        else:
            end_dt = parse_dt(end)

        return SanitizedMatrixStoreMinuteFeed(
            store_dir=cfg.get("market_feed","store_dir"),
            start=start,
            end=end_dt,
            symbols=cfg.get("market_feed","symbols", default=None),
            speed=cfg.get("market_feed","speed", default="fast"),
            volume_store_dir=cfg.get("market_feed", "volume_store_dir", default=None),
            max_abs_return=float(cfg.get("market_feed","max_abs_return", default=0.35)),
            min_price=float(cfg.get("market_feed","min_price", default=1e-9)),
            stats_every_rows=int(cfg.get("market_feed","stats_every_rows", default=0)),
            rebase_after_rejections=int(cfg.get("market_feed","rebase_after_rejections", default=3)),
            rebase_continuity_max_abs_return=float(cfg.get("market_feed","rebase_continuity_max_abs_return", default=0.10)),
            rebase_max_abs_return=float(cfg.get("market_feed","rebase_max_abs_return", default=0.95)),
        )

    raise ValueError(f"Unsupported market_feed.type: {ftype}")

def _make_execution(cfg: Config):
    return PaperExecutionEngine(
        initial_cash=float(cfg.get("execution","initial_cash", default=1_000_000)),
        slippage_bps=float(cfg.get("execution","slippage","bps", default=0)),
        fees_bps=float(cfg.get("execution","fees","bps", default=0)),
    )

_ACRONYMS = {
    # Common short acronyms we want to preserve in class names
    "rl", "xs", "dqn", "ddqn", "ppo", "a2c", "sac", "td3", "lstm", "cnn", "rnn", "mlp"
}

def _snake_to_strategy_class_name(stype: str) -> str:
    """
    Convert a snake_case strategy type (e.g., 'xs_mom_vol_target', 'rl_agent')
    into the conventional Strategy class name.

    Heuristics:
      - keep well-known acronyms in ALLCAPS (RL, XS, DQN, ...)
      - keep 1-2 letter parts in ALLCAPS
      - otherwise TitleCase each part
    """
    parts = [p for p in stype.split("_") if p]
    out = []
    for p in parts:
        pl = p.lower()
        if pl in _ACRONYMS or len(p) <= 2:
            out.append(p.upper())
        else:
            out.append(p.capitalize())
    return "".join(out) + "Strategy"


def _make_strategy(cfg, run_dir=None):
    stype = cfg.get("strategy", "type")
    params = dict(cfg.raw.get("strategy", {}))
    params.pop("type", None)

    # Optional: provide run_dir to strategies that accept it (used for checkpointing, etc.)
    if run_dir is not None:
        params.setdefault("run_dir", str(run_dir))

    def _instantiate(cls):
        """Instantiate strategy; if it does not accept run_dir, retry without it."""
        try:
            return cls(**params)
        except TypeError as e:
            msg = str(e)
            if "run_dir" in params and "run_dir" in msg and (
                "unexpected keyword argument" in msg or "got an unexpected keyword argument" in msg
            ):
                p2 = dict(params)
                p2.pop("run_dir", None)
                return cls(**p2)
            raise

    # Advanced usage: 'some.module.path:ClassName'
    if ":" in stype:
        mod_name, cls_name = stype.split(":", 1)
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, cls_name):
            raise ValueError(f"Strategy class not found: {mod_name}:{cls_name}")
        cls = getattr(mod, cls_name)
        return _instantiate(cls)

    base_mod = f"strategy.{stype}"
    cls_name = _snake_to_strategy_class_name(stype)

    # Try common layouts:
    candidates = [
        base_mod,
        f"{base_mod}.strategy",
        f"{base_mod}.agent",
        f"{base_mod}.impl",
        f"{base_mod}.core",
        f"{base_mod}.main",
    ]

    attempted = []
    last_strategy_classes = None

    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            attempted.append(f"{mod_name} (missing)")
            continue

        attempted.append(mod_name)

        # 1) Exact name match (preferred)
        if hasattr(mod, cls_name):
            cls = getattr(mod, cls_name)
            return _instantiate(cls)

        # 2) Module-provided hook/alias (optional, but convenient)
        if hasattr(mod, "STRATEGY_CLASS"):
            cls = getattr(mod, "STRATEGY_CLASS")
            return _instantiate(cls)

        # 3) Fallback: if there is exactly one *Strategy class in the module, use it
        strategy_classes = [
            obj for name, obj in vars(mod).items()
            if isinstance(obj, type) and name.endswith("Strategy")
        ]
        if strategy_classes:
            last_strategy_classes = [c.__name__ for c in strategy_classes]
        if len(strategy_classes) == 1:
            cls = strategy_classes[0]
            return _instantiate(cls)

    hint = ""
    if last_strategy_classes:
        hint = f" Available Strategy classes seen: {last_strategy_classes}."
    raise ValueError(
        f"Could not resolve strategy.type='{stype}'. Tried: {attempted}. "
        f"Expected class '{cls_name}' in one of the modules above, or set strategy.type "
        f"to an explicit 'module.path:ClassName'.{hint}"
    )



def _safe_float_for_ui(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(v)


def _extract_strategy_telemetry(strat, snap: MarketSnapshot, port: PositionSnapshot, tick_count: int) -> dict:
    payload = {
        "ts": snap.ts.isoformat(),
        "tick": int(tick_count),
        "strategy": strat.__class__.__name__,
    }

    metrics = None
    for name in ("get_dashboard_metrics", "get_ui_metrics", "get_telemetry"):
        fn = getattr(strat, name, None)
        if callable(fn):
            try:
                metrics = fn(snap=snap, portfolio=port)
            except TypeError:
                metrics = fn()
            except Exception:
                metrics = None
            if metrics is not None:
                break

    if not isinstance(metrics, dict):
        target_w = getattr(strat, "_last_target_weights", None)
        weights = {}
        if isinstance(target_w, dict):
            weights = {
                str(sym): _safe_float_for_ui(wt)
                for sym, wt in sorted(target_w.items(), key=lambda kv: abs(_safe_float_for_ui(kv[1])), reverse=True)[:10]
                if abs(_safe_float_for_ui(wt)) > 1e-12
            }

        metrics = {
            "mode": "generic",
            "status": f"tick={tick_count}",
            "scalars": {
                "tick": int(tick_count),
                "nav": _safe_float_for_ui(getattr(port, "nav", 0.0)),
                "cash": _safe_float_for_ui(getattr(port, "cash", 0.0)),
                "target_count": int(len(weights)),
            },
            "weights": {"target": weights},
            "lists": {},
            "latest_update": {},
        }

        for attr in (
            "_decision_steps",
            "_active_turnover",
            "_seg_cum_lr",
            "_last_reward",
            "rebalance_every_minutes",
            "max_assets",
            "lookback",
            "lookback_short",
            "lookback_long",
        ):
            if hasattr(strat, attr):
                metrics.setdefault("scalars", {})[attr] = _safe_float_for_ui(getattr(strat, attr))

        learner = getattr(strat, "_learner", None)
        if learner is not None and hasattr(learner, "updates"):
            metrics.setdefault("scalars", {})["learner_updates"] = int(getattr(learner, "updates", 0))

        buffer = getattr(strat, "_buffer", None)
        if buffer is not None:
            try:
                metrics.setdefault("scalars", {})["buffer_size"] = int(len(buffer))
            except Exception:
                pass

    payload.update(metrics)
    return payload





@dataclass
class LiquidityConstraints:
    """Data-driven execution capacity controls.

    These are not fixed share-count limits. They scale with the actual traded
    volume of each symbol, so liquid large-cap names can absorb larger orders
    while illiquid/penny names are naturally throttled.
    """

    enabled: bool = False
    max_bar_participation: float = 0.10
    max_position_adv_participation: float = 0.10
    adv_lookback_days: int = 20
    min_adv_history_days: int = 5
    require_volume_for_buys: bool = True
    apply_to_sells: bool = True


class LiquidityTracker:
    """Online rolling ADV estimator from per-minute volume snapshots."""

    def __init__(self, lookback_days: int = 20):
        self.lookback_days = max(1, int(lookback_days))
        self.current_date = None
        self.current_day_volume: Dict[str, float] = defaultdict(float)
        self.daily_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.lookback_days))
        self.last_bar_volume: Dict[str, float] = {}
        self.seen_volume_snapshots: int = 0
        self.seen_volume_symbols: set[str] = set()

    def update(self, ts: datetime, volumes: Optional[Dict[str, float]]) -> None:
        d = ts.date()
        if self.current_date is None:
            self.current_date = d
        elif d != self.current_date:
            self._finalize_current_day()
            self.current_date = d
            self.current_day_volume = defaultdict(float)

        clean: Dict[str, float] = {}
        for sym, v in (volumes or {}).items():
            try:
                vv = float(v)
            except Exception:
                continue
            if math.isfinite(vv) and vv > 0:
                clean[str(sym)] = vv
                self.current_day_volume[str(sym)] += vv
        if clean:
            self.seen_volume_snapshots += 1
            self.seen_volume_symbols.update(clean.keys())
        self.last_bar_volume = clean

    def _finalize_current_day(self) -> None:
        for sym, vol in self.current_day_volume.items():
            if math.isfinite(float(vol)) and float(vol) > 0:
                self.daily_history[sym].append(float(vol))

    def adv_shares(self, sym: str, min_history_days: int = 1) -> Optional[float]:
        hist = self.daily_history.get(sym)
        if not hist or len(hist) < int(min_history_days):
            return None
        return float(sum(hist) / len(hist))


def _make_liquidity_constraints(cfg: Config) -> LiquidityConstraints:
    return LiquidityConstraints(
        enabled=bool(cfg.get("execution", "liquidity", "enabled", default=False)),
        max_bar_participation=float(cfg.get("execution", "liquidity", "max_bar_participation", default=0.10)),
        max_position_adv_participation=float(cfg.get("execution", "liquidity", "max_position_adv_participation", default=0.10)),
        adv_lookback_days=int(cfg.get("execution", "liquidity", "adv_lookback_days", default=20)),
        min_adv_history_days=int(cfg.get("execution", "liquidity", "min_adv_history_days", default=5)),
        require_volume_for_buys=bool(cfg.get("execution", "liquidity", "require_volume_for_buys", default=True)),
        apply_to_sells=bool(cfg.get("execution", "liquidity", "apply_to_sells", default=True)),
    )


def _clean_weight(w: float) -> float:
    try:
        v = float(w)
    except Exception:
        return 0.0
    if not math.isfinite(v) or v <= 0:
        return 0.0
    return v


def _bar_qty_cap(sym: str, side: str, snap: MarketSnapshot, liq: LiquidityConstraints) -> Optional[int]:
    if not liq.enabled:
        return None
    if side == "SELL" and not liq.apply_to_sells:
        return None

    vol = getattr(snap, "volumes", {}) or {}
    try:
        bar_vol = float(vol.get(sym, 0.0))
    except Exception:
        bar_vol = 0.0

    if not math.isfinite(bar_vol) or bar_vol <= 0:
        # Unknown volume means no realistic way to estimate participation.
        # For buys, block by default; for sells, follow apply_to_sells.
        if side == "BUY" and liq.require_volume_for_buys:
            return 0
        if side == "SELL" and liq.apply_to_sells:
            return 0
        return None

    cap = int(math.floor(max(0.0, liq.max_bar_participation) * bar_vol))
    return max(0, cap)


def _position_qty_cap(sym: str, liq: LiquidityConstraints, tracker: Optional[LiquidityTracker]) -> Optional[int]:
    if not liq.enabled or tracker is None:
        return None
    adv = tracker.adv_shares(sym, min_history_days=liq.min_adv_history_days)
    if adv is None or not math.isfinite(adv) or adv <= 0:
        return None
    cap = int(math.floor(max(0.0, liq.max_position_adv_participation) * adv))
    return max(0, cap)


def _rebalance(
    ts,
    target_w: Dict[str, float],
    snap: MarketSnapshot,
    port: PositionSnapshot,
    liquidity: Optional[LiquidityConstraints] = None,
    liquidity_tracker: Optional[LiquidityTracker] = None,
) -> List[OrderRequest]:
    prices = snap.prices
    nav = float(port.nav)
    cash = float(port.cash)
    cur_pos = dict(port.positions)
    liq = liquidity or LiquidityConstraints(enabled=False)

    desired_notional = {sym: _clean_weight(w) * nav for sym, w in target_w.items()}
    orders: List[OrderRequest] = []

    # Sell logic. Liquidity caps can force a gradual sell-down when the current
    # position exceeds the ADV-based capacity, even if the optimizer still wants
    # a larger theoretical target.
    for sym, qty in cur_pos.items():
        px = float(prices.get(sym, 0.0))
        if px <= 0:
            continue
        cur_qty = int(qty)
        tgt_val = desired_notional.get(sym, 0.0)
        desired_qty = int(math.floor(max(0.0, tgt_val) / px))

        pos_cap = _position_qty_cap(sym, liq, liquidity_tracker)
        if pos_cap is not None:
            desired_qty = min(desired_qty, pos_cap)

        sell_qty = int(cur_qty - desired_qty)
        if sell_qty <= 0:
            continue

        bar_cap = _bar_qty_cap(sym, "SELL", snap, liq)
        if bar_cap is not None:
            sell_qty = min(sell_qty, bar_cap)
        if sell_qty > 0:
            orders.append(OrderRequest(ts=ts, order_id=str(uuid.uuid4()), symbol=sym, side="SELL", qty=sell_qty))

    # Buy logic. The optimizer may ask for a target weight, but the execution
    # layer only moves toward that target at a realistic participation rate and
    # refuses to accumulate more than a configurable fraction of rolling ADV.
    remaining = cash
    for sym, w in sorted(target_w.items(), key=lambda kv: _clean_weight(kv[1]), reverse=True):
        px = float(prices.get(sym, 0.0))
        if px <= 0:
            continue
        qty = int(cur_pos.get(sym, 0))
        tgt_val = desired_notional.get(sym, 0.0)
        desired_qty = int(math.floor(max(0.0, tgt_val) / px))

        pos_cap = _position_qty_cap(sym, liq, liquidity_tracker)
        if pos_cap is not None:
            desired_qty = min(desired_qty, pos_cap)

        buy_qty = int(desired_qty - qty)
        if buy_qty <= 0:
            continue

        bar_cap = _bar_qty_cap(sym, "BUY", snap, liq)
        if bar_cap is not None:
            buy_qty = min(buy_qty, bar_cap)

        affordable_qty = int(math.floor(remaining / px))
        buy_qty = min(buy_qty, affordable_qty)
        if buy_qty > 0:
            orders.append(OrderRequest(ts=ts, order_id=str(uuid.uuid4()), symbol=sym, side="BUY", qty=buy_qty))
            remaining -= buy_qty * px

    return orders

# --- Core Event Engine ---

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
                    orders = _rebalance(
                        event.ts,
                        target_w,
                        event,
                        port,
                        liquidity=liquidity_constraints,
                        liquidity_tracker=liquidity_tracker,
                    )
                    
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
    #The next line causes OOM, fix this problem in the future by building derived analytics incrementally instead of all at once at the end of the run.
    #build_derived_from_events(run_dir)