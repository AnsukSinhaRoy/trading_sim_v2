from __future__ import annotations

from datetime import timedelta

from execution.paper import PaperExecutionEngine
from market_feed import FolderMinuteFeed, MatrixStoreMinuteFeed, SyntheticMinuteFeed
from market_feed.sanitized_matrix_store_1m import SanitizedMatrixStoreMinuteFeed
from runner.config import Config, parse_dt


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


from strategy.sparse_switch_mv import SparseSwitchMVStrategy


SUPPORTED_STRATEGIES = {
    "sparse_switch_mv": SparseSwitchMVStrategy,
    "strategy.sparse_switch_mv:SparseSwitchMVStrategy": SparseSwitchMVStrategy,
}


def _make_strategy(cfg, run_dir=None):
    """Build the configured strategy.

    The refactored simulator is intentionally sparse-only. Keeping a narrow
    factory is less clever, but much easier to understand and safer to maintain.
    Add a new strategy here only when the simulator genuinely needs to support it.
    """
    stype = str(cfg.get("strategy", "type"))
    params = dict(cfg.raw.get("strategy", {}))
    params.pop("type", None)

    strategy_cls = SUPPORTED_STRATEGIES.get(stype)
    if strategy_cls is None:
        supported = ", ".join(sorted(SUPPORTED_STRATEGIES))
        raise ValueError(
            f"Unsupported strategy.type={stype!r}. This refactor is sparse-only. "
            f"Supported values: {supported}."
        )

    try:
        return strategy_cls(**params)
    except TypeError as exc:
        raise TypeError(
            f"Could not construct {strategy_cls.__name__} from strategy config. "
            f"Check for misspelled/obsolete parameters. Original error: {exc}"
        ) from exc


# Public aliases used by newer code. The underscored names remain for backward compatibility.
make_feed = _make_feed
make_execution = _make_execution
make_strategy = _make_strategy
