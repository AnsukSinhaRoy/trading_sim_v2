import math
from collections import defaultdict
from typing import Dict, Optional

from PyQt6.QtWidgets import QTableWidgetItem


def make_friction_state() -> dict:
    return {
        "nav": None,
        "initial_nav": None,
        "fills": 0,
        "buy_fills": 0,
        "sell_fills": 0,
        "gross_notional": 0.0,
        "buy_notional": 0.0,
        "sell_notional": 0.0,
        "fees": 0.0,
        "slippage_cost": 0.0,
        "total_cost": 0.0,
        "last_config": {},
        "last_liquidity": {},
        "last_rebalance": {},
        "cumulative_rebalance": {},
        "per_symbol": defaultdict(lambda: {
            "fills": 0,
            "gross_notional": 0.0,
            "fees": 0.0,
            "slippage_cost": 0.0,
            "total_cost": 0.0,
        }),
    }


def _finite_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(v)


def _fmt_num(value, digits: int = 4) -> str:
    v = _finite_float(value, default=float("nan"))
    if not math.isfinite(v):
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.2f}"
    return f"{v:,.{digits}f}"


def _fmt_money(value) -> str:
    v = _finite_float(value, default=float("nan"))
    if not math.isfinite(v):
        return "-"
    return f"{v:,.2f}"


def _fmt_pct(value) -> str:
    v = _finite_float(value, default=float("nan"))
    if not math.isfinite(v):
        return "-"
    return f"{100.0 * v:,.2f}%"


def _fmt_bps(value) -> str:
    v = _finite_float(value, default=float("nan"))
    if not math.isfinite(v):
        return "-"
    return f"{v:,.2f} bps"


def _fmt_bool(value) -> str:
    return "Yes" if bool(value) else "No"


def _fmt_list(value, limit: int = 12) -> str:
    vals = [str(x) for x in (value or [])]
    if not vals:
        return "-"
    head = vals[:limit]
    suffix = f" … +{len(vals) - limit}" if len(vals) > limit else ""
    return ", ".join(head) + suffix


def update_friction_from_nav(state: dict, nav_payload: dict) -> None:
    if not isinstance(nav_payload, dict):
        return

    nav = _finite_float(nav_payload.get("nav"), default=0.0)
    if nav > 0:
        state["nav"] = nav
        if state.get("initial_nav") is None:
            state["initial_nav"] = nav

    fr = nav_payload.get("frictions", {}) or {}
    if isinstance(fr, dict):
        state["last_config"] = dict(fr.get("config", {}) or {})
        state["last_liquidity"] = dict(fr.get("liquidity", {}) or {})
        state["last_rebalance"] = dict(fr.get("last_rebalance", {}) or {})
        state["cumulative_rebalance"] = dict(fr.get("cumulative_rebalance", {}) or {})


def update_friction_from_fill(state: dict, fill: dict) -> None:
    if not isinstance(fill, dict):
        return
    sym = str(fill.get("symbol", "")) or "-"
    side = str(fill.get("side", "")).upper()
    qty = int(_finite_float(fill.get("qty"), default=0.0))
    if qty <= 0:
        return

    exec_price = _finite_float(fill.get("price"), default=0.0)
    ref_price = _finite_float(fill.get("ref_price"), default=exec_price)
    fees = abs(_finite_float(fill.get("fees"), default=0.0))
    gross_notional = abs(exec_price * qty)

    if side == "BUY":
        slippage_cost = (exec_price - ref_price) * qty
        state["buy_fills"] += 1
        state["buy_notional"] += gross_notional
    elif side == "SELL":
        slippage_cost = (ref_price - exec_price) * qty
        state["sell_fills"] += 1
        state["sell_notional"] += gross_notional
    else:
        slippage_cost = abs(exec_price - ref_price) * qty

    total_cost = fees + slippage_cost

    state["fills"] += 1
    state["gross_notional"] += gross_notional
    state["fees"] += fees
    state["slippage_cost"] += slippage_cost
    state["total_cost"] += total_cost

    ps = state["per_symbol"][sym]
    ps["fills"] += 1
    ps["gross_notional"] += gross_notional
    ps["fees"] += fees
    ps["slippage_cost"] += slippage_cost
    ps["total_cost"] += total_cost


def _set_three_col_rows(table, rows) -> None:
    table.setRowCount(0)
    for key, value, note in rows:
        r = table.rowCount()
        table.insertRow(r)
        table.setItem(r, 0, QTableWidgetItem(str(key)))
        table.setItem(r, 1, QTableWidgetItem(str(value)))
        table.setItem(r, 2, QTableWidgetItem(str(note)))


def render_friction_tab(window) -> None:
    if not hasattr(window, "friction_cost_table"):
        return

    state = getattr(window, "_friction_state", None) or {}
    nav = _finite_float(state.get("nav"), default=0.0)
    initial_nav = _finite_float(state.get("initial_nav"), default=0.0)
    gross = _finite_float(state.get("gross_notional"), default=0.0)
    fees = _finite_float(state.get("fees"), default=0.0)
    slip = _finite_float(state.get("slippage_cost"), default=0.0)
    total = _finite_float(state.get("total_cost"), default=0.0)

    cost_bps = (total / gross * 10000.0) if gross > 0 else float("nan")
    fee_bps = (fees / gross * 10000.0) if gross > 0 else float("nan")
    slip_bps = (slip / gross * 10000.0) if gross > 0 else float("nan")
    drag_initial = (total / initial_nav) if initial_nav > 0 else float("nan")
    drag_nav = (total / nav) if nav > 0 else float("nan")
    turnover_initial = (gross / initial_nav) if initial_nav > 0 else float("nan")

    if hasattr(window, "frictions_summary_lbl"):
        window.frictions_summary_lbl.setText(
            f"Realized friction cost: {_fmt_money(total)} | Fees: {_fmt_money(fees)} | "
            f"Slippage: {_fmt_money(slip)} | Avg cost: {_fmt_bps(cost_bps)} | "
            f"Turnover/initial NAV: {_fmt_num(turnover_initial, 2)}x"
        )

    cost_rows = [
        ("Fill Count", f"{int(state.get('fills', 0)):,}", "Executed fills received by the dashboard"),
        ("Buy / Sell Fills", f"{int(state.get('buy_fills', 0)):,} / {int(state.get('sell_fills', 0)):,}", "Direction split"),
        ("Gross Traded Notional", _fmt_money(gross), "Sum of absolute executed notional"),
        ("Buy Notional", _fmt_money(state.get("buy_notional", 0.0)), "Executed buy notional"),
        ("Sell Notional", _fmt_money(state.get("sell_notional", 0.0)), "Executed sell notional"),
        ("Total Fees", _fmt_money(fees), "Explicit transaction fee drag"),
        ("Total Slippage Cost", _fmt_money(slip), "Execution price versus reference price; positive means worse execution"),
        ("Total Realized Friction Cost", _fmt_money(total), "Fees plus slippage drag"),
        ("Fee Rate", _fmt_bps(fee_bps), "Fees divided by gross traded notional"),
        ("Slippage Rate", _fmt_bps(slip_bps), "Slippage cost divided by gross traded notional"),
        ("Total Cost Rate", _fmt_bps(cost_bps), "Total friction cost divided by gross traded notional"),
        ("Drag vs Initial NAV", _fmt_pct(drag_initial), "How much initial capital has been consumed by realized frictions"),
        ("Drag vs Current NAV", _fmt_pct(drag_nav), "Realized friction cost relative to current NAV"),
        ("Turnover vs Initial NAV", f"{_fmt_num(turnover_initial, 2)}x", "Gross traded notional divided by initial NAV"),
    ]
    _set_three_col_rows(window.friction_cost_table, cost_rows)

    cfg = state.get("last_config", {}) or {}
    config_rows = [
        ("Initial Cash", _fmt_money(cfg.get("initial_cash")), "Execution starting capital from YAML"),
        ("Slippage Model", cfg.get("slippage_model", "-"), "Current implementation supports fixed bps in paper execution"),
        ("Slippage bps", _fmt_bps(cfg.get("slippage_bps")), "Applied against reference price: buy higher, sell lower"),
        ("Fees Model", cfg.get("fees_model", "-"), "Current implementation supports fixed bps fees"),
        ("Fees bps", _fmt_bps(cfg.get("fees_bps")), "Applied on absolute executed notional"),
        ("Liquidity Enabled", _fmt_bool(cfg.get("liquidity_enabled", False)), "Whether execution capacity constraints are active"),
        ("Max Bar Participation", _fmt_pct(cfg.get("max_bar_participation")), "Max fraction of current minute volume allowed per order decision"),
        ("Max Position ADV Participation", _fmt_pct(cfg.get("max_position_adv_participation")), "Max final position as a fraction of rolling ADV"),
        ("ADV Lookback Days", cfg.get("adv_lookback_days", "-"), "Rolling daily volume window"),
        ("Min ADV History Days", cfg.get("min_adv_history_days", "-"), "Minimum completed days needed before ADV cap is used"),
        ("Require Volume for Buys", _fmt_bool(cfg.get("require_volume_for_buys", False)), "Blocks buys when bar volume is missing"),
        ("Apply Liquidity to Sells", _fmt_bool(cfg.get("apply_to_sells", False)), "If yes, sells can also be throttled by volume"),
    ]
    _set_three_col_rows(window.friction_config_table, config_rows)

    liq = state.get("last_liquidity", {}) or {}
    last = state.get("last_rebalance", {}) or {}
    cum = state.get("cumulative_rebalance", {}) or {}
    liquidity_rows = [
        ("Volume Snapshots Seen", f"{int(liq.get('volume_snapshots_seen', last.get('volume_snapshots_seen', 0)) or 0):,}", "Number of market snapshots that carried positive volume data"),
        ("Volume Symbols Seen", f"{int(liq.get('volume_symbols_seen', last.get('volume_symbols_seen', 0)) or 0):,}", "Number of symbols with usable volume observations"),
        ("Rebalance Checks", f"{int(cum.get('rebalance_checks', 0)):,}", "Times the execution layer tried to align holdings to target weights"),
        ("Orders Submitted", f"{int(cum.get('orders', 0)):,}", "Total order requests generated after constraints"),
        ("Desired Buy Notional", _fmt_money(cum.get("desired_buy_notional")), "Raw buy movement needed before liquidity/cash throttling"),
        ("Submitted Buy Notional", _fmt_money(cum.get("submitted_buy_notional")), "Buy movement actually sent as orders"),
        ("Deferred Buy Notional", _fmt_money(cum.get("deferred_buy_notional")), "Buy movement delayed/blocked by liquidity or cash"),
        ("Desired Sell Notional", _fmt_money(cum.get("desired_sell_notional")), "Raw sell movement needed before liquidity throttling"),
        ("Submitted Sell Notional", _fmt_money(cum.get("submitted_sell_notional")), "Sell movement actually sent as orders"),
        ("Deferred Sell Notional", _fmt_money(cum.get("deferred_sell_notional")), "Sell movement delayed/blocked by liquidity"),
        ("Bar Cap Hits", f"{int(cum.get('bar_cap_hits', 0)):,}", "Current bar volume cap reduced an intended trade"),
        ("ADV Cap Hits", f"{int(cum.get('adv_cap_hits', 0)):,}", "Rolling ADV position cap reduced a desired target quantity"),
        ("Cash Cap Hits", f"{int(cum.get('cash_cap_hits', 0)):,}", "Cash availability reduced a buy order"),
        ("Missing-volume Blocks", f"{int(cum.get('missing_volume_blocks', 0)):,}", "Orders blocked because volume was unavailable and the liquidity rule required it"),
        ("Last Desired Buy/Sell", f"{_fmt_money(last.get('desired_buy_notional'))} / {_fmt_money(last.get('desired_sell_notional'))}", "Most recent rebalance demand before throttling"),
        ("Last Submitted Buy/Sell", f"{_fmt_money(last.get('submitted_buy_notional'))} / {_fmt_money(last.get('submitted_sell_notional'))}", "Most recent order notional after throttling"),
        ("Bar-capped Symbols", _fmt_list(cum.get("symbols_bar_capped")), "Examples, capped list is truncated for UI size"),
        ("ADV-capped Symbols", _fmt_list(cum.get("symbols_adv_capped")), "Examples, capped list is truncated for UI size"),
        ("Missing-volume Symbols", _fmt_list(cum.get("symbols_missing_volume")), "Examples, capped list is truncated for UI size"),
    ]
    _set_three_col_rows(window.liquidity_diag_table, liquidity_rows)

    # Per-symbol realized cost ranking.
    per_symbol: Dict[str, dict] = state.get("per_symbol", {}) or {}
    rows = []
    for sym, vals in per_symbol.items():
        sgross = _finite_float(vals.get("gross_notional"), 0.0)
        stotal = _finite_float(vals.get("total_cost"), 0.0)
        scost_bps = (stotal / sgross * 10000.0) if sgross > 0 else float("nan")
        rows.append((sym, int(vals.get("fills", 0)), sgross, vals.get("fees", 0.0), vals.get("slippage_cost", 0.0), stotal, scost_bps))
    rows.sort(key=lambda x: abs(x[5]), reverse=True)

    window.friction_symbol_table.setRowCount(0)
    for sym, fills, sgross, sfees, sslip, stotal, scost_bps in rows[:500]:
        r = window.friction_symbol_table.rowCount()
        window.friction_symbol_table.insertRow(r)
        window.friction_symbol_table.setItem(r, 0, QTableWidgetItem(str(sym)))
        window.friction_symbol_table.setItem(r, 1, QTableWidgetItem(f"{fills:,}"))
        window.friction_symbol_table.setItem(r, 2, QTableWidgetItem(_fmt_money(sgross)))
        window.friction_symbol_table.setItem(r, 3, QTableWidgetItem(_fmt_money(sfees)))
        window.friction_symbol_table.setItem(r, 4, QTableWidgetItem(_fmt_money(sslip)))
        window.friction_symbol_table.setItem(r, 5, QTableWidgetItem(_fmt_money(stotal)))
        window.friction_symbol_table.setItem(r, 6, QTableWidgetItem(_fmt_bps(scost_bps)))
