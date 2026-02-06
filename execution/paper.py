from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
import uuid

from common.events import MarketSnapshot, OrderRequest, OrderAck, FillEvent, PositionSnapshot, TradeOpen, TradeClose
from execution.portfolio import Portfolio
from execution.slippage import FixedBpsSlippage, FixedBpsFees

@dataclass
class PaperExecutionEngine:
    initial_cash: float
    slippage_bps: float = 0.0
    fees_bps: float = 0.0

    def __post_init__(self):
        self.portfolio = Portfolio(cash=float(self.initial_cash), positions={})
        self.last_prices: Dict[str, float] = {}
        self.slippage = FixedBpsSlippage(self.slippage_bps)
        self.fees = FixedBpsFees(self.fees_bps)
        self.open_trades: Dict[str, dict] = {}

    def update_market(self, snap: MarketSnapshot) -> None:
        self.last_prices = dict(snap.prices)

    def place_orders(self, ts: datetime, orders: List[OrderRequest]) -> List[dict]:
        events: List[dict] = []
        for o in orders:
            if o.qty < 1:
                events.append(OrderAck(ts=ts, order_id=o.order_id, accepted=False, reason="qty<1").model_dump())
                continue
            if o.symbol not in self.last_prices:
                events.append(OrderAck(ts=ts, order_id=o.order_id, accepted=False, reason="no_market_price").model_dump())
                continue

            ref_price = float(self.last_prices[o.symbol])
            exec_price = float(self.slippage.apply(o.side, ref_price))
            notional = exec_price * o.qty
            fee = float(self.fees.fee(notional))

            if o.side == "BUY":
                cost = notional + fee
                if self.portfolio.cash + 1e-9 < cost:
                    events.append(OrderAck(ts=ts, order_id=o.order_id, accepted=False, reason="insufficient_cash").model_dump())
                    continue
                self.portfolio.update_cash(-cost)
                self.portfolio.set_position(o.symbol, self.portfolio.position(o.symbol) + o.qty)
            else:
                if self.portfolio.position(o.symbol) < o.qty:
                    events.append(OrderAck(ts=ts, order_id=o.order_id, accepted=False, reason="insufficient_shares").model_dump())
                    continue
                proceeds = notional - fee
                self.portfolio.update_cash(+proceeds)
                self.portfolio.set_position(o.symbol, self.portfolio.position(o.symbol) - o.qty)

            events.append(OrderAck(ts=ts, order_id=o.order_id, accepted=True).model_dump())
            events.append(FillEvent(ts=ts, order_id=o.order_id, symbol=o.symbol, side=o.side,
                                   qty=o.qty, price=exec_price, ref_price=ref_price, fees=fee).model_dump())

            self._trade_bookkeeping(ts, o.symbol, o.side, o.qty, exec_price, events)

        events.append(self.snapshot(ts).model_dump())
        return events

    def snapshot(self, ts: datetime) -> PositionSnapshot:
        mtm = dict(self.last_prices)
        nav = self.portfolio.nav(mtm)
        return PositionSnapshot(ts=ts, cash=self.portfolio.cash, positions=dict(self.portfolio.positions), mtm_prices=mtm, nav=nav)

    def _trade_bookkeeping(self, ts: datetime, symbol: str, side: str, qty: int, price: float, events: List[dict]) -> None:
        # Toy avg-cost, long-only trade tracking
        if side == "BUY":
            if symbol not in self.open_trades:
                trade_id = str(uuid.uuid4())
                self.open_trades[symbol] = {"trade_id": trade_id, "qty": 0, "avg_entry": 0.0}
                events.append(TradeOpen(ts=ts, trade_id=trade_id, symbol=symbol, side="LONG", qty=0, entry_price=price).model_dump())
            t = self.open_trades[symbol]
            new_qty = t["qty"] + qty
            t["avg_entry"] = (t["avg_entry"] * t["qty"] + price * qty) / max(new_qty, 1)
            t["qty"] = new_qty
        else:
            if symbol not in self.open_trades:
                return
            t = self.open_trades[symbol]
            sell_qty = min(qty, t["qty"])
            if sell_qty <= 0:
                return
            pnl = (price - t["avg_entry"]) * sell_qty
            events.append(TradeClose(ts=ts, trade_id=t["trade_id"], symbol=symbol, side="LONG", qty=sell_qty,
                                     entry_price=t["avg_entry"], exit_price=price, pnl=pnl).model_dump())
            t["qty"] -= sell_qty
            if t["qty"] == 0:
                del self.open_trades[symbol]
