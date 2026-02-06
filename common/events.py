from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, Literal, Optional
from datetime import datetime

class MarketSnapshot(BaseModel):
    kind: Literal["market_snapshot"] = "market_snapshot"
    ts: datetime
    prices: Dict[str, float]

class OrderRequest(BaseModel):
    kind: Literal["order_request"] = "order_request"
    ts: datetime
    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: int = Field(ge=1, description="Whole number of shares (integer >= 1).")
    order_type: Literal["MARKET"] = "MARKET"

class OrderAck(BaseModel):
    kind: Literal["order_ack"] = "order_ack"
    ts: datetime
    order_id: str
    accepted: bool
    reason: Optional[str] = None

class FillEvent(BaseModel):
    kind: Literal["fill"] = "fill"
    ts: datetime
    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: int
    price: float
    ref_price: float
    fees: float

class PositionSnapshot(BaseModel):
    kind: Literal["position_snapshot"] = "position_snapshot"
    ts: datetime
    cash: float
    positions: Dict[str, int]
    mtm_prices: Dict[str, float]
    nav: float

class TradeOpen(BaseModel):
    kind: Literal["trade_open"] = "trade_open"
    ts: datetime
    trade_id: str
    symbol: str
    side: Literal["LONG", "SHORT"]
    qty: int
    entry_price: float

class TradeClose(BaseModel):
    kind: Literal["trade_close"] = "trade_close"
    ts: datetime
    trade_id: str
    symbol: str
    side: Literal["LONG", "SHORT"]
    qty: int
    entry_price: float
    exit_price: float
    pnl: float
