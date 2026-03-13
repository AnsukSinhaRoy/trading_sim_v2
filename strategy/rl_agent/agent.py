from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
from datetime import timedelta

import torch

from common.events import MarketSnapshot, PositionSnapshot

from .features import FeatureConfig, FeatureEncoder
from .policy import PolicyNet, _masked_softmax, sample_without_replacement
from .learner import Learner, PPOConfig
from .buffer import RolloutBuffer, Transition
from .reward import shaped_reward
from .checkpoint import CheckpointConfig, CheckpointManager


@dataclass
class RLAgentStrategy:
    """Serious long-only RL allocator (PPO Actor–Critic).

    Integration contract (THIS repo):
      - engine calls strat.on_snapshot(snapshot, portfolio)
      - engine reads strat._last_target_weights to generate rebalance orders

    Action space:
      - Select K assets (without replacement) via categorical policy
      - Allocate weights across selected assets via Dirichlet policy

    Objective:
      maximize stable returns via shaped reward:
        cum_log_return - tc_penalty*turnover - vol_penalty*vol - dd_penalty*max_drawdown

    Stoploss overlay:
      if an asset falls stoploss_pct below its trailing peak, we target weight=0 and block re-entry
      for stop_cooldown_minutes.
    """

    # ----- Feature windows (in bars/ticks) -----
    lookback_short: int = 1950
    lookback_long: int = 23400
    corr_short: int = 1950
    corr_long: int = 23400
    min_history: int = 3900

    # ----- Portfolio constraints -----
    max_assets: int = 10
    leverage: float = 1.0  # long-only; sum(weights) <= 1 is typical

    # ----- Trading cadence -----
    rebalance_every_minutes: int = 10080  # weekly by default
    max_turnover: float = 0.6

    # ----- Risk overlay -----
    stoploss_pct: float = 0.10
    stop_cooldown_minutes: int = 10080

    # ----- Reward shaping -----
    tc_penalty: float = 0.002
    vol_penalty: float = 0.10
    dd_penalty: float = 0.25

    # ----- PPO / learning -----
    lr: float = 3e-4
    device: str = "cuda"

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    vf_coef: float = 0.50
    ent_coef: float = 0.01

    train_epochs: int = 6
    update_every: int = 16
    batch_size: int = 16
    temperature: float = 1.0
    grad_clip: float = 1.0

    # ----- Checkpointing -----
    checkpoint_enabled: bool = True
    checkpoint_dir: Optional[str] = None
    checkpoint_every_steps: int = 8
    checkpoint_keep_last: int = 5

    # Provided by engine (optional)
    run_dir: Optional[str] = None

    # ----- internal state -----
    _encoder: FeatureEncoder = field(init=False)
    _policy: PolicyNet = field(init=False)
    _learner: Learner = field(init=False)
    _buffer: RolloutBuffer = field(init=False)
    _ckpt: CheckpointManager = field(init=False)

    _last_target_weights: Dict[str, float] = field(default_factory=dict)

    _last_rebalance_ts: Optional[object] = None  # datetime

    # For reward accumulation during holding period
    _hold_weights: Dict[str, float] = field(default_factory=dict)
    _last_prices: Optional[Dict[str, float]] = None
    _seg_step_lrs: List[float] = field(default_factory=list)
    _seg_cum_lr: float = 0.0

    # Active decision step info (state + action under behavior policy)
    _active_X: Optional[torch.Tensor] = None
    _active_idx: Optional[torch.Tensor] = None
    _active_w: Optional[torch.Tensor] = None
    _active_logp: float = 0.0
    _active_value: float = 0.0
    _active_turnover: float = 0.0
    _decision_steps: int = 0

    _last_reward: float = 0.0
    _last_update_stats: Dict[str, float] = field(default_factory=dict)
    _last_ready_symbol_count: int = 0
    _last_ready_symbols: List[str] = field(default_factory=list)
    _last_selected_symbols: List[str] = field(default_factory=list)

    # Stoploss tracking
    _blocked_until: Dict[str, object] = field(default_factory=dict)  # sym -> datetime
    _peak_price: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        fcfg = FeatureConfig(
            lookback_short=int(self.lookback_short),
            lookback_long=int(self.lookback_long),
            corr_short=int(self.corr_short),
            corr_long=int(self.corr_long),
        )
        self._encoder = FeatureEncoder(fcfg)

        # Determine feature dim from encoder
        in_dim = 10
        self._policy = PolicyNet(in_dim=in_dim, hidden=128)

        ppo_cfg = PPOConfig(
            gamma=float(self.gamma),
            gae_lambda=float(self.gae_lambda),
            clip_ratio=float(self.clip_ratio),
            vf_coef=float(self.vf_coef),
            ent_coef=float(self.ent_coef),
            train_epochs=int(self.train_epochs),
            batch_size=int(self.batch_size),
            grad_clip=float(self.grad_clip),
        )
        self._learner = Learner(self._policy, lr=float(self.lr), device=self.device, cfg=ppo_cfg)
        self._buffer = RolloutBuffer()

        ckc = CheckpointConfig(
            enabled=bool(self.checkpoint_enabled),
            dir=self.checkpoint_dir,
            save_every_steps=int(self.checkpoint_every_steps),
            keep_last=int(self.checkpoint_keep_last),
        )
        self._ckpt = CheckpointManager(ckc, run_dir=self.run_dir)

        self._maybe_load_checkpoint()

    def _maybe_load_checkpoint(self):
        payload = self._ckpt.load()
        if not payload:
            return
        try:
            if "learner" in payload:
                self._learner.load_state_dict(payload["learner"])
            if "buffer" in payload:
                self._buffer.load_state_dict(payload["buffer"])
            self._decision_steps = int(payload.get("decision_steps", self._decision_steps))
            self._blocked_until = payload.get("blocked_until", self._blocked_until)
            self._peak_price = payload.get("peak_price", self._peak_price)
        except Exception:
            # If checkpoint is incompatible, ignore.
            pass

    def _save_checkpoint(self):
        if not self.checkpoint_enabled:
            return
        payload = {
            "learner": self._learner.state_dict(),
            "buffer": self._buffer.state_dict(),
            "decision_steps": int(self._decision_steps),
            "blocked_until": self._blocked_until,
            "peak_price": self._peak_price,
        }
        self._ckpt.save(payload, step=self._decision_steps)

    def _should_rebalance(self, ts) -> bool:
        if self._last_rebalance_ts is None:
            return True
        dt = (ts - self._last_rebalance_ts).total_seconds()
        return dt >= float(self.rebalance_every_minutes) * 60.0

    def _portfolio_step_logret(self, w: Dict[str, float], prev_prices: Dict[str, float], curr_prices: Dict[str, float]) -> float:
        """Approximate portfolio log-return using fixed target weights and price relatives."""
        s = 0.0
        for sym, wt in w.items():
            p0 = prev_prices.get(sym)
            p1 = curr_prices.get(sym)
            if p0 is None or p1 is None or p0 <= 0 or p1 <= 0:
                continue
            s += float(wt) * math.log(float(p1) / float(p0))
        return float(s)

    def _current_weights_from_portfolio(self, port: PositionSnapshot, prices: Dict[str, float]) -> Dict[str, float]:
        nav = float(port.nav) if port else 0.0
        if nav <= 0:
            return {}
        out = {}
        for sym, qty in (port.positions or {}).items():
            px = float(prices.get(sym, 0.0))
            if px <= 0:
                continue
            out[sym] = float(qty) * px / nav
        return out

    def _renorm_long_only(self, w: Dict[str, float]) -> Dict[str, float]:
        w = {k: max(0.0, float(v)) for k, v in w.items() if float(v) > 1e-9}
        s = sum(w.values())
        if s <= 1e-12:
            return {}
        # keep sum at leverage (cap at 1.0 if user accidentally sets >1 for long-only)
        target_sum = min(float(self.leverage), 1.0)
        scale = target_sum / s
        return {k: float(v) * scale for k, v in w.items() if float(v) * scale > 1e-9}

    def _apply_turnover_cap(self, w_new: Dict[str, float], w_old: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        all_syms = set(w_new) | set(w_old)
        turnover = 0.0
        for s in all_syms:
            turnover += abs(w_new.get(s, 0.0) - w_old.get(s, 0.0))

        if self.max_turnover and self.max_turnover > 0 and turnover > self.max_turnover:
            alpha = float(self.max_turnover) / max(1e-12, turnover)
            w_blend = {}
            for s in all_syms:
                w_blend[s] = w_old.get(s, 0.0) + alpha * (w_new.get(s, 0.0) - w_old.get(s, 0.0))
            w_new = self._renorm_long_only(w_blend)
            # recompute
            all_syms2 = set(w_new) | set(w_old)
            turnover = 0.0
            for s in all_syms2:
                turnover += abs(w_new.get(s, 0.0) - w_old.get(s, 0.0))
        return w_new, float(turnover)

    def _stoploss_check(self, ts, prices: Dict[str, float]) -> bool:
        """Returns True if it modified targets (i.e., forced exits)."""
        if not self._last_target_weights or self.stoploss_pct <= 0:
            return False

        changed = False
        w = dict(self._last_target_weights)
        for sym, wt in list(w.items()):
            if wt <= 0:
                continue
            px = float(prices.get(sym, 0.0))
            if px <= 0:
                continue

            peak = float(self._peak_price.get(sym, px))
            if px > peak:
                peak = px
            self._peak_price[sym] = peak

            trigger = peak * (1.0 - float(self.stoploss_pct))
            if px < trigger:
                # force exit + cooldown
                w.pop(sym, None)
                self._blocked_until[sym] = ts + timedelta(minutes=int(self.stop_cooldown_minutes))
                self._peak_price[sym] = px
                changed = True

        if changed:
            w = self._renorm_long_only(w)
            self._last_target_weights = w
            self._hold_weights = dict(w)
        return changed

    def _allowed_symbols(self, ts, symbols: List[str]) -> List[str]:
        out = []
        for s in symbols:
            until = self._blocked_until.get(s)
            if until is not None and ts < until:
                continue
            out.append(s)
        return out

    def _finalize_active_step(self):
        if self._active_X is None:
            return
        r = shaped_reward(
            cum_logret=float(self._seg_cum_lr),
            step_rets=list(self._seg_step_lrs),
            turnover=float(self._active_turnover),
            tc_penalty=float(self.tc_penalty),
            vol_penalty=float(self.vol_penalty),
            dd_penalty=float(self.dd_penalty),
        )
        self._last_reward = float(r)
        self._buffer.add(
            Transition(
                X=self._active_X,
                idx=self._active_idx,
                w=self._active_w,
                logp=float(self._active_logp),
                value=float(self._active_value),
                reward=float(r),
                done=False,
            )
        )

        # Reset segment accumulator
        self._seg_step_lrs.clear()
        self._seg_cum_lr = 0.0

        # PPO update if buffer full
        if len(self._buffer) >= int(self.update_every):
            self._buffer.mark_last_done()
            trans = self._buffer.get()
            self._last_update_stats = self._learner.ppo_update(trans, temperature=float(self.temperature)) or {}
            self._buffer.clear()

        self._decision_steps += 1
        if self.checkpoint_enabled and (self._decision_steps % max(1, int(self.checkpoint_every_steps)) == 0):
            self._save_checkpoint()

    def get_dashboard_metrics(self, snap: MarketSnapshot | None = None, portfolio: PositionSnapshot | None = None) -> Dict[str, object]:
        target_weights = {
            str(sym): float(wt)
            for sym, wt in sorted((self._last_target_weights or {}).items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:10]
        }
        hold_weights = {
            str(sym): float(wt)
            for sym, wt in sorted((self._hold_weights or {}).items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:10]
        }
        blocked_preview = {
            str(sym): str(until)
            for sym, until in sorted((self._blocked_until or {}).items(), key=lambda kv: str(kv[0]))[:10]
        }
        update_stats = {str(k): float(v) if isinstance(v, (int, float)) else v for k, v in (self._last_update_stats or {}).items()}

        return {
            "mode": "online_learning",
            "strategy": self.__class__.__name__,
            "status": f"decisions={self._decision_steps} | updates={getattr(self._learner, 'updates', 0)} | buffer={len(self._buffer)}/{int(self.update_every)}",
            "scalars": {
                "decision_steps": int(self._decision_steps),
                "learner_updates": int(getattr(self._learner, 'updates', 0)),
                "buffer_size": int(len(self._buffer)),
                "update_every": int(self.update_every),
                "last_reward": float(self._last_reward),
                "seg_cum_logret": float(self._seg_cum_lr),
                "active_turnover": float(self._active_turnover),
                "active_value": float(self._active_value),
                "ready_symbols": int(self._last_ready_symbol_count),
                "selected_symbols": int(len(self._last_selected_symbols)),
                "holding_symbols": int(len(self._hold_weights)),
                "blocked_symbols": int(len(self._blocked_until)),
                "rebalance_every_minutes": int(self.rebalance_every_minutes),
                "max_assets": int(self.max_assets),
                "temperature": float(self.temperature),
                "lr": float(self.lr),
                "stoploss_pct": float(self.stoploss_pct),
                "max_turnover": float(self.max_turnover),
            },
            "lists": {
                "selected_symbols": list(self._last_selected_symbols[:12]),
                "ready_sample": list(self._last_ready_symbols[:12]),
                "blocked_symbols": list(sorted(self._blocked_until.keys())[:12]),
            },
            "weights": {
                "target": target_weights,
                "hold": hold_weights,
            },
            "latest_update": update_stats,
            "blocked_until": blocked_preview,
        }

    def on_snapshot(self, snap: MarketSnapshot, portfolio: PositionSnapshot):
        # Update features first
        self._encoder.update(snap.prices)

        # 1) Accumulate holding returns each tick
        if self._hold_weights and self._last_prices is not None:
            step_lr = self._portfolio_step_logret(self._hold_weights, self._last_prices, snap.prices)
            self._seg_step_lrs.append(step_lr)
            self._seg_cum_lr += float(step_lr)
        self._last_prices = dict(snap.prices)

        # 2) Stoploss overlay can trigger immediate exits
        self._stoploss_check(snap.ts, snap.prices)

        # 3) If not rebalance time, keep previous targets
        if not self._should_rebalance(snap.ts):
            return []

        self._last_rebalance_ts = snap.ts

        # 4) Finalize previous decision step (if any)
        self._finalize_active_step()

        # 5) Build universe & filter
        symbols = self._encoder.ready_symbols(min_len=int(self.min_history))
        symbols = self._allowed_symbols(snap.ts, symbols)
        self._last_ready_symbol_count = int(len(symbols))
        self._last_ready_symbols = list(symbols[:50])
        if len(symbols) < 2:
            self._last_selected_symbols = []
            # keep existing weights
            return []

        # 6) Encode features and sample action
        X = self._encoder.encode(symbols)  # CPU
        X_dev = X.to(self.device)

        logits, alpha, value = self._policy(X_dev)
        mask = torch.ones_like(logits, dtype=torch.bool)
        probs = _masked_softmax(logits, mask, temperature=float(self.temperature))

        k = min(int(self.max_assets), int(probs.numel()))
        if k <= 0:
            return []

        idx, logp_sel, ent_sel = sample_without_replacement(probs, k)
        idx_list = idx.detach().cpu().tolist()
        self._last_selected_symbols = [symbols[i] for i in idx_list]

        # Dirichlet for weights (long-only, sum to 1 over selected)
        alpha_sel = alpha.gather(0, idx)
        dist = torch.distributions.Dirichlet(alpha_sel)
        w_sel = dist.sample()
        logp_w = dist.log_prob(w_sel)

        logp = (logp_sel + logp_w).detach()

        # Build weight dict (scaled by leverage)
        target_sum = min(float(self.leverage), 1.0)
        w_new = {symbols[i]: float(w_sel[j].detach().cpu()) * target_sum for j, i in enumerate(idx_list)}
        w_new = self._renorm_long_only(w_new)

        w_old = self._last_target_weights or {}
        w_new, turnover = self._apply_turnover_cap(w_new, w_old)

        # Publish targets
        self._last_target_weights = w_new
        self._hold_weights = dict(w_new)

        # Init stoploss peaks for new holdings
        for sym in self._hold_weights:
            px = float(snap.prices.get(sym, 0.0))
            if px > 0:
                self._peak_price[sym] = max(float(self._peak_price.get(sym, px)), px)

        # Set active decision step info
        self._active_X = X.detach().cpu()
        self._active_idx = idx.detach().cpu()
        self._active_w = w_sel.detach().cpu().float()  # sums to 1
        self._active_logp = float(logp.item())
        self._active_value = float(value.detach().cpu().item())
        self._active_turnover = float(turnover)

        # Reset segment accumulator beginning from this price
        self._seg_step_lrs.clear()
        self._seg_cum_lr = 0.0
        self._last_prices = dict(snap.prices)

        return []
