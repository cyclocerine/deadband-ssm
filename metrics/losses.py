"""Cost-aware differential Sharpe objective and out-of-sample metrics.

The training objective follows the specification::

    net_t = a_t * r_t - fee * |a_t - a_{t-1}|
    L     = -( mean(net) / (std(net) + eps) ) * sqrt(bars_per_year)

which is the negative annualised Sharpe ratio of the *net* (post-cost) return
stream produced by the policy over the sampled batch. Because the fee term
enters the same stream, the optimiser cannot buy risk-adjusted return with
turnover: every unit of position change is charged at 5 bps.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Final

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "DifferentialSharpeLoss",
    "PerformanceMetrics",
    "compute_metrics",
    "max_drawdown",
    "annualise",
]

EPS: Final[float] = 1e-6


class DifferentialSharpeLoss(nn.Module):
    """Negative annualised Sharpe of net returns, with a turnover fee.

    Args:
        fee_ratio: per-unit-turnover cost. 5 bps = 0.0005.
        bars_per_year: annualisation factor (5-minute bars, 24/7 -> 105,120).
        eps: variance floor.
    """

    def __init__(self, fee_ratio: float = 5e-4, bars_per_year: int = 105_120, eps: float = EPS) -> None:
        super().__init__()
        self.fee_ratio = float(fee_ratio)
        self.bars_per_year = int(bars_per_year)
        self.eps = float(eps)

    def forward(
        self,
        positions: Tensor,
        returns: Tensor,
        prev_position: Tensor | float = 0.0,
    ) -> Tensor:
        """Args:
            positions: ``(B,)`` target weights decided at each bar.
            returns: ``(B,)`` realised return of the *next* bar.
            prev_position: position held before ``positions[0]`` (scalar or
                ``(1,)`` tensor), needed for the first trade's cost.
        """
        if positions.shape != returns.shape:
            raise ValueError(f"shape mismatch: {tuple(positions.shape)} vs {tuple(returns.shape)}")
        prev = torch.cat([torch.as_tensor(prev_position, dtype=positions.dtype, device=positions.device).reshape(1), positions[:-1]])
        turnover = torch.abs(positions - prev)
        net = positions * returns - self.fee_ratio * turnover
        return -net.mean() / (net.std(unbiased=False) + self.eps) * float(np.sqrt(self.bars_per_year))

    def extra_repr(self) -> str:
        return f"fee_ratio={self.fee_ratio}, bars_per_year={self.bars_per_year}"


@dataclass
class PerformanceMetrics:
    """Out-of-sample statistics of a net return stream."""

    sharpe: float
    ann_return: float
    ann_vol: float
    max_drawdown: float
    calmar: float
    sortino: float
    turnover: float
    hit_rate: float
    exposure: float
    n_bars: int
    total_return: float
    active_sharpe: float = 0.0
    fee_drag: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def annualise(returns: np.ndarray, bars_per_year: int) -> tuple[float, float]:
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    return mean * bars_per_year, std * float(np.sqrt(bars_per_year))


def max_drawdown(net_returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of the compounded equity curve."""
    equity = np.cumprod(1.0 + net_returns)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0)) if equity.size else 0.0


def compute_metrics(
    positions: np.ndarray,
    returns: np.ndarray,
    fee_ratio: float = 5e-4,
    bars_per_year: int = 105_120,
    prev_position: float = 0.0,
) -> PerformanceMetrics:
    """Metrics of the strategy defined by ``positions`` over ``returns``.

    ``positions[t]`` is held over bar ``t+1`` and earns ``returns[t]``; the cost
    of changing from ``positions[t-1]`` is charged at ``t``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    prev = np.concatenate([[prev_position], positions[:-1]])
    turnover = np.abs(positions - prev)
    net = positions * returns - fee_ratio * turnover

    ann_ret, ann_vol = annualise(net, bars_per_year)
    downside = net[net < 0]
    dd = max_drawdown(net)

    # Active stream: what the policy adds or subtracts relative to holding the
    # asset outright. The benchmark pays the same entry cost, so a constant-long
    # policy scores exactly 0.000 here and a raw Sharpe near buy-and-hold cannot
    # masquerade as timing skill.
    bh_prev = np.concatenate([[0.0], np.ones_like(returns)[:-1]])
    bh = returns - fee_ratio * np.abs(1.0 - bh_prev)
    active = net - bh
    a_ret, a_vol = annualise(active, bars_per_year)
    return PerformanceMetrics(
        sharpe=float(ann_ret / ann_vol) if ann_vol > 0 else 0.0,
        ann_return=ann_ret,
        ann_vol=ann_vol,
        max_drawdown=dd,
        calmar=float(ann_ret / abs(dd)) if dd < 0 else 0.0,
        sortino=float(ann_ret / (np.std(downside, ddof=1) * np.sqrt(bars_per_year)))
        if downside.size > 1
        else 0.0,
        turnover=float(turnover.mean()),
        hit_rate=float(np.mean((positions * returns) > 0)) if positions.size else 0.0,
        exposure=float(np.mean(np.abs(positions))),
        n_bars=int(positions.size),
        total_return=float(np.prod(1.0 + net) - 1.0),
        active_sharpe=float(a_ret / a_vol) if a_vol > 0 else 0.0,
        fee_drag=float(fee_ratio * turnover.sum()),
    )
