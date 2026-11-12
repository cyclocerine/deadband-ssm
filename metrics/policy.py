"""Causal transforms of a position path, evaluated under the grid's cost model.

The grid's failure is a limit cycle in the *position*, not a missing signal: in
the Q-ALU arm the post-cost Sharpe is almost a deterministic function of turnover
(corr = -0.98 over the ten cells) and the runs that chatter also lose their
information coefficient. These transforms act on the position path rather than on
the network, so their effect is measurable on saved cells without retraining, and
each of them can also be learned end-to-end later.

Every transform here is causal: the position at bar ``t`` depends only on bars
``<= t``, so none of them can import look-ahead into the backtest.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["no_trade_band", "cap_turnover", "smooth_position", "ensemble", "turnover_guard"]


def no_trade_band(target: np.ndarray, theta: float) -> np.ndarray:
    """Hysteresis: trade only when the target is more than ``theta`` away.

    This is the no-trade zone of microstructure practice, applied to the position
    instead of to the activation. Small target changes are ignored; the path
    becomes piecewise constant, so turnover falls roughly in proportion to how
    often the target crosses the band rather than to how much it moves.

    Args:
        target: desired positions in ``[-1, 1]``, one per bar.
        theta: half-width of the band in position units. ``0`` is a no-op.

    Returns:
        The held positions, same length as ``target``.
    """
    target = np.asarray(target, dtype=np.float64)
    if theta <= 0.0 or target.size == 0:
        return target.copy()
    out = np.empty_like(target)
    held = 0.0
    for i, want in enumerate(target):
        if abs(want - held) > theta:
            held = float(want)
        out[i] = held
    return out


def cap_turnover(target: np.ndarray, cap: float) -> np.ndarray:
    """Limit the position change per bar to ``cap`` of notional.

    A hard budget rather than a band: the path follows the target as closely as
    the budget allows. Positions are clipped to ``[-1, 1]``, which can add a small
    amount of turnover at the boundary; that cost is charged in the metrics.
    """
    target = np.asarray(target, dtype=np.float64)
    if cap <= 0.0 or target.size == 0:
        return target.copy()
    step = np.diff(np.concatenate([[0.0], target]))
    out = np.cumsum(np.clip(step, -cap, cap))
    return np.clip(out, -1.0, 1.0)


def smooth_position(target: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential smoothing of the position path.

    ``alpha = 1`` is a no-op, ``alpha = 0.1`` holds most of the previous bar. The
    smoothed path is the classic low-pass filter on the policy: it trades
    responsiveness for turnover at a fixed rate.
    """
    target = np.asarray(target, dtype=np.float64)
    if not 0.0 < alpha <= 1.0 or target.size == 0:
        return target.copy()
    out = np.empty_like(target)
    held = 0.0
    for i, want in enumerate(target):
        held = held + alpha * (float(want) - held)
        out[i] = held
    return out


def ensemble(paths: Sequence[np.ndarray]) -> np.ndarray:
    """Average several position paths bar by bar.

    Averaging cancels the phase of a limit cycle: two seeds that chatter with
    different phase produce a path whose changes partly offset, while a quiet path
    is left almost unchanged. It also removes the single-seed tail, which is what
    the bimodal distribution of the Q-ALU arm consists of.
    """
    stack = np.stack([np.asarray(p, dtype=np.float64) for p in paths], axis=0)
    if stack.shape[0] == 1:
        return stack[0].copy()
    return stack.mean(axis=0)


def turnover_guard(
    target: np.ndarray,
    window: int = 500,
    trip: float = 0.05,
    rearm: float = 0.5,
    cooldown: int = 2000,
    persist: int = 200,
) -> np.ndarray:
    """Hold flat while the policy's own turnover says it is chattering.

    The detector reads the *target* path, not the guarded one, so the guard cannot
    hide the signal it is watching. Once tripped it stays flat until it has seen
    ``cooldown`` consecutive bars whose rolling turnover is below
    ``rearm * trip``; without that hysteresis a chattering policy re-enters every
    window and pays a round trip per episode, which costs more than staying out.

    Args:
        target: desired positions in ``[-1, 1]``, one per bar.
        window: lookback in bars for the turnover estimate (500 = ~1.7 days of
            5-minute bars).
        trip: rolling turnover per bar above which the policy is considered to be
            chattering. The default is the threshold the sweep already uses to
            flag a cell, so the guard fires exactly where the diagnosis does.
        rearm: fraction of ``trip`` the rolling turnover must fall below before
            the cooldown counter starts.
        cooldown: consecutive quiet bars required before trading resumes.
        persist: consecutive bars above ``trip`` required before the guard fires.
            A single burst of trading at the start of a window is not chattering;
            without this debounce the guard flattens policies whose average
            turnover is an order of magnitude below the threshold.

    Returns:
        The guarded positions, same length as ``target``.
    """
    target = np.asarray(target, dtype=np.float64)
    n = target.size
    if n == 0:
        return target.copy()
    delta = np.abs(np.diff(np.concatenate([[0.0], target])))
    out = np.empty_like(target)
    total = 0.0
    tripped = False
    quiet = 0
    hot = 0
    for t in range(n):
        # Warm-up: one full window of evidence is required before the guard may
        # fire. Without it the single entry trade (0 -> position) reads as
        # turnover and flattens every policy that takes a position at all.
        if t >= window:
            mean_turn = total / window
            if not tripped:
                hot = hot + 1 if mean_turn > trip else 0
                if hot >= persist:
                    tripped = True
                    quiet = 0
            else:
                quiet = quiet + 1 if mean_turn < rearm * trip else 0
                if quiet >= cooldown:
                    tripped = False
                    quiet = 0
        out[t] = 0.0 if tripped else target[t]
        total += delta[t]
        if t >= window:
            total -= delta[t - window]
    return out

