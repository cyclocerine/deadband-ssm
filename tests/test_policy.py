"""Causality and behaviour of the position-path transforms.

The grid's headline claim is that no look-ahead enters the backtest. A transform
applied to the position path can break that claim without touching the model, so
every transform here is checked against the truncation property: the output for
bars ``[:k]`` must equal the first ``k`` outputs of the full-series run.
"""

from __future__ import annotations

import numpy as np
import pytest

from metrics.policy import cap_turnover, ensemble, no_trade_band, smooth_position, turnover_guard


def _chatter(n: int = 3000) -> np.ndarray:
    """Alternating full-size position: the limit cycle the guard must catch."""
    return np.where(np.arange(n) % 2 == 0, 1.0, -1.0)


def _quiet(n: int = 3000) -> np.ndarray:
    return np.full(n, 0.8)


def _walk(n: int = 3000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(np.cumsum(rng.normal(0.0, 0.05, n)) / 10.0, -1.0, 1.0)


TRANSFORMS = {
    "band": lambda p: no_trade_band(p, 0.2),
    "cap": lambda p: cap_turnover(p, 0.02),
    "smooth": lambda p: smooth_position(p, 0.2),
    "guard": lambda p: turnover_guard(p, window=200, trip=0.05, cooldown=500),
}


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
@pytest.mark.parametrize("series", ["walk", "chatter"])
def test_transform_is_causal(name: str, series: str) -> None:
    path = _walk() if series == "walk" else _chatter()
    fn = TRANSFORMS[name]
    full = fn(path)
    for k in (1, 37, 500, 1500):
        part = fn(path[:k])
        np.testing.assert_allclose(part, full[:k], atol=1e-12, err_msg=f"{name} looks ahead at k={k}")


def test_noops_return_the_input() -> None:
    path = _walk()
    np.testing.assert_array_equal(no_trade_band(path, 0.0), path)
    np.testing.assert_array_equal(cap_turnover(path, 0.0), path)
    np.testing.assert_allclose(smooth_position(path, 1.0), path, atol=1e-12)


def test_band_and_cap_reduce_turnover() -> None:
    path = _walk()
    raw = np.abs(np.diff(np.concatenate([[0.0], path]))).sum()
    assert np.abs(np.diff(np.concatenate([[0.0], no_trade_band(path, 0.2)]))).sum() < raw
    capped = cap_turnover(path, 0.01)
    assert np.abs(np.diff(np.concatenate([[0.0], capped]))).max() <= 0.01 + 1e-12


def test_guard_flattens_chatter_and_spares_a_quiet_path() -> None:
    chatter = _chatter()
    guarded = turnover_guard(chatter, window=200, trip=0.05, cooldown=500)
    # It must trip within the first window and stay flat: exposure is the cost.
    assert np.mean(np.abs(guarded[400:])) < 0.05
    quiet = _quiet()
    np.testing.assert_array_equal(turnover_guard(quiet, window=200, trip=0.05), quiet)


def test_ensemble_averages_and_keeps_shape() -> None:
    a, b = _quiet(), _chatter()
    out = ensemble([a, b])
    assert out.shape == a.shape
    np.testing.assert_allclose(out, (a + b) / 2.0)
    np.testing.assert_array_equal(ensemble([a]), a)
