"""Out-of-sample statistics: paired tests, HAC inference, bootstrap, DSR.

Nothing here re-estimates a model; it consumes the per-cell artifacts written by
:mod:`benchmark.cross_val` and answers one question: is the difference between
Q-ALU and a baseline larger than the sampling noise of a single market path?

Three complementary views are reported because each has a different weakness:

* **Paired fold x seed test** (Wilcoxon signed-rank) treats every (fold, seed)
  cell as one observation. It is the cleanest control for seed noise but
  assumes cells are exchangeable, which overlapping folds are not.
* **Diebold-Mariano on the stitched OOS stream** respects the time ordering and
  uses a Newey-West HAC variance, so it does not pretend that 500k correlated
  5-minute bars are 500k independent draws.
* **Deflated Sharpe ratio** corrects the headline Sharpe for the number of
  configurations tried and for non-normal returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "PairTest",
    "paired_comparison",
    "dm_test",
    "stationary_bootstrap_sharpe_diff",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "sharpe",
]

BARS_PER_YEAR = 105_120


def sharpe(net: np.ndarray, bars_per_year: int = BARS_PER_YEAR) -> float:
    net = np.asarray(net, dtype=np.float64)
    sd = net.std(ddof=1)
    return float(net.mean() / sd * math.sqrt(bars_per_year)) if sd > 0 else 0.0


@dataclass
class PairTest:
    """Q-ALU minus baseline on paired (fold, seed) cells."""

    baseline: str
    n_pairs: int
    mean_diff: float
    median_diff: float
    ci_low: float
    ci_high: float
    wilcoxon_p: float
    ttest_p: float
    cohens_dz: float
    wins: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "baseline": self.baseline,
            "n_pairs": self.n_pairs,
            "mean_diff": self.mean_diff,
            "median_diff": self.median_diff,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "wilcoxon_p": self.wilcoxon_p,
            "ttest_p": self.ttest_p,
            "cohens_dz": self.cohens_dz,
            "wins": self.wins,
        }


def _bootstrap_mean_ci(x: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_comparison(
    qalu: Sequence[float],
    baseline: Sequence[float],
    name: str,
    n_boot: int = 10_000,
) -> PairTest:
    """Compare two aligned vectors of per-cell Sharpe ratios."""
    a = np.asarray(qalu, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"unaligned cells: {a.shape} vs {b.shape}")
    d = a - b
    if np.allclose(d, 0.0):
        return PairTest(name, len(d), 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0)
    lo, hi = _bootstrap_mean_ci(d, n_boot=n_boot)
    w = stats.wilcoxon(a, b, zero_method="zsplit") if np.any(d != 0) else None
    t = stats.ttest_rel(a, b)
    sd = d.std(ddof=1)
    return PairTest(
        baseline=name,
        n_pairs=len(d),
        mean_diff=float(d.mean()),
        median_diff=float(np.median(d)),
        ci_low=lo,
        ci_high=hi,
        wilcoxon_p=float(w.pvalue) if w is not None else 1.0,
        ttest_p=float(t.pvalue),
        cohens_dz=float(d.mean() / sd) if sd > 0 else 0.0,
        wins=int((d > 0).sum()),
    )


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano statistic with a Newey-West HAC variance.

    ``loss_a``/``loss_b`` are per-bar losses; the null is equal expected loss.
    Returns ``(statistic, two_sided_p)``. Negative statistics favour ``a``.
    """
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    n = len(d)
    if n < 10:
        return 0.0, 1.0
    dbar = d.mean()
    dc = d - dbar
    gamma0 = float(dc @ dc) / n
    var = gamma0
    for lag in range(1, h + 1):
        w = 1.0 - lag / (h + 1.0)
        cov = float(dc[lag:] @ dc[:-lag]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return 0.0, 1.0
    stat = dbar / math.sqrt(var / n)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(stat)))
    return float(stat), float(p)


def stationary_bootstrap_sharpe_diff(
    net_a: np.ndarray,
    net_b: np.ndarray,
    n_boot: int = 2_000,
    mean_block: int = 288,
    seed: int = 0,
    bars_per_year: int = BARS_PER_YEAR,
) -> tuple[float, float, float]:
    """Politis-Romano stationary bootstrap CI for ``SR(a) - SR(b)``."""
    a = np.asarray(net_a, dtype=np.float64)
    b = np.asarray(net_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("series must be aligned")
    n = len(a)
    p = 1.0 / mean_block
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = 0
        while i < n:
            start = rng.integers(0, n)
            length = rng.geometric(p)
            take = min(length, n - i)
            idx[i : i + take] = (start + np.arange(take)) % n
            i += take
        diffs[k] = sharpe(a[idx], bars_per_year) - sharpe(b[idx], bars_per_year)
    obs = sharpe(a, bars_per_year) - sharpe(b, bars_per_year)
    return obs, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def probabilistic_sharpe_ratio(
    sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_benchmark: float = 0.0,
    bars_per_year: int = BARS_PER_YEAR,
) -> float:
    """P(SR > benchmark) under the Bailey-Lopez de Prado approximation."""
    if n_obs < 3:
        return 0.0
    sr0 = sr_benchmark / math.sqrt(bars_per_year)
    sr_h = sr / math.sqrt(bars_per_year)
    denom = math.sqrt(1.0 - skew * sr_h + 0.25 * (kurtosis - 1.0) * sr_h**2)
    z = (sr_h - sr0) * math.sqrt(n_obs - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    n_trials: int,
    sr_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    bars_per_year: int = BARS_PER_YEAR,
) -> tuple[float, float]:
    """Deflated Sharpe ratio and the Sharpe hurdle it is measured against.

    ``sr_variance`` is the cross-trial variance of the (annualised) Sharpe
    estimates; ``n_trials`` is the number of configurations searched.
    """
    if n_trials < 2 or sr_variance <= 0:
        return probabilistic_sharpe_ratio(sr, n_obs, skew, kurtosis, 0.0, bars_per_year), 0.0
    euler = 0.5772156649015329
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    hurdle = math.sqrt(sr_variance) * ((1.0 - euler) * z1 + euler * z2)
    return probabilistic_sharpe_ratio(sr, n_obs, skew, kurtosis, hurdle, bars_per_year), float(hurdle)


def aggregate(results: Iterable[dict], key: str = "sharpe") -> dict[str, dict[str, np.ndarray]]:
    """Group per-cell results by activation, keeping (fold, seed) ordering."""
    from collections import defaultdict

    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        out[r["activation"]][key].append(float(r[key]))
        out[r["activation"]]["fold"].append(float(r["fold"]))
        out[r["activation"]]["seed"].append(float(r["seed"]))
    return {a: {k: np.asarray(v) for k, v in d.items()} for a, d in out.items()}
