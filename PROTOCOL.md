# Pre-registered evaluation protocol

This file was written **before** the benchmark was run and is not edited
afterwards. Its purpose is to remove the freedom to choose a protocol after
seeing results.

## Question

Does replacing the activation function inside a dual-stream causal sequence
model change out-of-sample, post-cost trading performance on BTCUSDT 5-minute
bars?

Candidates: Q-ALU (Mutaqin, 2026), GELU, Swish/SiLU, ReLU. Everything else in
the model, the data, the optimiser, the epoch budget, the fee, the split
geometry and the seed schedule is identical across candidates.

## Data

* Instrument: BTCUSDT spot, 5-minute bars, Binance archive
  (`data.binance.vision`).
* Sample: 2020-01-01 03:10 UTC to 2026-11-10 23:50 UTC (721,227 bars).
* Features (4): `logret` (1-bar log return), `volz` (20-bar rolling z-score of
  absolute return), `vimb` (normalised taker-buy volume imbalance),
  `hlspread` (log high-low range).
* Target: the next-bar log return, aligned so that a position formed with
  information up to and including bar `t` earns the return of bar `t+1`.
* No forward filling of missing bars: bars are dropped, never interpolated.

## Split geometry

Five expanding-window folds. Fold `i` tests the `i`-th contiguous block of the
last 70% of the sample; training always ends before the block starts.

```
fold i:   [=========== train ===========][purge][= val =][purge][== test ==]
```

* purge gap = `seq_len + 1 + 288` bars (288 bars = 24h embargo).
* validation block = 8% of the sample, immediately before the test block.
* the feature scaler is fitted on the training slice only.
* model selection (early stopping) uses validation Sharpe only. The test block
  is touched once per cell.

## Training

| item | value |
| --- | --- |
| optimiser | AdamW, lr 3e-4, weight decay 1e-4, cosine decay to 5% |
| batch | 32 consecutive windows (label = next bar) |
| loss | negative annualised Sharpe of the post-cost net stream |
| fee | 5 bps per unit turnover |
| grad clip | 1.0 |
| seeds | 0-2 (paired: identical init and batch order across activations) |

## Inference

The model is a policy: at bar `t` it emits one weight in `[-1, 1]` from
`x_{<=t}`. Positions are held for one bar. Turnover is charged on
`|a_t - a_{t-1}|` including the entry from flat.

## Decision rule (fixed in advance)

The headline claim is "Q-ALU beats baseline B" only if **all** of the
following hold for that baseline:

1. mean paired Sharpe difference > 0 across the (fold, seed) cells;
2. the 95% bootstrap CI of that difference excludes 0;
3. the Wilcoxon signed-rank p-value < 0.05;
4. the Diebold-Mariano test on the stitched OOS stream does not reject equal
   mean net return at 5% in favour of the baseline.

If any condition fails, the paper reports the measured effect and states that
the evidence does not support superiority. Negative or null results are
reported as found.

## Amendment 1 — active-return diagnostic (before the full grid)

Recorded before any multi-fold grid run, after a two-cell probe on fold 0.
That probe showed both Q-ALU and GELU converging to a near-constant long
position (exposure 0.99, turnover 1e-3 per bar), where the raw post-cost
Sharpe is dominated by Bitcoin's drift rather than by the policy.

Two additions, neither of which replaces the pre-registered headline:

1. `active_sharpe` — Sharpe of the strategy's net stream minus the underlying
   return stream, i.e. the value added relative to holding the asset. A
   constant-long policy scores 0.000 by construction. Reported alongside the
   raw Sharpe, never instead of it.
2. `fee_drag` — the summed cost of turnover over the test block, in equity
   terms, so that a lower-turnover policy is credited for cost avoidance
   explicitly rather than implicitly.

The decision rule above still governs the headline claim, and it is evaluated
on the raw post-cost Sharpe. The amendment only adds diagnostics.

## Amendment 2 — benchmark pays the same entry cost (before the full grid)

`active_sharpe` subtracts a buy-and-hold stream that is charged the identical
entry fee, so that a policy that never trades after the first bar scores
exactly 0.000 and not a small negative number caused by comparing a
fee-paying strategy against a free one.

## Amendment 3 — training budget is part of the protocol

All four activations receive the same optimiser (AdamW), learning rate, batch
size, epoch count, steps per epoch and batch order; only the activation
differs. The budget is set by what a single commodity CPU can execute for the
whole grid, and it is reported in the paper next to the results, because at a
short budget the run-to-run variance of a single configuration is comparable to
the between-activation difference. Cells that end in either failure mode
(near-constant long position, or chattering with turnover above 0.05 per bar)
are reported as they are, not filtered out.

## Amendment 4 — drift-centred training target (before the full grid)

Recorded after the pilot grid and before the grid reported in the paper.

A Sharpe-ratio objective is invariant to the scale of the position: scaling
`a_t` by `lambda > 0` scales both the mean and the standard deviation of the net
stream by `lambda`. On an asset with a positive drift, the maximiser of that
objective is therefore a constant long position, and the pilot confirmed it:
across 9 pilot cells, every arm that converged sat at exposure 0.99 with
turnover of 1e-3 per bar, while the arms that did not converge chattered at
turnover 0.1 and lost the fee. Extending one Q-ALU cell to 1200 steps (twice
the grid budget) did not change the regime: validation Sharpe settled at
4.28--4.34, out-of-sample exposure was 1.000 with turnover 1.1e-4 per bar. A
validation Sharpe of 4.3 obtained by holding the asset is the asset's own
drift, not timing skill, and it is the maximum of the objective as written.
In neither regime can the choice of activation matter, so the pilot cannot
answer the question the paper asks.

The training target is therefore centred on the mean return of the training
slice:

```
r~_t = r_t - mean(r_train)          (one constant per fold, from train only)
```

The policy loss and the validation criterion used for model selection are both
computed on `r~`. Nothing else changes: the architecture, the optimiser, the
budget, the splits and the purge gaps are untouched, and every reported metric
(raw Sharpe, active Sharpe, IC, drawdown, turnover) is computed on the raw
returns. The offset is a single scalar estimated from the training slice, so no
information crosses the split boundary, and the transformation is applied
identically to all four activations.

The effect is that a constant long position earns zero before fees instead of
the market's drift, so the objective can only be improved by timing. This is a
design correction made before the grid whose results are reported, and it is
recorded here rather than discovered after the fact.

## Amendment 5 — the position is read out from the forecast (before the reported grid)

Centring the target removed the drift but not the degeneracy. A free policy head
has a flat direction: any constant position scores the same as flat on a
drift-free batch Sharpe, so there is no gradient to move it and the head
saturates. The GELU probe at the grid budget ended with exposure 1.000 and
position standard deviation 0.0014 --- the head was pinned at $+1$ and the
network was not using its input at all.

The position is therefore a monotone readout of the forecast,

```
pos_t = tanh(g * f_t)
```

where `f_t` is the one-step forecast and `g` a single learned scalar gain
initialised at one. The head keeps the specified shape --- a linear map followed
by tanh --- but its input is the forecast instead of a second, independent
projection of the trunk. A free head could decouple what the network predicts
from what it trades, which is precisely the failure the pilot exhibited.

Consequences, all intended: the forecast MSE anchors the scale of `f_t` (targets
are standardised, so `f_t = O(1)`), the position cannot saturate without paying
that MSE, and the activation now acts on the policy through one channel --- how
well the trunk forecasts. The parameter count changes from 4458 to 4427 because
the redundant second head is no longer built.
