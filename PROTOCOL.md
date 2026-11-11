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

1. mean paired Sharpe difference > 0 across the 15 (fold, seed) cells;
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
