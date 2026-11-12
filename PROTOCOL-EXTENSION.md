# Extension: cross-timeframe, cross-model sweep

`PROTOCOL.md` is the pre-registration for the main grid and is not edited after
the fact. This file records a **separate, exploratory** experiment that was
added after the main grid had been run, so that the distinction between the two
is on the record.

## Status

Exploratory. Not pre-registered. It cannot confirm or refute the main result and
no claim in the paper depends on it.

## What it varies

The main grid fixes the horizon (5-minute bars) and the sequence model (the
dual-stream block). The sweep varies both, from the same cached 5-minute
archive:

* Timeframes: 5m, 15m, 1h, 4h, 1d, 1w, resampled left-labelled in UTC with
  trailing partial buckets dropped.
* Sequence models: the dual-stream block, a TCN with the same dilation
  schedule, and hand-written LSTM and GRU cells.
* Activations: Q-ALU, GELU, Swish, ReLU.

## What was fixed before the sweep ran

* The grid itself (6 x 4 x 4) and the budget: 2 epochs, 80 steps, one seed.
* The purge gap `T + 1 + embargo` with the per-timeframe embargo table in
  `data/fetcher.py` (288, 96, 24, 6, 1, 1 bars).
* The chattering flag: mean turnover above 0.05 per bar
  (`CHAT_TURNOVER` in `analysis/sweep_report.py`).
* The statistics the report prints: mean and fold spread per cell, the
  per-activation mean and median over cells, and the paired within-group win
  count and median gap.

## What is post-hoc

The prose that reads the table was written after the numbers existed. The
thresholds above were not chosen by looking at the sweep; the emphasis in the
prose was. The paired statistic exists because the mean over cells turned out to
be dominated by single cells that blow up, which was not anticipated.

## Outcome, recorded after the fact

The sweep does not reproduce the ordering of the main grid. On the mean over
(timeframe, model) groups Q-ALU ranks last; on the median over the same groups it
ranks first; on the paired within-group comparison it wins 9 of 24. On the
5-minute panel the same arm is the best of four at the sweep budget and the worst
at the main budget. The paper states this in the sweep subsection instead of
presenting the sweep as replication, and the chattering mechanism proposed for
the main grid is explicitly left uncorroborated by it.

## Why the budget is smaller

A single 5-minute cell costs roughly 500 s on the reference machine (Intel
Celeron N2930, 4 cores, no GPU), and the sweep is 256 cells. The main grid's
budget (5 epochs, 120 steps, two seeds) would put the sweep at several days. The
reduced budget ranks cells against each other; it does not produce numbers
comparable with Table I of the paper, and the sweep table says so in its caption.

## Reproduce

```bash
python run_pipeline.py --stage sweep
python run_pipeline.py --stage sweep-report
```

The sweep resumes: cells already on disk are skipped.
