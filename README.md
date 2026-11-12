# Deadband-SSM

A causal, attention-free dual-stream sequence policy for 5-minute BTCUSDT timing,
built to answer one narrow question: **does the Q-ALU activation change
out-of-sample, post-cost trading performance relative to GELU, Swish and ReLU?**

Q-ALU comes from [`cyclocerine/qalu-pytorch`](https://github.com/cyclocerine/qalu-pytorch)
(Mutaqin, 2026, SSRN 7365098). Everything else here — the data pipeline, the
model, the purged walk-forward engine, the statistics, the paper — is built to
test it under a protocol that was fixed before the benchmark ran
([`PROTOCOL.md`](PROTOCOL.md)).

```
core/        Q-ALU adapter, causal conv + linear SSM layers, model
data/        Binance 5m fetcher (REST with archive fallback), features, splits
metrics/     differential-Sharpe objective, post-cost performance metrics
benchmark/   purged walk-forward cross-validation engine
analysis/    paired tests, Diebold-Mariano, deflated Sharpe, figures, LaTeX
export/      TorchScript + ONNX export, latency and RSS measurement
examples/    runnable quickstart, numpy Q-ALU reference, CUDA kernel
paper/       IEEE 2-column paper, TikZ architecture, generated tables/macros
tests/       causality, scan/FIR equivalence, Q-ALU calculus, purging
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # torch, numpy, pandas, scipy, matplotlib, psutil, qalu-pytorch
pip install -e '.[export]'  # onnx, onnxruntime
```

`qalu-pytorch` is a hard dependency: `core/qalu.py` imports the canonical
package and falls back to an embedded, numerically identical copy only when it
is unavailable, so an offline checkout still reproduces the paper.

## Run

```bash
python run_pipeline.py --stage all --workers 3     # data -> benchmark -> stats -> figures -> export
python run_pipeline.py --stage analysis --figures  # re-analyse existing runs
python run_pipeline.py --stage macros             # re-render LaTeX + README from analysis.json
python run_pipeline.py --stage export --ckpt results/raw/qalu_f0_s0.pt
```

Stages are idempotent and write to `results/`. Per-cell artifacts (positions,
returns, net P&L, timestamps, checkpoints) are stored as compressed `npz`/`pt`
so every number in the paper can be recomputed without retraining.

## Model

```
X (B,T,4) --Linear--> Z (B,T,32)
  -> block x2:
       Stream A: depthwise causal Conv1d, k=3, dilation 2^i, pad (k-1)d, slice [:T]
       Stream B: diagonal linear SSM, h_t = a h_{t-1} + b x_t,  y_t = C h_t + D x_t
       Z = LayerNorm(A + B)
       Z = Z + W_out( Q-ALU(Z) )
  -> LayerNorm -> terminal step -> forecast head w.z
                                -> position = tanh(g * w.z) in [-1,1]
```

* Both streams are causal by construction; the recurrence is evaluated either
  as a linear scan or, for `T <= 64`, as a single grouped convolution with the
  exact closed-form impulse response `k_j = sum_n C_n a_n^j b_n`.
* No self-attention, no `O(T^2)` term. Complexity is `O(T D N)` per layer.
* 4427 parameters (Q-ALU) vs 4419 (ReLU). The position is read out from the
  forecast through one learned gain, so what the network predicts and what it
  trades cannot diverge; PROTOCOL.md, amendments 4 and 5, records why the
  obvious alternative -- a free policy head -- collapses to a constant long.

## Deployment, measured

Numbers below come from `export/report.json`, produced by
`run_pipeline.py --stage export` on the reference machine (Intel Celeron N2930,
4 cores, no AVX/FMA, single thread):

<!-- DEPLOY:START -->
| artifact | size | p50 latency, one bar | peak incremental RSS |
|---|---|---|---|
| PyTorch eager | -- | 19.6 ms | -- |
| TorchScript | 43 KB | 8.5 ms | 229 MB |
| ONNX Runtime | 158 KB | 5.1 ms | **37.1 MB** |

Both exports reproduce the PyTorch policy to float32 rounding: TorchScript bit-exactly, ONNX to 6.0e-08 in maximum absolute deviation. The 50 MB inference budget is met by the ONNX Runtime path; the TorchScript RSS is dominated by the PyTorch runtime itself, not by the 4427-parameter model.
<!-- DEPLOY:END -->

## Examples

Three runnable pieces, each with its captured output in
[`examples/README.md`](examples/README.md):

```bash
python examples/quickstart.py --onnx export/model.onnx   # one real bar, end to end
python examples/qalu_reference.py --check                # numpy Q-ALU vs the module
nvcc -O3 -arch=sm_60 -o qalu_kernel examples/qalu_kernel.cu && ./qalu_kernel
```

`quickstart.py` rebuilds fold 0 with the benchmark's own splitter, refits the
standardizer on the training slice, and prints the forecast and position next to
the values recorded in the cell's `.npz` artifact; the two agree to float32
rounding. `qalu_kernel.cu` is the activation as a CUDA kernel -- its arithmetic
is checked against the numpy reference, but the kernel itself has not been run:
the reference machine has no CUDA device, and the README does not quote a
bandwidth number it did not measure.

## Protocol in one paragraph

Five expanding-window folds over 2020-01-01..2026-11-10 (721k bars). Every
train/validation/test boundary is separated by `T + 1 + 288` bars (the input
window, the label horizon, and a 24-hour embargo). The scaler is fitted on the
training slice only. Four activations share architecture, optimiser, epoch
budget and batch order; seeds are paired so that cell `(fold, seed)` differs
only in the activation. Positions are emitted per bar, held one bar, and
charged 5 bp per unit turnover. Claims require a positive paired mean
difference, a bootstrap CI excluding zero, Wilcoxon `p < 0.05`, and a
Diebold-Mariano test that does not favour the baseline.

## Results

Generated by `--stage analysis` (or `--stage macros`, which re-renders without
re-running the bootstrap) from `results/raw/analysis.json`; edit the report, not
this block.

<!-- RESULTS:START -->
| activation | Sharpe | IR | IC | ann. ret. | max DD | turnover | exposure |
|---|---|---|---|---|---|---|---|
| Q-ALU | -9.722 +/- 6.889 | -8.755 | +0.0090 | -402.6% | -83.9% | 0.0814 | 0.895 |
| GELU | -0.632 +/- 1.525 | -3.956 | +0.0122 | -27.2% | -50.2% | 0.0080 | 0.952 |
| Swish | -0.845 +/- 1.435 | -4.268 | +0.0142 | -39.7% | -54.0% | 0.0081 | 0.950 |
| ReLU | -0.530 +/- 1.182 | -9.636 | +0.0112 | -27.4% | -52.4% | 0.0084 | 0.949 |

* Q-ALU - GELU: dSharpe -9.090 [-12.434, -5.520], Wilcoxon p = 0.0039, Q-ALU wins 1/10 cells
* Q-ALU - Swish: dSharpe -8.877 [-12.416, -5.113], Wilcoxon p = 0.0137, Q-ALU wins 2/10 cells
* Q-ALU - ReLU: dSharpe -9.192 [-13.204, -4.946], Wilcoxon p = 0.0273, Q-ALU wins 3/10 cells

Q-ALU is bimodal rather than uniformly bad: 3 of its 10 cells trade below 0.02 turnover and average a Sharpe of +0.094, while the remaining 7 average -13.929 and pay 573% of notional per year in fees. The protocol forbids selecting the quiet subset, so the reported mean is the mean over both regimes.
<!-- RESULTS:END -->

## Robustness layer

`metrics/policy.py` holds five causal transforms on the position path -- a
no-trade band, a turnover cap, exponential smoothing, a seed ensemble, and a
turnover guard. They sit after the model, so they change no training and no
protocol number; `analysis/robustness.py` replays them over the saved cells and
recomputes post-cost metrics.

```bash
python -m analysis.robustness --dir results/raw --out results/robustness.json
```

The guard is the one transform that behaves like a safety layer rather than a
performance knob. It holds the book flat once the policy's own rolling turnover
has stayed above 0.05 per bar for 200 bars, and re-arms only after 2,000 bars
below half that threshold. It reads the target path, never its own output, and
uses only bars up to `t`.

| variant | Q-ALU | GELU | Swish | ReLU | Q-ALU exposure |
|---|---|---|---|---|---|
| raw | -9.72 | -0.63 | -0.84 | -0.53 | 0.89 |
| turnover guard | -0.89 | -0.65 | -0.87 | -0.53 | 0.28 |
| no-trade band 0.3 | -8.13 | +0.21 | -0.01 | +0.32 | 0.84 |
| turnover cap 0.02 | +0.49 | +0.19 | -0.06 | +0.18 | 0.97 |

Read the last two rows as a warning, not a fix. Banding and capping lift Q-ALU to
roughly +0.2, but they lift every arm by the same amount and leave exposure at
0.93-0.97 with turnover near 0.0004: what improves is the amount of time the book
is long, not the timing. Active Sharpe stays between -0.5 and -1.4 for all four
arms under every transform. The guard is the only row where the other arms do not
move (max 0.03), because it only acts where the diagnosis says the policy is
broken; it converts a -13.9 Sharpe regime into a flat book, and a flat book is
not alpha. Seed ensembling makes Q-ALU worse (-19.11 over 5 folds, exposure 0.25):
the chattering seeds are phase-locked, so averaging cancels exposure and leaves
the cost.

The regime is also not a seed lottery. Within the Q-ALU arm every cell in folds
2-4 chatters (turnover 0.10-0.12, 6 of 6), fold 1 splits by seed (0.0042 vs
0.1151) and fold 0 is quiet; `corr(train_bars, turnover)` is +0.75 for Q-ALU and
-0.77 to -0.37 for the other three arms. Expanding windows make training size and
calendar time collinear, so this does not identify which of the two flips the
switch -- only that picking a luckier seed is not a fix.

```bash
python -m analysis.fold_scan --dir results/raw   # per-fold turnover and Sharpe
```

### A fee surcharge is not the fix

`analysis/surcharge.py` retrains with a fee charged inside the loss only, so the
metrics keep the true 5 bp. At 10x, matched on activation and fold (seed 0):

| act | fold | turnover raw -> 10x | Sharpe raw -> 10x |
|---|---|---|---|
| Q-ALU | 0 | 0.0100 -> 0.0006 | +0.09 -> -1.20 |
| Q-ALU | 1 | 0.0042 -> 0.0005 | +1.70 -> -2.25 |
| Q-ALU | 2 | 0.1146 -> 0.1055 | -12.59 -> -10.49 |
| Q-ALU | 3 | 0.1154 -> 0.1137 | -14.25 -> -13.33 |
| Q-ALU | 4 | 0.1127 -> 0.1128 | -13.50 -> -15.16 |
| GELU | 0 | 0.0079 -> 0.0009 | +0.44 -> -1.22 |
| GELU | 1 | 0.0064 -> 0.0008 | +1.44 -> -2.28 |
| GELU | 2 | 0.0069 -> 0.0009 | +0.95 -> -1.66 |
| GELU | 3 | 0.0059 -> 0.1261 | -0.98 -> -14.66 |
| GELU | 4 | 0.0055 -> 0.1240 | -1.03 -> -16.39 |

Neither outcome helps. On the early folds the policy stops changing position
(turnover 0.006 -> 0.0008 at exposure 0.99): with no timing edge the cheapest way
to satisfy a 10x fee is to hold one position forever, and that loses the only
positive Sharpes in the table. On the later folds it chatters exactly as before.
GELU, quiet at the true fee in every fold, picks up the chattering regime at
folds 3-4 under the surcharge -- the intervention spreads the failure instead of
removing it.

The training history says why better selection cannot rescue these cells: on the
chattering folds the validation Sharpe is already -15 to -17 at every epoch and
the selected epoch wins by 0.003, so the model is not fooling the validation
window -- it cannot find a quiet policy on that slice, where the loss sits an
order of magnitude higher than on the quiet folds (81 vs 7).

### The objective weight is the knob that matters

The total training loss is `DSR + sup_weight * MSE(forecast)`, and `sup_weight`
was fixed at 1.0 before the grid ran. Retraining only the two folds where the
chattering regime lives (3 and 4, seed 0) moves it directly:

| sup_weight | Q-ALU turnover | Q-ALU Sharpe | GELU turnover | GELU Sharpe |
|---|---|---|---|---|
| 1 (protocol) | 0.1141 | -13.88 | 0.0057 | -1.01 |
| 5 | 0.0090 | -1.31 | 0.0113 | -1.95 |
| 20 | 0.0166 | -3.02 | 0.0191 | -3.39 |

At 5 the Q-ALU limit cycle is gone: turnover falls to GELU's own level and the
mean Sharpe over the two folds goes from -13.88 to -1.31. It is a trade, not a
free win -- GELU loses about a Sharpe point and doubles its turnover -- and 20
overshoots, which is why the curve is not monotone. The differential-Sharpe term
is what buys the chattering; the supervised term is what stops it.

## Cross-timeframe sweep

A second grid asks whether the ordering of the four activations is a 5-minute
artefact, and whether it survives replacing the recurrent core. Every timeframe
is resampled from the same cached 5m archive, so the rows differ in horizon, not
in data vintage; the embargo scales with the bar length (288, 96, 24, 6, 1, 1
bars), and the weekly panel runs a single fold with `T = 8` because the archive
holds only ~319 weekly bars.

```bash
python run_pipeline.py --stage sweep         # 6 timeframes x 4 models x 4 activations
python run_pipeline.py --stage sweep-report  # -> paper/tables/sweep.tex + sweep_macros.tex
```

The sweep is resumable: each cell writes `results/sweep/<tf>/<arch>_<act>_f<fold>_s<seed>.json`
next to its `.npz` and `.pt`, and completed cells are skipped on a re-run. Cells
aggregate into `results/sweep/all.json`, which is the only input to the report.

The sweep is an extension added after the pre-registered grid was run, so it is
exploratory by construction; `PROTOCOL-EXTENSION.md` records what was fixed
before it ran and what was written afterwards.

Its budget is deliberately smaller than the main grid's -- 2 epochs, 80 steps,
one seed per cell -- because a single 5m cell costs ~500 s on the reference CPU.
Numbers from the sweep are therefore not comparable with Table I, and no
significance claim is made across cells.

The sweep does not reproduce the ordering of Table I, and the paper says so
instead of presenting it as replication. Averaged over the 24 (timeframe, model)
groups Q-ALU ranks last (-5.449) and Swish first (-2.618); taken as a median over
the same groups Q-ALU ranks first (-0.913), because the mean is set by a few
cells that blow up. The paired comparison inside a group is the statistic that
survives both readings: Q-ALU matches or beats the best other arm in 9 of 24
groups, with a median gap of -0.340 Sharpe. On the 5-minute panel with the
dual-stream block the same arm is the best of four at the sweep budget (-1.089)
and the worst at the paper's budget (-9.722), so the ordering is a property of
the budget, not of the activation. Chattering is near-universal at this budget
(88 of 96 cells), which is why the sweep does not corroborate the chattering
mechanism either.

## Paper

`paper/paper.tex` is the IEEE conference paper. `paper/macros.tex` and
`paper/tables/results.tex` are generated by the analysis stage, and
`paper/tables/sweep.tex` with `paper/tables/sweep_macros.tex` by the sweep
report, so no number in the PDF is typed by hand. Build with:

```bash
cd paper && pdflatex paper && pdflatex paper
```

Two passes are enough because the paper uses `\bibitem` rather than BibTeX; the
PDF carries its title, author and keywords in the document metadata.

## Tests

```bash
pytest -q
```

The suite checks that (i) no output at step `t` depends on inputs after `t`,
(ii) the FIR, loop and scan forms of the SSM agree, (iii) Q-ALU's calculus
matches Proposition 1 (including the exact `1 + e^-2 = 1.1353` overshoot),
(iv) the splitter's purge gaps are respected, (v) the numpy Q-ALU in `examples/`
still matches the module the benchmark trains with, and (vi) the generated LaTeX
stays buildable --- macro names contain letters only, and every macro the prose
quotes is emitted even when a timeframe is missing from the results.

## Citation

```bibtex
@techreport{mutaqin2026deadband,
  author      = {Mutaqin, Faiq Hammam},
  title       = {Deadband-SSM: A Causal Dual-Stream State-Space Policy Network
                 with Q-ALU Activations for Cost-Aware Bitcoin Timing},
  institution = {Independent Researcher},
  year        = {2026}
}
```

## License

MIT. The Binance archive data is redistributed by Binance under its own terms;
`data/fetcher.py` downloads it rather than vendoring it.
