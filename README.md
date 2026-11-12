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

## Paper

`paper/paper.tex` is the IEEE conference paper; `paper/macros.tex` and
`paper/tables/results.tex` are generated by the analysis stage, so no number in
the PDF is typed by hand. Build with:

```bash
cd paper && pdflatex paper && pdflatex paper
```

## Tests

```bash
pytest -q
```

The suite checks that (i) no output at step `t` depends on inputs after `t`,
(ii) the FIR, loop and scan forms of the SSM agree, (iii) Q-ALU's calculus
matches Proposition 1 (including the exact `1 + e^-2 = 1.1353` overshoot), and
(iv) the splitter's purge gaps are respected, and (v) the numpy Q-ALU in
`examples/` still matches the module the benchmark trains with.

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
