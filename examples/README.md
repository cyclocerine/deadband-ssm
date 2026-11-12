# Examples

Three runnable pieces, each answering a different question about the model.

| file | question it answers |
| --- | --- |
| `quickstart.py` | What does one trained cell actually output on a real out-of-sample bar? |
| `qalu_reference.py` | Is the Q-ALU formula in the paper the arithmetic the code runs? |
| `qalu_kernel.cu` | What does the activation look like as a CUDA kernel? |

All commands are run from the repository root with the project environment
(`/home/faiq/.venv-deadband` in the paper's setup, or any venv with
`pyproject.toml` installed).

## `quickstart.py` — one bar, end to end

```console
$ python examples/quickstart.py --onnx export/model.onnx
cell        qalu f0 s0  (4427 params)
bar         221368  2022-02-09 07:40:00+00:00  (test slice 216368..317339)
window      torch.Size([1, 30, 4])  features ('logret', 'volz', 'vimb', 'hlspread')
forecast    v_T = -1.589132   (artifact -1.589132)
position    pi_T = tanh(v_T) = -0.945674   (artifact -0.945674)
difference  5.96e-08 on the position, 2.38e-07 on the forecast
next bar    r = +0.115924%   traded 0.002437   net = -0.109748% of notional
onnx        export/model.onnx  pi_T = -0.945675   |onnx - torch| = 5.96e-08
```

The script rebuilds the fold geometry with the benchmark's own splitter, refits
the standardizer on the training slice only, loads the checkpoint, and prints
the forecast and position next to the values recorded in the cell's `.npz`
artifact. The two agree to float32 rounding (`5.96e-08`), which is what makes
the printed line a check rather than a demonstration.

Two details worth noticing, because they are the difference between a
reproduction and a coincidence:

- The panel is truncated to the `n_samples` recorded in the checkpoint's JSON.
  Binance adds bars continuously, and `PurgedWalkForward` derives every boundary
  from the sample count, so an unpinned panel shifts fold 0 by however many bars
  arrived since the run — here 86 bars, enough to move the position in the third
  decimal and make a correct model look slightly wrong.
- The standardizer is fitted on `features[train]` only. Fitting on the full panel
  would put the test period's mean into the inputs, which is the leak the purge
  is there to prevent.

The printed bar is not a win. `net = -0.109748%` is what the model does on this
bar: it is short (`pi_T = -0.946`) into a bar that rose `+0.116%`. That is the
honest way to show a single sample of a strategy with a per-bar Sharpe near
zero — the grid result, not this bar, is the evidence.

## `qalu_reference.py` — the formula, checked

```console
$ python examples/qalu_reference.py --check
f'(-2*delta) = 1.703003 = gamma*(1+e^-2) = 1.703003  (13.53% above the asymptote slope gamma)
torch.float64    max |f_fwd - ref| = 8.88e-16   max |f_bwd - ref| = 1.78e-15   (shape params 1.000100/1.500100)
torch.float32    max |f_fwd - ref| = 9.54e-07   max |f_bwd - ref| = 8.34e-07   (shape params 1.000100/1.500100)
```

The numpy implementation is 30 lines and is the reference the CUDA kernel is
written against. The check compares it with the `qalu-pytorch` module the
benchmark trains with, over `x ∈ [-8, 8]` plus the points either side of zero.

Two things in that output are not rounding, and both are worth knowing before
you trust a comparison:

- The shape parameters read back as `1.0001 / 1.5001`, not `1.0 / 1.5`. The
  module keeps them strictly positive through a softplus with a `1e-4` floor, so
  a comparison against the nominal constants measures the floor and reports an
  error of `1.6e-3` that looks like a bug in one of the two implementations.
  Read the parameters off the module.
- `x = 0` is excluded from the gradient comparison. The module clamps each
  branch input before selecting between them, which makes `d/dx` at exactly zero
  come out as `0` instead of `alpha`. One point of measure zero, but a whole-grid
  comparison reports an error of `1.0` and looks like a real defect.

## `qalu_kernel.cu` — the activation as a CUDA kernel

```console
$ nvcc -O3 -arch=sm_60 -o qalu_kernel examples/qalu_kernel.cu && ./qalu_kernel
n=4194304  non-finite=0
max |f_fwd - ref| = ...   max |f_bwd - ref| = ...  (float32 vs double)
f'(-2*delta) = 1.703003   gamma*(1+e^-2) = 1.703003
forward: ... ms/call over 4194304 elements  (... GB/s effective)
```

**Status: not executed.** The paper's reference machine is an Intel Celeron
N2930 with no CUDA device and no `nvcc`, so the kernel has never run. The two
`...` fields above are deliberately left unquoted rather than filled with a
number from somewhere else.

What *was* checked: the two device math functions (`qalu_fwd`, `qalu_grad`) were
extracted into a plain C++ translation unit, compiled with `g++ -O2 -Wall`, and
evaluated on a 4001-point grid. Against the numpy reference the agreement is
`1.25e-06` forward and `5.51e-07` backward (float32 arithmetic against a float64
reference), and at `x = -2δ` the derivative is `1.703003` against the closed form
`γ(1+e^-2) = 1.703003`. The arithmetic is verified; the launch, the memory
traffic and the bandwidth figure are not.

Design notes that the file comments expand on:

- The derivative branches on the sign of `x`, not on the sign of `f`. `tanh`
  saturates in float32 beyond `|x/β| > 9` and `expm1(x/δ)` overflows to `-inf`
  below `x/δ < -88`; a branch that multiplies its output by a saturated
  exponential produces `0 * inf = NaN` in the backward pass.
- `sech²` is written as `1 - tanh²`, which stays finite where `tanh` saturates,
  and the `x·exp(x/δ)` term is kept separate from `expm1` so it does not lose
  precision near zero.
- The self-test sizes the array at 4.2 M elements so the timing is a bandwidth
  measurement rather than a launch-overhead measurement: the forward pass reads
  and writes 4 bytes per element, so 8 bytes of traffic per element.
