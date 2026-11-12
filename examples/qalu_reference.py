"""Q-ALU in 30 lines of numpy, forward and backward, plus a parity check.

This is the reference the CUDA kernel in ``qalu_kernel.cu`` is written against.
Run ``python examples/qalu_reference.py --check`` to compare it with the
``qalu-pytorch`` implementation the paper trains with, in float32 and in
float64, over a grid that includes the region where the derivative overshoots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for core.qalu

ALPHA, BETA, GAMMA, DELTA = 1.0, 1.0, 1.5, 0.5


def qalu_forward(x: np.ndarray, alpha: float = ALPHA, beta: float = BETA,
                 gamma: float = GAMMA, delta: float = DELTA) -> np.ndarray:
    """f(x) = alpha*x*(1+tanh(x/beta)) for x >= 0, -gamma*x*expm1(x/delta) below."""
    pos = alpha * x * (1.0 + np.tanh(x / beta))
    neg = -gamma * x * np.expm1(x / delta)
    return np.where(x >= 0.0, pos, neg)


def qalu_backward(x: np.ndarray, grad_out: np.ndarray, alpha: float = ALPHA,
                  beta: float = BETA, gamma: float = GAMMA,
                  delta: float = DELTA) -> np.ndarray:
    """d f / d x, with the branch taken on the sign of x rather than on f."""
    t = np.tanh(x / beta)
    pos = alpha * ((1.0 + t) + (x / beta) * (1.0 - t * t))
    e = np.exp(x / delta)
    neg = -gamma * (np.expm1(x / delta) + (x / delta) * e)
    return grad_out * np.where(x >= 0.0, pos, neg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="compare against qalu-pytorch")
    args = ap.parse_args()

    # The overshoot: at x = -2*delta the negative branch is 13.53% steeper than
    # its own large-negative asymptote, whose slope is gamma.
    x = np.array([-2.0 * DELTA], dtype=np.float64)
    slope = qalu_backward(x, np.ones_like(x))[0]
    print(f"f'(-2*delta) = {slope:.6f} = gamma*(1+e^-2) = {GAMMA * (1 + np.exp(-2)):.6f}"
          f"  ({100 * (slope / GAMMA - 1):.2f}% above the asymptote slope gamma)")

    if not args.check:
        return

    import torch

    from core.qalu import QALU

    torch.manual_seed(0)
    for dtype in (torch.float64, torch.float32):
        act = QALU().to(dtype)
        # The module keeps its shape parameters strictly positive through a
        # softplus with a 1e-4 floor, so the defaults land on 1.0001 / 1.0001 /
        # 1.5001 / 0.5001 rather than on the nominal 1.0 / 1.0 / 1.5 / 0.5.
        # Read them back off the module instead of assuming the nominal values,
        # or the comparison measures the softplus floor, not the arithmetic.
        p = {k: float(getattr(act, k).detach()) for k in ("alpha", "beta", "gamma", "delta")}
        grid = torch.cat([
            torch.linspace(-8.0, 8.0, 4001, dtype=dtype),
            torch.tensor([-1e-8, 1e-8, -2.0], dtype=dtype),
        ]).requires_grad_(True)
        out = act(grid)
        (grad,) = torch.autograd.grad(out.sum(), grid)
        g = grid.detach().numpy()
        # x == 0 is dropped from the gradient comparison: the reference
        # implementation clamps each branch input before selecting between them,
        # which makes d/dx at exactly zero come out as 0 instead of alpha. One
        # point of measure zero, but it is why a whole-grid comparison reports
        # an error of 1.0 and looks like a bug.
        keep = g != 0.0
        fwd = qalu_forward(g, **p).astype(g.dtype)
        bwd = qalu_backward(g, np.ones_like(g), **p).astype(g.dtype)
        print(f"{str(dtype):<16} max |f_fwd - ref| = {np.max(np.abs(out.detach().numpy() - fwd)):.2e}   "
              f"max |f_bwd - ref| = {np.max(np.abs(grad.numpy()[keep] - bwd[keep])):.2e}"
              f"   (shape params {p['alpha']:.6f}/{p['gamma']:.6f})")


if __name__ == "__main__":
    main()
