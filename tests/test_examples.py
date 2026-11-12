"""The examples are documentation; these tests keep them honest.

``examples/qalu_reference.py`` is quoted in the paper as the arithmetic the CUDA
kernel is written against, and ``examples/quickstart.py`` is quoted as a
reproduction. Both claims rot silently if the model or the activation changes, so
the two cheap parts of them are asserted here: the numpy activation against the
module the benchmark trains with, and the reference kernel's closed-form check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import qalu_reference  # noqa: E402


def test_reference_matches_module_forward() -> None:
    torch = pytest.importorskip("torch")
    from core.qalu import QALU

    act = QALU().to(torch.float64)
    p = {k: float(getattr(act, k).detach()) for k in ("alpha", "beta", "gamma", "delta")}
    x = torch.linspace(-8.0, 8.0, 2001, dtype=torch.float64)
    mine = qalu_reference.qalu_forward(x.numpy(), **p)
    assert np.max(np.abs(act(x).detach().numpy() - mine)) < 1e-12


def test_reference_matches_module_gradient() -> None:
    torch = pytest.importorskip("torch")
    from core.qalu import QALU

    act = QALU().to(torch.float64)
    p = {k: float(getattr(act, k).detach()) for k in ("alpha", "beta", "gamma", "delta")}
    x = torch.linspace(-8.0, 8.0, 2001, dtype=torch.float64).requires_grad_(True)
    (grad,) = torch.autograd.grad(act(x).sum(), x)
    g = x.detach().numpy()
    keep = g != 0.0  # the module returns 0 at exactly zero by construction of its clamp
    mine = qalu_reference.qalu_backward(g, np.ones_like(g), **p)
    assert np.max(np.abs(grad.numpy()[keep] - mine[keep])) < 1e-12


def test_overshoot_matches_the_closed_form() -> None:
    """f'(-2*delta) / gamma = 1 + e^-2, the 13.53% the paper quotes."""
    p = dict(alpha=1.0, beta=1.0, gamma=1.5, delta=0.5)
    x = np.array([-2.0 * p["delta"]], dtype=np.float64)
    slope = qalu_reference.qalu_backward(x, np.ones_like(x), **p)[0]
    assert slope == pytest.approx(p["gamma"] * (1.0 + np.exp(-2.0)), rel=1e-12)
