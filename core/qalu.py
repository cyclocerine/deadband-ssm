"""Q-ALU (Quant Asymmetric Leaky Unit) adapters.

The mathematics is from Mutaqin (2026), *Q-ALU: Quant Asymmetric Leaky Unit*,
SSRN 7365098, https://doi.org/10.2139/ssrn.7365098. The reference
implementation is the ``qalu-pytorch`` package, which this module prefers. An
exact, dependency-free fallback is embedded so the repository reproduces
without network access to PyPI; it mirrors the canonical forward pass,
including the clamp-before-branch trick that keeps ``0 * inf = NaN`` out of the
backward pass, and the softplus reparameterisation that keeps every shape
parameter strictly positive.

Only the canonical (non-approximated) Q-ALU is used here. ``FastQALU`` and the
volatility-conditioned ``QALU_Ultra`` are intentionally out of scope: the
benchmark compares activations, so all four candidates must be exact.
"""

from __future__ import annotations

import math
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QALU", "ACTIVATIONS", "make_activation", "activation_parameter_count"]

ACTIVATIONS: Final[tuple[str, ...]] = ("qalu", "gelu", "swish", "relu")

# Canonical defaults from the Q-ALU reference implementation.
_QALU_DEFAULTS: Final[dict[str, float]] = {
    "alpha": 1.0,
    "beta": 1.0,
    "gamma": 1.5,
    "delta": 0.5,
}

_SOFTPLUS_FLOOR: Final[float] = 1e-4


def _inv_softplus(value: float) -> float:
    """Inverse of ``softplus(v) + _SOFTPLUS_FLOOR`` (stable for all ``v > 0``)."""
    target = float(value) - _SOFTPLUS_FLOOR
    if not target > 0.0:
        raise ValueError(f"Q-ALU parameters must be > {_SOFTPLUS_FLOOR}, got {value!r}")
    if target < 1.0:
        return math.log(math.expm1(target))
    return target + math.log1p(-math.exp(-target))


class _PositiveParam(nn.Module):
    """Wires strictly-positive shape parameters through softplus."""

    def _init_positive(self, trainable: bool, **values: float) -> None:
        for name, value in values.items():
            raw = torch.tensor(_inv_softplus(value), dtype=torch.float32)
            if trainable:
                setattr(self, f"_raw_{name}", nn.Parameter(raw))
            else:
                self.register_buffer(f"_raw_{name}", raw)

    def _positive(self, name: str) -> torch.Tensor:
        return F.softplus(getattr(self, f"_raw_{name}")) + _SOFTPLUS_FLOOR


try:  # pragma: no cover - exercised implicitly by the benchmark
    from qalu import QALU as _ReferenceQALU

    _HAS_QALU_PACKAGE = True
except Exception:  # noqa: BLE001 - any import problem falls back to the local copy
    _ReferenceQALU = None
    _HAS_QALU_PACKAGE = False


if _HAS_QALU_PACKAGE:

    class QALU(_ReferenceQALU):  # type: ignore[misc,valid-type]
        """Canonical Q-ALU from ``qalu-pytorch`` (thin, documented subclass)."""

        def extra_repr(self) -> str:  # pragma: no cover - cosmetic
            return (
                f"alpha={float(self.alpha):.4f}, beta={float(self.beta):.4f}, "
                f"gamma={float(self.gamma):.4f}, delta={float(self.delta):.4f}, "
                f"trainable={self.trainable}"
            )

else:

    class QALU(_PositiveParam):
        """Embedded copy of the canonical Q-ALU forward pass.

        ``f(x) = alpha * x * (1 + tanh(x / beta))`` for ``x >= 0`` and
        ``f(x) = -gamma * x * expm1(x / delta)`` for ``x < 0``.
        """

        def __init__(
            self,
            alpha: float = 1.0,
            beta: float = 1.0,
            gamma: float = 1.5,
            delta: float = 0.5,
            trainable: bool = True,
        ) -> None:
            super().__init__()
            self.trainable = trainable
            self._init_positive(trainable, alpha=alpha, beta=beta, gamma=gamma, delta=delta)

        @property
        def alpha(self) -> torch.Tensor:
            return self._positive("alpha")

        @property
        def beta(self) -> torch.Tensor:
            return self._positive("beta")

        @property
        def gamma(self) -> torch.Tensor:
            return self._positive("gamma")

        @property
        def delta(self) -> torch.Tensor:
            return self._positive("delta")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_pos = torch.clamp(x, min=0.0)
            x_neg = torch.clamp(x, max=0.0)
            pos_term = self.alpha * x_pos * (1.0 + torch.tanh(x_pos / self.beta))
            neg_term = -self.gamma * x_neg * torch.expm1(x_neg / self.delta)
            return torch.where(x >= 0.0, pos_term, neg_term)


def make_activation(name: str, **kwargs: float) -> nn.Module:
    """Return the activation module for ``name``.

    The four candidates are parameter-free except Q-ALU, which carries four
    trainable scalars (alpha, beta, gamma, delta) per instance. That asymmetry
    is reported in the paper; it is 4 scalars against a ~20k-parameter model.
    """
    key = name.lower()
    if key == "qalu":
        params = {**_QALU_DEFAULTS, **kwargs}
        return QALU(**params)  # type: ignore[arg-type]
    if key == "gelu":
        return nn.GELU(approximate="none")
    if key == "swish":
        return nn.SiLU()
    if key == "relu":
        return nn.ReLU()
    raise ValueError(f"unknown activation {name!r}; expected one of {ACTIVATIONS}")


def activation_parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def qalu_gradient_ratio(u: float) -> float:
    """Exact negative-branch gradient in units of ``gamma`` at ``u = -x/delta``.

    ``f'(x)/gamma = (1 - e^{-u}) + u e^{-u}``; maximised at ``u = 2`` with value
    ``1 + e^{-2} = 1.135335...``, i.e. the published +13.53% overshoot.
    """
    return (1.0 - math.exp(-u)) + u * math.exp(-u)


def qalu_positive_gradient_ratio(u: float) -> float:
    """Exact positive-branch gradient in units of ``alpha`` at ``u = x/beta``.

    ``f'(x)/alpha = 1 + tanh(u) + u sech^2(u)``; maximum is 2.1996 at
    ``u ~ 1.1997``, i.e. +10.0% above the terminal slope ``2 alpha``.
    """
    t = math.tanh(u)
    return 1.0 + t + u * (1.0 - t * t)
