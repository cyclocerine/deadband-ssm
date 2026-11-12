"""Alternative sequence policies for the cross-architecture sweep.

Every model here answers the same question as :class:`core.model.DeadbandSSM`:
given ``x[<=t]`` it emits one position ``pi_t = tanh(g * v_t)`` and the forecast
``v_t`` it is derived from. The point of the sweep is to ask whether the Q-ALU
turnover pathology is a property of the activation or of the dual-stream block,
so the readout, the head, the parameter width and the training budget are held
fixed and only the sequence mixer changes.

The activation is swapped where each architecture actually uses a nonlinearity:

* ``lstm`` / ``gru``: the ``tanh`` on the cell candidate (and, for the LSTM, on
  the cell state) is replaced by the candidate activation. The gates stay
  sigmoid -- they are part of the cell, not part of the ablation.
* ``tcn``: the activation sits after each dilated causal convolution.
* ``ssm``: unchanged, i.e. :class:`core.model.DeadbandSSM`.

The recurrent cells are written out rather than taken from ``nn.LSTM`` because
``nn.LSTM`` fuses its ``tanh`` inside the C++ kernel and cannot be ablated.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from core.layers import CausalConv1d
from core.qalu import make_activation

__all__ = ["SequencePolicy", "LSTMPolicy", "GRUPolicy", "TCNPolicy", "build_model"]


class SequencePolicy(nn.Module):
    """Shared readout: terminal step -> forecast -> ``tanh(g * forecast)``."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        dropout: float = 0.1,
        act: str = "qalu",
        mode: str = "derived",
    ) -> None:
        super().__init__()
        if mode != "derived":
            raise ValueError("baselines implement the derived readout only")
        self.mode = mode
        self.n_features = n_features
        self.d_model = d_model
        self.norm_f = nn.LayerNorm(d_model)
        self.head_value = nn.Linear(d_model, 1)
        gain = nn.Linear(1, 1)
        nn.init.ones_(gain.weight)
        nn.init.zeros_(gain.bias)
        self.head_policy = nn.Sequential(gain, nn.Tanh())

    def encode(self, x: Tensor) -> Tensor:
        """``(B, T, F)`` -> ``(B, T, D)``. Causal by construction in every child."""
        raise NotImplementedError

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.norm_f(self.encode(x))[:, -1, :]
        forecast = self.head_value(h).squeeze(-1)
        return self.head_policy(forecast.unsqueeze(-1)).squeeze(-1), forecast

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LSTMPolicy(SequencePolicy):
    """LSTM whose cell candidate uses the candidate activation instead of tanh."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        dropout: float = 0.1,
        act: str = "qalu",
        mode: str = "derived",
    ) -> None:
        super().__init__(n_features, d_model, dropout, act, mode)
        self.d = d_model
        self.act = make_activation(act)
        self.w_x = nn.Linear(n_features, 4 * d_model)
        self.w_h = nn.Linear(d_model, 4 * d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def encode(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        h = x.new_zeros(b, self.d)
        c = x.new_zeros(b, self.d)
        gate_x = self.w_x(x)  # (B, T, 4D)
        outs = []
        for step in range(t):
            i, f, g, o = torch.split(gate_x[:, step] + self.w_h(h), self.d, dim=-1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            c = f * c + i * self.act(g)
            h = o * self.act(c)
            outs.append(h)
        return self.drop(torch.stack(outs, dim=1))


class GRUPolicy(SequencePolicy):
    """GRU whose candidate state uses the candidate activation instead of tanh."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        dropout: float = 0.1,
        act: str = "qalu",
        mode: str = "derived",
    ) -> None:
        super().__init__(n_features, d_model, dropout, act, mode)
        self.d = d_model
        self.act = make_activation(act)
        self.w_x = nn.Linear(n_features, 3 * d_model)
        self.w_h = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_hn = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def encode(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        h = x.new_zeros(b, self.d)
        gate_x = self.w_x(x)
        outs = []
        for step in range(t):
            r, z, n = torch.split(gate_x[:, step] + self.w_h(h), self.d, dim=-1)
            r, z = torch.sigmoid(r), torch.sigmoid(z)
            n = self.act(n + r * self.w_hn(h))
            h = (1.0 - z) * n + z * h
            outs.append(h)
        return self.drop(torch.stack(outs, dim=1))


class TCNPolicy(SequencePolicy):
    """Two dilated causal convolutions with the candidate activation."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        dropout: float = 0.1,
        act: str = "qalu",
        mode: str = "derived",
        kernel_size: int = 3,
    ) -> None:
        super().__init__(n_features, d_model, dropout, act, mode)
        self.in_proj = nn.Linear(n_features, d_model)
        self.convs = nn.ModuleList(
            [CausalConv1d(d_model, kernel_size, dilation=1), CausalConv1d(d_model, kernel_size, dilation=2)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.acts = nn.ModuleList([make_activation(act) for _ in range(2)])
        self.drop = nn.Dropout(dropout)

    def encode(self, x: Tensor) -> Tensor:
        h = self.in_proj(x)
        for conv, norm, act in zip(self.convs, self.norms, self.acts):
            h = h + self.drop(act(norm(conv(h))))
        return h


def build_model(
    arch: str,
    n_features: int,
    d_model: int = 32,
    n_layers: int = 2,
    d_state: int = 8,
    dropout: float = 0.1,
    act: str = "qalu",
    mode: str = "derived",
    kernel_size: int = 3,
    use_conv: bool = True,
    use_ssm: bool = True,
) -> nn.Module:
    """Factory used by the benchmark so every arm is constructed the same way."""
    if arch == "ssm":
        from core.model import DeadbandSSM

        return DeadbandSSM(
            n_features=n_features,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            dropout=dropout,
            act=act,
            mode=mode,  # type: ignore[arg-type]
            kernel_size=kernel_size,
            use_conv=use_conv,
            use_ssm=use_ssm,
        )
    cls = {"lstm": LSTMPolicy, "gru": GRUPolicy, "tcn": TCNPolicy}.get(arch)
    if cls is None:
        raise ValueError(f"unknown architecture {arch!r}")
    return cls(n_features=n_features, d_model=d_model, dropout=dropout, act=act, mode=mode)
