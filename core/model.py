"""Deadband-SSM: dual-stream causal sequence model with a Q-ALU core.

Pipeline (per the architecture specification)::

    X (B,T,F) --Linear--> (B,T,D)
      -> DeadbandSSMBlock x n_layers
           Stream A: depthwise causal Conv1d, k=3, dilation 2^i
           Stream B: diagonal linear SSM (S4D-real), O(T)
           Z = LayerNorm(A + B)
           H = act(Z)                      <- Q-ALU lives here
           O = X_in + Dropout(Linear(H))
      -> LayerNorm -> terminal step (B,D)
      -> head: Linear(D,1)  [supervised]  or  Linear(D,1)+Tanh [policy]

The terminal-step readout is what makes the model a *policy*: at bar ``t`` it
emits one position weight from the history ``x_{<=t}`` and nothing else.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from core.layers import CausalConv1d, LinearSSM
from core.qalu import make_activation

__all__ = ["DeadbandSSMBlock", "DeadbandSSM"]

Mode = Literal["policy", "supervised", "both"]


class DeadbandSSMBlock(nn.Module):
    """One dual-stream block: causal conv + linear SSM + Q-ALU + residual."""

    def __init__(
        self,
        d_model: int = 32,
        layer_idx: int = 0,
        kernel_size: int = 3,
        d_state: int = 8,
        dropout: float = 0.1,
        act: str = "qalu",
        use_conv: bool = True,
        use_ssm: bool = True,
    ) -> None:
        super().__init__()
        self.use_conv = use_conv
        self.use_ssm = use_ssm
        if not (use_conv or use_ssm):
            raise ValueError("a block needs at least one temporal stream")
        dilation = 2**layer_idx
        self.conv = CausalConv1d(d_model, kernel_size, dilation) if use_conv else None
        self.ssm = LinearSSM(d_model, d_state) if use_ssm else None
        self.norm = nn.LayerNorm(d_model)
        self.act = make_activation(act)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Args: ``x`` of shape ``(B, T, D)``. Returns ``(B, T, D)``."""
        streams = []
        if self.conv is not None:
            streams.append(self.conv(x))
        if self.ssm is not None:
            streams.append(self.ssm(x))
        z = streams[0] if len(streams) == 1 else streams[0] + streams[1]
        z = self.norm(z)
        return x + self.drop(self.proj(self.act(z)))


class DeadbandSSM(nn.Module):
    """Causal dual-stream state-space policy network.

    Args:
        n_features: input feature dimension ``F``.
        d_model: hidden width ``D`` (default 32, per the specification).
        n_layers: number of :class:`DeadbandSSMBlock` layers.
        d_state: SSM state dimension per channel.
        dropout: residual dropout.
        act: one of ``qalu``, ``gelu``, ``swish``, ``relu``.
        mode: ``policy`` returns ``tanh``-bounded positions in ``[-1, 1]``;
            ``supervised`` returns an unbounded one-step return forecast;
            ``both`` returns ``(position, forecast)`` from two heads on the
            same trunk, which is what the benchmark trains (the forecast head
            keeps the trunk from collapsing onto a constant long bias).
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        n_layers: int = 2,
        d_state: int = 8,
        dropout: float = 0.1,
        act: str = "qalu",
        mode: Mode = "policy",
        kernel_size: int = 3,
        use_conv: bool = True,
        use_ssm: bool = True,
    ) -> None:
        super().__init__()
        if mode not in ("policy", "supervised", "both"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.n_features = n_features
        self.d_model = d_model
        self.in_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList(
            DeadbandSSMBlock(
                d_model=d_model,
                layer_idx=i,
                kernel_size=kernel_size,
                d_state=d_state,
                dropout=dropout,
                act=act,
                use_conv=use_conv,
                use_ssm=use_ssm,
            )
            for i in range(n_layers)
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.head_policy = nn.Sequential(nn.Linear(d_model, 1), nn.Tanh())
        self.head_value = nn.Linear(d_model, 1)
        head: nn.Module = self.head_policy
        if mode == "supervised":
            head = self.head_value
        elif mode == "both":
            head = nn.Identity()  # unused; forward returns both heads
        self.head = head

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Args: ``x`` of shape ``(B, T, F)``.

        Returns ``(B,)`` for the single-head modes and ``(position, forecast)``
        for ``mode="both"``.
        """
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_f(h)[:, -1, :]
        if self.mode == "both":
            return self.head_policy(h).squeeze(-1), self.head_value(h).squeeze(-1)
        return self.head(h).squeeze(-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
