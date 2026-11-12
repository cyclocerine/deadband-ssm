"""Causal convolution and linear state-space layers.

Both layers are strictly causal: the output at step ``t`` depends on inputs at
steps ``<= t`` only. The convolution enforces this by left-padding and slicing
the tail; the state-space layer by construction, since ``h_t`` is a function of
``x_{<=t}``.

The recurrence ``h_t = a_t * h_{t-1} + v_t`` is evaluated with a chunked
log-space parallel scan: inside a block of length ``L`` the cumulative decay is
renormalised to the block's first step, so the exponent stays bounded by
``L * max|log a|`` and no term overflows. Work is ``O(T * D * N)`` and the
Python-level loop runs ``T / L`` times rather than ``T`` times.

Two numerical choices make this viable in FP32 on a CPU:

* ``log a`` is clamped to ``[-4, 0]``, i.e. a per-step decay of at least
  ``e^-4 = 0.018``. With ``L = 16`` the largest exponent in the block is
  ``64``, well inside the FP32 range (``e^88``). Real parameters sit at
  ``|log a| <= 1.6``, so the clamp only bounds pathological drift.
* For a time-invariant system the prefix sums are analytic
  (``S_t = (t + 1) log a``), so the scan never materialises a cumulative-sum
  tensor. That halves the memory traffic and removes an ``O(T)`` chain from
  the backward graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["CausalConv1d", "LinearSSM", "parallel_scan"]

_LOG_A_MIN: float = -4.0
_CHUNK: int = 16


def _scan(log_a: Tensor, v: Tensor, chunk: int = _CHUNK) -> Tensor:
    """Evaluate ``h_t = exp(log_a_t) * h_{t-1} + v_t`` with a chunked scan.

    ``log_a`` may be time-invariant (``shape[1] == 1``) or full-length; it is
    clamped to ``[_LOG_A_MIN, 0]``.
    """
    b, t, d, n = v.shape
    if t == 0:
        return v.clone()
    la = log_a.clamp(min=_LOG_A_MIN, max=0.0)
    if la.shape[1] == 1:
        steps = torch.arange(1, t + 1, device=v.device, dtype=v.dtype).view(1, t, 1, 1)
        cum = steps * la  # (1, T, D, N), broadcast, never expanded
    else:
        cum = torch.cumsum(la, dim=1)

    out = torch.empty_like(v)
    h_prev = torch.zeros(b, 1, d, n, dtype=v.dtype, device=v.device)
    for t0 in range(0, t, chunk):
        t1 = min(t0 + chunk, t)
        s_rel = cum[:, t0:t1] - cum[:, t0 : t0 + 1]  # <= 0 inside the block
        decay = torch.exp(s_rel)
        local = torch.cumsum(torch.exp(-s_rel) * v[:, t0:t1], dim=1)
        la_t0 = la[:, t0 : t0 + 1] if la.shape[1] > 1 else la
        carry = torch.exp(la_t0) * h_prev
        h = decay * (local + carry)
        out[:, t0:t1] = h
        h_prev = h[:, -1:]
    return out


def parallel_scan(a: Tensor, v: Tensor, chunk: int = _CHUNK) -> Tensor:
    """Public wrapper: ``h_t = a_t * h_{t-1} + v_t`` for ``a`` in ``(0, 1]``."""
    if a.shape != v.shape:
        raise ValueError(f"shape mismatch: a={tuple(a.shape)} v={tuple(v.shape)}")
    return _scan(torch.log(a.clamp_min(1e-12)), v, chunk)


class CausalConv1d(nn.Module):
    """Depthwise causal convolution over the time axis.

    ``padding = (k - 1) * d`` is applied on both sides (as PyTorch requires for
    dilation) and the tail is sliced away, leaving exactly ``T`` outputs whose
    last one sees ``x_T`` and nothing after it.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
            groups=channels,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Args: ``x`` of shape ``(B, T, C)``. Returns ``(B, T, C)``."""
        t = x.shape[1]
        y = self.conv(x.transpose(1, 2))
        return y[..., :t].transpose(1, 2)


class LinearSSM(nn.Module):
    """Diagonal linear state-space layer (S4D-real), ``O(T)`` in sequence length.

    Continuous dynamics per channel ``d`` and state ``n``::

        h'(t) = A_dn h(t) + B_dn x_d(t),   y_d(t) = sum_n C_dn h_dn(t) + D_d x_d(t)

    with ``A_dn = -exp(log_A_dn) < 0``, discretised by zero-order hold at step
    size ``dt_d``::

        a = exp(dt * A),  b = (a - 1) / A,  h_t = a h_{t-1} + b B x_t

    ``B`` and ``C`` are input-independent, so the layer is a linear time
    invariant system: no ``O(T^2)`` attention, no data-dependent gating.

    Two evaluation strategies, all ``O(T)`` in the recurrence:

    * ``form="fir"`` (default for ``T <= 64``) evaluates the exact
      finite impulse response of the recurrence with one grouped convolution.
      Because the state starts at zero for every window, this is not an
      approximation: ``y_t = sum_{j<=t} k_j x_{t-j}`` with
      ``k_j = sum_n C_dn a_dn^j gain_dn``. It is the fastest path on CPUs
      without vector units, where element-wise scans are bandwidth-bound.
    * ``form="loop"`` steps the recurrence with ``(B, D, N)`` tensors. Minimal
      memory, maximal Python overhead.
    * ``form="scan"`` uses the chunked log-space scan in :func:`_scan`, which
      removes the Python loop and materialises ``(B, T, D, N)``; it is the
      right choice for long sequences.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        dt_min: float = 0.02,
        dt_max: float = 0.2,
        form: str = "auto",
    ) -> None:
        super().__init__()
        if form not in ("auto", "fir", "loop", "scan"):
            raise ValueError(f"form must be auto/fir/loop/scan, got {form!r}")
        self.d_model = d_model
        self.d_state = d_state
        self.form = form
        # Timescales 1..N give a geometric bank of memories from ~1 step to ~N/dt.
        self.log_A = nn.Parameter(torch.log(torch.arange(1.0, d_state + 1)).repeat(d_model, 1))
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(d_model))
        log_dt = torch.log(torch.tensor(dt_min)) + torch.rand(d_model) * (
            torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min))
        )
        self.log_dt = nn.Parameter(log_dt)

    def _coefficients(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(log_a, gain, C)`` with ``log_a`` of shape ``(D, N)``."""
        a_mat = -torch.exp(self.log_A)  # (D, N), negative
        dt = torch.exp(self.log_dt)  # (D,)
        log_a = (dt.unsqueeze(-1) * a_mat).clamp(min=_LOG_A_MIN, max=0.0)
        a = torch.exp(log_a)
        b_bar = (a - 1.0) / a_mat  # exact ZOH input gain, > 0
        return log_a, b_bar * self.B, self.C

    def impulse_response(self, length: int) -> Tensor:
        """Kernel ``k[j, d] = sum_n C_dn a_dn^j gain_dn`` of shape ``(length, D)``."""
        log_a, gain, c = self._coefficients()
        j = torch.arange(length, dtype=log_a.dtype, device=log_a.device).view(-1, 1, 1)
        decay = torch.exp(log_a.unsqueeze(0) * j)  # (K, D, N) <= 1
        return (decay * (c * gain).unsqueeze(0)).sum(-1)  # (K, D)

    def _resolve_form(self, t: int) -> str:
        if self.form != "auto":
            return self.form
        return "fir" if t <= 64 else "scan"

    def forward(self, x: Tensor) -> Tensor:
        """Args: ``x`` of shape ``(B, T, D)``. Returns ``(B, T, D)``."""
        b, t, d = x.shape
        form = self._resolve_form(t)

        if form == "fir":
            k = self.impulse_response(t)  # (T, D), k[j] is the lag-j coefficient
            # Depthwise FIR written as T shifted multiplies. A grouped conv1d
            # with one channel per group is several times slower than this on
            # CPUs without AVX, where it takes a scalar fallback path. The
            # accumulation is in-place on a fresh tensor, so autograd is safe.
            y = k[0].view(1, 1, -1) * x
            for j in range(1, t):
                y[:, j:] = y[:, j:] + k[j] * x[:, :-j]
            return y + self.D * x

        log_a, gain, c = self._coefficients()
        if form == "scan":
            v = x.unsqueeze(-1) * gain.unsqueeze(0).unsqueeze(0)
            h = _scan(log_a.unsqueeze(0).unsqueeze(0), v)
            return (h * c.unsqueeze(0).unsqueeze(0)).sum(-1) + self.D * x

        a = torch.exp(log_a)
        h = torch.zeros(b, d, self.d_state, dtype=x.dtype, device=x.device)
        out = []
        for step in range(t):
            h = a * h + x[:, step].unsqueeze(-1) * gain
            out.append((h * c).sum(-1) + self.D * x[:, step])
        return torch.stack(out, dim=1)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_state={self.d_state}, form={self.form}"
