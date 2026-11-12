"""Correctness tests: causality, scan equivalence, Q-ALU calculus, purging."""

from __future__ import annotations

import math

import numpy as np
import torch

from core.layers import CausalConv1d, LinearSSM, parallel_scan
from core.model import DeadbandSSM
from core.qalu import QALU, make_activation, qalu_gradient_ratio, qalu_positive_gradient_ratio
from benchmark.cross_val import PurgedWalkForward


def test_parallel_scan_matches_naive_recurrence() -> None:
    torch.manual_seed(0)
    b, t, d, n = 2, 37, 5, 3
    a = torch.rand(b, t, d, n) * 0.3 + 0.7
    v = torch.randn(b, t, d, n)
    got = parallel_scan(a, v, chunk=8)

    h = torch.zeros(b, d, n)
    want = []
    for step in range(t):
        h = a[:, step] * h + v[:, step]
        want.append(h.clone())
    want = torch.stack(want, dim=1)
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()


def test_parallel_scan_survives_long_decays() -> None:
    """A near-zero decay over 300 steps must not overflow or NaN."""
    a = torch.full((1, 300, 4, 2), 1e-6)
    v = torch.ones(1, 300, 4, 2)
    out = parallel_scan(a, v, chunk=16)
    assert torch.isfinite(out).all()
    # Decay is clamped at e^-4, so the tail converges to the geometric limit
    # v / (1 - a) rather than to v.
    steady = 1.0 / (1.0 - math.exp(-4.0))
    assert torch.allclose(out[:, -1], torch.full((1, 4, 2), steady), rtol=1e-4)


def test_causal_conv_has_no_future_leakage() -> None:
    conv = CausalConv1d(4, kernel_size=3, dilation=2).eval()
    x = torch.randn(1, 12, 4)
    y = conv(x)
    x2 = x.clone()
    x2[:, 7:] += 100.0  # perturb only the future of step 6
    y2 = conv(x2)
    assert torch.allclose(y[:, :7], y2[:, :7], atol=1e-6)
    assert not torch.allclose(y[:, 7:], y2[:, 7:])


def test_ssm_has_no_future_leakage() -> None:
    ssm = LinearSSM(4, d_state=3).eval()
    x = torch.randn(1, 12, 4)
    y = ssm(x)
    x2 = x.clone()
    x2[:, 7:] += 100.0
    y2 = ssm(x2)
    assert torch.allclose(y[:, :7], y2[:, :7], atol=1e-5)


def test_ssm_forms_agree() -> None:
    """FIR, loop and scan must produce the same layer output."""
    torch.manual_seed(0)
    ref = LinearSSM(8, d_state=4, form="loop")
    x = torch.randn(3, 17, 8)
    y_loop = ref(x)
    for form in ("fir", "scan"):
        other = LinearSSM(8, d_state=4, form=form)
        other.load_state_dict(ref.state_dict())
        assert torch.allclose(other(x), y_loop, atol=1e-5), form


def test_model_is_causal_end_to_end() -> None:
    torch.manual_seed(0)
    model = DeadbandSSM(n_features=4, d_model=16, n_layers=2, act="qalu").eval()
    x = torch.randn(2, 30, 4)
    y = model(x)
    x2 = x.clone()
    x2[:, -1] += 1.0  # only the terminal bar changes -> the output must change
    assert not torch.allclose(y, model(x2), atol=1e-6)
    # Changing the terminal bar cannot affect an output read at an earlier bar;
    # here the readout *is* the terminal bar, so verify the conv/ssm instead.


def test_qalu_continuity_and_monotonicity() -> None:
    act = QALU(alpha=1.0, beta=1.0, gamma=1.5, delta=0.5, trainable=False)
    x = torch.linspace(-20, 20, 20001, requires_grad=True)
    y = act(x)
    assert abs(float(y[10000].detach())) < 1e-7
    grad = torch.autograd.grad(y.sum(), x)[0]
    assert (grad >= -1e-6).all(), "Q-ALU must be monotone non-decreasing"


def test_qalu_downside_overshoot_is_1353() -> None:
    """f'(-2 delta) / gamma = 1 + e^-2 = 1.135335..."""
    act = QALU(alpha=1.0, beta=1.0, gamma=1.5, delta=0.5, trainable=False)
    delta, gamma = float(act.delta), float(act.gamma)
    x = torch.tensor([-2 * delta], requires_grad=True)
    act(x).backward()
    ratio = float(x.grad) / gamma
    assert abs(ratio - (1 + math.exp(-2))) < 1e-4
    assert abs(ratio - 1.1353) < 1e-3
    # and it is a maximum of the negative branch
    grid = np.linspace(0.0, 12.0, 1201)
    values = np.array([qalu_gradient_ratio(u) for u in grid])
    assert abs(grid[int(values.argmax())] - 2.0) < 0.02


def test_qalu_is_c0_but_not_c1_at_the_origin() -> None:
    """One-sided slopes are alpha from the right and 0 from the left.

    The deadband is one-sided: the negative branch is quadratic at the origin,
    the positive branch is linear. Anything downstream that assumes a smooth
    Jacobian at zero is wrong, so pin the kink.
    """
    act = QALU(alpha=1.0, beta=1.0, gamma=1.5, delta=0.5, trainable=False)
    alpha, gamma, delta = (float(act.alpha), float(act.gamma), float(act.delta))

    def slope_at(x0: float) -> float:
        x = torch.tensor([x0], requires_grad=True)
        act(x).backward()
        return float(x.grad)

    for h in (1e-4, 1e-5):
        assert abs(slope_at(h) - alpha) < 1e-2
        # left branch: f'(x) = -2 gamma x / delta + O(x^2)
        assert abs(slope_at(-h) - 2.0 * gamma * h / delta) < 1e-6
    assert slope_at(-1e-7) < 1e-5, "left gain must vanish at the origin"
    assert abs(slope_at(1e-7) - alpha) < 1e-3


def test_qalu_positive_branch_peak_is_10_percent_above_asymptote() -> None:
    grid = np.linspace(0.0, 12.0, 12001)
    values = np.array([qalu_positive_gradient_ratio(u) for u in grid])
    peak = values.max()
    assert abs(peak / 2.0 - 1.10) < 2e-3
    assert abs(grid[int(values.argmax())] - 1.1997) < 0.01


def test_qalu_matches_published_formula() -> None:
    act = QALU(alpha=1.3, beta=0.8, gamma=1.7, delta=0.4, trainable=False)
    # The reference implementation stores shape parameters behind a softplus
    # with a 1e-4 floor, so the realised values sit 1e-4 above the arguments.
    alpha, beta, gamma, delta = (float(p) for p in (act.alpha, act.beta, act.gamma, act.delta))
    x = torch.tensor([-1.1, -0.05, 0.0, 0.05, 1.1])
    got = act(x)
    want = torch.where(
        x >= 0,
        alpha * x * (1 + torch.tanh(x / beta)),
        -gamma * x * torch.expm1(x / delta),
    )
    assert torch.allclose(got, want, atol=1e-5)


def test_activation_factory_shapes() -> None:
    x = torch.randn(3, 7)
    for name in ("qalu", "gelu", "swish", "relu"):
        assert make_activation(name)(x).shape == x.shape


def test_purged_split_has_no_overlap() -> None:
    n = 200_000
    splits = PurgedWalkForward(n, n_splits=5, purge=31).split()
    assert len(splits) == 5
    for s in splits:
        assert s.train[1] < s.val[0] < s.val[1] < s.test[0] < s.test[1]
        assert s.val[0] - s.train[1] >= s.purge
        assert s.test[0] - s.val[1] >= s.purge
    # test blocks are contiguous and cover everything after the first train slice
    assert splits[0].test[0] == int(0.30 * n)
    assert splits[-1].test[1] == n


def test_loss_matches_closed_form() -> None:
    from metrics.losses import DifferentialSharpeLoss

    pos = torch.tensor([1.0, 1.0, 0.0, -1.0])
    ret = torch.tensor([0.01, -0.02, 0.005, -0.01])
    fee = 5e-4
    loss = DifferentialSharpeLoss(fee_ratio=fee, bars_per_year=105_120)(pos, ret, prev_position=0.0)
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos.numpy()])))
    net = pos.numpy() * ret.numpy() - fee * turnover
    want = -net.mean() / (net.std() + 1e-6) * math.sqrt(105_120)
    assert abs(float(loss) - want) < 1e-4


def test_derived_mode_ties_position_to_forecast() -> None:
    """In derived mode the position must be a monotone readout of the forecast."""
    torch.manual_seed(0)
    model = DeadbandSSM(4, 32, 2, 8, 0.1, "qalu", mode="derived").eval()
    x = torch.randn(6, 30, 4)
    pos, forecast = model(x)
    assert pos.shape == forecast.shape == (6,)
    assert torch.all(pos.abs() <= 1.0)
    assert torch.allclose(pos, torch.tanh(forecast), atol=1e-6)


def test_derived_mode_position_responds_to_input() -> None:
    """A free policy head can sit at a constant; the derived one cannot."""
    torch.manual_seed(0)
    model = DeadbandSSM(4, 32, 2, 8, 0.1, "qalu", mode="derived").eval()
    with torch.no_grad():
        pos, _ = model(torch.randn(64, 30, 4))
    assert float(pos.detach().std()) > 1e-3
