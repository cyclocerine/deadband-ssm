"""Paper figures. Vector PDFs sized for the IEEE two-column layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.report import ACTIVATIONS, LABELS, stitch

__all__ = ["fig_activation", "fig_equity", "fig_folds", "make_all"]

COLORS = {"qalu": "#c1440e", "gelu": "#1f4e79", "swish": "#2e7d32", "relu": "#6a1b9a"}
COL_W = 3.5  # inches, single column
DCOL_W = 7.16  # inches, double column


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fig_activation(path: str | Path, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0, delta: float = 0.4) -> Path:
    """Activation and derivative, with the Q-ALU downside overshoot marked."""
    _style()
    from core.qalu import make_activation
    import torch

    x = np.linspace(-4.0, 4.0, 2001)
    xt = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.75))
    for act in ACTIVATIONS:
        fn = make_activation(act)
        y = fn(xt).detach().numpy()
        (grad,) = torch.autograd.grad(fn(xt).sum(), xt, retain_graph=False)
        g = grad.detach().numpy()
        lw = 1.4 if act == "qalu" else 0.9
        axes[0].plot(x, y, lw=lw, color=COLORS[act], label=LABELS[act])
        axes[1].plot(x, g, lw=lw, color=COLORS[act], label=LABELS[act])
        xt.grad = None

    x_star = -2.0 * delta
    peak = gamma * (1.0 + np.exp(-2.0))  # d/dx at x* = -2 delta
    axes[1].axvline(x_star, color="0.55", lw=0.6, ls=":")
    axes[1].annotate(
        r"$1{+}e^{-2}=1.1353$",
        xy=(x_star, peak),
        xytext=(-3.7, peak + 0.55),
        fontsize=6.5,
        arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"),
    )
    axes[1].plot([x_star], [peak], marker="o", ms=3, color=COLORS["qalu"])
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$f(x)$")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$f'(x)$")
    axes[0].set_title("activation")
    axes[1].set_title("derivative")
    axes[0].legend(frameon=False, loc="lower right", handlelength=1.2)
    fig.tight_layout(pad=0.3)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_equity(out_dir: str | Path, path: str | Path) -> Path:
    """Stitched walk-forward OOS equity curves, one per activation."""
    _style()
    fig, ax = plt.subplots(figsize=(DCOL_W, 2.2))
    for act in ACTIVATIONS:
        try:
            s = stitch(out_dir, act)
        except FileNotFoundError:
            continue
        net = s["net"]
        equity = np.cumprod(1.0 + net)
        t = np.arange(len(equity)) / 105_120.0  # years of 5-minute bars
        ax.plot(t, equity, lw=1.0 if act != "qalu" else 1.5, color=COLORS[act], label=LABELS[act])
    ax.set_yscale("log")
    ax.set_xlabel("out-of-sample years (5-minute bars, stitched folds)")
    ax.set_ylabel("equity (log scale)")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.grid(alpha=0.15, lw=0.4)
    fig.tight_layout(pad=0.3)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_folds(cells: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Per-fold test Sharpe, every seed a marker, paired across activations."""
    _style()
    folds = sorted({int(c["fold"]) for c in cells})
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    width = 0.8 / max(len(ACTIVATIONS), 1)
    for i, act in enumerate(ACTIVATIONS):
        rows = [c for c in cells if c["activation"] == act]
        if not rows:
            continue
        means, stds = [], []
        for f in folds:
            v = np.array([c["sharpe"] for c in rows if int(c["fold"]) == f])
            means.append(v.mean() if v.size else np.nan)
            stds.append(v.std(ddof=1) if v.size > 1 else 0.0)
        pos = np.arange(len(folds)) + (i - (len(ACTIVATIONS) - 1) / 2) * width
        ax.bar(pos, means, width=width * 0.92, yerr=stds, color=COLORS[act], label=LABELS[act],
               error_kw=dict(lw=0.6, capsize=1.5), alpha=0.9)
    ax.axhline(0.0, color="0.3", lw=0.6)
    ax.set_xticks(np.arange(len(folds)))
    ax.set_xticklabels([f"fold {f}" for f in folds])
    ax.set_ylabel("OOS Sharpe")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    fig.tight_layout(pad=0.3)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def make_all(cells: Sequence[dict[str, Any]], out_dir: str | Path, fig_dir: str | Path) -> list[Path]:
    fig_dir = Path(fig_dir)
    return [
        fig_activation(fig_dir / "activation.pdf"),
        fig_equity(out_dir, fig_dir / "equity.pdf"),
        fig_folds(cells, fig_dir / "folds.pdf"),
    ]
