"""Purged walk-forward cross-validation engine.

Protocol (pre-registered in ``PROTOCOL.md``):

* 5 expanding-window folds. Fold ``i`` tests the ``i``-th contiguous block of
  the last 70% of the sample; training always ends before the block starts.
* A purge gap of ``seq_len + 1 + embargo`` bars separates every training,
  validation and test boundary, so no training window (length ``seq_len``) can
  contain a bar whose label lives inside the next split.
* The feature scaler is fitted on the training slice only.
* Four activations (qalu, gelu, swish, relu) share the architecture, the
  optimiser, the seed schedule and the epoch budget. Only the activation
  module differs, so the comparison is a controlled ablation.
* Seeds are paired across activations: seed ``s`` initialises every candidate
  with the same torch RNG state, and every candidate sees the same batch
  indices in the same order.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch import Tensor

from core.model import DeadbandSSM
from data.fetcher import BARS_PER_YEAR, Panel, Standardizer
from metrics.losses import DifferentialSharpeLoss, compute_metrics

__all__ = [
    "TrainConfig",
    "Split",
    "PurgedWalkForward",
    "run_experiment",
    "benchmark",
    "predict_range",
]

EPS = 1e-6


@dataclass
class TrainConfig:
    """Everything that affects a single training run."""

    seq_len: int = 30
    d_model: int = 32
    n_layers: int = 2
    d_state: int = 8
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 12
    steps_per_epoch: int = 300
    batch_size: int = 128
    grad_clip: float = 1.0
    fee_ratio: float = 5e-4
    patience: int = 3
    mode: str = "both"
    sup_weight: float = 1.0
    use_conv: bool = True
    use_ssm: bool = True
    bars_per_year: int = BARS_PER_YEAR

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Split:
    """Half-open index ranges ``[start, end)`` for one walk-forward fold."""

    fold: int
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]
    purge: int


class PurgedWalkForward:
    """Expanding-window walk-forward splitter with purge and embargo."""

    def __init__(
        self,
        n_samples: int,
        n_splits: int = 5,
        initial_train_frac: float = 0.30,
        val_frac: float = 0.08,
        purge: int = 32,
        embargo: int = 288,
    ) -> None:
        if not 0.0 < initial_train_frac < 0.9:
            raise ValueError("initial_train_frac must be in (0, 0.9)")
        self.n_samples = n_samples
        self.n_splits = n_splits
        self.initial_train_frac = initial_train_frac
        self.val_frac = val_frac
        self.gap = purge + embargo

    def split(self) -> list[Split]:
        n = self.n_samples
        first_train_end = int(self.initial_train_frac * n)
        block = (n - first_train_end) // self.n_splits
        val_len = int(min(max(self.val_frac * n, 2_000), 15_000))
        out: list[Split] = []
        for i in range(self.n_splits):
            test_start = first_train_end + i * block
            test_end = n if i == self.n_splits - 1 else test_start + block
            val_end = test_start - self.gap
            val_start = max(val_end - val_len, self.gap + 1)
            train_end = val_start - self.gap
            if train_end <= 0:
                raise ValueError("not enough samples for the requested split geometry")
            out.append(
                Split(
                    fold=i,
                    train=(0, train_end),
                    val=(val_start, val_end),
                    test=(test_end - (test_end - test_start), test_end),
                    purge=self.gap,
                )
            )
        return out


def _windows(x: np.ndarray, starts: np.ndarray, seq_len: int) -> np.ndarray:
    """Stack ``len(starts)`` windows of length ``seq_len`` -> ``(B, T, F)``."""
    span = x[int(starts[0]) : int(starts[-1]) + seq_len]
    view = np.lib.stride_tricks.sliding_window_view(span, seq_len, axis=0)
    return np.ascontiguousarray(view.transpose(0, 2, 1))


@torch.no_grad()
def predict_range(
    model: nn.Module,
    features: np.ndarray,
    start: int,
    end: int,
    seq_len: int,
    device: torch.device,
    batch: int = 2048,
    head: str = "policy",
) -> np.ndarray:
    """Outputs for bars ``[start, end)``; window for bar ``t`` is ``x[t-T+1:t+1]``.

    ``head="policy"`` returns the bounded position, ``head="value"`` the
    one-step forecast. Only the terminal step of each window is read, so the
    output at bar ``t`` depends on ``x[<=t]`` and nothing after it.
    """
    model.eval()
    if start < seq_len - 1:
        raise ValueError("start must leave room for one full window")
    starts = np.arange(start - seq_len + 1, end - seq_len + 1)
    out = np.empty(len(starts), dtype=np.float32)
    for i in range(0, len(starts), batch):
        chunk = starts[i : i + batch]
        xb = torch.from_numpy(_windows(features, chunk, seq_len)).to(device)
        pred = model(xb)
        if isinstance(pred, tuple):  # mode="both": (policy, forecast)
            pred = pred[0] if head == "policy" else pred[1]
        elif head != "policy":
            raise ValueError('head="value" requires a model built with mode="both"')
        out[i : i + len(chunk)] = pred.detach().cpu().numpy()
    return out


def _train(
    model: nn.Module,
    x_train: np.ndarray,
    r_train: np.ndarray,
    x_val: np.ndarray,
    r_val: np.ndarray,
    cfg: TrainConfig,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * cfg.steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=cfg.lr * 0.05)
    loss_fn = DifferentialSharpeLoss(fee_ratio=cfg.fee_ratio, bars_per_year=cfg.bars_per_year)

    n_train = len(x_train)
    hi = n_train - (cfg.batch_size + cfg.seq_len + 2)
    if hi <= 1:
        raise ValueError(f"training slice too short: {n_train}")
    # Unit-variance target for the forecast head: raw 5-minute log returns are
    # O(1e-4), so an unstandardised MSE would be numerically invisible.
    r_scale = float(np.std(r_train)) or 1.0

    best = {"val_sharpe": -np.inf, "epoch": -1, "state": None}
    history: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        model.train()
        losses: list[float] = []
        for _ in range(cfg.steps_per_epoch):
            s = int(rng.integers(0, hi))
            starts = np.arange(s, s + cfg.batch_size + 1)
            xb = torch.from_numpy(_windows(x_train, starts, cfg.seq_len)).to(device)
            out = model(xb)  # (B+1,) positions for bars s+T-1 .. s+T+B-1
            rb = torch.from_numpy(r_train[s + cfg.seq_len : s + cfg.seq_len + cfg.batch_size]).to(device)
            if isinstance(out, tuple):
                pos, forecast = out
            else:
                pos, forecast = out, None
            loss = loss_fn(pos[1:], rb, prev_position=pos[0])
            if forecast is not None and cfg.sup_weight > 0:
                target = rb / r_scale
                loss = loss + cfg.sup_weight * torch.nn.functional.mse_loss(forecast[1:], target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))

        pos_val = predict_range(model, x_val, cfg.seq_len - 1, len(x_val), cfg.seq_len, device)
        # Bars [seq_len-1, len(x_val)) of the val slice; label r[t+1] sits at the same index.
        m = compute_metrics(pos_val, r_val[cfg.seq_len - 1 :], cfg.fee_ratio, cfg.bars_per_year)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_sharpe": m.sharpe})
        if m.sharpe > best["val_sharpe"]:
            best = {
                "val_sharpe": m.sharpe,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        elif epoch - int(best["epoch"]) >= cfg.patience:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {"best_val_sharpe": float(best["val_sharpe"]), "best_epoch": int(best["epoch"]), "history": history}


def run_experiment(task: dict[str, Any]) -> dict[str, Any]:
    """Train and evaluate one (activation, fold, seed) cell. Picklable entry point."""
    torch.set_num_threads(1)
    panel: Panel = task["panel"]
    split: Split = task["split"]
    cfg: TrainConfig = task["cfg"]
    act: str = task["activation"]
    seed: int = task["seed"]
    out_dir = Path(task["out_dir"])
    device = torch.device(task.get("device", "cpu"))

    t0 = time.perf_counter()
    tr0, tr1 = split.train
    va0, va1 = split.val
    te0, te1 = split.test

    scaler = Standardizer().fit(panel.features[tr0:tr1])
    x_all = scaler.transform(panel.features)
    r_all = panel.returns.astype(np.float32)

    model = DeadbandSSM(
        n_features=panel.features.shape[1],
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        d_state=cfg.d_state,
        dropout=cfg.dropout,
        act=act,
        mode=cfg.mode,  # type: ignore[arg-type]
        use_conv=cfg.use_conv,
        use_ssm=cfg.use_ssm,
    )
    model, info = _train(model, x_all[tr0:tr1], r_all[tr0:tr1], x_all[va0:va1], r_all[va0:va1], cfg, seed, device)

    pos = predict_range(model, x_all, te0, te1, cfg.seq_len, device)
    ret = r_all[te0:te1]
    metrics = compute_metrics(pos, ret, cfg.fee_ratio, cfg.bars_per_year)
    prev = np.concatenate([[0.0], pos[:-1]])
    net = pos * ret - cfg.fee_ratio * np.abs(pos - prev)

    # Forecast diagnostics: a lower-variance view of what the activation does
    # to the trunk than the post-cost Sharpe, which is dominated by turnover.
    forecast = np.zeros_like(pos)
    ic = 0.0
    sign_acc = 0.0
    if cfg.mode == "both":
        forecast = predict_range(model, x_all, te0, te1, cfg.seq_len, device, head="value")
        if np.std(forecast) > 0:
            ic = float(spearmanr(forecast, ret).statistic)
        sign_acc = float(np.mean(np.sign(forecast) == np.sign(ret)))

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{act}_f{split.fold}_s{seed}"
    np.savez_compressed(
        out_dir / f"{tag}.npz",
        positions=pos,
        returns=ret,
        net=net,
        forecast=forecast,
        timestamps=panel.timestamps[te0:te1].asi8,
    )
    torch.save(model.state_dict(), out_dir / f"{tag}.pt")

    result = {
        "activation": act,
        "fold": split.fold,
        "seed": seed,
        "n_params": model.n_params,
        "n_samples": int(len(panel)),
        "test_range": [int(te0), int(te1)],
        "test_start": str(panel.timestamps[te0]),
        "test_end": str(panel.timestamps[te1 - 1]),
        "train_seconds": time.perf_counter() - t0,
        "ic": ic,
        "sign_acc": sign_acc,
        "config": cfg.to_dict(),
        **info,
        **metrics.to_dict(),
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2))
    return result


def _ascii_table(results: Sequence[dict[str, Any]]) -> str:
    """Mean +/- std per activation over folds x seeds."""
    from collections import defaultdict

    by_act: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_act[r["activation"]].append(r)

    header = (
        f"{'activation':<8} {'sharpe':>14} {'active':>8} {'ann.ret':>10} {'maxDD':>9} "
        f"{'turnover':>10} {'exposure':>9} {'params':>8}"
    )
    lines = [header, "-" * len(header)]
    for act in ("qalu", "gelu", "swish", "relu"):
        rows = by_act.get(act, [])
        if not rows:
            continue
        sh = np.array([r["sharpe"] for r in rows])
        ac = np.array([r.get("active_sharpe", 0.0) for r in rows])
        ar = np.array([r["ann_return"] for r in rows])
        dd = np.array([r["max_drawdown"] for r in rows])
        to = np.array([r["turnover"] for r in rows])
        ex = np.array([r["exposure"] for r in rows])
        sd = sh.std(ddof=1) if sh.size > 1 else 0.0
        lines.append(
            f"{act:<8} {sh.mean():>7.3f}+-{sd:<5.3f} {ac.mean():>8.3f} "
            f"{ar.mean():>9.1%} {dd.mean():>8.1%} {to.mean():>10.5f} "
            f"{ex.mean():>9.3f} {int(rows[0]['n_params']):>8d}"
        )
    return "\n".join(lines)


def benchmark(
    panel: Panel,
    activations: Iterable[str] = ("qalu", "gelu", "swish", "relu"),
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    cfg: TrainConfig | None = None,
    out_dir: str | Path = "results/raw",
    workers: int = 1,
    device: str = "cpu",
    splits: Sequence[Split] | None = None,
) -> list[dict[str, Any]]:
    """Run the full grid and persist per-cell JSON + npz artifacts."""
    cfg = cfg or TrainConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_list = list(splits) if splits is not None else PurgedWalkForward(len(panel), purge=cfg.seq_len + 1).split()

    tasks = [
        {
            "panel": panel,
            "split": s,
            "cfg": cfg,
            "activation": act,
            "seed": int(seed),
            "out_dir": str(out),
            "device": device,
        }
        for s in split_list
        for act in activations
        for seed in seeds
    ]
    print(f"[bench] {len(tasks)} runs: {len(split_list)} folds x {len(list(activations))} activations x {len(list(seeds))} seeds")

    results: list[dict[str, Any]] = []
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_experiment, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                print(
                    f"[{i:>3}/{len(tasks)}] {r['activation']:<5} f{r['fold']} s{r['seed']} "
                    f"sharpe={r['sharpe']:>7.3f} ret={r['ann_return']:>8.1%} "
                    f"dd={r['max_drawdown']:>7.1%} turn={r['turnover']:.5f} ({r['train_seconds']:.0f}s)"
                )
    else:
        for i, t in enumerate(tasks, 1):
            r = run_experiment(t)
            results.append(r)
            print(
                f"[{i:>3}/{len(tasks)}] {r['activation']:<5} f{r['fold']} s{r['seed']} "
                f"sharpe={r['sharpe']:>7.3f} ret={r['ann_return']:>8.1%} "
                f"dd={r['max_drawdown']:>7.1%} turn={r['turnover']:.5f} ({r['train_seconds']:.0f}s)"
            )

    results.sort(key=lambda r: (r["activation"], r["fold"], r["seed"]))
    (out / "results.json").write_text(json.dumps(results, indent=2))
    table = _ascii_table(results)
    (out / "summary.txt").write_text(table + "\n")
    print("\n" + table)
    return results
