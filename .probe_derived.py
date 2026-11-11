"""Probe: derived mode (position = tanh(forecast)) at the grid budget.

Usage: python .probe_derived.py <activation> <epochs> <steps_per_epoch>
"""
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)

from benchmark.cross_val import PurgedWalkForward, TrainConfig, _train, predict_range
from core.model import DeadbandSSM
from data.fetcher import Standardizer, load_panel
from metrics.losses import compute_metrics

act = sys.argv[1]
epochs = int(sys.argv[2])
steps = int(sys.argv[3])

panel = load_panel(start="2020-01-01")
split = PurgedWalkForward(len(panel), purge=31).split()[0]
scaler = Standardizer().fit(panel.features[split.train[0] : split.train[1]])
X = scaler.transform(panel.features)
R = panel.returns.astype(np.float32)

cfg = TrainConfig(epochs=epochs, steps_per_epoch=steps, batch_size=128, mode="derived")
model = DeadbandSSM(4, 32, 2, 8, 0.1, act, mode="derived")
t0 = time.perf_counter()
model, info = _train(
    model,
    X[split.train[0] : split.train[1]],
    R[split.train[0] : split.train[1]],
    X[split.val[0] : split.val[1]],
    R[split.val[0] : split.val[1]],
    cfg,
    0,
    torch.device("cpu"),
)
pos = predict_range(model, X, split.test[0], split.test[1], 30, torch.device("cpu"))
mt = compute_metrics(pos, R[split.test[0] : split.test[1]])
hist = [(h["epoch"], round(h["train_loss"], 2), round(h["val_sharpe"], 3)) for h in info["history"]]
print(
    f"DERIVED {act} {time.perf_counter() - t0:.0f}s hist={hist} "
    f"TEST sharpe={mt.sharpe:.3f} active={mt.active_sharpe:.3f} ic={mt.ic:.4f} "
    f"turn={mt.turnover:.5f} expo={mt.exposure:.3f} pos_std={float(pos.std()):.4f} "
    f"best_val={info['best_val_sharpe']:.3f}",
    flush=True,
)
