"""The same cell must train to the same numbers no matter what ran before it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from benchmark.cross_val import Split, TrainConfig, run_experiment
from data.fetcher import Panel


@pytest.fixture()
def panel() -> Panel:
    n = 400
    rng = np.random.default_rng(11)
    features = rng.normal(size=(n, 4)).astype(np.float32)
    returns = (0.0004 * features[:, 0] + rng.normal(scale=2e-3, size=n)).astype(np.float32)
    return Panel(
        timestamps=pd.date_range("2024-01-01", periods=n, freq="5min"),
        features=features,
        returns=returns,
        feature_names=("a", "b", "c", "d"),
        symbol="TEST",
        interval="5m",
    )


@pytest.fixture()
def split() -> Split:
    return Split(fold=0, train=(0, 260), val=(260, 320), test=(320, 400), purge=9)


@pytest.fixture()
def cfg() -> TrainConfig:
    return TrainConfig(
        seq_len=8,
        d_model=8,
        n_layers=1,
        d_state=4,
        dropout=0.0,
        epochs=1,
        steps_per_epoch=3,
        batch_size=16,
        patience=2,
    )


def _run(panel: Panel, split: Split, cfg: TrainConfig, tmp_path, name: str) -> dict:
    return run_experiment(
        {
            "panel": panel,
            "split": split,
            "cfg": cfg,
            "activation": "qalu",
            "seed": 0,
            "out_dir": str(tmp_path / name),
            "device": "cpu",
        }
    )


def test_cell_is_independent_of_prior_rng_use(panel, split, cfg, tmp_path) -> None:
    # Weight init used to draw from the process-global RNG, so a cell's result
    # depended on how much randomness earlier cells had consumed.
    torch.manual_seed(0)
    first = _run(panel, split, cfg, tmp_path, "first")

    torch.rand(17)
    np.random.rand(5)
    torch.manual_seed(1234)
    second = _run(panel, split, cfg, tmp_path, "second")

    assert first["sharpe"] == pytest.approx(second["sharpe"], abs=1e-12)
    assert first["turnover"] == pytest.approx(second["turnover"], abs=1e-12)

    a = np.load(tmp_path / "first" / "qalu_f0_s0.npz")["positions"]
    b = np.load(tmp_path / "second" / "qalu_f0_s0.npz")["positions"]
    assert np.array_equal(a, b)


def test_same_seed_same_weights(panel, split, cfg, tmp_path) -> None:
    _run(panel, split, cfg, tmp_path, "a")
    torch.rand(3)
    _run(panel, split, cfg, tmp_path, "b")
    wa = torch.load(tmp_path / "a" / "qalu_f0_s0.pt", weights_only=True)
    wb = torch.load(tmp_path / "b" / "qalu_f0_s0.pt", weights_only=True)
    assert wa.keys() == wb.keys()
    for key in wa:
        assert torch.equal(wa[key], wb[key]), key
