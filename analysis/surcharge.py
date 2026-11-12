"""Does a fee surcharge remove chattering at the source?

The policy's chattering regime is a training outcome, not a property of the
activation: it appears in cells whose turnover the loss was happy to buy. This
script retrains with a fee surcharge inside the loss (10x the real cost by
default) while every reported metric keeps the true 5 bp, so the comparison is
"same evaluation, different incentive during training".

    python -m analysis.surcharge --activations qalu gelu --epochs 5 --steps 120

Cells are written to ``results/surcharge`` with the same tag scheme as the main
grid, so they can be inspected with the same tooling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.cross_val import PurgedWalkForward, TrainConfig, benchmark
from data.fetcher import load_panel

ACTIVATIONS = ("qalu", "gelu", "swish", "relu")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", nargs="+", default=["qalu", "gelu"], choices=ACTIVATIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--surcharge", type=float, default=5e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=str, default="results/surcharge")
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--symbol", type=str, default="BTCUSDT")
    ap.add_argument("--interval", type=str, default="5m")
    args = ap.parse_args()

    panel = load_panel(start=args.start, end=args.end, symbol=args.symbol, interval=args.interval)
    print(f"bars {len(panel):,}  features {panel.feature_names}")

    cfg = TrainConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        train_fee_ratio=args.surcharge,
    )
    splits = PurgedWalkForward(len(panel), n_splits=args.folds, purge=cfg.seq_len + 1).split()
    out = Path(args.out)
    rows = benchmark(
        panel,
        activations=args.activations,
        seeds=args.seeds,
        cfg=cfg,
        out_dir=out,
        workers=args.workers,
        device="cpu",
        splits=splits,
    )
    print(f"\n{len(rows)} cells -> {out}")


if __name__ == "__main__":
    main()
