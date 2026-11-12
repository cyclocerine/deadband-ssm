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
import glob
import json
from collections import defaultdict
from pathlib import Path

from benchmark.cross_val import PurgedWalkForward, TrainConfig, benchmark
from data.fetcher import load_panel

ACTIVATIONS = ("qalu", "gelu", "swish", "relu")


def _load(directory: str) -> dict[tuple[str, int, int], dict]:
    cells: dict[tuple[str, int, int], dict] = {}
    for path in glob.glob(f"{directory}/*_f*_s*.json"):
        with open(path) as fh:
            d = json.load(fh)
        cells[(d["activation"], int(d["fold"]), int(d["seed"]))] = d
    return cells


def compare(raw_dir: str, sur_dir: str) -> None:
    """Raw-cost training against surcharged training, cell by cell."""
    raw = _load(raw_dir)
    sur = _load(sur_dir)
    keys = sorted(set(raw) & set(sur))
    if not keys:
        raise SystemExit("no cells in common between the two directories")

    print(f"{'act':<5} {'fold':>4} {'raw turn':>9} {'sur turn':>9} {'raw shp':>8} {'sur shp':>8} {'raw expo':>9} {'sur expo':>9}")
    by_act: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for key in keys:
        a, b = raw[key], sur[key]
        by_act[key[0]].append((a, b))
        print(
            f"{key[0]:<5} {key[1]:>4} {a['turnover']:>9.4f} {b['turnover']:>9.4f} "
            f"{a['sharpe']:>8.2f} {b['sharpe']:>8.2f} {a['exposure']:>9.2f} {b['exposure']:>9.2f}"
        )
    print()
    for act, pairs in sorted(by_act.items()):
        ra = [p[0] for p in pairs]
        su = [p[1] for p in pairs]
        print(
            f"{act:<5} n={len(pairs)}  mean sharpe {sum(r['sharpe'] for r in ra) / len(ra):+.2f} -> "
            f"{sum(r['sharpe'] for r in su) / len(su):+.2f}   mean turnover "
            f"{sum(r['turnover'] for r in ra) / len(ra):.4f} -> {sum(r['turnover'] for r in su) / len(su):.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", nargs="+", default=["qalu", "gelu"], choices=ACTIVATIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--folds", type=int, default=5, help="number of walk-forward splits")
    ap.add_argument("--only-folds", nargs="+", type=int, default=None, help="train only these fold indices")
    ap.add_argument("--sup-weight", type=float, default=1.0, help="weight on the forecast MSE term")
    ap.add_argument("--surcharge", type=float, default=None,
                    help="fee charged inside the loss only; default is the true 5 bp")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=str, default="results/surcharge")
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--symbol", type=str, default="BTCUSDT")
    ap.add_argument("--interval", type=str, default="5m")
    ap.add_argument("--compare", nargs=2, metavar=("RAW", "SURCHARGED"), default=None,
                    help="skip training; compare two result directories cell by cell")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    panel = load_panel(start=args.start, end=args.end, symbol=args.symbol, interval=args.interval)
    print(f"bars {len(panel):,}  features {panel.feature_names}")

    cfg = TrainConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        train_fee_ratio=args.surcharge,
        sup_weight=args.sup_weight,
    )
    splits = PurgedWalkForward(len(panel), n_splits=args.folds, purge=cfg.seq_len + 1).split()
    if args.only_folds:
        keep = set(args.only_folds)
        splits = [s for s in splits if s.fold in keep]
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
