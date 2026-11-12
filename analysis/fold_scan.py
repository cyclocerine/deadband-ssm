"""Turnover and Sharpe by walk-forward fold, per activation.

The chattering regime is not spread across seeds: it switches on at a fold
boundary and stays on. Expanding-window folds make "more training bars" and
"later calendar time" collinear, so this script reports both the per-fold
turnover and the correlation with training size, and says nothing about which of
the two causes it.

    python -m analysis.fold_scan --dir results/raw
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/raw")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(f"{args.dir}/*_f*_s*.json")):
        with open(path) as fh:
            d = json.load(fh)
        rows.append(
            {
                "activation": d["activation"],
                "fold": int(d["fold"]),
                "turnover": float(d["turnover"]),
                "sharpe": float(d["sharpe"]),
                "train_bars": int(d["test_range"][0]),
            }
        )
    if not rows:
        raise SystemExit(f"no cells under {args.dir}")

    by_act: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_act[r["activation"]].append(r)

    print(f"{'act':<6} {'fold':>4} {'train_bars':>11} {'turnover':>9} {'sharpe':>8}")
    for act in sorted(by_act):
        for r in sorted(by_act[act], key=lambda x: (x["fold"], x["sharpe"])):
            print(
                f"{act:<6} {r['fold']:>4} {r['train_bars']:>11,} "
                f"{r['turnover']:>9.4f} {r['sharpe']:>8.2f}"
            )

    print(f"\n{'act':<6} {'corr(train_bars,turnover)':>26} {'corr(train_bars,sharpe)':>24}")
    for act in sorted(by_act):
        sub = by_act[act]
        tr = np.array([r["train_bars"] for r in sub], dtype=float)
        to = np.array([r["turnover"] for r in sub])
        sh = np.array([r["sharpe"] for r in sub])
        print(
            f"{act:<6} {np.corrcoef(tr, to)[0, 1]:>+26.2f} {np.corrcoef(tr, sh)[0, 1]:>+24.2f}"
        )


if __name__ == "__main__":
    main()
