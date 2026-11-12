"""Evaluate policy-level robustness transforms on the saved grid cells.

Reads ``results/raw/*.npz`` (positions and returns per cell), applies the
transforms in :mod:`metrics.policy`, and reports post-cost Sharpe and turnover
per activation. Nothing is retrained, so every number here is a property of the
*policy* the trained network produced, not of a new model.

Usage::

    python -m analysis.robustness --dir results/raw
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from data.fetcher import BARS_PER_YEAR
from metrics.losses import compute_metrics
from metrics.policy import cap_turnover, ensemble, no_trade_band, smooth_position, turnover_guard

ACTS = ("qalu", "gelu", "swish", "relu")
FEE = 5e-4
CHAT = 0.05  # same threshold the sweep uses to flag chattering


def load_cells(directory: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    for path in sorted(directory.glob("*.npz")):
        act, fold, seed = path.stem.split("_")
        z = np.load(path)
        cells[(act, int(fold[1:]), int(seed[1:]))] = {
            "pos": z["positions"].astype(np.float64),
            "ret": z["returns"].astype(np.float64),
            "ic": json.loads((directory / f"{path.stem}.json").read_text()).get("ic", 0.0),
        }
    return cells


def score(pos: np.ndarray, ret: np.ndarray) -> dict[str, float]:
    m = compute_metrics(pos, ret, FEE, BARS_PER_YEAR)
    return {
        "sharpe": m.sharpe,
        "active": m.active_sharpe,
        "turnover": m.turnover,
        "exposure": m.exposure,
        "vol": m.ann_vol,
    }


def apply(name: str, pos: np.ndarray) -> np.ndarray:
    if name == "raw":
        return pos
    if name.startswith("guard"):
        parts = name.split(":")
        window = int(parts[1]) if len(parts) > 1 else 500
        trip = float(parts[2]) if len(parts) > 2 else 0.05
        cooldown = int(parts[3]) if len(parts) > 3 else 2000
        guarded = turnover_guard(pos, window=window, trip=trip, cooldown=cooldown)
        if len(parts) > 4:
            return no_trade_band(guarded, float(parts[4]))
        return guarded
    kind, _, arg = name.partition(":")
    value = float(arg)
    if kind == "band":
        return no_trade_band(pos, value)
    if kind == "cap":
        return cap_turnover(pos, value)
    if kind == "smooth":
        return smooth_position(pos, value)
    raise ValueError(f"unknown transform {name}")


VARIANTS = ["raw"]
VARIANTS += ["guard:500:0.05:2000", "guard:1000:0.03:5000", "guard:200:0.10:1000"]
VARIANTS += ["guard:500:0.05:2000:0.1"]
VARIANTS += [f"band:{t}" for t in (0.1, 0.3)]
VARIANTS += [f"cap:{c}" for c in (0.01, 0.02)]
VARIANTS += [f"smooth:{a}" for a in (0.1, 0.3)]


def summarise(cells: dict[tuple[str, int, int], dict[str, Any]], transform: Callable[[np.ndarray], np.ndarray]) -> dict[str, Any]:
    per_act: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (act, _, _), cell in cells.items():
        per_act[act].append(score(transform(cell["pos"]), cell["ret"]))
    out: dict[str, Any] = {}
    for act, rows in per_act.items():
        sr = np.array([r["sharpe"] for r in rows])
        out[act] = {
            "n": len(rows),
            "mean": float(sr.mean()),
            "median": float(np.median(sr)),
            "active": float(np.mean([r["active"] for r in rows])),
            "active_med": float(np.median([r["active"] for r in rows])),
            "turnover": float(np.mean([r["turnover"] for r in rows])),
            "exposure": float(np.mean([r["exposure"] for r in rows])),
            "vol": float(np.mean([r["vol"] for r in rows])),
            "chat": int(sum(r["turnover"] >= CHAT for r in rows)),
            "worst": float(sr.min()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("results/raw"))
    ap.add_argument("--out", type=Path, default=None, help="optional JSON dump")
    args = ap.parse_args()

    cells = load_cells(args.dir)
    print(f"{len(cells)} cells from {args.dir}\n")

    results: dict[str, Any] = {}
    header = f"{'variant':12s}" + "".join(f"{a:>34s}" for a in ACTS)
    print(header)
    print(f"{'':12s}" + "".join(f"{'sharpe  active  turn  expo':>34s}" for _ in ACTS))
    for name in VARIANTS:
        row = summarise(cells, lambda p, n=name: apply(n, p))
        results[name] = row
        line = f"{name:12s}"
        for act in ACTS:
            d = row[act]
            line += f"{d['mean']:+7.2f}{d['active']:+7.2f}{d['turnover']:7.4f}{d['exposure']:7.2f}   "
        print(line)

    # Seed ensembles: average the two seeds inside each fold. This is the only
    # variant that changes the population rather than filtering each path.
    folds = sorted({f for (_, f, _) in cells})
    ens: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for act in ACTS:
        for fold in folds:
            paths = [cells[(act, fold, s)]["pos"] for s in (0, 1) if (act, fold, s) in cells]
            if len(paths) < 2:
                continue
            pos = ensemble(paths)
            ret = cells[(act, fold, 0)]["ret"]
            ens[act].append(score(pos, ret))
    print("\nseed ensemble (2 seeds averaged per fold)")
    results["ensemble"] = {}
    for act in ACTS:
        rows = ens.get(act, [])
        if not rows:
            continue
        sr = np.array([r["sharpe"] for r in rows])
        results["ensemble"][act] = {
            "n": len(rows),
            "mean": float(sr.mean()),
            "median": float(np.median(sr)),
            "active": float(np.mean([r["active"] for r in rows])),
            "turnover": float(np.mean([r["turnover"] for r in rows])),
            "exposure": float(np.mean([r["exposure"] for r in rows])),
            "chat": int(sum(r["turnover"] >= CHAT for r in rows)),
            "worst": float(sr.min()),
        }
        print(f"  {act:6s} n={len(rows):2d}  sharpe {sr.mean():+7.2f}  active {np.mean([r['active'] for r in rows]):+7.2f}  turn {np.mean([r['turnover'] for r in rows]):.4f}  expo {np.mean([r['exposure'] for r in rows]):.2f}  chatter {int(sum(r['turnover'] >= CHAT for r in rows))}")

    # Per-cell detail for the arm the guard is aimed at: a tail truncation is
    # only credible if it is visible cell by cell, not just in the mean.
    print("\nper-cell, Q-ALU (raw -> guarded)")
    for (act, fold, seed), cell in sorted(cells.items()):
        if act != "qalu":
            continue
        before = score(cell["pos"], cell["ret"])
        after = score(apply("guard:500:0.05:2000", cell["pos"]), cell["ret"])
        print(
            f"  f{fold} s{seed}  {before['sharpe']:+8.2f} -> {after['sharpe']:+8.2f}"
            f"   turn {before['turnover']:.4f} -> {after['turnover']:.4f}"
            f"   expo {before['exposure']:.2f} -> {after['exposure']:.2f}"
        )

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
