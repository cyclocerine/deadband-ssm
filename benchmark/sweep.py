"""Cross-timeframe x cross-architecture sweep.

The main experiment in the paper fixes one architecture and one horizon. This
module asks whether the Q-ALU turnover pathology survives a change of either.
Everything that is not the object of study is held fixed: the same readout, the
same loss, the same optimiser, the same step budget for every arm in the sweep,
the same purged walk-forward geometry (scaled to the bar length) and the same
underlying trades -- the coarser panels are aggregated from the 5-minute archive
rather than downloaded separately, so a difference between timeframes is a
difference of horizon and not of data vintage.

The sweep is deliberately cheaper than the headline grid (fewer folds, one seed,
a shorter step budget). It is therefore a *within-sweep* comparison: every arm
in it shares the budget, but its absolute Sharpe numbers are not comparable with
Table I and are not reported as such.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from benchmark.cross_val import PurgedWalkForward, TrainConfig, run_experiment
from data.fetcher import BARS_PER_YEAR_BY_TF, EMBARGO_BARS, Panel, load_panel_resampled

__all__ = ["TIMEFRAMES", "ARCHS", "ACTS", "SPLIT_GEOMETRY", "run_sweep", "main"]

TIMEFRAMES: tuple[str, ...] = ("5m", "15m", "1h", "4h", "1d", "1w")
ARCHS: tuple[str, ...] = ("ssm", "lstm", "gru", "tcn")
ACTS: tuple[str, ...] = ("qalu", "gelu", "swish", "relu")

#: (n_splits, initial_train_frac) per timeframe. Weekly has 319 bars in total, so
#: it gets a single split and a longer initial training window; a five-fold
#: walk-forward on 319 bars would leave folds too short to score.
SPLIT_GEOMETRY: dict[str, tuple[int, float]] = {
    "5m": (3, 0.30),
    "15m": (3, 0.30),
    "1h": (3, 0.30),
    "4h": (3, 0.30),
    "1d": (3, 0.30),
    "1w": (1, 0.75),
}

#: Input window per timeframe. Thirty bars is a two-and-a-half hour context at
#: 5 minutes and seven months at weekly resolution, and a thirty-bar window on
#: 319 weekly bars leaves no training slice once the purge gap is removed, so the
#: weekly arm uses eight bars (two months). This is the only arm whose window
#: length differs, and it is the reason no claim in the paper rests on it.
SEQ_LEN_BY_TF: dict[str, int] = {"5m": 30, "15m": 30, "1h": 30, "4h": 30, "1d": 30, "1w": 8}

#: Last bar of the sample the headline results were computed on. Pinning it keeps
#: the 5-minute arm of the sweep comparable with Table I even if the cache has
#: grown since.
SAMPLE_END = np.datetime64("2026-11-10T23:50:00", "ns")


def _truncate(panel: Panel, end: np.datetime64) -> Panel:
    keep = int(np.searchsorted(panel.timestamps.asi8, end.astype("datetime64[ns]").astype("int64"), side="right"))
    if keep >= len(panel):
        return panel
    return replace(
        panel,
        timestamps=panel.timestamps[:keep],
        features=panel.features[:keep],
        returns=panel.returns[:keep],
    )


def _splits(panel: Panel, timeframe: str, seq_len: int, folds: int | None) -> list:
    n_splits, frac = SPLIT_GEOMETRY[timeframe]
    if folds is not None:
        n_splits = folds
    splitter = PurgedWalkForward(
        len(panel),
        n_splits=n_splits,
        initial_train_frac=frac,
        purge=seq_len + 1,
        embargo=EMBARGO_BARS[timeframe],
    )
    return splitter.split()


def run_sweep(
    timeframes: Iterable[str] = TIMEFRAMES,
    archs: Iterable[str] = ARCHS,
    acts: Iterable[str] = ACTS,
    seeds: Sequence[int] = (0,),
    epochs: int = 2,
    steps: int = 100,
    batch: int = 128,
    seq_len: int | None = None,
    workers: int = 3,
    out_root: str | Path = "results/sweep",
    resume: bool = True,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Run the sweep and return every per-cell result."""
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    everything: list[dict[str, Any]] = []

    for tf in timeframes:
        panel = _truncate(load_panel_resampled(timeframe=tf), SAMPLE_END)
        tf_seq_len = SEQ_LEN_BY_TF[tf] if seq_len is None else seq_len
        splits = _splits(panel, tf, tf_seq_len, folds=None)
        out = root / tf
        out.mkdir(parents=True, exist_ok=True)
        cfg = TrainConfig(
            seq_len=tf_seq_len,
            epochs=epochs,
            steps_per_epoch=steps,
            batch_size=batch,
            mode="derived",
            bars_per_year=BARS_PER_YEAR_BY_TF[tf],
            arch="ssm",
            timeframe=tf,
        )
        print(
            f"[sweep:{tf}] {len(panel):,} bars, {len(splits)} folds, "
            f"test blocks {[(s.test[0], s.test[1]) for s in splits]}",
            flush=True,
        )

        tasks: list[dict[str, Any]] = []
        done: list[dict[str, Any]] = []
        for split in splits:
            for arch in archs:
                for act in acts:
                    for seed in seeds:
                        tag = f"{act}_f{split.fold}_s{seed}" if arch == "ssm" else f"{arch}_{act}_f{split.fold}_s{seed}"
                        path = out / f"{tag}.json"
                        if resume and path.exists():
                            done.append(json.loads(path.read_text()))
                            continue
                        tasks.append(
                            {
                                "panel": panel,
                                "split": split,
                                "cfg": replace(cfg, arch=arch),
                                "activation": act,
                                "seed": int(seed),
                                "out_dir": str(out),
                                "device": device,
                            }
                        )
        print(f"[sweep:{tf}] {len(tasks)} to run, {len(done)} already on disk", flush=True)

        results = list(done)
        if tasks:
            t0 = time.perf_counter()
            if workers > 1:
                from concurrent.futures import ProcessPoolExecutor, as_completed

                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(run_experiment, t): t for t in tasks}
                    for i, fut in enumerate(as_completed(futures), 1):
                        r = fut.result()
                        results.append(r)
                        elapsed = time.perf_counter() - t0
                        eta = elapsed / i * (len(tasks) - i)
                        print(
                            f"[sweep:{tf} {i:>3}/{len(tasks)}] {r['arch']:<4} {r['activation']:<5} "
                            f"f{r['fold']} sharpe={r['sharpe']:>7.3f} turn={r['turnover']:.5f} "
                            f"({r['train_seconds']:.0f}s, eta {eta/60:.0f}m)",
                            flush=True,
                        )
            else:
                for i, t in enumerate(tasks, 1):
                    r = run_experiment(t)
                    results.append(r)
                    print(
                        f"[sweep:{tf} {i:>3}/{len(tasks)}] {r['arch']:<4} {r['activation']:<5} "
                        f"f{r['fold']} sharpe={r['sharpe']:>7.3f} turn={r['turnover']:.5f} "
                        f"({r['train_seconds']:.0f}s)",
                        flush=True,
                    )

        results.sort(key=lambda r: (r["arch"], r["activation"], r["fold"], r["seed"]))
        (out / "results.json").write_text(json.dumps(results, indent=2))
        everything.extend(results)

    (root / "all.json").write_text(json.dumps(everything, indent=2))
    return everything


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeframes", nargs="+", default=list(TIMEFRAMES))
    ap.add_argument("--archs", nargs="+", default=list(ARCHS))
    ap.add_argument("--activations", nargs="+", default=list(ACTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="results/sweep")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    run_sweep(
        timeframes=args.timeframes,
        archs=args.archs,
        acts=args.activations,
        seeds=args.seeds,
        epochs=args.epochs,
        steps=args.steps,
        batch=args.batch,
        workers=args.workers,
        out_root=args.out,
        resume=not args.no_resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
