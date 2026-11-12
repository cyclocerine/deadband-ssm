#!/usr/bin/env python3
"""End-to-end pipeline: data -> benchmark -> statistics -> figures -> export.

Examples::

    python run_pipeline.py --stage all --workers 3
    python run_pipeline.py --stage bench --folds 5 --seeds 0 1 2 --epochs 6 --steps 100
    python run_pipeline.py --stage analysis --figures      # reuse existing runs
    python run_pipeline.py --stage export --ckpt results/raw/qalu_f0_s0.pt

Every stage is idempotent and writes into ``results/``; the paper's numbers are
read from ``results/analysis.json`` and never typed by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _banner(msg: str) -> None:
    print(f"\n=== {msg} " + "=" * max(0, 66 - len(msg)))


def stage_data(args: argparse.Namespace):
    from data.fetcher import load_panel

    _banner("data")
    t0 = time.perf_counter()
    panel = load_panel(start=args.start, end=args.end, symbol=args.symbol, interval=args.interval)
    print(f"bars          : {len(panel):,}  ({panel.timestamps[0]} .. {panel.timestamps[-1]})")
    print(f"features      : {panel.feature_names} {panel.features.shape}")
    print(f"return std    : {panel.returns.std():.3e} /bar  (ann. vol {panel.returns.std() * np.sqrt(105_120):.1%})")
    print(f"nan / inf     : {int(np.isnan(panel.features).sum())} / {int(np.isinf(panel.features).sum())}")
    print(f"loaded in     : {time.perf_counter() - t0:.1f}s")
    return panel


def stage_bench(args: argparse.Namespace, panel):
    from benchmark.cross_val import TrainConfig, PurgedWalkForward, benchmark

    _banner("benchmark")
    cfg = TrainConfig(
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        batch_size=args.batch,
        mode=args.mode,
        sup_weight=args.sup_weight,
    )
    splits = PurgedWalkForward(len(panel), n_splits=args.folds, purge=cfg.seq_len + 1).split()
    for s in splits:
        print(
            f"fold {s.fold}: train=[{s.train[0]:>7},{s.train[1]:>7}) "
            f"val=[{s.val[0]:>7},{s.val[1]:>7}) test=[{s.test[0]:>7},{s.test[1]:>7}) purge={s.purge}"
        )
    return benchmark(
        panel,
        activations=args.activations,
        seeds=args.seeds,
        cfg=cfg,
        out_dir=args.out,
        workers=args.workers,
        device=args.device,
        splits=splits,
    )


def stage_analysis(args: argparse.Namespace):
    from analysis.report import (
    analyse,
    ascii_report,
    latex_macros,
    latex_table,
    load_cells,
    readme_block,
)

    _banner("analysis")
    cells = load_cells(args.out)
    analysis = analyse(cells, args.out, n_boot=args.boot)
    print(ascii_report(analysis))
    tex = latex_table(analysis, ROOT / "paper" / "tables" / "results.tex")
    deploy_path = ROOT / "export" / "report.json"
    deploy = json.loads(deploy_path.read_text()) if deploy_path.exists() else None
    latex_macros(analysis, ROOT / "paper" / "macros.tex", deploy)
    readme_block(analysis, ROOT / "README.md")
    print(f"\nLaTeX table -> paper/tables/results.tex ({len(tex)} chars)")
    print("LaTeX macros -> paper/macros.tex")
    print("README block -> README.md")
    return analysis


def stage_macros(args: argparse.Namespace):
    """Re-render the LaTeX table and macros from the cached analysis JSON.

    Editing prose in the report means editing strings, not re-running 40k
    bootstrap resamples on a 100k-bar stream, which costs minutes on a slow CPU.
    """
    from analysis.report import (
        latex_macros,
        latex_table,
        readme_block,
        readme_deploy_block,
    )

    _banner("macros")
    path = ROOT / args.out / "analysis.json"
    if not path.exists():
        raise SystemExit(f"{path} not found; run --stage analysis first")
    analysis = json.loads(path.read_text())
    latex_table(analysis, ROOT / "paper" / "tables" / "results.tex")
    deploy_path = ROOT / "export" / "report.json"
    deploy = json.loads(deploy_path.read_text()) if deploy_path.exists() else None
    latex_macros(analysis, ROOT / "paper" / "macros.tex", deploy)
    readme_block(analysis, ROOT / "README.md")
    if deploy:
        readme_deploy_block(deploy, ROOT / "README.md")
    print("LaTeX table -> paper/tables/results.tex")
    print("LaTeX macros -> paper/macros.tex")
    print("README block -> README.md")
    return analysis


def stage_figures(args: argparse.Namespace, cells):
    from analysis.figures import make_all

    _banner("figures")
    paths = make_all(cells, args.out, ROOT / "paper" / "figures")
    for p in paths:
        print(f"wrote {p}")
    return paths


def stage_export(args: argparse.Namespace, panel):
    from export.deploy import benchmark_artifact
    from core.model import DeadbandSSM
    import torch

    _banner("export")
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    meta = json.loads(ckpt.with_suffix(".json").read_text())
    cfg = meta["config"]
    model = DeadbandSSM(
        n_features=panel.features.shape[1],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        d_state=cfg["d_state"],
        dropout=cfg["dropout"],
        act=meta["activation"],
        mode=cfg.get("mode", "policy"),
    )
    state = torch.load(ckpt, map_location="cpu")
    # Load every tensor. An earlier version filtered out the ``head_value``
    # weights and loaded with strict=False, which in derived mode silently
    # discarded the forecast head -- the exported artifact was then a model with
    # a randomly initialised readout, and its parity check passed only because
    # it compared that same broken model against itself.
    model.load_state_dict(state, strict=True)
    model.eval()
    report = benchmark_artifact(model, ROOT / "export", args.seq_len, panel.features.shape[1])
    (ROOT / "export" / "report.json").write_text(json.dumps(report.to_dict(), indent=2))
    from analysis.report import readme_deploy_block

    readme_deploy_block(report.to_dict(), ROOT / "README.md")
    print(json.dumps(report.to_dict(), indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--stage",
        default="all",
        choices=["all", "data", "bench", "analysis", "macros", "figures", "export"],
        help="'macros' re-renders tables/macros from results/analysis.json without re-bootstrapping",
    )
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--activations", nargs="+", default=["qalu", "gelu", "swish", "relu"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--d-state", type=int, default=8)
    ap.add_argument("--mode", default="derived")
    ap.add_argument("--sup-weight", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--ckpt", default="results/raw/qalu_f0_s0.pt")
    ap.add_argument("--figures", action="store_true", help="also render figures during the analysis stage")
    args = ap.parse_args(argv)

    panel = None
    if args.stage in ("all", "data", "bench", "export"):
        panel = stage_data(args)
    if args.stage in ("all", "bench"):
        stage_bench(args, panel)
    analysis = None
    if args.stage in ("all", "analysis"):
        analysis = stage_analysis(args)
    elif args.stage == "macros":
        analysis = stage_macros(args)
    if args.stage in ("all", "figures") or (args.stage == "analysis" and args.figures):
        from analysis.report import load_cells

        stage_figures(args, load_cells(args.out))
    if args.stage in ("all", "export"):
        stage_export(args, panel)

    if analysis is not None:
        _banner("verdict")
        paired = analysis["paired"]
        passed = [
            name
            for name, p in paired.items()
            if p["mean_diff"] > 0 and p["ci_low"] > 0 and p["wilcoxon_p"] < 0.05
        ]
        print("Q-ALU significantly better than: " + (", ".join(passed) if passed else "none"))
        print(f"headline Sharpe: " + ", ".join(f"{a}={v['sharpe']:.3f}" for a, v in analysis["summary"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
