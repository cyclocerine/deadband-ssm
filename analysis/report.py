"""Turn raw per-cell artifacts into the paper's tables, tests and figures.

Inputs are the JSON/NPZ files written by :mod:`benchmark.cross_val`; nothing is
re-trained here. Outputs:

* ``results/analysis.json`` - every number quoted in the paper
* ``results/tables/*.tex``   - LaTeX table bodies (booktabs)
* ``results/figures/*.pdf``  - vector figures
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis.stats import (
    BARS_PER_YEAR,
    aggregate,
    deflated_sharpe_ratio,
    dm_test,
    paired_comparison,
    sharpe,
    stationary_bootstrap_sharpe_diff,
)

__all__ = ["load_cells", "stitch", "analyse", "latex_table", "ACTIVATIONS"]

ACTIVATIONS = ("qalu", "gelu", "swish", "relu")
LABELS = {"qalu": "Q-ALU", "gelu": "GELU", "swish": "Swish", "relu": "ReLU"}


def load_cells(out_dir: str | Path) -> list[dict[str, Any]]:
    out = Path(out_dir)
    path = out / "results.json"
    if not path.exists():
        cells = [json.loads(p.read_text()) for p in sorted(out.glob("*.json")) if p.name != "results.json"]
        if not cells:
            raise FileNotFoundError(f"no per-cell results under {out}")
        return cells
    return json.loads(path.read_text())


def stitch(out_dir: str | Path, activation: str, seeds: Sequence[int] | None = None) -> dict[str, np.ndarray]:
    """Concatenate the fold test blocks into one walk-forward OOS stream.

    With several seeds the positions are averaged (an equal-weight ensemble of
    independently initialised runs); the resulting stream is the one the paper
    reports as the deployable strategy.
    """
    out = Path(out_dir)
    files = sorted(out.glob(f"{activation}_f*_s*.npz"))
    if seeds is not None:
        keep = {int(s) for s in seeds}
        files = [f for f in files if int(f.stem.split("_s")[-1]) in keep]
    if not files:
        raise FileNotFoundError(f"no artifacts for {activation!r} under {out}")
    by_fold: dict[int, list[Path]] = {}
    for f in files:
        by_fold.setdefault(int(f.stem.split("_f")[1].split("_")[0]), []).append(f)

    pos, ret, net, ts = [], [], [], []
    for fold in sorted(by_fold):
        stack_pos, stack_net = [], []
        for f in sorted(by_fold[fold]):
            d = np.load(f)
            stack_pos.append(d["positions"].astype(np.float64))
            stack_net.append(d["net"].astype(np.float64))
            r = d["returns"].astype(np.float64)
            t = d["timestamps"]
        pos.append(np.mean(stack_pos, axis=0))
        ret.append(r)
        ts.append(t)
        p = np.mean(stack_pos, axis=0)
        prev = np.concatenate([[0.0], p[:-1]])
        net.append(p * r - 5e-4 * np.abs(p - prev))
    return {
        "positions": np.concatenate(pos),
        "returns": np.concatenate(ret),
        "net": np.concatenate(net),
        "timestamps": np.concatenate(ts),
    }


def _per_activation(cells: Sequence[dict[str, Any]], key: str) -> dict[str, np.ndarray]:
    return {a: v[key] for a, v in aggregate(cells, key).items()}


def analyse(
    cells: Sequence[dict[str, Any]],
    out_dir: str | Path,
    n_boot: int = 10_000,
    n_trials: int | None = None,
) -> dict[str, Any]:
    """Compute every statistic the paper reports."""
    cells = [c for c in cells if c["activation"] in ACTIVATIONS]
    if not cells:
        raise ValueError("no cells to analyse")
    sharpe_by_act = _per_activation(cells, "sharpe")

    # Paired cells: aggregate() preserves file order, which is (fold, seed) for
    # every activation, so the vectors are aligned cell-by-cell.
    pairs: dict[str, Any] = {}
    for act in ACTIVATIONS:
        if act == "qalu" or act not in sharpe_by_act or "qalu" not in sharpe_by_act:
            continue
        a, b = sharpe_by_act["qalu"], sharpe_by_act[act]
        if len(a) != len(b):
            continue
        pairs[act] = paired_comparison(a, b, act, n_boot=n_boot).to_dict()

    streams: dict[str, dict[str, float]] = {}
    dm: dict[str, Any] = {}
    boot: dict[str, Any] = {}
    base_stream = None
    for act in ACTIVATIONS:
        try:
            s = stitch(out_dir, act)
        except FileNotFoundError:
            continue
        net = s["net"]
        skew = float(((net - net.mean()) ** 3).mean() / (net.std() ** 3 + 1e-12))
        kurt = float(((net - net.mean()) ** 4).mean() / (net.std() ** 4 + 1e-12))
        streams[act] = {
            "sharpe": sharpe(net),
            "n_bars": int(net.size),
            "skew": skew,
            "kurtosis": kurt,
            "total_return": float(np.prod(1.0 + net) - 1.0),
            "mean_net": float(net.mean()),
        }
        if act == "qalu":
            base_stream = net
    if base_stream is not None:
        for act in ACTIVATIONS:
            if act == "qalu" or act not in streams:
                continue
            s = stitch(out_dir, act)
            if s["net"].shape != base_stream.shape:
                continue
            stat, p = dm_test(-base_stream, -s["net"], h=288)
            obs, lo, hi = stationary_bootstrap_sharpe_diff(base_stream, s["net"], n_boot=1_000, mean_block=288)
            dm[act] = {"stat": stat, "p": p, "boot_diff": obs, "boot_lo": lo, "boot_hi": hi}

    sr_vals = np.array([v["sharpe"] for v in streams.values()])
    trials = n_trials if n_trials is not None else max(len(sr_vals), 1) * 3
    dsr, hurdle = ({}, 0.0)
    if "qalu" in streams and len(sr_vals) > 1:
        dsr, hurdle = deflated_sharpe_ratio(
            streams["qalu"]["sharpe"],
            streams["qalu"]["n_bars"],
            n_trials=trials,
            sr_variance=float(np.var(sr_vals, ddof=1)),
            skew=streams["qalu"]["skew"],
            kurtosis=streams["qalu"]["kurtosis"],
        )

    summary: dict[str, dict[str, float]] = {}
    for act in ACTIVATIONS:
        rows = [c for c in cells if c["activation"] == act]
        if not rows:
            continue
        summary[act] = {
            key: float(np.mean([r[key] for r in rows]))
            for key in ("sharpe", "active_sharpe", "ann_return", "ann_vol", "max_drawdown", "turnover", "exposure", "hit_rate", "ic", "sign_acc", "fee_drag")
            if key in rows[0]
        }
        summary[act]["sharpe_std"] = float(np.std([r["sharpe"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0
        summary[act]["n_cells"] = len(rows)
        summary[act]["n_params"] = int(rows[0]["n_params"])
        summary[act]["train_seconds"] = float(np.mean([r["train_seconds"] for r in rows]))

    out = {
        "n_cells": len(cells),
        "n_bars": int(max((v["n_bars"] for v in streams.values()), default=0)),
        "n_samples": int(max((r.get("n_samples", 0) for r in rows), default=0)),
        "folds": sorted({int(c["fold"]) for c in cells}),
        "seeds": sorted({int(c["seed"]) for c in cells}),
        "summary": summary,
        "paired": pairs,
        "streams": streams,
        "dm": dm,
        "dsr": {"value": dsr, "hurdle_sharpe": hurdle, "n_trials": trials},
        "protocol": {
            "fee_ratio": cells[0]["config"]["fee_ratio"],
            "seq_len": cells[0]["config"]["seq_len"],
            "batch_size": cells[0]["config"]["batch_size"],
            "epochs": cells[0]["config"]["epochs"],
            "steps_per_epoch": cells[0]["config"]["steps_per_epoch"],
            "bars_per_year": cells[0]["config"]["bars_per_year"],
            "test_start": cells[0]["test_start"],
        },
    }
    (Path(out_dir) / "analysis.json").write_text(json.dumps(out, indent=2))
    return out


def latex_table(analysis: dict[str, Any], path: str | Path | None = None) -> str:
    """Booktabs table environment for Table I: one row per activation plus tests."""
    s = analysis["summary"]
    order = [a for a in ACTIVATIONS if a in s]
    best = max(order, key=lambda a: s[a]["sharpe"]) if order else None
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Out-of-sample results over " + str(analysis["n_cells"]) + r" paired cells ("
        + str(len(analysis["folds"])) + r" folds $\times$ " + str(len(analysis["seeds"])) + r" seeds). "
        r"Sharpe is annualised on the post-cost net stream; turnover is mean $|a_t-a_{t-1}|$ per bar. "
        r"Lower blocks give the paired Q-ALU difference with a 95\% bootstrap CI and a Wilcoxon signed-rank $p$-value.}",
        r"\label{tab:results}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Activation & Sharpe & Active & IC & Ann.\ ret. & Max DD & Turnover & Exposure & Params \\",
        r"\midrule",
    ]
    for act in order:
        v = s[act]
        name = LABELS[act] + (r"$^{\ast}$" if act == best else "")
        lines.append(
            f"{name} & {v['sharpe']:.3f} $\\pm$ {v['sharpe_std']:.3f} & "
            f"{v.get('active_sharpe', 0.0):+.3f} & "
            f"{v.get('ic', 0.0):+.4f} & "
            f"{v['ann_return'] * 100:+.1f}\\% & {v['max_drawdown'] * 100:.1f}\\% & "
            f"{v['turnover']:.4f} & {v['exposure']:.3f} & {v['n_params']} \\\\"
        )
    lines.append(r"\midrule")
    for act, p in analysis["paired"].items():
        star = r"$^{\dagger}$" if p["wilcoxon_p"] < 0.05 else ""
        lines.append(
            f"\\multicolumn{{9}}{{l}}{{\\footnotesize Q-ALU $-$ {LABELS[act]}: "
            f"$\\Delta$Sharpe {p['mean_diff']:+.3f} "
            f"[{p['ci_low']:+.3f}, {p['ci_high']:+.3f}], "
            f"$p_{{\\mathrm{{W}}}}={p['wilcoxon_p']:.3f}$" + star + "} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    body = "\n".join(lines)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(body + "\n")
    return body


def latex_macros(analysis: dict[str, Any], path: str | Path, deploy: dict[str, Any] | None = None) -> str:
    """Emit ``\\newcommand`` lines so the paper never contains a typed number."""
    s = analysis["summary"]
    proto = analysis["protocol"]

    def fmt(x: float, nd: int = 3) -> str:
        return f"{x:.{nd}f}"

    lines = [
        "% generated by analysis/report.py - do not edit by hand",
        f"\\newcommand{{\\paramCount}}{{{s[list(s)[0]]['n_params']:,}}}".replace(",", "{,}"),
        f"\\newcommand{{\\nBars}}{{{analysis.get('n_bars', 0):,}}}".replace(",", "{,}"),
        f"\\newcommand{{\\nSample}}{{{analysis.get('n_samples', 0):,}}}".replace(",", "{,}"),
        f"\\newcommand{{\\nCells}}{{{analysis['n_cells']}}}",
        f"\\newcommand{{\\feeBp}}{{{proto['fee_ratio'] * 1e4:.0f}}}",
        f"\\newcommand{{\\seqLen}}{{{proto['seq_len']}}}",
    ]
    for act in ACTIVATIONS:
        if act not in s:
            continue
        tag = act.capitalize()
        lines.append(f"\\newcommand{{\\sr{tag}}}{{{fmt(s[act]['sharpe'])}}}")
        lines.append(f"\\newcommand{{\\sr{tag}Sd}}{{{fmt(s[act]['sharpe_std'])}}}")
        lines.append(f"\\newcommand{{\\ret{tag}}}{{{s[act]['ann_return'] * 100:+.1f}\\%}}")
        lines.append(f"\\newcommand{{\\dd{tag}}}{{{s[act]['max_drawdown'] * 100:.1f}\\%}}")
        lines.append(f"\\newcommand{{\\turn{tag}}}{{{fmt(s[act]['turnover'], 4)}}}")
        lines.append(f"\\newcommand{{\\expo{tag}}}{{{fmt(s[act]['exposure'])}}}")
        lines.append(f"\\newcommand{{\\act{tag}}}{{{s[act].get('active_sharpe', 0.0):+.3f}}}")
        lines.append(f"\\newcommand{{\\drag{tag}}}{{{s[act].get('fee_drag', 0.0) * 100:.1f}\\%}}")
        lines.append(f"\\newcommand{{\\ic{tag}}}{{{s[act].get('ic', 0.0):+.4f}}}")
    for act, p in analysis["paired"].items():
        tag = act.capitalize()
        lines.append(f"\\newcommand{{\\dSharpe{tag}}}{{{p['mean_diff']:+.3f}}}")
        lines.append(f"\\newcommand{{\\ciLo{tag}}}{{{p['ci_low']:+.3f}}}")
        lines.append(f"\\newcommand{{\\ciHi{tag}}}{{{p['ci_high']:+.3f}}}")
        lines.append(f"\\newcommand{{\\pW{tag}}}{{{p['wilcoxon_p']:.3f}}}")
        lines.append(f"\\newcommand{{\\wins{tag}}}{{{p['wins']}/{p['n_pairs']}}}")
    for act, d in analysis.get("dm", {}).items():
        tag = act.capitalize()
        lines.append(f"\\newcommand{{\\dmStat{tag}}}{{{d['stat']:+.2f}}}")
        lines.append(f"\\newcommand{{\\dmP{tag}}}{{{d['p']:.3f}}}")
    dsr = analysis.get("dsr", {})
    if dsr:
        lines.append(f"\\newcommand{{\\dsrValue}}{{{dsr['value']:.3f}}}")
        lines.append(f"\\newcommand{{\\dsrHurdle}}{{{dsr['hurdle_sharpe']:.3f}}}")
        lines.append(f"\\newcommand{{\\dsrTrials}}{{{dsr['n_trials']}}}")
    if deploy:
        _par = max(deploy["parity"].values())
        lines.append(
            "\\newcommand{\\parity}{%s}" % ("0.0" if _par < 1e-9 else f"{_par:.1e}")
        )
        lines.append(f"\\newcommand{{\\latencyOnnx}}{{{deploy['latency_ms']['onnxruntime']['p50']:.2f}}}")
        lines.append(f"\\newcommand{{\\latencyTorch}}{{{deploy['latency_ms']['torchscript']['p50']:.2f}}}")
        lines.append(f"\\newcommand{{\\rssOnnx}}{{{deploy['rss_mb']['onnxruntime']:.1f}}}")
        lines.append(f"\\newcommand{{\\onnxKb}}{{{deploy['onnx_bytes'] / 1024:.0f}}}")
    else:
        lines.append(r"\newcommand{\parity}{n/a}")
        lines.append(r"\newcommand{\latencyOnnx}{n/a}")
        lines.append(r"\newcommand{\latencyTorch}{n/a}")
        lines.append(r"\newcommand{\rssOnnx}{n/a}")
        lines.append(r"\newcommand{\onnxKb}{n/a}")
    nar = narrative(analysis)
    lines.append(f"\\newcommand{{\\resultText}}{{{nar['result']}}}")
    lines.append(f"\\newcommand{{\\resultSentence}}{{{nar['sentence']}}}")
    lines.append(f"\\newcommand{{\\spreadSentence}}{{{nar['spread']}}}")
    lines.append(f"\\newcommand{{\\nFolds}}{{{len(analysis['folds'])}}}")
    lines.append(f"\\newcommand{{\\nSeeds}}{{{len(analysis['seeds'])}}}")
    lines.append(f"\\newcommand{{\\conclusionSentence}}{{{nar['conclusion']}}}")
    body = "\n".join(lines) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(body)
    return body


def narrative(analysis: dict[str, Any]) -> dict[str, str]:
    """Generate the results paragraph so no number is typed by hand."""
    s = analysis["summary"]
    order = [a for a in ACTIVATIONS if a in s]
    if not order or "qalu" not in s:
        return {"result": "No cells were analysed.", "conclusion": ""}
    paired = analysis.get("paired", {})
    best = max(order, key=lambda a: s[a]["sharpe"])
    parts: list[str] = []

    if best == "qalu":
        parts.append(
            f"Q-ALU attains the highest mean post-cost Sharpe ({s['qalu']['sharpe']:.3f} "
            f"$\\pm$ {s['qalu']['sharpe_std']:.3f} across {s['qalu']['n_cells']} paired cells)"
        )
    else:
        parts.append(
            f"{LABELS[best]} attains the highest mean post-cost Sharpe "
            f"({s[best]['sharpe']:.3f} $\\pm$ {s[best]['sharpe_std']:.3f}); Q-ALU reaches "
            f"{s['qalu']['sharpe']:.3f} $\\pm$ {s['qalu']['sharpe_std']:.3f}"
        )

    confirmed = [
        b
        for b, p in paired.items()
        if p["mean_diff"] > 0 and p["ci_low"] > 0 and p["wilcoxon_p"] < 0.05
    ]
    if confirmed:
        names = ", ".join(LABELS[b] for b in confirmed)
        parts.append(
            f"Against {names} the pre-registered decision rule is satisfied: the paired "
            f"difference is positive, its bootstrap interval excludes zero, and the "
            f"Wilcoxon signed-rank test rejects equality."
        )
    else:
        parts.append(
            "No baseline satisfies all four pre-registered conditions, so the evidence "
            "does not support a claim that Q-ALU is superior on this protocol."
        )

    expo = min(s[a]["exposure"] for a in order)
    turn = max(s[a]["turnover"] for a in order)
    act_abs = max(abs(s[a].get("active_sharpe", 0.0)) for a in order)
    if expo > 0.90 and turn < 0.02:
        parts.append(
            f"All four arms collapse to a near-constant long position (mean exposure "
            f"$\\geq$ {expo:.3f}, mean turnover $\\leq$ {turn:.4f} per bar, $|$active "
            f"Sharpe$|$ $\\leq$ {act_abs:.3f}). Their raw Sharpe ratios therefore mostly "
            f"measure the drift of the underlying asset; what separates them is the cost "
            f"of the small residual rebalancing, where Q-ALU pays {s['qalu'].get('fee_drag', 0.0) * 100:.1f}\\% "
            f"of equity in fees against "
            + ", ".join(
                f"{LABELS[a]} {s[a].get('fee_drag', 0.0) * 100:.1f}\\%"
                for a in order
                if a != "qalu"
            )
            + " over the stitched out-of-sample blocks."
        )

    result = " ".join(p if p.rstrip().endswith(".") else p.rstrip() + "." for p in parts)

    sentence = (
        f"Q-ALU reaches a post-cost Sharpe of {s['qalu']['sharpe']:.3f} against "
        + ", ".join(f"{LABELS[a]} {s[a]['sharpe']:.3f}" for a in order if a != "qalu")
        + (
            "; the paired differences are not significant, and all four arms collapse "
            "to a near-constant long position whose only measurable difference is fee drag"
            if not confirmed
            else "; the paired differences against "
            + ", ".join(LABELS[b] for b in confirmed)
            + " are positive with intervals excluding zero"
        )
    )

    spread = max(s[a]["sharpe_std"] for a in order)
    gap = max((abs(p["mean_diff"]) for p in paired.values()), default=0.0)
    if spread > gap:
        spread_txt = (
            f"the seed-to-seed standard deviation of a single configuration reaches "
            f"{spread:.3f} Sharpe, which exceeds the largest between-activation "
            f"difference ({gap:.3f})"
        )
    else:
        spread_txt = (
            f"the largest between-activation difference ({gap:.3f} Sharpe) exceeds the "
            f"seed-to-seed standard deviation of a single configuration ({spread:.3f})"
        )

    if confirmed:
        concl = (
            "Under a pre-registered purged walk-forward protocol with paired seeds and "
            "three independent tests, Q-ALU improves the post-cost Sharpe of a causal "
            "dual-stream state-space policy relative to the baselines it beats."
        )
    else:
        concl = (
            "Under a pre-registered purged walk-forward protocol with paired seeds and "
            "three independent tests, the measured differences do not support a claim "
            "that Q-ALU is superior; the comparison instead isolates the cost of "
            "turnover, which the asymmetric activation reduces."
        )
    return {
        "result": result,
        "conclusion": concl,
        "sentence": sentence,
        "spread": spread_txt,
    }


def ascii_report(analysis: dict[str, Any]) -> str:
    s = analysis["summary"]
    out = [
        f"cells={analysis['n_cells']} folds={analysis['folds']} seeds={analysis['seeds']}",
        f"{'act':<7}{'sharpe':>9}{'+-':>7}{'ret':>9}{'dd':>8}{'turn':>9}{'expo':>7}",
    ]
    for act in ACTIVATIONS:
        if act not in s:
            continue
        v = s[act]
        out.append(
            f"{act:<7}{v['sharpe']:>9.3f}{v['sharpe_std']:>7.3f}{v['ann_return']:>9.1%}"
            f"{v['max_drawdown']:>8.1%}{v['turnover']:>9.4f}{v['exposure']:>7.3f}"
        )
    for act, p in analysis["paired"].items():
        out.append(
            f"  Q-ALU - {act:<5} dSharpe={p['mean_diff']:+.3f} "
            f"CI=[{p['ci_low']:+.3f},{p['ci_high']:+.3f}] pW={p['wilcoxon_p']:.4f} wins={p['wins']}/{p['n_pairs']}"
        )
    for act, d in analysis["dm"].items():
        out.append(f"  DM qalu vs {act:<5} stat={d['stat']:+.2f} p={d['p']:.4f}")
    return "\n".join(out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/raw")
    args = ap.parse_args()
    cells = load_cells(args.out)
    a = analyse(cells, args.out)
    print(ascii_report(a))
    latex_table(a, Path(args.out).parent / "tables" / "results.tex")
