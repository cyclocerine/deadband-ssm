"""Render the cross-timeframe, cross-architecture sweep as a paper table.

The sweep in :mod:`benchmark.sweep` answers a different question from the main
grid in Table I: not "does the activation matter on 5-minute bars" but "does the
ordering survive when the horizon and the sequence model change". It is run at a
smaller budget on one seed, so its cells are not comparable to Table I and the
table says so. Nothing here re-uses a Table I number, and no significance claim
is made across cells: with one seed and three folds per cell there is no paired
test to make.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TF_ORDER = ("5m", "15m", "1h", "4h", "1d", "1w")
ARCH_ORDER = ("ssm", "lstm", "gru", "tcn")
CHAT_TURNOVER = 0.05  # mean |Δposition| per bar above which a cell is called chattering

ACT_ORDER = ("qalu", "gelu", "swish", "relu")
LABELS = {"qalu": "Q-ALU", "gelu": "GELU", "swish": "Swish", "relu": "ReLU"}
ARCH_LABELS = {
    "ssm": "SSM",
    "lstm": "LSTM",
    "gru": "GRU",
    "tcn": "TCN",
}


def load(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text())


def summarise(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, float]]:
    """Mean and spread of each (timeframe, arch, activation) cell over folds."""
    cells: dict[tuple[str, str, str], dict[str, float]] = {}
    for r in rows:
        key = (r["timeframe"], r["arch"], r["activation"])
        cells.setdefault(key, {"sharpe": [], "turnover": [], "ann_return": []})  # type: ignore[arg-type]
        cells[key]["sharpe"].append(float(r["sharpe"]))  # type: ignore[union-attr]
        cells[key]["turnover"].append(float(r["turnover"]))  # type: ignore[union-attr]
        cells[key]["ann_return"].append(float(r["ann_return"]))  # type: ignore[union-attr]
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, v in cells.items():
        s = np.asarray(v["sharpe"], dtype=float)
        out[key] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
            "n": float(s.size),
            "turnover": float(np.mean(v["turnover"])),
            "ann_return": float(np.mean(v["ann_return"])),
        }
    return out


def latex_table(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    cells = summarise(rows)
    tfs = [t for t in TF_ORDER if any(k[0] == t for k in cells)]
    archs = [a for a in ARCH_ORDER if any(k[1] == a for k in cells)]
    acts = [a for a in ACT_ORDER if any(k[2] == a for k in cells)]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Out-of-sample mean Sharpe by timeframe, sequence model and",
        r"activation, one seed and the same three purged folds per timeframe",
        r"(one fold for 1w). Every row is trained at the reduced sweep budget of",
        r"Section~\ref{sec:sweep}, so cells are comparable within this table and",
        r"not to Table~\ref{tab:results}; the 5m row is the same protocol as",
        r"Table~\ref{tab:results} at a smaller budget and is expected to differ.",
        r"Parentheses give the spread across folds; the last column is the",
        r"per-bar turnover, which is what decides the fee.}",
        r"\label{tab:sweep}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll" + "r" * len(acts) + r"r}",
        r"\toprule",
        r"TF & model & " + " & ".join(LABELS[a] for a in acts) + r" & turnover \\",
        r"\midrule",
    ]
    for tf in tfs:
        for ai, arch in enumerate(archs):
            if ai == 0 and tf != tfs[0]:
                lines.append(r"\midrule")
            row = [tf if ai == 0 else "", ARCH_LABELS[arch]]
            turns = []
            for act in acts:
                c = cells.get((tf, arch, act))
                if c is None:
                    row.append("--")
                    continue
                spread = f" ({c['std']:.2f})" if c["n"] > 1 else ""
                row.append(f"${c['mean']:+.3f}$" + spread)
                turns.append(c["turnover"])
            row.append(f"{np.mean(turns):.4f}" if turns else "--")
            lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def latex_macros(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Emit the handful of numbers the prose is allowed to quote."""
    cells = summarise(rows)
    cfg = rows[0].get("config", {}) if rows else {}
    epochs = cfg.get("epochs", "?")
    steps = cfg.get("steps_per_epoch", cfg.get("steps", "?"))
    n_folds = int(max(c["n"] for c in cells.values())) if cells else 0
    acts = [a for a in ACT_ORDER if any(k[2] == a for k in cells)]
    per_act = {a: float(np.mean([c["mean"] for k, c in cells.items() if k[2] == a])) for a in acts}
    ranked = sorted(per_act, key=lambda a: per_act[a], reverse=True)
    qalu_rank = ranked.index("qalu") + 1 if "qalu" in ranked else 0
    best_key = max(cells, key=lambda k: cells[k]["mean"])
    chat = sum(1 for c in cells.values() if c["turnover"] > CHAT_TURNOVER)
    chat_qalu = sum(1 for k, c in cells.items() if k[2] == "qalu" and c["turnover"] > CHAT_TURNOVER)
    qalu_cells = sum(1 for k in cells if k[2] == "qalu")
    # Paired comparison. Inside a (timeframe, model) group the four activations
    # see the same bars, folds and budget, so the activation is the only
    # difference. A mean over all cells is dominated by whichever cell blows up,
    # so the headline is the within-group win count plus the median gap.
    groups: dict[tuple[str, str], dict[str, float]] = {}
    for (tf, arch, act), c in cells.items():
        groups.setdefault((tf, arch), {})[act] = c["mean"]
    paired = {k: v for k, v in groups.items() if len(v) == len(acts) and "qalu" in v}
    wins = sum(1 for v in paired.values() if v["qalu"] >= max(v.values()))
    gaps = [v["qalu"] - max(x for a, x in v.items() if a != "qalu") for v in paired.values()]
    per_act_med = {a: float(np.median([c["mean"] for k, c in cells.items() if k[2] == a])) for a in acts}
    ranked_med = sorted(per_act_med, key=lambda a: per_act_med[a], reverse=True)
    # The 5-minute row of the sweep runs the same arm as Table I at the reduced
    # budget, so the pair (Table I, this row) shows the budget effect directly.
    five = {a: cells[("5m", "ssm", a)]["mean"] for a in acts if ("5m", "ssm", a) in cells}
    five_best = max(five, key=lambda a: five[a]) if five else None
    five_qalu = f"{five['qalu']:+.3f}" if "qalu" in five else "?"
    five_best_act = LABELS[five_best] if five_best else "?"
    five_best_val = f"{five[five_best]:+.3f}" if five_best else "?"
    # Q-ALU is often both the subject and the extreme, so the prose needs the
    # runner-up explicitly; quoting "the best other arm" twice reads as a bug.
    five_rest = sorted((a for a in five if a != "qalu"), key=lambda a: -five[a])
    five_next_act = LABELS[five_rest[0]] if five_rest else "?"
    five_next_val = f"{five[five_rest[0]]:+.3f}" if five_rest else "?"
    med_rest = [a for a in ranked_med if a != "qalu"]
    med_next_act = LABELS[med_rest[0]] if med_rest else "?"
    med_next_val = f"{per_act_med[med_rest[0]]:+.3f}" if med_rest else "?"
    lines = [
        "% Generated by analysis/sweep_report.py. Do not edit by hand.",
        f"\\newcommand{{\\sweepEpochs}}{{{epochs}}}",
        f"\\newcommand{{\\sweepSteps}}{{{steps}}}",
        f"\\newcommand{{\\sweepFolds}}{{{n_folds}}}",
        f"\\newcommand{{\\sweepCells}}{{{len(rows)}}}",
        f"\\newcommand{{\\sweepTFs}}{{{len({k[0] for k in cells})}}}",
        f"\\newcommand{{\\sweepBestCell}}{{{ARCH_LABELS[best_key[1]]} with {LABELS[best_key[2]]} at {best_key[0]}}}",
        f"\\newcommand{{\\sweepBestSharpe}}{{{cells[best_key]['mean']:+.3f}}}",
        f"\\newcommand{{\\sweepBestAct}}{{{LABELS[ranked[0]]}}}",
        f"\\newcommand{{\\sweepBestActMean}}{{{per_act[ranked[0]]:+.3f}}}",
        f"\\newcommand{{\\sweepWorstAct}}{{{LABELS[ranked[-1]]}}}",
        f"\\newcommand{{\\sweepWorstActMean}}{{{per_act[ranked[-1]]:+.3f}}}",
        f"\\newcommand{{\\sweepQaluRank}}{{{qalu_rank}}}",
        f"\\newcommand{{\\sweepQaluMean}}{{{per_act.get('qalu', float('nan')):+.3f}}}",
        f"\\newcommand{{\\sweepChatCells}}{{{chat}}}",
        f"\\newcommand{{\\sweepChatDenom}}{{{len(cells)}}}",
        f"\\newcommand{{\\sweepFiveMQalu}}{{{five_qalu}}}",
        f"\\newcommand{{\\sweepFiveMBestAct}}{{{five_best_act}}}",
        f"\\newcommand{{\\sweepFiveMBest}}{{{five_best_val}}}",
        f"\\newcommand{{\\sweepFiveMNextAct}}{{{five_next_act}}}",
        f"\\newcommand{{\\sweepFiveMNext}}{{{five_next_val}}}",
        f"\\newcommand{{\\sweepMedNextAct}}{{{med_next_act}}}",
        f"\\newcommand{{\\sweepMedNext}}{{{med_next_val}}}",
        f"\\newcommand{{\\sweepChatQalu}}{{{chat_qalu}}}",
        f"\\newcommand{{\\sweepChatOther}}{{{chat - chat_qalu}}}",
        f"\\newcommand{{\\sweepOtherCells}}{{{len(cells) - qalu_cells}}}",
        f"\\newcommand{{\\sweepQaluCells}}{{{qalu_cells}}}",
        f"\\newcommand{{\\sweepGroups}}{{{len(paired)}}}",
        f"\\newcommand{{\\sweepQaluWins}}{{{wins}}}",
        f"\\newcommand{{\\sweepQaluGapMed}}{{{np.median(gaps) if gaps else float('nan'):+.3f}}}",
        f"\\newcommand{{\\sweepQaluGapBest}}{{{max(gaps) if gaps else float('nan'):+.3f}}}",
        f"\\newcommand{{\\sweepQaluGapWorst}}{{{min(gaps) if gaps else float('nan'):+.3f}}}",
        f"\\newcommand{{\\sweepMedRank}}{{{ranked_med.index('qalu') + 1 if 'qalu' in ranked_med else 0}}}",
        f"\\newcommand{{\\sweepMedQalu}}{{{per_act_med.get('qalu', float('nan')):+.3f}}}",
        f"\\newcommand{{\\sweepMedBestAct}}{{{LABELS[ranked_med[0]]}}}",
        f"\\newcommand{{\\sweepMedBest}}{{{per_act_med[ranked_med[0]]:+.3f}}}",
        f"\\newcommand{{\\sweepChatThr}}{{{CHAT_TURNOVER:.2f}}}",
        f"\\newcommand{{\\sweepTurnMedian}}{{{np.median([c['turnover'] for c in cells.values()]):.4f}}}",
        f"\\newcommand{{\\sweepTurnQalu}}{{{np.mean([c['turnover'] for k, c in cells.items() if k[2] == 'qalu']):.4f}}}",
        f"\\newcommand{{\\sweepTurnOther}}{{{np.mean([c['turnover'] for k, c in cells.items() if k[2] != 'qalu']):.4f}}}",
    ]
    # Panel size per timeframe, so the prose can quote the weekly panel without
    # hard-coding a bar count.
    # Letters only: a digit would end the control sequence and typeset the digit
    # into the preamble.
    tag = {"5m": "FiveM", "15m": "FifteenM", "1h": "OneH", "4h": "FourH", "1d": "OneD", "1w": "OneW"}
    for tf in TF_ORDER:
        n = next((int(r["n_samples"]) for r in rows if r["timeframe"] == tf), None)
        # Always emit all six: a missing row must not break the paper build.
        val = "?" if n is None else f"{n:,}".replace(",", "{,}")
        lines.append(f"\\newcommand{{\\sweepBars{tag[tf]}}}{{{val}}}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def ascii_report(rows: Sequence[dict[str, Any]]) -> str:
    cells = summarise(rows)
    tfs = [t for t in TF_ORDER if any(k[0] == t for k in cells)]
    archs = [a for a in ARCH_ORDER if any(k[1] == a for k in cells)]
    acts = [a for a in ACT_ORDER if any(k[2] == a for k in cells)]
    head = f"{'tf':>4} {'model':>5} " + " ".join(f"{LABELS[a]:>8}" for a in acts) + f" {'turn':>7}"
    out = [head, "-" * len(head)]
    for tf in tfs:
        for arch in archs:
            parts = []
            for act in acts:
                c = cells.get((tf, arch, act))
                parts.append(f"{c['mean']:>+8.3f}" if c else f"{'--':>8}")
            turns = [cells[(tf, arch, a)]["turnover"] for a in acts if (tf, arch, a) in cells]
            out.append(f"{tf:>4} {ARCH_LABELS[arch]:>5} " + " ".join(parts) + f" {np.mean(turns):>7.4f}")
    # Rank the activations by their mean over every cell in the sweep.
    out.append("")
    out.append("mean over all cells:")
    for act in acts:
        vals = [c["mean"] for k, c in cells.items() if k[2] == act]
        if vals:
            out.append(f"  {LABELS[act]:>6}: {np.mean(vals):+.3f}  (n={len(vals)} cells)")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", default=str(ROOT / "results" / "sweep" / "all.json"))
    ap.add_argument("--tex", default=str(ROOT / "paper" / "tables" / "sweep.tex"))
    ap.add_argument("--macros", default=str(ROOT / "paper" / "tables" / "sweep_macros.tex"))
    args = ap.parse_args(argv)
    rows = load(args.src)
    print(ascii_report(rows))
    print(f"\nwrote {latex_table(rows, args.tex)}")
    print(f"wrote {latex_macros(rows, args.macros)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
