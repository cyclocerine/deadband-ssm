"""Guards on the generated LaTeX.

The paper is built from macros, so a generator bug shows up as a broken build
rather than a wrong number. These tests pin the two failure modes that have
already cost a debugging session: a macro name containing a digit, and a macro
that the prose uses but the generator does not emit.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from analysis.sweep_report import ARCH_ORDER, ACT_ORDER, TF_ORDER, latex_macros, latex_table, summarise

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "paper.tex"


def _rows() -> list[dict[str, Any]]:
    """A full synthetic grid: every timeframe, model and activation, two folds."""
    rows: list[dict[str, Any]] = []
    for i, tf in enumerate(TF_ORDER):
        for j, arch in enumerate(ARCH_ORDER):
            for k, act in enumerate(ACT_ORDER):
                for fold in range(2):
                    rows.append(
                        {
                            "timeframe": tf,
                            "arch": arch,
                            "activation": act,
                            "fold": fold,
                            "sharpe": -1.0 - 0.1 * i - 0.01 * j + 0.001 * k + 0.01 * fold,
                            "turnover": 0.02 + 0.001 * k + 0.05 * (act == "qalu"),
                            "ann_return": -0.2 - 0.01 * k,
                            "n_samples": 100_000 // (i + 1),
                            "config": {"epochs": 2, "steps": 80},
                        }
                    )
    return rows


def test_macro_names_are_letters_only(tmp_path: Path) -> None:
    """A digit ends a control sequence and typesets the digit into the preamble."""
    path = latex_macros(_rows(), tmp_path / "m.tex")
    names = re.findall(r"\\newcommand\{\\([^}]*)\}", path.read_text())
    assert names
    bad = [n for n in names if not n.isalpha()]
    assert not bad, f"macro names with non-letters: {bad}"


def test_every_bar_macro_is_emitted_even_on_partial_data(tmp_path: Path) -> None:
    """A missing timeframe must not leave an undefined macro in the build."""
    rows = [r for r in _rows() if r["timeframe"] == "5m"]
    path = latex_macros(rows, tmp_path / "m.tex")
    text = path.read_text()
    for tag in ("FiveM", "FifteenM", "OneH", "FourH", "OneD", "OneW"):
        assert f"\\sweepBars{tag}" in text


def test_prose_macros_are_all_emitted(tmp_path: Path) -> None:
    """Every \\sweep* the prose quotes must exist in the generated file."""
    used = set(re.findall(r"\\(sweep[A-Za-z]+)", PAPER.read_text()))
    assert used
    emitted = set(re.findall(r"\\newcommand\{\\(sweep[A-Za-z]+)\}", latex_macros(_rows(), tmp_path / "m.tex").read_text()))
    missing = sorted(used - emitted)
    assert not missing, f"used in paper.tex but never emitted: {missing}"


def test_table_has_one_row_per_timeframe_and_model(tmp_path: Path) -> None:
    path = latex_table(_rows(), tmp_path / "t.tex")
    text = path.read_text()
    assert "\\begin{table*}" in text and "\\end{table*}" in text
    # Each timeframe label opens a group of one row per model.
    for tf in TF_ORDER:
        assert re.search(rf"^{re.escape(tf)} & ", text, flags=re.M), tf
    assert text.count(" & ") >= len(TF_ORDER) * len(ARCH_ORDER) * len(ACT_ORDER)
    # Every activation column appears in the header.
    header = next(line for line in text.splitlines() if "Q-ALU" in line)
    for label in ("GELU", "Swish", "ReLU"):
        assert label in header
    assert text.count("{") == text.count("}")


def test_summarise_averages_folds() -> None:
    cells = summarise(_rows())
    assert len(cells) == len(TF_ORDER) * len(ARCH_ORDER) * len(ACT_ORDER)
    one = cells[("5m", "ssm", "qalu")]
    assert one["n"] == 2
    assert one["mean"] == pytest.approx((-1.0 + 0.001 * 0 + 0.01 * 0 + -1.0 + 0.001 * 0 + 0.01 * 1) / 2, abs=1e-9)
