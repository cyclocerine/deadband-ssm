"""Run one trained Deadband-SSM cell on a real out-of-sample bar, end to end.

The point of this script is that nothing in it is a re-implementation: it
rebuilds the fold geometry from the same splitter the benchmark uses, refits the
standardizer on the same training slice, loads the checkpoint, and prints the
forecast and the position for a bar whose recorded values are in the cell's
``.npz`` artifact. If the pipeline and this script disagree, one of them is
wrong, and the printed difference says which.

    python examples/quickstart.py                       # Q-ALU, fold 0, seed 0
    python examples/quickstart.py --ckpt results/raw/gelu_f1_s0.pt --bar 40000
    python examples/quickstart.py --onnx export/model.onnx   # also check ONNX

Expected output for the default cell is reproduced in examples/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.cross_val import PurgedWalkForward  # noqa: E402
from core.model import DeadbandSSM  # noqa: E402
from data.fetcher import Standardizer, load_panel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(ROOT / "results/raw/qalu_f0_s0.pt"))
    ap.add_argument("--bar", type=int, default=5000, help="bar offset inside the test slice")
    ap.add_argument("--onnx", default=None, help="also run this ONNX artifact on the same window")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    meta = json.loads(ckpt_path.with_suffix(".json").read_text())
    cfg = meta["config"]

    panel = load_panel()
    # The panel grows as new bars arrive, and the splitter's geometry is a
    # function of the sample count, so pin the panel to the length the cell was
    # trained on. Without this the fold boundaries move by however many bars were
    # downloaded since, and the comparison below measures the drift, not the model.
    panel = replace(panel, timestamps=panel.timestamps[:meta["n_samples"]],
                    features=panel.features[:meta["n_samples"]],
                    returns=panel.returns[:meta["n_samples"]])
    split = PurgedWalkForward(len(panel), purge=cfg["seq_len"] + 1).split()[meta["fold"]]
    te0, te1 = split.test

    # The standardizer is fitted on the training slice only; fitting it on the
    # whole panel would leak the test period's mean into the inputs.
    scaler = Standardizer().fit(panel.features[split.train[0]:split.train[1]])
    x_all = scaler.transform(panel.features)

    i = te0 + args.bar
    t = cfg["seq_len"]
    window = torch.from_numpy(x_all[i - t + 1:i + 1][None])  # (1, T, F)

    model = DeadbandSSM(
        n_features=x_all.shape[1],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        d_state=cfg["d_state"],
        dropout=cfg["dropout"],
        act=meta["activation"],
        mode=cfg["mode"],
    ).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

    with torch.no_grad():
        pos, forecast = model(window)
    pos, forecast = float(pos[0]), float(forecast[0])

    # What the benchmark recorded for this bar, from the cell's own artifact.
    cell = np.load(ckpt_path.with_suffix(".npz"))
    rec_pos, rec_fc = float(cell["positions"][i - te0]), float(cell["forecast"][i - te0])
    r_next = float(panel.returns[i])
    fee = 0.0005
    trade = abs(pos - (float(cell["positions"][i - te0 - 1]) if i - te0 else 0.0))

    print(f"cell        {meta['activation']} f{meta['fold']} s{meta['seed']}  ({meta['n_params']} params)")
    print(f"bar         {i}  {panel.timestamps[i]}  (test slice {te0}..{te1})")
    print(f"window      {window.shape}  features {panel.feature_names}")
    print(f"forecast    v_T = {forecast:+.6f}   (artifact {rec_fc:+.6f})")
    print(f"position    pi_T = tanh(v_T) = {pos:+.6f}   (artifact {rec_pos:+.6f})")
    print(f"difference  {abs(pos - rec_pos):.2e} on the position, {abs(forecast - rec_fc):.2e} on the forecast")
    print(f"next bar    r = {r_next:+.6%}   traded {trade:.6f}   "
          f"net = {pos * r_next - fee * trade:+.6%} of notional")

    if args.onnx:
        import onnxruntime as ort

        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        onnx_pos = float(sess.run(None, {sess.get_inputs()[0].name: window.numpy()})[0][0])
        print(f"onnx        {args.onnx}  pi_T = {onnx_pos:+.6f}   "
              f"|onnx - torch| = {abs(onnx_pos - pos):.2e}")


if __name__ == "__main__":
    main()
