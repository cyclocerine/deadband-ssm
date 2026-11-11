"""Export and deployment benchmarks: TorchScript, ONNX Runtime, latency, RSS.

The deployment claim in the paper is a *measured* one, so every number this
module reports comes from a real execution:

* ``latency``  - wall-clock per-bar inference, p50/p95/p99, after warm-up.
* ``rss``      - peak resident set size of an isolated process that loads the
  artifact and runs inference, minus the interpreter baseline.
* ``parity``   - max absolute deviation between PyTorch, TorchScript and ONNX
  outputs on the same inputs. A non-trivial deviation fails the export.

ONNX Runtime is measured in a child process so that PyTorch's own allocator
does not inflate the footprint.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from core.model import DeadbandSSM

__all__ = [
    "DeployReport",
    "export_torchscript",
    "export_onnx",
    "measure_latency",
    "measure_process_rss",
    "benchmark_artifact",
]


@dataclass
class DeployReport:
    seq_len: int
    n_features: int
    n_params: int
    torchscript_bytes: int
    onnx_bytes: int
    latency_ms: dict[str, dict[str, float]]
    rss_mb: dict[str, float]
    parity: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _example(seq_len: int, n_features: int, batch: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(batch, seq_len, n_features, generator=g)


class PositionOnly(nn.Module):
    """Exported view of the model: one position per bar, no forecast tuple."""

    def __init__(self, model: DeadbandSSM) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out[0] if isinstance(out, tuple) else out


def _positions(model: DeadbandSSM, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(x)
    return (out[0] if isinstance(out, tuple) else out).numpy()


def export_torchscript(model: DeadbandSSM, path: str | Path, seq_len: int, n_features: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(PositionOnly(model), _example(seq_len, n_features), strict=False)
        traced = torch.jit.freeze(traced)  # folds the FIR kernel into constants
    traced.save(str(path))
    return path


def export_onnx(model: DeadbandSSM, path: str | Path, seq_len: int, n_features: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        PositionOnly(model),
        (_example(seq_len, n_features),),
        str(path),
        input_names=["features"],
        output_names=["position"],
        dynamic_axes={"features": {0: "batch"}, "position": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    return path


def measure_latency(
    fn: Callable[[torch.Tensor], Any],
    seq_len: int,
    n_features: int,
    warmup: int = 30,
    runs: int = 200,
) -> dict[str, float]:
    """Per-call wall clock in milliseconds, single bar at a time."""
    x = _example(seq_len, n_features)
    for _ in range(warmup):
        fn(x)
    samples = np.empty(runs, dtype=np.float64)
    for i in range(runs):
        t0 = time.perf_counter()
        fn(x)
        samples[i] = (time.perf_counter() - t0) * 1e3
    return {
        "mean": float(samples.mean()),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "p99": float(np.percentile(samples, 99)),
        "max": float(samples.max()),
    }


_RSS_CHILD = r"""
import json, sys, psutil, numpy as np
kind, path, seq_len, n_features, runs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
proc = psutil.Process()
base = proc.memory_info().rss
x = np.random.randn(1, seq_len, n_features).astype(np.float32)
if kind == "torchscript":
    import torch
    m = torch.jit.load(path).eval()
    with torch.no_grad():
        for _ in range(runs):
            m(torch.from_numpy(x))
    peak = proc.memory_info().rss
elif kind == "onnx":
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    for _ in range(runs):
        sess.run(None, {"features": x})
    peak = proc.memory_info().rss
else:
    raise SystemExit(f"unknown kind {kind}")
print(json.dumps({"base_mb": base / 2**20, "peak_mb": peak / 2**20, "delta_mb": (peak - base) / 2**20}))
"""


def measure_process_rss(kind: str, path: str | Path, seq_len: int, n_features: int, runs: int = 300) -> dict[str, float]:
    """Run inference in a clean child process and report its RSS delta."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(_RSS_CHILD)
        script = fh.name
    try:
        out = subprocess.run(
            [sys.executable, script, kind, str(path), str(seq_len), str(n_features), str(runs)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
    finally:
        os.unlink(script)
    return json.loads(out.stdout.strip().splitlines()[-1])


def benchmark_artifact(
    model: DeadbandSSM,
    out_dir: str | Path,
    seq_len: int,
    n_features: int,
    runs: int = 200,
) -> DeployReport:
    """Export both artifacts, verify parity, measure latency and footprint."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts_path = export_torchscript(model, out / "model.ts", seq_len, n_features)
    onnx_path = export_onnx(model, out / "model.onnx", seq_len, n_features)

    x = _example(seq_len, n_features)
    ref = _positions(model, x)

    ts_model = torch.jit.load(str(ts_path)).eval()
    with torch.no_grad():
        ts_out = ts_model(x).numpy()

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"features": x.numpy()})[0]

    parity = {
        "torchscript_max_abs": float(np.max(np.abs(ts_out - ref))),
        "onnx_max_abs": float(np.max(np.abs(onnx_out - ref))),
    }
    if max(parity.values()) > 1e-4:
        raise RuntimeError(f"export parity check failed: {parity}")

    latency = {
        "torch_eager": measure_latency(lambda t: model(t), seq_len, n_features, runs=runs),
        "torchscript": measure_latency(lambda t: ts_model(t), seq_len, n_features, runs=runs),
        "onnxruntime": measure_latency(
            lambda t: sess.run(None, {"features": t.numpy()}), seq_len, n_features, runs=runs
        ),
    }
    rss = {
        "torchscript": measure_process_rss("torchscript", ts_path, seq_len, n_features)["delta_mb"],
        "onnxruntime": measure_process_rss("onnx", onnx_path, seq_len, n_features)["delta_mb"],
    }
    return DeployReport(
        seq_len=seq_len,
        n_features=n_features,
        n_params=model.n_params,
        torchscript_bytes=ts_path.stat().st_size,
        onnx_bytes=onnx_path.stat().st_size,
        latency_ms=latency,
        rss_mb=rss,
        parity=parity,
    )


def _load_checkpoint(ckpt: str | Path) -> tuple[DeadbandSSM, dict[str, Any]]:
    ckpt = Path(ckpt)
    meta = json.loads(ckpt.with_suffix(".json").read_text())
    cfg = meta["config"]
    model = DeadbandSSM(
        n_features=4,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        d_state=cfg["d_state"],
        dropout=cfg["dropout"],
        act=meta["activation"],
        mode=cfg["mode"],
        use_conv=cfg["use_conv"],
        use_ssm=cfg["use_ssm"],
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model, meta


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export Deadband-SSM and benchmark deployment.")
    ap.add_argument("--ckpt", default="results/raw/qalu_f0_s0.pt")
    ap.add_argument("--out", default="export")
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--n-features", type=int, default=4)
    args = ap.parse_args(argv)

    model, meta = _load_checkpoint(args.ckpt)
    report = benchmark_artifact(model, args.out, args.seq_len, args.n_features)
    (Path(args.out) / "report.json").write_text(json.dumps({"meta": meta, **report.to_dict()}, indent=2))
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
