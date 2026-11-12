"""Reproducibility guards for anything that trains a model.

Two things made the same ``(activation, fold, seed)`` cell train into different
basins on different runs, with Sharpe anywhere between -1 and -16:

1. ``build_model`` ran before ``torch.manual_seed`` was called, so the weights
   were drawn from whatever RNG state the worker process happened to be in.
   Fixed in ``benchmark.cross_val.run_experiment``.
2. Python randomises string hashing per process, which can change the order of
   dict and set iteration and, through it, the order of float accumulation. That
   is not something a library call can undo once the interpreter is running, so
   the entry points re-exec themselves once with ``PYTHONHASHSEED=0``.

After both, two runs of the same cell are bit-identical.
"""

from __future__ import annotations

import os
import sys

__all__ = ["ensure_reproducible_env"]


def ensure_reproducible_env() -> None:
    """Re-exec the current entry point once with ``PYTHONHASHSEED=0``.

    Costs a fraction of a second and is a no-op when the variable is already
    set, so it is safe to call unconditionally from ``main()``.
    """
    if os.environ.get("PYTHONHASHSEED") == "0":
        return

    env = dict(os.environ, PYTHONHASHSEED="0")
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    module = getattr(spec, "name", None)
    if module:
        argv = [sys.executable, "-m", module, *sys.argv[1:]]
    else:
        argv = [sys.executable, *sys.argv]
    os.execve(sys.executable, argv, env)
