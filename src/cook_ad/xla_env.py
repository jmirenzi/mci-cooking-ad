"""Process-level XLA settings that have to be applied before the GPU backend initialises.

Import and call from a runner's top, next to the existing
`os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")` -- these are the same kind of
thing and have the same constraint: XLA reads them when the backend comes up, so setting them
after a computation has run is silently a no-op.
"""
import os


def disable_gpu_autotuning():
    """Turn off XLA's GPU kernel autotuner, which dominates compile time for this model.

    The E-step is a stack of scans nested under two vmaps, producing large fusions with shapes
    the autotuner has no heuristic for, so it benchmarks kernel variants for minutes. Measured
    at K=128, K_R=16, D=200: compilation did not finish in 4 minutes with autotuning on, and
    took 3.0s with it off. Runtime is unaffected -- this workload is elementwise and reduction
    bound, not matmul bound. Respects an explicitly-set --xla_gpu_autotune_level.
    """
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_autotune_level" not in flags:
        os.environ["XLA_FLAGS"] = (flags + " --xla_gpu_autotune_level=0").strip()
