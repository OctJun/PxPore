import os


def configure_test_threads():
    """在导入 PxPore/Numba 前配置全部可用 CPU 核心。"""
    if hasattr(os, "sched_getaffinity"):
        threads = max(1, len(os.sched_getaffinity(0)))
    else:
        threads = max(1, os.cpu_count() or 1)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    os.environ["NUMBA_THREADING_LAYER"] = "omp"
    return threads
