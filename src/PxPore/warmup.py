import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
import platform

from . import APP_NAME, __version__


def run_warmup():
    env = os.environ.copy()
    env["PXPORE_IN_WARMUP"] = "1"
    env.setdefault("NUMBA_THREADING_LAYER", "omp")
    example_structure = files("PxPore").joinpath("data", "single_H.gro")
    common = [
        sys.executable, "-m", "PxPore.cli", str(example_structure),
        "--g", "0.05",
        "--probe", "0.01",
        "--threads", "1",
        "--surface-samples", "64",
        "--oct-level", "2",
        "--oct-grid", "0.01",
    ]
    variants = [
        ["--pore", "--psd-method", "both", "--psd-mc-samples", "256"],
        ["--pore", "--psd-method", "both", "--psd-mc-samples", "256",
         "--connectivity", "periodic", "--transport-direction", "x"],
        ["--no-octree"],
        ["--no-octree", "--transport-direction", "x"],
        ["--no-octree", "--connectivity", "periodic",
         "--transport-direction", "x"],
    ]
    for args in variants:
        cmd = common + args
        print("[PxPore] First-time warmup: running", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)


def ensure_warmup(force=False):
    if os.environ.get("PXPORE_IN_WARMUP") == "1":
        return

    marker = get_marker_file()
    expected = f"warmup_version={__version__}\npython={platform.python_version()}\n"

    if marker.exists() and not force:
        try:
            if marker.read_text(encoding="utf-8") == expected:
                return
        except Exception:
            pass

    run_warmup()

    marker.write_text(expected, encoding="utf-8")


def get_marker_file() -> Path:
    return get_cache_dir() / "warmup_done.txt"


def get_cache_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA",
                    Path.home() / "AppData/Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = base / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    ensure_warmup(force=True)
