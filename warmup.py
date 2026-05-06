import os
import subprocess
import sys
from pathlib import Path
import platform
import shutil

from . import APP_NAME, __version__


def run_warmup():
    cache_dir = Path(__file__).resolve().parent / "__pycache__"
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
        except Exception as exc:
            print(
                f"[PxPore] Warmup cache cleanup skipped: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
    env = os.environ.copy()
    env["PXPORE_IN_WARMUP"] = "1"
    env.setdefault("NUMBA_THREADING_LAYER", "omp")
    cmd = [sys.executable, "-m",
           "PxPore.cli",
           str(Path(__file__).resolve().parent / "sharing" / "single_H.gro"),
           "--pore",
           "--g", "0.05",
           "--oct-level", "2",
           "--probe", "0.01"]
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
