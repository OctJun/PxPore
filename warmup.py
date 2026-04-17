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
        shutil.rmtree(cache_dir)
    env = os.environ.copy()
    env["PXPORE_IN_WARMUP"] = "1"
    cmd = [sys.executable, "-m",
           "PxPore.cli",
           str(Path(__file__).resolve().parent / "sharing" / "single_H.gro"),
           "--pore",
           "--g","0.05",
           "--oct-level","2",
           "--probe","0.01"]
    print("[PxPore] First-time warmup: running", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL,)


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
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA",
                    Path.home() / "AppData/Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = base / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d / "warmup_done.txt"


if __name__ == "__main__":
    ensure_warmup(force=True)
