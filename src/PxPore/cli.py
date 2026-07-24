import argparse
import logging
import sys
import traceback

from .api import analyse as api_analyse
from .config import AnalyseConfig
from .warmup import ensure_warmup

logger = logging.getLogger('PxPore')


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m PxPore",
        description="PxPore pore analysis toolkit",
    )

    ap.add_argument(
        "input", help="input .gro/.xyz/.pdb/.cif orthogonal system required")
    ap.add_argument("--grid", "-g", type=float, default=0.01,
                    help="target grid spacing in nm")
    ap.add_argument("--probe", "-p", type=float,
                    default=0.0, help="probe radius in nm")
    ap.add_argument("--connectivity", choices=("legacy", "periodic"),
                    default="legacy",
                    help="connectivity boundary mode")
    ap.add_argument("--transport-direction", choices=("any", "x", "y", "z"),
                    default="any",
                    help="transport direction used for accessibility")
    ap.add_argument("--atoms", type=str, default=None,
                    help="atom info file to load, overrides default tables. Format: symbol Z mass(g/mol) LJsigma(nm) epsilon(K)")
    ap.add_argument("--threads", type=int, default=0,
                    help="numba threads, 0=half of available threads")
    ap.add_argument("--out_prefix", type=str, default=None)

    ap.add_argument("--no-surface", action="store_true",
                    default=False, help="disable surface area analysis")
    ap.add_argument("--surface-samples", type=int, default=1000,
                    help="Fibonacci surface samples per atom")
    ap.add_argument("--pore", action="store_true",
                    default=False, help="enable pore analysis")
    ap.add_argument("--porevis", action="store_true", default=False,
                    help="enableing output visulization of pore")
    ap.add_argument("--psd-method", choices=("centers", "mc"),
                    default="centers", help="pore-size distribution method")
    ap.add_argument("--psd-mc-samples", type=int, default=50000,
                    help="number of samples for Monte Carlo PSD")
    ap.add_argument("--psd-mc-seed", type=int, default=11451466,
                    help="random seed for Monte Carlo PSD")
    ap.add_argument("--psd-mc-bin-size", type=float, default=None,
                    help="Monte Carlo PSD bin size in nm; default uses grid")

    ap.add_argument("--no-octree", action="store_true", default=False)
    ap.add_argument("--oct-level", type=int, default=4,
                    help="max octree levels")
    ap.add_argument("--oct-grid", type=float, default=0.001,
                    help="minimum octree leaf size in nm")

    ap.add_argument("--cube", action="store_true", default=False)
    ap.add_argument("--cube-space", type=float, default=None)
    ap.add_argument("--smooth", action="store_true", default=False)
    ap.add_argument("--stats", action="store_true", default=False)

    ap.add_argument("--debug", action="store_true",
                    default=False, help="save intermediate arrays")
    ap.add_argument("--debug-print", action="store_true",
                    default=False, help="print extra debug info")
    return ap


def namespace_to_config(args) -> AnalyseConfig:
    return AnalyseConfig(
        input=args.input,
        grid=args.grid,
        probe=args.probe,
        connectivity=args.connectivity,
        transport_direction=args.transport_direction,
        atoms=args.atoms,
        threads=args.threads,
        out_prefix=args.out_prefix,
        no_surface=args.no_surface,
        surface_samples=args.surface_samples,
        pore=args.pore,
        porevis=args.porevis,
        psd_method=args.psd_method,
        psd_mc_samples=args.psd_mc_samples,
        psd_mc_seed=args.psd_mc_seed,
        psd_mc_bin_size=args.psd_mc_bin_size,
        no_octree=args.no_octree,
        oct_level=args.oct_level,
        oct_grid=args.oct_grid,
        cube=args.cube,
        cube_space=args.cube_space,
        smooth=args.smooth,
        stats=args.stats,
        debug=args.debug,
        debug_print=args.debug_print,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = namespace_to_config(args)
    return api_analyse(**cfg.__dict__)


def run():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        ensure_warmup()
        main()
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last = tb[-1]
            logger.error(
                f"[ERROR] {type(e).__name__}: {e} "
                f"(file {last.filename}, line {last.lineno}, in {last.name})"
            )
        else:
            logger.error(f"[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
