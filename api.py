from .config import AnalyseConfig
from .core import analyse as _analyse


def analyse(
    input: str,
    grid: float = 0.01,
    probe: float = 0.0,
    atoms: str | None = None,
    threads: int = 0,
    out_prefix: str | None = None,
    no_surface: bool = False,
    pore: bool = False,
    porevis: bool = False,
    no_octree: bool = False,
    oct_level: int = 4,
    oct_grid: float = 0.001,
    cube: bool = False,
    cube_space: float | None = None,
    smooth: bool = False,
    stats: bool = False,
    debug: bool = False,
    debug_print: bool = False,
):
    cfg = AnalyseConfig(
        input=input,
        grid=grid,
        probe=probe,
        atoms=atoms,
        threads=threads,
        out_prefix=out_prefix,
        no_surface=no_surface,
        pore=pore,
        porevis=porevis,
        no_octree=no_octree,
        oct_level=oct_level,
        oct_grid=oct_grid,
        cube=cube,
        cube_space=cube_space,
        smooth=smooth,
        stats=stats,
        debug=debug,
        debug_print=debug_print,
    )
    return _analyse(cfg)