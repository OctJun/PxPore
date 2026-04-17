from dataclasses import dataclass
from typing import Optional


@dataclass
class AnalyseConfig:
    input: str
    grid: float = 0.01
    probe: float = 0.0
    atoms: Optional[str] = None
    threads: int = 0
    out_prefix: Optional[str] = None

    no_surface: bool = False
    pore: bool = False

    no_octree: bool = False
    oct_level: int = 4
    oct_grid: float = 0.001

    cube: bool = False
    cube_space: Optional[float] = None
    smooth: bool = False
    filter: bool = False
    stats: bool = False

    debug: bool = False
    debug_print: bool = False