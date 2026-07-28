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
    porevis: bool = False
    
    no_octree: bool = False
    oct_level: int = 2
    oct_grid: float = 0.001

    cube: bool = False
    cube_space: Optional[float] = None
    smooth: bool = False

    stats: bool = False

    debug: bool = False
    debug_print: bool = False

    connectivity: str = "legacy"
    transport_direction: str = "any"

    psd_method: str = "centers"
    psd_mc_samples: int = 50000
    psd_mc_seed: int = 11451466
    psd_mc_bin_size: Optional[float] = None

    surface_samples: int = 1000

    psd_local_max_mode: str = "strict"
    psd_min_center_radius: float = 0.005
    psd_overlap_prune: bool = True
    psd_overlap_threshold: float = 1.0
    psd_hist_weighting: str = "volume"
