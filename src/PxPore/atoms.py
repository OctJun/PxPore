from pathlib import Path

import numpy as np


def guess_element(atomname: str) -> str:
    """
    Very simple element guess: take leading letters.
    Examples: 'C1'->'C', 'CL'->'Cl', 'NA'->'Na'
    You should adjust if your naming is special.
    """
    s = atomname.strip()
    if not s:
        return "X"
    # take first 1-2 letters
    a = s[0].upper()
    b = s[1].lower() if len(s) > 1 and s[1].isalpha() else ""
    # common two-letter elements
    two = a + b
    if two in {"Cl", "Br", "Na", "Li", "Al", "Si", "Ca", "Fe", "Zn", "Mg", "Cu", "Mn", "Co", "Ni"}:
        return two
    return a


def load_atom_info(filepath, overwrite=True):
    """
    从外部文件加载原子信息，并直接更新全局默认表。

    文件格式（空白分隔，允许#注释）:
        symbol   Z   mass   radius_ang

    例如:
        H   1   1.008   1.20
        C   6   12.011  1.70
        Cl  17  35.45   1.75

    Parameters
    ----------
    filepath : str
    overwrite : bool
        True  -> 外部表覆盖已有表项
        False -> 仅补充不存在的表项
    verbose : bool
        是否打印更新信息
    """
    n_update = 0

    path = Path(filepath)
    if not path.exists():
        path = Path(__file__).resolve().parent / "data" / filepath

    with open(path, "r", encoding="utf-8") as f:

        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) < 4:
                raise ValueError(
                    f"[ATOMS] line {lineno}: 需要4列 "
                    f"(symbol Z mass radius_ang)，实际内容: {line.rstrip()}"
                )

            symbol = parts[0]
            try:
                Z = int(parts[1])
                mass = float(parts[2])
                sigma_nm = float(parts[3])
            except Exception as e:
                raise ValueError(
                    f"[ATOMS] line {lineno}: 解析失败 -> {line.rstrip()}"
                ) from e

            if overwrite or (symbol not in SYMBOL_TO_Z and Z not in Z_TO_SYMBOL):
                SYMBOL_TO_Z[symbol] = Z
                Z_TO_SYMBOL[Z] = symbol
                Z_TO_MASS[Z] = mass
                Z_TO_RADIUS_ANG[Z] = sigma_nm * 10 * 0.5  # nm -> Å -> radius
                n_update += 1
            else:
                # 不覆盖时，尽量补缺
                if symbol not in SYMBOL_TO_Z:
                    SYMBOL_TO_Z[symbol] = Z
                if Z not in Z_TO_SYMBOL:
                    Z_TO_SYMBOL[Z] = symbol
                if Z not in Z_TO_MASS:
                    Z_TO_MASS[Z] = mass
                if Z not in Z_TO_RADIUS_ANG:
                    Z_TO_RADIUS_ANG[Z] = sigma_nm * 10 * 0.5  # nm -> Å -> radius
                n_update += 1
    return n_update



def symbols_to_Z(symbol_list, strict=True, default_symbol='C'):
    """
    元素符号 -> 原子序数 Z
    """
    out = []
    if default_symbol not in SYMBOL_TO_Z:
        raise ValueError(f"default_symbol {default_symbol} not found in SYMBOL_TO_Z")
    default_Z = SYMBOL_TO_Z[default_symbol]

    for s in symbol_list:
        if s in SYMBOL_TO_Z:
            out.append(SYMBOL_TO_Z[s])
        else:
            if strict:
                raise ValueError(f"Unknown element symbol: {s}")
            out.append(default_Z)

    return out


def elems_to_Z(elems, strict=True, default_symbol='C'):
    """
    更通用:
    elems 既可以是元素符号列表，也可以是原子序数列表/数组

    支持:
      ['C','H','O']
      [6,1,8]
      np.array([6,1,8])
      混合列表 ['C', 1, 'O'] 也能处理
    """
    out = []
    if default_symbol not in SYMBOL_TO_Z:
        raise ValueError(f"default_symbol {default_symbol} not found in SYMBOL_TO_Z")
    default_Z = SYMBOL_TO_Z[default_symbol]

    for x in elems:
        if isinstance(x, str):
            if x in SYMBOL_TO_Z:
                out.append(SYMBOL_TO_Z[x])
            else:
                if strict:
                    raise ValueError(f"Unknown element symbol: {x}")
                out.append(default_Z)
        else:
            z = int(x)
            if z in Z_TO_SYMBOL or z in Z_TO_MASS or z in Z_TO_RADIUS_ANG:
                out.append(z)
            else:
                if strict:
                    raise ValueError(f"Unknown atomic number: {z}")
                out.append(default_Z)

    return np.asarray(out, dtype=np.int32)


def build_radii_nm(elems, strict=False, default_symbol='C'):
    """
    根据元素符号或原子序数返回 vdW 半径（nm）
    内部统一先转为 Z，再查表
    """
    Z_arr = elems_to_Z(elems, strict=strict, default_symbol=default_symbol)
    out = np.zeros(len(Z_arr), dtype=np.float64)

    default_Z = SYMBOL_TO_Z[default_symbol]
    if default_Z not in Z_TO_RADIUS_ANG:
        raise ValueError(f"default Z={default_Z} has no radius data")

    default_ang = Z_TO_RADIUS_ANG[default_Z]

    for i, z in enumerate(Z_arr):
        ang = Z_TO_RADIUS_ANG.get(int(z), default_ang)
        out[i] = ang * 0.1  # Å -> nm

    return out


def build_mass(elems, strict=False, default_symbol='C'):
    """
    根据元素符号或原子序数返回原子质量
    内部统一先转为 Z，再查表
    """
    Z_arr = elems_to_Z(elems, strict=strict, default_symbol=default_symbol)
    out = np.zeros(len(Z_arr), dtype=np.float64)

    default_Z = SYMBOL_TO_Z[default_symbol]
    if default_Z not in Z_TO_MASS:
        raise ValueError(f"default Z={default_Z} has no mass data")

    default_mass = Z_TO_MASS[default_Z]

    for i, z in enumerate(Z_arr):
        out[i] = Z_TO_MASS.get(int(z), default_mass)

    return out


#Bondi
VDW_ANG = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    "Si": 2.10, "B": 1.92,
    "Na": 2.27, "Li": 1.82, "K": 2.75, "Ca": 2.31, "Mg": 1.73,
    "Fe": 2.00, "Zn": 2.10, "Cu": 2.00, "Mn": 2.00, "Co": 2.00, "Ni": 2.00,
    "Al": 2.00,
}

ATOM_MASS = {
    'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81,
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974,
    'S': 32.06, 'Cl': 35.45, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
    'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
    'Ga': 69.723, 'Ge': 72.63, 'As': 74.922, 'Se': 78.96, 'Br': 79.904,
    'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224,
    'Nb': 92.906, 'Mo': 95.96, 'Tc': 98, 'Ru': 101.07, 'Rh': 102.91,
    'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41, 'In': 114.82, 'Sn': 118.71,
    'Sb': 121.76, 'Te': 127.60, 'I': 126.90, 'Xe': 131.29, 'Cs': 132.91,
    'Ba': 137.33, 'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24,
    'Pm': 145, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93,
    'Dy': 162.50, 'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.04,
    'Lu': 174.97, 'Hf': 178.49, 'Ta': 180.95, 'W': 183.84, 'Re': 186.21,
    'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59,
    'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.98, 'Th': 232.04, 'Pa': 231.04,
    'U': 238.03
}

PERIODIC_TABLE = {
    # 1–10
    "Hydrogen": 1, "H": 1,
    "Helium": 2, "He": 2,
    "Lithium": 3, "Li": 3,
    "Beryllium": 4, "Be": 4,
    "Boron": 5, "B": 5,
    "Carbon": 6, "C": 6,
    "Nitrogen": 7, "N": 7,
    "Oxygen": 8, "O": 8,
    "Fluorine": 9, "F": 9,
    "Neon": 10, "Ne": 10,

    # 11–20
    "Sodium": 11, "Na": 11,
    "Magnesium": 12, "Mg": 12,
    "Aluminum": 13, "Al": 13,
    "Silicon": 14, "Si": 14,
    "Phosphorus": 15, "P": 15,
    "Sulfur": 16, "S": 16,
    "Chlorine": 17, "Cl": 17,
    "Argon": 18, "Ar": 18,
    "Potassium": 19, "K": 19,
    "Calcium": 20, "Ca": 20,

    # 21–30
    "Scandium": 21, "Sc": 21,
    "Titanium": 22, "Ti": 22,
    "Vanadium": 23, "V": 23,
    "Chromium": 24, "Cr": 24,
    "Manganese": 25, "Mn": 25,
    "Iron": 26, "Fe": 26,
    "Cobalt": 27, "Co": 27,
    "Nickel": 28, "Ni": 28,
    "Copper": 29, "Cu": 29,
    "Zinc": 30, "Zn": 30,

    # 31–40
    "Gallium": 31, "Ga": 31,
    "Germanium": 32, "Ge": 32,
    "Arsenic": 33, "As": 33,
    "Selenium": 34, "Se": 34,
    "Bromine": 35, "Br": 35,
    "Krypton": 36, "Kr": 36,
    "Rubidium": 37, "Rb": 37,
    "Strontium": 38, "Sr": 38,
    "Yttrium": 39, "Y": 39,
    "Zirconium": 40, "Zr": 40,

    # 41–54（常用到 Mo, Ag, I, Xe）
    "Niobium": 41, "Nb": 41,
    "Molybdenum": 42, "Mo": 42,
    "Technetium": 43, "Tc": 43,
    "Ruthenium": 44, "Ru": 44,
    "Rhodium": 45, "Rh": 45,
    "Palladium": 46, "Pd": 46,
    "Silver": 47, "Ag": 47,
    "Cadmium": 48, "Cd": 48,
    "Indium": 49, "In": 49,
    "Tin": 50, "Sn": 50,
    "Antimony": 51, "Sb": 51,
    "Tellurium": 52, "Te": 52,
    "Iodine": 53, "I": 53,
    "Xenon": 54, "Xe": 54,

    # 常见金属补充
    "Cesium": 55, "Cs": 55,
    "Barium": 56, "Ba": 56,
    "Tungsten": 74, "W": 74,
    "Platinum": 78, "Pt": 78,
    "Gold": 79, "Au": 79,
    "Mercury": 80, "Hg": 80,
    "Lead": 82, "Pb": 82,
}


SYMBOL_TO_Z = dict(PERIODIC_TABLE)  # symbol -> Z
Z_TO_SYMBOL = {z: s for s, z in PERIODIC_TABLE.items()}  # Z -> symbol
Z_TO_MASS = {PERIODIC_TABLE[s]: m for s, m in ATOM_MASS.items() if s in PERIODIC_TABLE}
Z_TO_RADIUS_ANG = {PERIODIC_TABLE[s]: r for s, r in VDW_ANG.items() if s in PERIODIC_TABLE}
