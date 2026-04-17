import os

import numpy as np
from .atoms import guess_element

def read_gro(path: str):
    """
    Returns:
        pos: (N,3) float64 in nm
        elems: list[str] length N, element guessed from atom name
        box: (Lx, Ly, Lz) float64 in nm
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        title = f.readline()
        n = int(f.readline().strip())
        pos = np.zeros((n, 3), dtype=np.float64)
        elems = []
        for i in range(n):
            line = f.readline()
            # gro fixed columns: resid(5) resname(5) atomname(5) atomnr(5) x(8) y(8) z(8)
            atomname = line[10:15].strip()
            # coords are nm
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])
            pos[i] = (x, y, z)
            elems.append(guess_element(atomname))
        box_line = f.readline().split()
        # orthorhombic: 3 numbers
        if len(box_line) < 3:
            raise ValueError("GRO box line does not have 3 numbers; not orthorhombic?")
        Lx, Ly, Lz = map(float, box_line[:3])
    return pos, elems, np.array([Lx, Ly, Lz], dtype=np.float64)

def read_xyz(path: str):
    """
    Reads an XYZ file.

    Returns:
        pos: (N,3) float64 array, coordinates in nm
        elems: list[str] length N, element symbols as read from file
        box: (Lx, Ly, Lz) float64 array, zeros because XYZ has no box info
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # Skip possible leading blank lines
        line = f.readline()
        while line.strip() == "":
            line = f.readline()
        n = int(line.strip())                     # number of atoms
        comment = f.readline()                    # comment line (ignored)
        pos = np.zeros((n, 3), dtype=np.float64)
        elems = []
        for i in range(n):
            line = f.readline()
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Line {i+3} does not have enough columns: {line}")
            elem = parts[0]
            x = float(parts[1])/10
            y = float(parts[2])/10
            z = float(parts[3])/10
            pos[i] = (x, y, z)
            elems.append(elem)
        # XYZ files do not contain box information
        box = np.zeros(3, dtype=np.float64)
    return pos, elems, box


def read_pdb(path: str):
    """
    Reads a PDB file.

    Supported:
        - ATOM / HETATM records
        - CRYST1 record for box
        - orthorhombic box only

    Returns:
        pos: (N,3) float64 array, coordinates in nm
        elems: list[str] length N
        box: (Lx, Ly, Lz) float64 array in nm;
             zeros if CRYST1 is absent
    """
    pos_list = []
    elems = []
    box = np.zeros(3, dtype=np.float64)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = line[:6].strip()

            if rec == "CRYST1":
                # PDB CRYST1:
                # cols  7-15 a
                #      16-24 b
                #      25-33 c
                #      34-40 alpha
                #      41-47 beta
                #      48-54 gamma
                a = float(line[6:15]) / 10.0
                b = float(line[15:24]) / 10.0
                c = float(line[24:33]) / 10.0
                alpha = float(line[33:40])
                beta = float(line[40:47])
                gamma = float(line[47:54])

                # keep same philosophy as GRO reader: only orthorhombic
                if not (
                    abs(alpha - 90.0) < 1e-6 and
                    abs(beta  - 90.0) < 1e-6 and
                    abs(gamma - 90.0) < 1e-6
                ):
                    raise ValueError(
                        f"PDB CRYST1 is not orthorhombic: "
                        f"alpha={alpha}, beta={beta}, gamma={gamma}"
                    )
                box[:] = (a, b, c)

            elif rec == "ATOM" or rec == "HETATM":
                # PDB fixed columns:
                # x: 31-38, y: 39-46, z: 47-54 (1-based)
                # python slices:
                # x: [30:38], y: [38:46], z: [46:54]
                try:
                    x = float(line[30:38]) / 10.0
                    y = float(line[38:46]) / 10.0
                    z = float(line[46:54]) / 10.0
                except ValueError:
                    raise ValueError(f"Failed to parse PDB coordinates: {line.rstrip()}")

                pos_list.append((x, y, z))

                # element symbol usually in cols 77-78 -> [76:78]
                elem = line[76:78].strip() if len(line) >= 78 else ""
                if elem == "":
                    atomname = line[12:16].strip()
                    elem = guess_element(atomname)
                elems.append(elem)

    if len(pos_list) == 0:
        raise ValueError("No ATOM/HETATM records found in PDB file.")

    pos = np.asarray(pos_list, dtype=np.float64)
    return pos, elems, box

def read_cif(path: str):
    """
    Reads a CIF file via ASE, imported lazily for compatibility.

    Returns:
        pos: (N,3) float64 array in nm
        elems: list[str] length N
        box: (Lx, Ly, Lz) float64 array in nm for orthorhombic cells

    Notes:
        - ASE is imported only when this function is called
        - currently returns orthorhombic box only
    """
    try:
        from ase import io as ase_io
    except ImportError as e:
        raise ImportError(
            "Reading CIF requires ASE, but ASE is not installed. "
            "Please install it with `pip install ase`."
        ) from e

    atoms = ase_io.read(path)
    pos = np.asarray(atoms.get_positions(), dtype=np.float64) / 10.0
    elems = atoms.get_chemical_symbols()
    cell = np.asarray(atoms.get_cell(), dtype=np.float64) / 10.0

    # keep current interface style: return only orthorhombic box lengths
    offdiag = cell - np.diag(np.diag(cell))
    if not np.allclose(offdiag, 0.0, atol=1e-6):
        raise ValueError("CIF cell is not orthorhombic; current interface only supports box=(Lx,Ly,Lz).")

    box = np.diag(cell).astype(np.float64)
    return pos, elems, box



def read_structure(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gro":
        return read_gro(path)   # 最好也改成返回 cell(3,3)
    elif ext == ".xyz":
        return read_xyz(path)
    elif ext == ".pdb":
        return read_pdb(path)
    elif ext == ".cif":
        return read_cif(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")