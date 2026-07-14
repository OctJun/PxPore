import numpy as np


def write_cube(path, field, box_nm, grid_space, atoms_pos_nm, atoms_Z=None):
    BOHR_PER_ANG = 1.0 / 0.529177210903
    BOHR_PER_NM  = 10.0 * BOHR_PER_ANG

    field = np.asarray(field, dtype=np.float32)
    Nx, Ny, Nz = field.shape

    atoms_pos_nm = np.asarray(atoms_pos_nm, dtype=float)
    natoms = atoms_pos_nm.shape[0]
    if atoms_Z is None:
        atoms_Z = np.full((natoms,), 6, dtype=int)
    else:
        atoms_Z = np.asarray(atoms_Z, dtype=int)
        if atoms_Z.shape[0] != natoms:
            raise ValueError("atoms_Z length mismatch")

    origin_bohr = np.array([0.0, 0.0, 0.0], dtype=float)
    step_bohr = float(grid_space) * BOHR_PER_NM

    with open(path, "w", buffering=1024 * 1024 * 32) as f:  # 1MB buffer
        f.write("Gaussian cube for VMD\n")
        f.write("Volumetric data\n")
        f.write(f"{natoms:6d} {origin_bohr[0]:12.6f} {origin_bohr[1]:12.6f} {origin_bohr[2]:12.6f}\n")
        f.write(f"{Nx:5d} {step_bohr:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        f.write(f"{Ny:5d} {0.0:12.6f} {step_bohr:12.6f} {0.0:12.6f}\n")
        f.write(f"{Nz:5d} {0.0:12.6f} {0.0:12.6f} {step_bohr:12.6f}\n")

        atoms_pos_bohr = atoms_pos_nm * BOHR_PER_NM
        for Z, (x, y, z) in zip(atoms_Z, atoms_pos_bohr):
            f.write(f"{int(Z):5d} {float(Z):12.6f} {x:12.6f} {y:12.6f} {z:12.6f}\n")

        # 数据部分：每行 6 个数（cube 标准）
        data = field.reshape(Nx * Ny, Nz)  # (x,y) 合并，z 仍是最内层
        np.savetxt(f, data, fmt="%.8f", delimiter=" ", newline="\n")
