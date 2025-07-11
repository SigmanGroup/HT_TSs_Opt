#!/usr/bin/env python3
import ase
import sys
from ase.io import read
import numpy as np
import os
import shutil
from glob import glob


def angle_between_4_1(p):
    """Calculate the dihedral angle based on vector operations."""
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b1, b2)

    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    y = np.dot(b0xb1_x_b1xb2, b1) * (1.0 / np.linalg.norm(b1))
    x = np.dot(b0xb1, b1xb2)

    return np.degrees(np.arctan2(y, x))


if __name__ == "__main__":
    # Adjust indices to 0-based by subtracting 1 from the command line arguments
    idx1, idx2, idx3, idx4 = int(sys.argv[1]) - 1, int(sys.argv[2]) - 1, int(sys.argv[3]) - 1, int(sys.argv[4]) - 1
    filenames = glob(os.path.join(".", "*.xyz"))

    structures = [read(i, format=i.strip()[-3:]) for i in filenames]

    os.makedirs("below_90", exist_ok=True)
    os.makedirs("above_90", exist_ok=True)

    for structure, filename in zip(structures, filenames):
        positions = structure.get_positions()
        p = [positions[idx1], positions[idx2], positions[idx3], positions[idx4]]
        for idx in range(1, len(p)):
            p[idx] -= p[0]  # Adjusting positions relative to the first point

        angle = angle_between_4_1(p)
        folder = "above_90" if abs(angle) > 90 else "below_90"
        shutil.copy(filename, os.path.join(folder, os.path.basename(filename)))
        print(f"{filename} has dihedral {angle} classified into {folder}")


