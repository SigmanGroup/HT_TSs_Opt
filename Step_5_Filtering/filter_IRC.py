#!/usr/bin/env python3
import ase
import sys
from ase import neighborlist
from ase.io import read
from ase.build import molecule

import numpy as np
import os
import subprocess
from glob import glob


def split_non_consecutive(data):
    data = iter(data)
    val = next(data)
    chunk = []
    try:
        while True:
            chunk.append(val)
            val = next(data)
            if val != chunk[-1] + 1:
                yield chunk
                chunk = []
    except StopIteration:
        if chunk:
            yield chunk


def unit_vector(v):
    return v / np.linalg.norm(v)


def angle_between(u, v):
    v_1 = unit_vector(v)
    u_1 = unit_vector(u)
    return np.arccos(np.clip(np.dot(v_1, u_1), -1.0, 1.0))


def dihedral_angle(p: np.ndarray) -> float:
    """Calculates the dihedral between four points"""
    b0 = -1.0 * (p[1] - p[0])
    b1 = p[2] - p[1]
    b2 = p[3] - p[2]

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


refdirectory = "."
dists = []
structures = []
freqs = []
if __name__ == "__main__":
    filenames = glob(os.path.join(refdirectory, "*.xyz"))

    for i in filenames:
        term = i.strip()[-3:]
        if term == "log":
            term = "gaussian-out"
        print(f"Reading {i} as a {term} file")
        structure = read(i, format=term, index=-1)
        structures.append(structure)

    for i, j in zip(structures, filenames):
        cutoff = ase.neighborlist.natural_cutoffs(i, mult=1)
        neighborList = neighborlist.NeighborList(
            cutoff, self_interaction=False, bothways=True
        )
        neighborList.update(i)
        coords = i.get_positions()
        # Indices to measure are defined here (count starting from 0!)
        d = np.linalg.norm(coords[0, :] - coords[6, :])
        dists.append(np.round(d, 2))

    distances, filenames = zip(*sorted(zip(dists, filenames)))
    for j, a in zip(filenames, distances):
        print(os.path.basename(j), a)
        root = os.path.basename(j).split(".")[0]
        if a < 1.6:
            # os.rename(j, f"Product/{j}
            print("Product!")
            cf = f"{root}.com"
            lf = f"{root}.log"
            xf = f"{root}.xyz"
            fs = [cf, lf, xf]
            for fl in fs:
                os.rename(fl, f"Product/{fl}")
        else:
            print("Reactant!")
            cf = f"{root}.com"
            lf = f"{root}.log"
            xf = f"{root}.xyz"
            fs = [cf, lf, xf]
            for fl in fs:
                os.rename(fl, f"Reactant/{fl}")
