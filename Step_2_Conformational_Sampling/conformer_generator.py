#!/usr/bin/env python
import scine_utilities as su
import scine_molassembler as masm
import numpy as np
from typing import Optional, Tuple, List, Iterable, Dict, Callable, Union
from sklearn.neighbors import KDTree as skKDTree
import ase
import sys
import random
import glob
from functools import partial

# Max number of conformers
max_n_confs = 250


def cg_ensemble(mol: masm.Molecule, config: masm.dg.Configuration) -> List[np.array]:
    """Tries to generate an ensemble of conformations"""
    TRIAL_ENSEMBLE_SIZE = max_n_confs
    MIN_ENSEMBLE_SIZE = 1
    try:
        maybe_confs = masm.dg.generate_random_ensemble(mol, TRIAL_ENSEMBLE_SIZE, config)
        confs = []

        for conf in maybe_confs:
            if isinstance(conf, masm.dg.Error):
                continue

            error_norms = [
                np.linalg.norm(conf[i] - fixed_pos)
                for i, fixed_pos in config.fixed_positions
            ]

            if any(norm > 0.5 for norm in error_norms):
                continue

            for i, fixed_pos in config.fixed_positions:
                conf[i] = fixed_pos

            confs.append(conf)

        for conf in confs:
            if any(np.linalg.norm(r) > 1e3 for r in conf):
                # masm.io.write("failed-conformer.svg", mol)
                # masm.io.write("failed-conf.mol", mol, conf)
                raise RuntimeError("Bad CG result! See failed-conf* files.")

        if len(confs) < MIN_ENSEMBLE_SIZE:
            # masm.io.write("failed-ensemble.svg", mol)
            raise RuntimeError(
                f"Generated only {len(confs)} confs: see failed-ensemble.svg"
            )

        return confs
    except RuntimeError as e:
        # masm.io.write("problem-child.svg", mol)
        print(f"Encountered exception {e} in CG. See problem-child.svg")
        raise e


def fixed_atoms(mol: masm.Molecule) -> List[int]:
    """Makes a list of the central metal and all its immediate adjacents"""
    CENTRAL_ELEMENT = su.ElementType.Ni
    metal = mol.graph.elements().index(CENTRAL_ELEMENT)
    first_shell = list(mol.graph.adjacents(metal))
    second_shell = []
    for idx in first_shell:
        if mol.graph[idx] == su.ElementType.H or mol.graph[idx] == su.ElementType.O:
            linked = mol.graph.adjacents(idx)
            second_shell.extend(linked)
    shells = first_shell + second_shell
    return list(set(shells + list(range(20,21))))


def radius_adjacency(
    z, coords, radii=ase.data.covalent_radii, mult=1.1, pruning_mult=10
):
    tree = skKDTree(coords)
    am = np.zeros((len(z), len(z)), dtype=int)
    z_radii = np.array(
        [radii[zi] * mult if zi > 1 else radii[zi] for zi in z], dtype=float
    )
    idxs = tree.query_radius(coords, z_radii[z] * pruning_mult)
    nhb = 0
    for i, zi in enumerate(z):
        for j in idxs[i]:
            if j > i:
                if (z[i] == 35 and z[j] == 28) or (z[i] == 28 and z[j] == 35):
                    if np.linalg.norm(coords[i] - coords[j]) < 2.45:
                        am[i, j] = am[j, i] = 1
                        nhb += 1
                if np.linalg.norm(coords[i] - coords[j]) <= mult * (
                    z_radii[i] + z_radii[j]
                ):
                    am[i, j] = am[j, i] = 1
                if (z[i] == 7 and z[j] == 28) or (z[i] == 28 and z[j] == 7):
                    if np.linalg.norm(coords[i] - coords[j]) < 2.3:
                        print("Added Ni-N bond!")
                        am[i, j] = am[j, i] = 1
    return am


def iter_structures() -> Iterable[Tuple[str, su.AtomCollection]]:
    """Iterate through template TS variations"""
    prefix = templatepath
    filenames = glob.glob(prefix)
    for filename in filenames:
        ac, _ = su.io.read(filename)
        yield (filename, ac)


def interpret(ac: su.AtomCollection) -> masm.Molecule:
    """Interprets an atom collection, no index shuffles"""
    z = np.array([element.value for element in ac.elements])
    # ztag = [ element.name() for element in ac.elements ]
    # for i,j in zip(z,ztag):
    #   print(i,j)
    z[np.where(z == 2441)] = 9  # Dont ask me, this must be a bug in scine utilities
    z[np.where(z == 3983)] = 15  # Dont ask me, this must be a bug in scine utilities
    if any(z > 200):
        print("There is a problem with the atomic number extraction sometimes.")
    coords = ac.positions * 0.529177
    bo = su.BondDetector.detect_bonds(ac)
    old_cm = bo.matrix
    cm = radius_adjacency(z, coords)
    bo.matrix = cm  # The bonddetector of scine utilities will not catch hydrogen bonds as we want, so we monkeypatch it!
    print(f"Created {np.count_nonzero(cm - old_cm)/2} new bonds.")
    discretization = masm.interpret.BondDiscretization.Binary
    interpreted = masm.interpret.molecules(ac, bo, discretization)
    for i, mol in enumerate(interpreted.molecules):
        # masm.io.write(f"interpreted_molecule_{i}.svg", mol)
        pass
    if len(interpreted.molecules) != 1:
        raise Exception("Intepreted more than one disjoint molecules.")
    mol = interpreted.molecules[0]
    return mol


# This is 99% of what matters to you as a user
for sl in [1, 2, 3, 4, 5, 6, 7]:
    try:
        # Conformer generation settings
        dg_config = masm.dg.Configuration()
        dg_config.partiality = masm.dg.Partiality.FourAtom
        rsl = 1e7

        if sl < 2:
            thresh = 1e-4
        if sl == 2:
            thresh = 1e-2
        if sl in [3, 4, 5, 6]:
            thresh = 1e-1
        if 6 < sl:
            thresh = 10
            rsl = 1e8

        dg_config.refinement_step_limit = int(rsl)
        dg_config.refinement_gradient_target = thresh  # 1e-4
        dg_config.spatial_model_loosening = sl  # 1

        # Now read input
        filename = sys.argv[1]
        print(
            f"Read filename {filename} with settings {sl} and gradient threshold {thresh}"
        )
        ac, _ = su.io.read(filename)

        # Building the graph from atoms and coordinates
        # Critical point: making sure the graph is connected and nice
        mol = interpret(ac)
        elements = mol.graph.elements()

        # We select parts of the graph to keep fixed
        # Critical point: fixed_atoms is a function that defines the fixed atoms
        graph_fixed = frozenset(fixed_atoms(mol))
        index_map = {i: ac.positions[i, :] for i in mol.graph.atoms()}
        dg_config.fixed_positions = [
            (i, pos) for i, pos in index_map.items() if i in graph_fixed
        ]

        # Generate conformational ensemble (or attempt to) and write it to files
        ensemble = cg_ensemble(mol, dg_config)
        converted = [su.AtomCollection(elements, conf) for conf in ensemble]
        for j, conf in enumerate(converted):
            su.io.write(filename.replace(".xyz", f"-{j}.xyz"), conf)
        break
    except Exception as m:
        print(m)
        continue
