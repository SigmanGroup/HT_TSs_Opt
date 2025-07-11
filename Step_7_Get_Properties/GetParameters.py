#!/usr/bin/env python3
#Molecular parameters with and without Morfeus

import os
from morfeus import read_xyz, BuriedVolume, Dispersion, SASA, Sterimol, Pyramidalization
from morfeus import XTB
import sys         
import re
import numpy as np
import pandas as pd
import shutil
from glob import glob

float_or_int_regex = "[-+]?[0-9]*\.[0-9]+|[0-9]+"

path = os.getcwd()

#list with log files
filenames_log = []
for file in os.listdir(path):
    if file.endswith(".log"):
        if "freq" not in file:
            filenames_log.append(file)

template = "L0000"
name_output = template + "_Morfeus_Results.csv"

#################################################
#TEMPLATE-DEPENDENT INPUT!
#Replace atom label with numerical indices (use Change_labels.py and Atom_indices.csv) 
Vbur_a = [Ni,Br,C1s,C2s,N1,N2]
Hirshfeld_a = [N1,N2,C2,R1,X1,C1,C4,X2,C5,R2,Ni,C1s,C2s,H1s,Br]
CM5_a = [N1,N2,C2,R1,X1,C1,C4,X2,C5,R2,Ni,C1s,C2s,H1s,Br]
SpinD_a = [N1,N2,C2,R1,X1,C1,C4,X2,C5,R2,Ni,C1s,C2s,H1s,Br]
Pyr_a = [N1,N2,C1,C4,C1s,C2s]
Disp_a = [N1,C2,N2,C5,Ni,C1s,C2s,Br]
SASA_a = [N1,N2,Ni,Br,H1s]
Sterimol_a1 = [C2,C5,N1,N2,Ni,Ni]
Sterimol_a2 = [R1,R2,Ni,Ni,C1s,C2s]
Fukui_a = [N1,N2,C2,R1,X1,C1,C4,X2,C5,R2,Ni,C1s,C2s,H1s,Br]
Bond_a1 = [N1,N2,Ni,Ni,Br,C1s]
Bond_a2 = [Ni,Ni,C1s,C2s,Ni,C2s]
Angle_a1 = [N1,N1,N1,N2,N2,C1s,C1s,C2s,N1,N2]
Angle_a2 = [Ni,Ni,Ni,Ni,Ni,Ni,Ni,Ni,Ni,Ni]
Angle_a3 = [N2,C1s,C2s,C1s,C2s,C2s,Br,Br,Br,Br]
Dihedral_a1 = [N1,N1,N2,C1,C4]
Dihedral_a2 = [C1,Ni,Ni,N1,N2]
Dihedral_a3 = [C4,C2s,C2s,Ni,Ni]
Dihedral_a4 = [N2,H1s,H1s,C1s,C1s]
#################################################
Hirshfeld_names = []

for i in range(len(Hirshfeld_a)):
    name = "Hirshfeld_" + str(Hirshfeld_a[i])
    Hirshfeld_names.append(name)

CM5_names = []

for i in range(len(CM5_a)):
    name = "CM5_" + str(CM5_a[i])
    CM5_names.append(name)

SpinD_names = []

for i in range(len(SpinD_a)):
    name = "Spin_Density_" + str(SpinD_a[i])
    SpinD_names.append(name)

Vbur_names = []

for i in range(len(Vbur_a)):
    name_Vbur = "Vbur_" + str(Vbur_a[i])
    Vbur_names.append(name_Vbur)

Sterimol_names = []

for i in range(len(Sterimol_a1)):
    name_L = "L_" + str(Sterimol_a1[i]) + "_" + str(Sterimol_a2[i])
    name_B1 = "B1_" + str(Sterimol_a1[i]) + "_" + str(Sterimol_a2[i])
    name_B5 = "B5_" + str(Sterimol_a1[i]) + "_" + str(Sterimol_a2[i])
    Sterimol_names.append(name_L)
    Sterimol_names.append(name_B1)
    Sterimol_names.append(name_B5)

Disp_names = []

for i in range(len(Disp_a)):
    name_Disp = "P_int_" + str(Disp_a[i])
    Disp_names.append(name_Disp)

SASA_names = []

for i in range(len(SASA_a)):
    name_SASA = "SASA_" + str(SASA_a[i])
    SASA_names.append(name_SASA)

Pyr_names = []

for i in range(len(Pyr_a)):
    name_Pyr = "P_" + str(Pyr_a[i])
    name_Pyr_angle = "P_angle_" + str(Pyr_a[i])
    Pyr_names.append(name_Pyr)
    Pyr_names.append(name_Pyr_angle)

xtb_names = []

for i in range(len(Fukui_a)):
    name_f_elec = "f_elec_" + str(Fukui_a[i])
    name_f_nuc = "f_nuc_" + str(Fukui_a[i])
    name_local_elec = "local_elec_" + str(Fukui_a[i])
    name_local_nuc = "local_nuc_" + str(Fukui_a[i])
    xtb_names.append(name_f_elec)
    xtb_names.append(name_f_nuc)
    xtb_names.append(name_local_elec)
    xtb_names.append(name_local_nuc)

Bond_names = []

for i in range(len(Bond_a1)):
    name = "d_" + str(Bond_a1[i]) + "_" + str(Bond_a2[i])
    Bond_names.append(name)

Angle_names = []

for i in range(len(Angle_a1)):
    name_angle = "theta_" + str(Angle_a1[i]) + "_" + str(Angle_a2[i]) + "_" + str(Angle_a3[i])
    Angle_names.append(name_angle)

Dihedral_names = []

for i in range(len(Dihedral_a1)):
    name_dihedral = "phi_" + str(Dihedral_a1[i]) + "_" + str(Dihedral_a2[i]) + "_" + str(Dihedral_a3[i]) + "_" + str(Dihedral_a4[i])
    Dihedral_names.append(name_dihedral)

with open(name_output,'w') as f:
    f.write("Filename,"+"HOMO_au,"+"LUMO_au,"+"Dipole,"+"Surface_area,"+"Surface_volume,"+"P_int,"+"SASA,"+"V_SASA,")
    print(','.join(Hirshfeld_names), file=f, end=',')
    print(','.join(CM5_names), file=f, end=',')
    print(','.join(SpinD_names), file=f, end=',')
    print(','.join(Vbur_names), file=f, end=',')
    print(','.join(Disp_names), file=f, end=',')
    print(','.join(SASA_names), file=f, end=',')
    print(','.join(Pyr_names), file=f, end=',')
    print(','.join(xtb_names), file=f, end=',')
    print(','.join(Bond_names), file=f, end=',')
    print(','.join(Angle_names), file=f, end=',')
    print(','.join(Dihedral_names), file=f, end=',')
    print(','.join(Sterimol_names), file=f)

for i in range(len(filenames_log)):
    filename = filenames_log[i]
    filename_xyz = filenames_log[i].replace(".log",".xyz")
    elements, coordinates = read_xyz(filename_xyz)
    #print(elements,len(elements))

#################################################
    # %Vbur with 3.5 Å radius
   
    Vbur_values = []

    for i in range(len(Vbur_a)):

        bv_atoms = BuriedVolume(elements, coordinates, Vbur_a[i], radius = 3.5)
        bv_atoms_fraction = bv_atoms.fraction_buried_volume
        #print(bv_atoms_fraction)
        Vbur_values.append(str(bv_atoms_fraction))

#################################################
    # Dispersion descriptor
    Disp_values = []
    disp = Dispersion(elements, coordinates)
    disp_area = disp.area
    disp_volume = disp.volume
    disp_P_int = disp.p_int
    
    for i in range(len(Disp_a)):

        disp_atoms = disp.atom_p_int[Disp_a[i]]
        Disp_values.append(str(disp_atoms))

#################################################
    # Solvent Accessible Surface Area
    SASA_values = []
    sasa = SASA(elements, coordinates)
    sasa_area = sasa.area
    sasa_volume = sasa.volume
    
    for i in range(len(SASA_a)):

        sasa_atoms = sasa.atom_areas[SASA_a[i]]
        SASA_values.append(str(sasa_atoms))

################################################
    # Sterimol values
    Sterimol_values = []
    for i in range(len(Sterimol_a1)):
    
        sterimol = Sterimol(elements, coordinates, Sterimol_a1[i], Sterimol_a2[i] )
        Sterimol_values.append(str(sterimol.L_value))
        Sterimol_values.append(str(sterimol.B_1_value))
        Sterimol_values.append(str(sterimol.B_5_value))
    
    #print(Sterimol_names)
    #print(Sterimol_values)
    
#################################################
    # Pyramidalization
    Pyr_values = []
    for i in range(len(Pyr_a)):

        pyr = Pyramidalization(coordinates, Pyr_a[i])
        #print(pyr)
        Pyr_values.append(str(pyr.P))
        Pyr_values.append(str(pyr.P_angle))

    #print(Pyr_values)

#################################################
    # Local Conceptual DFT descriptors (from xTB)
    xtb_values = []
    xtb = XTB(elements, coordinates)
    for i in range(len(Fukui_a)):    

        f_elec = xtb.get_fukui("electrophilicity")[Fukui_a[i]]
        f_nuc = xtb.get_fukui("nucleophilicity")[Fukui_a[i]]
        local_elec = xtb.get_fukui("local_electrophilicity")[Fukui_a[i]]
        local_nuc = xtb.get_fukui("local_nucleophilicity")[Fukui_a[i]]
        xtb_values.append(str(f_elec))
        xtb_values.append(str(f_nuc))
        xtb_values.append(str(local_elec))
        xtb_values.append(str(local_nuc))

#################################################
    # Bond lengths
    distance_values = []
    for i in range(len(Bond_a1)):
        p1 = np.array(coordinates[Bond_a1[i] - 1])
        p2 = np.array(coordinates[Bond_a2[i] - 1])
        squared_dist = np.sum((p1-p2)**2, axis=0)
        dist1 = np.sqrt(squared_dist)
        # print(dist1)
        distance_values.append(str(dist1))

#################################################
    # Bond angles
    angle_values = []
    for i in range(len(Angle_a1)):
        a1 = np.array(coordinates[Angle_a1[i] - 1])
        a2 = np.array(coordinates[Angle_a2[i] - 1])
        a3 = np.array(coordinates[Angle_a3[i] - 1])
        v1 = a1 - a2
        v2 = a3 - a2
        dot_product = np.dot(v1, v2)
        mag_v1 = np.linalg.norm(v1)
        mag_v2 = np.linalg.norm(v2)
        cos_angle = dot_product / (mag_v1 * mag_v2)
        cos_angle = min(max(cos_angle, -1), 1)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        #print(angle_deg)
        angle_values.append(str(angle_deg))

#################################################
    # Dihedral angles
    dihedral_values = []
    for i in range(len(Dihedral_a1)):
        a1 = np.array(coordinates[Dihedral_a1[i] - 1])
        a2 = np.array(coordinates[Dihedral_a2[i] - 1])
        a3 = np.array(coordinates[Dihedral_a3[i] - 1])
        a4 = np.array(coordinates[Dihedral_a4[i] - 1])
        v1 = a2 - a1
        v2 = a2 - a3
        v3 = a3 - a4
        v4 = np.cross(v1, v2)
        v5 = np.cross(v3, v2)
        v6 = np.cross(v1, v3)
        cost = (v4@v5) / np.sqrt(v4@v4 * v5@v5)
        t = - np.arccos(cost) * 180.0/np.pi * np.sign(v2@v6)
        # print(t)
        dihedral_values.append(str(t))

#################################################
    # Hirshfeld charges
    with open(filename,'r') as f:
        log = f.read()
        f.close()

    Hirshfeld_values = []
    SpinD_values = []
    CM5_values = []
    
    pattern = r"Hirshfeld charges,\s*spin densities,\s*dipoles,\s*and CM5 charges using IRadAn\s*=\s*\d+:\n\s*Q-H\s+S-H\s+Dx\s+Dy\s+Dz\s+Q-CM5\s*\n((?:\s*\d+\s+\w+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s*\n)+)"
    match = re.search(pattern, log, re.DOTALL)
    if match:
        data_string = match.group(1)
        #print("Captured Data:")
        #print(data_string)
        data_pattern = r'(\d+)\s+([A-Za-z]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)'
        population = re.findall(data_pattern, data_string)
        npa = pd.DataFrame(population, columns=['Index', 'Element', 'Q-H', 'S-H', 'Dx', 'Dy', 'Dz', 'Q-CM5'])
        #print(npa)
    else:
        print("No match found")

    #Processing Hirshfeld Indices
    for i in range(len(Hirshfeld_a)):
        idx = Hirshfeld_a[i] - 1
        charge = npa['Q-H'].iloc[idx]
        cm5 = npa['Q-CM5'].iloc[idx]
        spin = npa['S-H'].iloc[idx]
        Hirshfeld_values.append(float(charge))
        CM5_values.append(float(cm5))
        SpinD_values.append(float(spin))

#################################################
    # SOMO and LUMO energies
    # Define the regular expression pattern
    pattern = r"Population analysis using the SCF Density\..*?Alpha\s*occ\.\s*eigenvalues(.*?)Alpha\s*virt\.\s*eigenvalues(.*?)Beta\s*occ\.\s*eigenvalues(.*?)Beta\s*virt\.\s*eigenvalues(.*?)Condensed to atoms"

    # Search for the pattern in the log file
    match = re.search(pattern, log, re.DOTALL)

    if match:
        # Extract sections for alpha and beta orbitals
        alpha_occupied = match.group(1)
        alpha_virtual = match.group(2)
        beta_occupied = match.group(3)
        beta_virtual = match.group(4)

        # Extract energies for alpha and beta orbitals
        alpha_occ_energies = re.findall(r"([-+]?\d*\.\d+|\d+)", alpha_occupied)
        alpha_virt_energies = re.findall(r"([-+]?\d*\.\d+|\d+)", alpha_virtual)
        beta_occ_energies = re.findall(r"([-+]?\d*\.\d+|\d+)", beta_occupied)
        beta_virt_energies = re.findall(r"([-+]?\d*\.\d+|\d+)", beta_virtual)

        # Calculate SOMO and LUMO for alpha and beta orbitals
        somo_alpha = max(map(float, alpha_occ_energies))
        lumo_alpha = min(map(float, alpha_virt_energies))
        somo_beta = max(map(float, beta_occ_energies))
        lumo_beta = min(map(float, beta_virt_energies))

        # Calculate HOMO and LUMO
        homo = somo_alpha
        lumo = lumo_alpha

        # Calculate electronegativity and hardness
        electronegativity = -0.5 * (lumo + homo)
        hardness = 0.5 * (lumo - homo)

        #print(f"HOMO: {homo}, LUMO: {lumo}")
    else:
        raise ValueError("Error: HOMO LUMO pattern not found in log file.")

#################################################
    # Dipole moment
    try:
        string_dipole = re.search(r"(?<=Dipole moment \(field-independent basis, Debye\):).*?(?=.Quadrupole)",log, re.DOTALL).group(0)
        lst_dipole =  re.findall(r"[-+]?(?:\d*\.*\d+)", string_dipole)
        #print(lst_dipole[-1])
        dipole = lst_dipole[-1]
    except Exception:
        print("Error dipole")
        dipole = np.nan

#################################################
    with open(name_output,'a') as f:
        f.write(filename_xyz.replace(".xyz","") + "," + 
        str(homo) + "," + 
        str(lumo) + "," + 
        str(dipole) + "," + 
        str(disp_area) + "," + 
        str(disp_volume) + "," + 
        str(disp_P_int) + "," + 
        str(sasa_area) + "," + 
        str(sasa_volume) + "," + 
        ",".join(map(str, Hirshfeld_values)) + "," +
        ",".join(map(str, CM5_values)) + "," +
        ",".join(str(x) for x in SpinD_values)[1:-1])
        row = ','
        for mylist in [Disp_values, SASA_values, Vbur_values, Pyr_values, xtb_values, distance_values, angle_values, dihedral_values, Sterimol_values]:
            for i in mylist :
                 row += f"{i},"
        print(row, file=f)

#################################################
