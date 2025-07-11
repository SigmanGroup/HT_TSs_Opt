#!/usr/bin/env python
import os
import pandas as pd

cur_dir = os.getcwd()

def get_orca_SPE(output):
    with open(output) as f:
        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                SPE = float(line.split()[-1])
                return SPE

out_files = []
energies = []

ext = '.out'

for file in os.listdir(cur_dir):
    if file.endswith(ext):
        out_files.append(file[:-4])
        energy = get_orca_SPE(os.path.join(cur_dir, file))
        energies.append(energy)

data = {'Output': [f + '_SPC' for f in out_files],
        'Single_point_energy(Hartrees)': energies}

energies_dataframe = pd.DataFrame(data)

print(energies_dataframe)

energies_dataframe.to_excel('orca_SPC.xlsx', index=False)

