#!/usr/bin/env python
import os
import pandas as pd

def get_xtb_E(output):
    SPE = None  # Default value if the line is not found
    with open(output, 'r') as f:
        for line in reversed(list(f)):  # Iterate over lines in reverse order
            if "Recovered energy" in line:
                SPE = float(line.split()[2])
                break  # Break the loop once the first occurrence is found
    return SPE

cur_dir = os.getcwd()

out_files = []
energies = []

ext = '.log'

for file in os.listdir(cur_dir):
    if file.endswith(ext):
        out_files.append(file[:-4])
        energy = get_xtb_E(os.path.join(cur_dir, file))
        energies.append(energy)

data = {'Output': out_files,
        'Electronic_energy(Hartrees)': energies}

energies_dataframe = pd.DataFrame(data)

print(energies_dataframe)

energies_dataframe.to_excel('xTB_E.xlsx', index=False)

