#!/usr/bin/env python
import os
import pandas as pd

cur_dir = os.getcwd()

def get_corr(output):
    with open(output) as f:
        for gibbs_line in f:
            if "Thermal correction to Gibbs Free Energy" in gibbs_line:
                corr = float(gibbs_line.split()[6])
                return corr

out_files = []
thermal_corr = []

ext = '.log'

for file in os.listdir(cur_dir):
    if file.endswith(ext):
        out_files.append(file[:-4])
        gibbs_corr = get_corr(os.path.join(cur_dir, file))
        thermal_corr.append(gibbs_corr)

data = [{'Output': f, 'Gibbs_corr(Hartrees)': gibbs_corr} for f, gibbs_corr in zip(out_files, thermal_corr)]

thermal_corr_dataframe = pd.DataFrame(data)

print(thermal_corr_dataframe)

thermal_corr_dataframe.to_excel('Gibbs_corr.xlsx', index=False)

