#!/bin/bash
# Check if a filename argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 ligands.txt"
    exit 1
fi

# Loop through each ligand ID in the provided text file
while IFS= read -r ligand_id
do
    sed -i "s/L0000/$ligand_id/g" GetParameters.py
    python Change_labels.py
    mv GetParameters_"$ligand_id".py "$ligand_id"/
    cd "$ligand_id"/
    bash /uufs/chpc.utah.edu/common/home/u6055669/bin/submit_cli.sh python GetParameters_"$ligand_id".py
    cd ..
    sed -i "s/$ligand_id/L0000/g" GetParameters.py
done < "$1"

