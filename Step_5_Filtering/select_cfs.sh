#!/bin/bash
# Check if a filename argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 filename.txt"
    exit 1
fi

# Loop through each line in the provided text file
while IFS= read -r file_id
do
    python -m navicat_marc -i "${file_id}_TSRE_R"*.xyz -m rmsd -ewin 10 -mine
    python -m navicat_marc -i "${file_id}_TSRE_S"*.xyz -m rmsd -ewin 10 -mine
done < "$1"

