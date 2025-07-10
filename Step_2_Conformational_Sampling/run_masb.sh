#!/bin/bash
# Check if a filename argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 filename.txt"
    exit 1
fi

# Loop through each line in the provided text file
while IFS= read -r file_id
do
    python conformer_generator.py "${file_id}_TSRE_R.xyz"
    python conformer_generator.py "${file_id}_TSRE_S.xyz"
done < "$1"

