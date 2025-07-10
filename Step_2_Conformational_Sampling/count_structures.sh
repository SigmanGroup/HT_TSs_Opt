#!/bin/bash
# Check if a filename argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 filename.txt"
    exit 1
fi

# Loop through each line in the provided text file
while IFS= read -r file_id
do
    count_R=$(ls "${file_id}_TSRE_R"*.xyz 2>/dev/null | wc -l)
    echo "${file_id}_TSRE_R: $count_R"

    count_S=$(ls "${file_id}_TSRE_S"*.xyz 2>/dev/null | wc -l)
    echo "${file_id}_TSRE_S: $count_S"
    
done < "$1"

