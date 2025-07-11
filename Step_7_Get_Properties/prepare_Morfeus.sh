#!/bin/bash
# Check if a filename argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 filename.txt"
    exit 1
fi

# Loop through each line in the provided text file
while IFS= read -r file_id
do
    mkdir "$file_id"
    mv "${file_id}_TSRE_R"*.xyz "$file_id"
    mv "${file_id}_TSRE_R"*.log "$file_id"
    mv "${file_id}_TSRE_S"*.xyz "$file_id"
    mv "${file_id}_TSRE_S"*.log "$file_id"
done < "$1"

