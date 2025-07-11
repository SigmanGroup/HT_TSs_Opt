#!/usr/bin/env python3

import os
import sys

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def g16_data_test(value):
    if not value:
        return "-----"
    else:
        return value

def g16_lowest_imaginary_frequency(files, threshold_low, threshold_high):
    # Create Non_TS directory if it doesn't exist
    os.makedirs('Non_TS', exist_ok=True)

    output_values = {}
    files_with_imaginary_freqs = set()

    # Process each file to find the lowest imaginary frequency
    for file in files:
        with open(file) as f:
            found_frequencies = False
            for line in f:
                if 'Frequencies' in line:
                    frequency = line.split()[2]
                    output_values[file] = g16_data_test(frequency)
                    found_frequencies = True
                    break
            if not found_frequencies:
                output_values[file] = "-----"

        # Add files with imaginary frequencies to the set
        if '1 imaginary frequencies (negative Signs)' in open(file).read():
            files_with_imaginary_freqs.add(file)

    # Sort output values by frequency
    sorted_values = sorted((value for value in output_values.values() if value != "-----"), key=float, reverse=True)

    # Write output to Im_freq.csv and move files with frequencies out of range
    with open('Im_freq.csv', 'w') as f:
        for value in sorted_values:
            for file, output in output_values.items():
                if output == value:
                    f.write(f"{file},{output}\n")
                    if is_float(output) and (float(output) < threshold_low or float(output) > threshold_high):
                        move_file_to_non_ts(file)
                    break

    # Move files without imaginary frequencies to Non_TS directory and write their names to Multiple_ifreqs.csv
    with open('Multiple_ifreqs.csv', 'w') as f:
        f.write("Structure\n")
        for file in files:
            if file not in files_with_imaginary_freqs:
                move_file_to_non_ts(file)
                f.write(f"{file}\n")

def move_file_to_non_ts(file):
    try:
        os.rename(file, os.path.join('Non_TS', os.path.basename(file)))
    except FileNotFoundError:
        print(f"{file} is not a true TS and has already been filtered out.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: {} threshold_low threshold_high file1 [file2 ...]".format(sys.argv[0]))
        sys.exit(1)

    threshold_low = float(sys.argv[1])
    threshold_high = float(sys.argv[2])
    files = sys.argv[3:]

    g16_lowest_imaginary_frequency(files, threshold_low, threshold_high)

