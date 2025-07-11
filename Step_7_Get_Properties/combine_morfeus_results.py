#!/usr/bin/env python3

import os
import pandas as pd
import argparse

def combine_csv_files(directory, output_xlsx):
    data_frames = []
    processed_files = []  # Keep track of processed files

    # Loop through all files in the directory
    for filename in sorted(os.listdir(directory)):  # Sort to process files in a predictable order
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)

            # Check if we've already processed this file
            if file_path in processed_files:
                print(f"Skipping duplicate processing of {filename}")
                continue

            # Read the CSV file, skipping the first row (header), and handle the trailing comma
            df = pd.read_csv(file_path, skiprows=1, header=None)
            # Ensure the DataFrame has the correct number of columns, slicing off any extra empty column
            if df.shape[1] == 185:
                df = df.iloc[:, :-1]

            # Check if the DataFrame still has the expected 184 columns
            if df.shape[1] != 184:
                print(f"Warning: File '{filename}' has {df.shape[1]} columns after adjustments.")

            # Append to our list of DataFrames
            data_frames.append(df)
            processed_files.append(file_path)  # Add this file to the list of processed files

    # Concatenate all dataframes into one
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Write the combined DataFrame to an Excel file
    combined_df.to_excel(output_xlsx, index=False)
    print(f"Successfully created {output_xlsx} with {combined_df.shape[0]} rows and {combined_df.shape[1]} columns.")

def main():
    parser = argparse.ArgumentParser(description='Combine multiple CSV files into a single Excel file with 184 columns each.')
    parser.add_argument('directory', type=str, help='Directory containing CSV files')
    parser.add_argument('output_xlsx', type=str, help='Name of the output Excel file')

    args = parser.parse_args()

    combine_csv_files(args.directory, args.output_xlsx)

if __name__ == "__main__":
    main()

