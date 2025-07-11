#!/usr/bin/env python3

import pandas as pd
import sys

def combine_features(input_file, output_file):
    # Define the sheet names
    sheets = ["TSRC", "IntC", "TSRE"]

    # Define suffix replacements for each sheet
    suffix_replacements = {
        "TSRC": "_TSRC",
        "IntC": "_IntC",
        "TSRE": "_TSRE"
    }

    # Initialize an empty DataFrame for the combined data
    combined_df = None

    for sheet in sheets:
        # Read the current sheet
        df = pd.read_excel(input_file, sheet_name=sheet)

        # Rename feature columns to include the sheet name in the suffix
        renamed_columns = {}
        for col in df.columns[4:]:  # Skip the first four columns
            for suffix in ["_Boltz", "_min", "_max", "_low_E"]:
                if suffix in col:
                    renamed_columns[col] = col.replace(suffix, suffix_replacements[sheet] + suffix)
                    break

        # Apply the renaming
        df.rename(columns=renamed_columns, inplace=True)

        # Append to the combined DataFrame
        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df.iloc[:, 4:]], axis=1)  # Keep only features

    # Save the combined DataFrame to a new Excel file
    combined_df.to_excel(output_file, index=False)
    print(f"Output file saved as {output_file}")


if __name__ == "__main__":
    # Check if the script received the required arguments
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_file> <output_file>")
        sys.exit(1)

    # Get the input and output file names from the command-line arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Run the function
    combine_features(input_file, output_file)

