#!/usr/bin/env python3

import pandas as pd
import argparse

def remove_unwanted_columns(xlsx_file, output_file):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(xlsx_file)
    # Identify columns where the header includes '_stdev' or '_range'
    columns_to_drop = [col for col in df.columns if '_stdev' in col or '_range' in col]
    # Drop these columns from the DataFrame
    df_cleaned = df.drop(columns=columns_to_drop)
    # Save the cleaned DataFrame back to an Excel file
    df_cleaned.to_excel(output_file, index=False)
    print(f"Processed file saved as {output_file} with {df_cleaned.shape[1]} columns remaining.")

def main():
    parser = argparse.ArgumentParser(description='Remove columns containing "_stdev" or "_range" in their headers from an Excel file.')
    parser.add_argument('input_xlsx', type=str, help='Input Excel file path')
    parser.add_argument('output_xlsx', type=str, help='Output Excel file path')
    args = parser.parse_args()
    remove_unwanted_columns(args.input_xlsx, args.output_xlsx)

if __name__ == "__main__":
    main()

