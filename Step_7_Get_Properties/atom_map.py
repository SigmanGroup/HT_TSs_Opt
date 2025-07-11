#!/usr/bin/env python3
import pandas as pd
import openpyxl
import sys

def fill_excel_from_csv(excel_file_path, csv_file_path):
    # Load the Excel file with formatting
    workbook = openpyxl.load_workbook(excel_file_path)
    sheet = workbook.active

    # Convert Excel data (excluding the first cell) to a DataFrame
    data = sheet.values
    columns = next(data)[1:]  # Skip the first cell in the first row
    data = list(data)
    indices = [row[0] for row in data]  # Keep the first column as indices
    data = (row[1:] for row in data)  # Skip the first column in the data
    excel_df = pd.DataFrame(data, index=indices, columns=columns)

    # Read the CSV file
    csv_df = pd.read_csv(csv_file_path)

    # Extract the "Template" from the "log_name" column in the Excel DataFrame
    excel_df['Template'] = excel_df['log_name'].str.split('_').str[0]

    # List of columns to be filled in the Excel file
    columns_to_fill = ['R1', 'C2', 'N1', 'C1', 'C4', 'N2', 'X2', 'C5', 'R2', 'X1', 'Ni', 'C1s', 'Br', 'C2s', 'H1s']

    # Fill in the columns using data from the CSV file
    for index, row in excel_df.iterrows():
        template = row['Template']
        if template in csv_df['Template'].values:
            csv_row = csv_df[csv_df['Template'] == template].iloc[0]
            for col in columns_to_fill:
                excel_df.at[index, col] = csv_row[col]

    # Drop the temporary 'Template' column
    excel_df = excel_df.drop(columns=['Template'])

    # Write the updated DataFrame back to the Excel file, preserving the first cell and formatting
    for r_idx, row in enumerate(excel_df.itertuples(index=False, name=None), 2):
        for c_idx, value in enumerate(row, 2):
            sheet.cell(row=r_idx, column=c_idx).value = value

    # Save the workbook with the same formatting
    output_file_path = excel_file_path.replace('.xlsx', '_updated.xlsx')
    workbook.save(output_file_path)

    print(f"Updated Excel file saved as {output_file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <excel_file_path> <csv_file_path>")
    else:
        excel_file_path = sys.argv[1]
        csv_file_path = sys.argv[2]
        fill_excel_from_csv(excel_file_path, csv_file_path)

