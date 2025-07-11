#!/usr/bin/env python3
import csv
import re

# Read CSV file and store values for each template
template_values = {}
with open('Atom_indices.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        template = row['Template']
        values = {k: int(v) for k, v in row.items() if k != 'Template'}
        template_values[template] = values

# Open and read the original script
with open('GetParameters.py', 'r') as file:
    original_script = file.read()

# Find template name in the original script
template_match = re.search(r'template = "([^"]+)"', original_script)
if template_match:
    template_name = template_match.group(1)
    if template_name in template_values:
        values = template_values[template_name]
        # Replace labels in the original script with values from CSV file
        for label, value in values.items():
            # Using \b to match whole word labels
            original_script = re.sub(r'\b' + label + r'\b', str(value), original_script)

        # Write the modified script to a new file
        modified_filename = f'GetParameters_{template_name}.py'
        with open(modified_filename, 'w') as file:
            file.write(original_script)
    else:
        print(f"Template '{template_name}' not found in the CSV file.")
else:
    print("Template not found in the original script.")

