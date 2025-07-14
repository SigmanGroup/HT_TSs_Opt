#!/usr/bin/env python3

import pandas as pd
import sys
from collections import defaultdict

# Define known suffixes
SUFFIXES = ["_Boltz", "_min", "_max", "_low_E"]

def get_base_name(feature_name):
    """
    Extract the base name of a feature by removing known suffixes.
    """
    for suffix in SUFFIXES:
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)]
    return feature_name


def analyze_features(features, classifications):
    """
    Analyze the number of features for each type and category.
    """
    overall_counts = defaultdict(int)
    unique_feature_sets = defaultdict(set)

    for feature in features:
        classification = classifications.get(feature, None)
        if classification:
            overall_counts[classification] += 1
            # Get the base name of the feature by removing known suffixes
            base_name = get_base_name(feature)
            unique_feature_sets[classification].add(base_name)

    return overall_counts, unique_feature_sets


def summarize_general_categories(overall_counts, unique_feature_sets):
    """
    Summarize the counts into general "electronic" and "steric" categories.
    """
    general_overall = {"electronic": 0, "steric": 0}
    general_unique = {"electronic": set(), "steric": set()}

    for classification, count in overall_counts.items():
        if "electronic" in classification:
            general_overall["electronic"] += count
            general_unique["electronic"].update(unique_feature_sets.get(classification, set()))
        elif "steric" in classification:
            general_overall["steric"] += count
            general_unique["steric"].update(unique_feature_sets.get(classification, set()))

    # Count unique features for general categories
    general_unique_counts = {k: len(v) for k, v in general_unique.items()}

    return general_overall, general_unique_counts


def print_feature_analysis(title, overall_counts, unique_counts, general_overall, general_unique):
    """
    Print the feature analysis results.
    """
    print(f"\n{title}")
    print("Overall Counts by Classification:")
    for classification, count in overall_counts.items():
        print(f"  {classification}: {count}")
    print("\nUnique Feature Counts by Classification:")
    for classification, count in unique_counts.items():
        print(f"  {classification}: {count}")
    print("\nAggregated Counts:")
    print(f"  electronic (overall): {general_overall['electronic']}")
    print(f"  steric (overall): {general_overall['steric']}")
    print(f"  electronic (unique): {general_unique['electronic']}")
    print(f"  steric (unique): {general_unique['steric']}")


def filter_features(input_file, output_file):
    # Load the Excel file
    features_df = pd.read_excel(input_file, sheet_name="Features", engine='openpyxl')
    type_df = pd.read_excel(input_file, sheet_name="Type", engine='openpyxl', header=None)

    # Separate the first four columns from "Features"
    main_columns = features_df.iloc[:, :4]

    # Extract the feature names and classifications from the "Type" sheet
    feature_names = type_df.iloc[0]  # First row contains feature names
    classifications = type_df.iloc[1]  # Second row contains classifications
    feature_classification_map = dict(zip(feature_names, classifications))

    # Analyze features in the input
    input_overall, input_unique_sets = analyze_features(features_df.columns[4:], feature_classification_map)
    input_general_overall, input_general_unique = summarize_general_categories(input_overall, input_unique_sets)
    print_feature_analysis("Input Feature Analysis", input_overall, {k: len(v) for k, v in input_unique_sets.items()}, input_general_overall, input_general_unique)

    # Create a list to store the columns to keep
    columns_to_keep = list(main_columns.columns)  # Start with the main columns

    # Iterate through the features in "Features" to decide which ones to keep
    for column in features_df.columns[4:]:  # Skip the first four non-feature columns
        classification = feature_classification_map.get(column, None)
        if classification:
            if "electronic" in classification and column.endswith("_Boltz"):
                columns_to_keep.append(column)
            elif "steric" in classification and any(column.endswith(suffix) for suffix in ["_Boltz", "_min", "_max"]):
                columns_to_keep.append(column)

    # Filter the DataFrame and analyze features in the output
    filtered_df = features_df[columns_to_keep]
    output_overall, output_unique_sets = analyze_features(filtered_df.columns[4:], feature_classification_map)
    output_general_overall, output_general_unique = summarize_general_categories(output_overall, output_unique_sets)
    print_feature_analysis("Output Feature Analysis", output_overall, {k: len(v) for k, v in output_unique_sets.items()}, output_general_overall, output_general_unique)

    # Save the resulting DataFrame to a new Excel file
    filtered_df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"\nFiltered file saved to {output_file}")


if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python filter_features.py <input_file> <output_file>")
        sys.exit(1)

    # Parse command-line arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Call the filtering function
    filter_features(input_file, output_file)

