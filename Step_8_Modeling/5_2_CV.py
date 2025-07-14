#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
import os
import warnings

# Suppress UserWarning related to feature names and StandardScaler
warnings.filterwarnings("ignore", message="X has feature names, but StandardScaler was fitted without feature names")

def read_model_features(file_path):
    feature_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            sheet_name = parts[0].strip()
            features = [f.strip() for f in parts[1].split(',')]
            feature_dict[sheet_name] = features
    return feature_dict

def read_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.drop(['Structure', 'ee'], axis=1, inplace=True)
    X = df.drop(columns=['DeltaDeltaG', 'Class'])
    y = df['DeltaDeltaG']
    ligand_class = df['Class']
    return X, y, ligand_class

def generate_folds(y, ligand_class, n_splits=2, n_repeats=5, random_state=42):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    indices = list(rskf.split(np.zeros(len(y)), ligand_class))
    return indices

def scale_features_with_dataframe(train_data, test_data):
    """
    Consistently scale features as DataFrames, preserving names and indices.
    """
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_data),
        columns=train_data.columns,
        index=train_data.index
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_data),
        columns=test_data.columns,
        index=test_data.index
    )
    return train_scaled, test_scaled, scaler

def excel_style_r2(y_true, y_pred):
    """
    Calculate R^2 as the square of the Pearson correlation coefficient.
    """
    correlation_matrix = np.corrcoef(y_true, y_pred)
    r_value = correlation_matrix[0, 1]
    return r_value**2

def calculate_adjusted_r2_excel(r_squared, n_samples, n_features):
    """
    Calculate adjusted R^2 in Excel-style, using non-negative R^2 values.
    """
    return 1 - ((1 - r_squared) * ((n_samples - 1) / (n_samples - n_features - 1)))

def print_linear_equation(selected_features, coefficients, intercept):
    terms = [f"{coeff:.3f}*{feature}" for coeff, feature in zip(coefficients, selected_features)]
    equation = " + ".join(terms) + f" + {intercept:.3f}"
    print("Linear Equation of the Data-Specific Champion Model:")
    print(f"ΔΔG‡ = {equation}")

def predict_and_save(df, train_indices, test_indices, coefficients, intercept, features, scaler, sheet_name, file_path='predicted_results.xlsx'):
    file_exists = os.path.exists(file_path)
    mode = 'a' if file_exists else 'w'
    if_sheet_exists = 'replace' if file_exists else None
    X_all = df[features]
    X_scaled = pd.DataFrame(
        scaler.transform(X_all),
        columns=X_all.columns,
        index=X_all.index
    )
    predictions = np.dot(X_scaled, coefficients) + intercept
    results_df = df[['DeltaDeltaG']].copy()
    results_df['Predicted_DeltaDeltaG'] = predictions
    results_df['DataSet'] = 'Test'
    results_df.loc[train_indices, 'DataSet'] = 'Train'
    with pd.ExcelWriter(file_path, mode=mode, engine='openpyxl', if_sheet_exists=if_sheet_exists) as writer:
        results_df.to_excel(writer, sheet_name=sheet_name)
    print(f"Predictions saved for {sheet_name} in '{file_path}'")

def train_and_evaluate(X_train, y_train, X_test, y_test, n_features):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r_squared = excel_style_r2(y_test, predictions)
    adjusted_r2 = calculate_adjusted_r2_excel(r_squared, X_train.shape[0], n_features)
    coeffs = model.coef_
    intercept = model.intercept_
    return rmse, r_squared, adjusted_r2, coeffs, intercept

def main_workflow(file_path, model_features_path):
    model_features = read_model_features(model_features_path)
    xls = pd.ExcelFile(file_path)
    results = {}

    for sheet_name in xls.sheet_names:
        print(f"\nProcessing {sheet_name}")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df.drop(['Structure', 'ee'], axis=1, inplace=True)
        
        champion_features = model_features.get(sheet_name, [])
        X = df[champion_features] if champion_features else df.drop(columns=['DeltaDeltaG', 'Class'])
        y = df['DeltaDeltaG']
        ligand_class = df['Class']

        indices = generate_folds(y, ligand_class)
        rmses = []
        r2s = []
        adjusted_r2s = []
        coefficients = []
        intercepts = []
        scaler_means = []
        scaler_scales = []

        for train_index, test_index in indices:
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            # Use the DataFrame-based scaling function
            X_train_scaled, X_test_scaled, scaler = scale_features_with_dataframe(X_train, X_test)
            
            # Collect scaler parameters for averaging
            scaler_means.append(scaler.mean_)
            scaler_scales.append(scaler.scale_)
            
            rmse, r_squared, adj_r2, coef, intercept = train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, len(champion_features))

            rmses.append(rmse)
            r2s.append(r_squared)
            adjusted_r2s.append(adj_r2)
            coefficients.append(coef)
            intercepts.append(intercept)

        # Averaging scaler parameters
        avg_mean = np.mean(scaler_means, axis=0)
        avg_scale = np.mean(scaler_scales, axis=0)

        # Creating a custom scaler for predictions
        custom_scaler = StandardScaler()
        custom_scaler.mean_ = avg_mean
        custom_scaler.scale_ = avg_scale

        mean_rmse = np.mean(rmses)
        std_rmse = np.std(rmses)
        mean_r2 = np.mean(r2s)
        mean_adjusted_r2 = np.mean(adjusted_r2s)
        avg_coeffs = np.mean(coefficients, axis=0)
        avg_intercept = np.mean(intercepts)

        predict_and_save(df, np.concatenate([train_index]), np.concatenate([test_index]), avg_coeffs, avg_intercept, champion_features, custom_scaler, sheet_name)

        results[sheet_name] = {
            "Mean RMSE": mean_rmse,
            "Std RMSE": std_rmse,
            "Mean R2": mean_r2,
            "Mean Adjusted R2": mean_adjusted_r2
        }

        print(f"Results for {sheet_name}:")
        print(f"Mean RMSE: {mean_rmse:.3f}, Std Dev RMSE: {std_rmse:.3f}")
        print(f"Mean R²: {mean_r2:.3f}, Mean Adjusted R²: {mean_adjusted_r2:.3f}")
        print_linear_equation(champion_features, avg_coeffs, avg_intercept)

    return results

if __name__ == '__main__':
    excel_file_path = sys.argv[1]
    model_features_file_path = sys.argv[2]
    results = main_workflow(excel_file_path, model_features_file_path)

