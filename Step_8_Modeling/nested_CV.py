#!/usr/bin/env python3

import pandas as pd
import numpy as np
from itertools import combinations, product
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import sys

# Set display options to show full DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.colheader_justify', 'left')
pd.set_option('display.width', None)

# Define evaluate_combinations at the top level for multiprocessing
def evaluate_combinations(chunk, X, y, mandatory_sets, feature_indices):
    """
    Evaluate a chunk of feature combinations.
    """
    results = []
    for indices in chunk:
        for mandatory_set in mandatory_sets:
            # Combine mandatory features with the current combination
            selected_indices = [feature_indices[feature] for feature in mandatory_set] + list(indices)
            X_subset = X.iloc[:, selected_indices]

            # Fit a linear regression model and calculate RSS
            model = LinearRegression()
            model.fit(X_subset, y)
            y_pred = model.predict(X_subset)
            rss = np.sum((y_pred - y) ** 2)

            results.append((selected_indices, rss))
    return results

def read_data(file_path):
    df = pd.read_excel(file_path)
    df.drop(['Structure', 'ee'], axis=1, inplace=True)
    X = df.drop(columns=['DeltaDeltaG', 'Class'])
    y = df['DeltaDeltaG']
    ligand_class = df['Class']
    return X, y, ligand_class

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, scaler

def remove_highly_correlated_features(X, y, threshold=0.7):
    """
    Remove highly correlated features while ensuring that all features from the predefined groups are retained.
    """
    # Define feature groups
    feature_groups = {}

    # Step 1: Flatten the list of features to retain from all groups
    retained_features = [
        feature
        for group_features in feature_groups.values()
        for feature in group_features
        if feature in X.columns
    ]

    # Step 2: Compute the correlation matrix
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Step 3: Identify features to drop based on high correlation
    to_drop = set()
    for column in upper.columns:
        if column in retained_features:
            continue  # Skip retained features
        for row in upper.index[upper[column] > threshold]:
            if row not in retained_features and column not in retained_features:
                # Drop the feature with the lower correlation to y
                if abs(X[row].corr(y)) > abs(X[column].corr(y)):
                    to_drop.add(column)
                else:
                    to_drop.add(row)

    # Step 4: Ensure retained features are not dropped
    to_drop = list(to_drop - set(retained_features))
    X_reduced = X.drop(columns=to_drop, errors="ignore")

    # Step 5: Print excluded features
    print(f"Excluded features: {to_drop}")
    print(f"Retained features: {retained_features}")
    return X_reduced, to_drop

def exhaustive_search(X, y, feature_names, n_features=4, top_n=5):
    """
    Perform exhaustive search for the best feature combinations in parallel.
    """
    # Define feature groups
    feature_groups = {}

    # Generate all valid combinations of mandatory features
    mandatory_combinations = []
    for group, features in feature_groups.items():
        valid_features = [feature for feature in features if feature in feature_names]
        mandatory_combinations.append(valid_features)

    mandatory_sets = list(product(*mandatory_combinations))
    feature_indices = {name: idx for idx, name in enumerate(feature_names)}

    # Determine remaining features
    remaining_indices = [
        feature_indices[feature]
        for feature in feature_names
        if feature not in [item for sublist in mandatory_combinations for item in sublist]
    ]

    # Prepare feature combinations and ensure total features = n_features
    top_models = []
    for mandatory_set in mandatory_sets:
        mandatory_set_indices = [feature_indices[feature] for feature in mandatory_set]
        num_mandatory = len(mandatory_set_indices)

        # Skip combinations if mandatory features exceed n_features
        if num_mandatory > n_features:
            continue

        comb = list(combinations(remaining_indices, n_features - num_mandatory))

        # Create chunks for parallel processing
        num_processors = 52
        min_chunks = num_processors  # Ensure at least one chunk per CPU
        chunk_size = max(1, math.ceil(len(comb) / min_chunks))  # Avoid empty chunks
        comb_chunks = [comb[i:i + chunk_size] for i in range(0, len(comb), chunk_size)]

        print(f"Number of combinations: {len(comb)}")
        print(f"Number of chunks: {len(comb_chunks)}")

        with ProcessPoolExecutor(max_workers=num_processors) as executor:
            futures = [
                executor.submit(evaluate_combinations, chunk, X, y, [mandatory_set], feature_indices)
                for chunk in comb_chunks
            ]

            for future in as_completed(futures):
                top_models.extend(future.result())

    # Sort and select top_n models
    top_models.sort(key=lambda x: x[1])
    top_models = top_models[:top_n]

    return [([feature_names[i] for i in model[0]], model[1]) for model in top_models]

def calculate_r2_excel(y_true, y_pred):
    """Calculate R² using the squared Pearson correlation coefficient."""
    correlation_matrix = np.corrcoef(y_true, y_pred)
    r = correlation_matrix[0, 1]
    return r**2

def calculate_adjusted_r2(r_squared, n_samples, n_features):
    return 1 - (1 - r_squared) * ((n_samples - 1) / (n_samples - n_features - 1))

def print_linear_equation(selected_features, coefficients, intercept, label):
    terms = [f"{coeff:.3f}*{feature}" for coeff, feature in zip(coefficients, selected_features)]
    equation = " + ".join(terms) + f" + {intercept:.3f}"
    print(f"Linear Equation of the {label} Champion Model:")
    print(f"ΔΔG‡ = {equation}")

def predict_and_save(y_actual, y_pred, output_file):
    results_df = pd.DataFrame({
        'Measured ΔΔG‡': y_actual,
        'Predicted ΔΔG‡': y_pred
    })
    results_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

def nested_cv(X, y, ligand_class, X_oos=None, y_oos=None):
    print("Starting feature reduction...")
    X_reduced, excluded_features = remove_highly_correlated_features(X, y)

    if X_oos is not None and y_oos is not None:
        print("Processing out-of-sample (OOS) data...")
        X_oos = X_oos.drop(columns=excluded_features, errors="ignore")
        X_oos = X_oos.reindex(columns=X_reduced.columns, fill_value=0)

    print("Scaling features...")
    X_scaled, scaler = scale_features(X_reduced)

    if X_oos is not None:
        X_oos_scaled = pd.DataFrame(scaler.transform(X_oos), columns=X_oos.columns)

    print("Starting outer cross-validation...")
    outer_cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=5, random_state=42)
    feature_names = X_reduced.columns.tolist()
    outer_fold_metrics = []
    model_details = {}

    for fold_idx, (train_val_idx, test_idx) in enumerate(outer_cv.split(X, ligand_class)):
        print(f"\nProcessing outer fold {fold_idx + 1}/{outer_cv.get_n_splits()}")
        X_train_val, X_test = X_scaled.iloc[train_val_idx], X_scaled.iloc[test_idx]
        y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]

        inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
        inner_fold_models = defaultdict(list)

        for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train_val, ligand_class.iloc[train_val_idx])):
            print(f"  Inner fold {inner_fold_idx + 1}/{inner_cv.get_n_splits()} for outer fold {fold_idx + 1}")
            X_inner_train, X_inner_val = X_train_val.iloc[inner_train_idx], X_train_val.iloc[inner_val_idx]
            y_inner_train, y_inner_val = y_train_val.iloc[inner_train_idx], y_train_val.iloc[inner_val_idx]

            print(f"    Performing exhaustive search for inner fold {inner_fold_idx + 1}...")
            top_models = exhaustive_search(X_inner_train, y_inner_train, feature_names)
            print(f"    Found {len(top_models)} top models for inner fold {inner_fold_idx + 1}")

            for features_set, _ in top_models:
                model = LinearRegression().fit(X_inner_train[features_set], y_inner_train)
                y_val_pred = model.predict(X_inner_val[features_set])
                rmse = np.sqrt(mean_squared_error(y_inner_val, y_val_pred))
                inner_fold_models[tuple(features_set)].append(rmse)

        best_inner_models = {k: np.mean(v) for k, v in inner_fold_models.items()}
        best_inner_model_features = min(best_inner_models, key=best_inner_models.get)
        print(f"Best model for outer fold {fold_idx + 1} uses features: {best_inner_model_features}")

        best_model = LinearRegression().fit(X_train_val[list(best_inner_model_features)], y_train_val)
        y_train_val_pred = best_model.predict(X_train_val[list(best_inner_model_features)])
        y_test_pred = best_model.predict(X_test[list(best_inner_model_features)])

        train_rmse = np.sqrt(mean_squared_error(y_train_val, y_train_val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2_train = calculate_r2_excel(y_train_val, y_train_val_pred)
        adjusted_r2 = calculate_adjusted_r2(r2_train, len(y_train_val), len(best_inner_model_features))
        print(f"  Train RMSE: {train_rmse:.3f}, Test RMSE: {test_rmse:.3f}, Adjusted R²: {adjusted_r2:.3f}")

        if X_oos is not None:
            y_oos_pred = np.dot(X_oos_scaled[list(best_inner_model_features)], best_model.coef_) + best_model.intercept_
            r2_oos = calculate_r2_excel(y_oos, y_oos_pred)
            mae_oos = mean_absolute_error(y_oos, y_oos_pred)
            print(f"  OOS R²: {r2_oos:.3f}, OOS MAE: {mae_oos:.3f}")
        else:
            r2_oos = mae_oos = None

        outer_fold_metrics.append((best_inner_model_features, train_rmse, test_rmse, adjusted_r2, r2_oos, mae_oos))
        model_details[tuple(best_inner_model_features)] = (best_model.coef_, best_model.intercept_)

    print("\nOuter cross-validation complete!")

    feature_stats = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    for features, train_rmse, test_rmse, adjusted_r2, r2_oos, mae_oos in outer_fold_metrics:
        stats = feature_stats[features]
        stats[0] += 1
        stats[1] += train_rmse
        stats[2] += test_rmse
        stats[3] += adjusted_r2
        if r2_oos is not None:
            stats[4] += r2_oos
            stats[5] += mae_oos

    data = []
    for features, (count, sum_train_rmse, sum_test_rmse, sum_adjusted_r2, sum_r2_oos, sum_mae_oos) in feature_stats.items():
        avg_train_rmse = sum_train_rmse / count
        avg_test_rmse = sum_test_rmse / count
        avg_adjusted_r2 = sum_adjusted_r2 / count
        avg_r2_oos = sum_r2_oos / count if count > 0 and sum_r2_oos != 0 else None
        avg_mae_oos = sum_mae_oos / count if count > 0 and sum_mae_oos != 0 else None
        data.append([", ".join(features), count, avg_train_rmse, avg_test_rmse, avg_adjusted_r2, avg_r2_oos, avg_mae_oos])

    df = pd.DataFrame(data, columns=['Features', 'Count', 'Average Train RMSE', 'Average Test RMSE', 'Average Adjusted R²', 'Average OOS R²', 'Average OOS MAE'])

    # Table sorted by Count (or Adjusted R² in case of ties)
    df_sorted_by_count = df.sort_values(by=['Count', 'Average Test RMSE'], ascending=[False, True])
    if X_oos is None:
        df_sorted_by_count = df_sorted_by_count.drop(columns=['Average OOS R²', 'Average OOS MAE'])
    print("Outer-Fold Champion Models Sorted by Count:")
    print(df_sorted_by_count.to_string(index=False))

    # Table sorted by OOS MAE if OOS data is provided
    if X_oos is not None:
        df_sorted_by_oos_mae = df.sort_values(by='Average OOS MAE', ascending=True)
        print("\nOuter-Fold Champion Models Sorted by OOS MAE:")
        print(df_sorted_by_oos_mae.to_string(index=False))

    # OOS-Specific Champion Model
    if X_oos is not None:
        best_oos_features = df_sorted_by_oos_mae.iloc[0]['Features'].split(", ")
        coefficients, intercept = model_details[tuple(best_oos_features)]
        print_linear_equation(best_oos_features, coefficients, intercept, "OOS-Specific")

        y_oos_pred = np.dot(X_oos_scaled[best_oos_features], coefficients) + intercept
        oos_results_df = pd.DataFrame({
            'Measured ΔΔG‡ (OOS)': y_oos,
            'Predicted ΔΔG‡ (OOS)': y_oos_pred
        })
        oos_results_df.to_excel('oos_predicted_results.xlsx', index=False)
        print("Out-of-sample predictions saved to 'oos_predicted_results.xlsx'")

    # Data-Specific Champion Model
    champion_features = df_sorted_by_count.iloc[0]['Features'].split(", ")
    coefficients, intercept = model_details[tuple(champion_features)]
    print_linear_equation(champion_features, coefficients, intercept, "Data-Specific")

    y_pred = np.dot(X_scaled[champion_features], coefficients) + intercept
    predict_and_save(y, y_pred, 'predicted_results.xlsx')

if __name__ == '__main__':
    file_path = sys.argv[1]
    if len(sys.argv) > 2:
        oos_file_path = sys.argv[2]
        X, y, ligand_class = read_data(file_path)
        X_oos, y_oos, _ = read_data(oos_file_path)
        nested_cv(X, y, ligand_class, X_oos, y_oos)
    else:
        X, y, ligand_class = read_data(file_path)
        nested_cv(X, y, ligand_class)

