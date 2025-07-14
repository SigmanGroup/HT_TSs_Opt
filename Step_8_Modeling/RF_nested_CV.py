#!/usr/bin/env python3

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import sys

# Set display options to show full DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.colheader_justify', 'left')
pd.set_option('display.width', None)

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
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = []
    for column in upper.columns:
        for row in upper.index[upper[column] > threshold]:
            if row != column and row not in to_drop and column not in to_drop:
                if abs(X[row].corr(y)) > abs(X[column].corr(y)):
                    to_drop.append(column)
                else:
                    to_drop.append(row)
    to_drop = list(set(to_drop))
    X_reduced = X.drop(columns=to_drop, errors='ignore')
    print(f"Excluded features: {to_drop}")
    return X_reduced, to_drop

def exhaustive_search(X, y, feature_names, n_features=5, top_n=5):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importances = rf.feature_importances_

    important_indices = [i for i, imp in enumerate(feature_importances) if imp > 0.01]
    important_features = [feature_names[i] for i in important_indices]

    comb = combinations(important_indices, n_features)
    top_models = []
    for indices in comb:
        model = LinearRegression()
        X_subset = X.iloc[:, list(indices)]
        model.fit(X_subset, y)
        rss = np.sum((y - model.predict(X_subset)) ** 2)
        top_models.append((indices, rss))

    top_models.sort(key=lambda x: x[1])
    return [([feature_names[i] for i in model[0]], model[1]) for model in sorted(top_models, key=lambda x: x[1])[:top_n]]

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
    X_reduced, excluded_features = remove_highly_correlated_features(X, y)

    if X_oos is not None and y_oos is not None:
        X_oos = X_oos.drop(columns=excluded_features, errors='ignore')
        X_oos = X_oos.reindex(columns=X_reduced.columns, fill_value=0)

    X_scaled, scaler = scale_features(X_reduced)

    if X_oos is not None:
        X_oos_scaled = pd.DataFrame(scaler.transform(X_oos), columns=X_oos.columns)

    outer_cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=5, random_state=42)
    feature_names = X_reduced.columns.tolist()
    outer_fold_metrics = []
    model_details = {}

    for train_val_idx, test_idx in outer_cv.split(X, ligand_class):
        X_train_val, X_test = X_scaled.iloc[train_val_idx], X_scaled.iloc[test_idx]
        y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]

        inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
        inner_fold_models = defaultdict(list)

        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, ligand_class.iloc[train_val_idx]):
            X_inner_train, X_inner_val = X_train_val.iloc[inner_train_idx], X_train_val.iloc[inner_val_idx]
            y_inner_train, y_inner_val = y_train_val.iloc[inner_train_idx], y_train_val.iloc[inner_val_idx]
            top_models = exhaustive_search(X_inner_train, y_inner_train, feature_names)
            for features_set, _ in top_models:
                model = LinearRegression().fit(X_inner_train[features_set], y_inner_train)
                y_val_pred = model.predict(X_inner_val[features_set])
                rmse = np.sqrt(mean_squared_error(y_inner_val, y_val_pred))
                inner_fold_models[tuple(features_set)].append(rmse)

        best_inner_models = {k: np.mean(v) for k, v in inner_fold_models.items()}
        best_inner_model_features = min(best_inner_models, key=best_inner_models.get)

        best_model = LinearRegression().fit(X_train_val[list(best_inner_model_features)], y_train_val)
        y_train_val_pred = best_model.predict(X_train_val[list(best_inner_model_features)])
        y_test_pred = best_model.predict(X_test[list(best_inner_model_features)])

        train_rmse = np.sqrt(mean_squared_error(y_train_val, y_train_val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2_train = calculate_r2_excel(y_train_val, y_train_val_pred)
        adjusted_r2 = calculate_adjusted_r2(r2_train, len(y_train_val), len(best_inner_model_features))

        if X_oos is not None:
            y_oos_pred = np.dot(X_oos_scaled[list(best_inner_model_features)], best_model.coef_) + best_model.intercept_
            r2_oos = calculate_r2_excel(y_oos, y_oos_pred)
            mae_oos = mean_absolute_error(y_oos, y_oos_pred)
        else:
            r2_oos = mae_oos = None

        outer_fold_metrics.append((best_inner_model_features, train_rmse, test_rmse, adjusted_r2, r2_oos, mae_oos))
        model_details[tuple(best_inner_model_features)] = (best_model.coef_, best_model.intercept_)

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
    file_path = sys.argv[1]  # Training file

    if len(sys.argv) > 2:
        oos_file_path = sys.argv[2]  # Out-of-sample file
        X, y, ligand_class = read_data(file_path)
        X_oos, y_oos, _ = read_data(oos_file_path)
        nested_cv(X, y, ligand_class, X_oos, y_oos)
    else:
        X, y, ligand_class = read_data(file_path)
        nested_cv(X, y, ligand_class)

