#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def excel_style_r2(y_true, y_pred):
    """
    Calculate R^2 as the square of the Pearson correlation coefficient.
    """
    correlation_matrix = np.corrcoef(y_true, y_pred)
    r_value = correlation_matrix[0, 1]
    return r_value ** 2


def calculate_adjusted_r2_excel(r_squared, n_samples, n_features):
    """
    Calculate adjusted R^2 in Excel-style.
    """
    return 1 - ((1 - r_squared) * ((n_samples - 1) / (n_samples - n_features - 1)))


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


def train_and_evaluate(X_train, y_train, X_test, y_test, n_features):
    """
    Train a linear regression model and evaluate its performance.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r_squared = excel_style_r2(y_test, predictions)
    adjusted_r2 = calculate_adjusted_r2_excel(r_squared, X_train.shape[0], n_features)
    coeffs = model.coef_
    intercept = model.intercept_
    return rmse, r_squared, adjusted_r2, coeffs, intercept, predictions


def print_linear_equation(features, coefficients, intercept):
    """
    Print the linear equation of the model.
    """
    terms = [f"{coef:.3f}*{feature}" for coef, feature in zip(coefficients, features)]
    equation = " + ".join(terms) + f" + {intercept:.3f}"
    print("Linear Equation of the Model:")
    print(f"ΔΔG‡ = {equation}")

def parse_out_file(out_file):
    """
    Parse .out files to extract model information.
    """
    with open(out_file, 'r') as file:
        lines = [line.strip() for line in file]

    start_table_index = next(
        i for i, line in enumerate(lines) if line.startswith("Outer-Fold Champion Models Sorted by Count:")
    )
    table_lines = []
    for line in lines[start_table_index + 1:]:
        if not line.strip() or line.startswith("Outer-Fold Champion Models Sorted by OOS MAE:"):
            break
        table_lines.append(line)

    header = ["Features", "Count", "Average Train RMSE", "Average Test RMSE", "Average Adjusted R²", "Average OOS R²", "Average OOS MAE"]
    table_data = []
    for line in table_lines:
        parts = line.rsplit(maxsplit=6)
        if len(parts) == 7:
            table_data.append(parts)

    models_df = pd.DataFrame(table_data, columns=header)
    for col in header[1:]:
        models_df[col] = pd.to_numeric(models_df[col], errors='coerce')

    models_df['Features'] = models_df['Features'].apply(
        lambda x: [feature.strip() for feature in x.split(',')] if isinstance(x, str) else []
    )

    models_df = models_df.dropna(subset=["Count"]).reset_index(drop=True)
    models_df.insert(0, "Model", range(1, len(models_df) + 1))
    return models_df


def parse_excel_file(excel_file):
    """
    Parse an Excel file to extract model information.
    """
    models_df = pd.read_excel(excel_file)
    models_df['Features'] = models_df['Features'].apply(
        lambda x: x.split(', ') if isinstance(x, str) else []
    )
    return models_df


def analyze_features(models_df):
    """
    Analyze feature occurrences and associations with models.
    """
    feature_counts = {}
    feature_models = {}
    tsrc_features = set()
    intc_features = set()
    tsre_features = set()

    for _, row in models_df.iterrows():
        model_name = row['Model']
        features = row['Features']
        for feature in features:
            if feature not in feature_counts:
                feature_counts[feature] = 0
                feature_models[feature] = []
            feature_counts[feature] += 1
            feature_models[feature].append(str(model_name))

            if "TSRC" in feature:
                tsrc_features.add(feature)
            if "IntC" in feature:
                intc_features.add(feature)
            if "TSRE" in feature:
                tsre_features.add(feature)

    feature_analysis = pd.DataFrame({
        'Feature': list(feature_counts.keys()),
        'Frequency': list(feature_counts.values()),
        'Models': [', '.join(feature_models[f]) for f in feature_counts.keys()]
    })

    feature_analysis = feature_analysis.sort_values(by='Frequency', ascending=False).reset_index(drop=True)

    tsrc_count = sum(feature_counts[feature] for feature in tsrc_features)
    intc_count = sum(feature_counts[feature] for feature in intc_features)
    tsre_count = sum(feature_counts[feature] for feature in tsre_features)

    print("\nFeature Analysis:")
    print(feature_analysis)
    print(f"TSRC Features: {tsrc_count}")
    print(f"IntC Features: {intc_count}")
    print(f"TSRE Features: {tsre_count}")

    return feature_analysis, tsrc_count, intc_count, tsre_count

def main_workflow(models_file, training_file, oos_file, output_file):
    # Determine file type and parse models
    if models_file.endswith(".out"):
        print("\nParsing .out file...")
        models_df = parse_out_file(models_file)
    else:
        print("\nParsing Excel file...")
        models_df = parse_excel_file(models_file)

    training_data = pd.read_excel(training_file)
    oos_data = pd.read_excel(oos_file)

    cv_predictions = training_data[['Structure', 'Class', 'ee', 'DeltaDeltaG']].copy()
    oos_predictions = oos_data[['Structure', 'Class', 'ee', 'DeltaDeltaG']].copy()
    metrics = []
    oos_metrics = []

    # Perform feature analysis
    feature_analysis, tsrc_count, intc_count, tsre_count = analyze_features(models_df)

    # Filter models for OOS predictions based on median Average Test RMSE
    median_rmse = models_df['Average Test RMSE'].median()
    selected_models = models_df[models_df['Average Test RMSE'] < median_rmse].copy()
    print("\nSelected Models for OOS Predictions:")
    print(selected_models[['Model', 'Average Test RMSE']])

    selected_features_analysis, _, _, _ = analyze_features(selected_models)

    for _, model in models_df.iterrows():
        model_name = model['Model']
        features = model['Features']
        X = training_data[features]
        y = training_data['DeltaDeltaG']
        ligand_class = training_data['Class']

        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
        all_predictions = np.zeros(len(X))
        data_labels = np.empty(len(X), dtype=object)
        scaler_means, scaler_scales = [], []
        coeffs_list, intercepts_list = [], []
        rmses, r2s, adjusted_r2s = [], [], []

        for train_index, test_index in rskf.split(X, ligand_class):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            X_train_scaled, X_test_scaled, scaler = scale_features_with_dataframe(X_train, X_test)
            scaler_means.append(scaler.mean_)
            scaler_scales.append(scaler.scale_)

            rmse, r_squared, adj_r2, coeffs, intercept, test_predictions = train_and_evaluate(
                X_train_scaled, y_train, X_test_scaled, y_test, len(features)
            )
            rmses.append(rmse)
            r2s.append(r_squared)
            adjusted_r2s.append(adj_r2)
            coeffs_list.append(coeffs)
            intercepts_list.append(intercept)

            all_predictions[test_index] = test_predictions
            data_labels[train_index] = "Train"
            data_labels[test_index] = "Test"

        # Average scaler parameters
        avg_mean = np.mean(scaler_means, axis=0)
        avg_scale = np.mean(scaler_scales, axis=0)
        custom_scaler = StandardScaler()
        custom_scaler.mean_ = avg_mean
        custom_scaler.scale_ = avg_scale

        # Average coefficients and intercept
        avg_coeffs = np.mean(coeffs_list, axis=0)
        avg_intercept = np.mean(intercepts_list)

        # Final predictions for training data
        X_scaled = pd.DataFrame(
            custom_scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        final_predictions = np.dot(X_scaled, avg_coeffs) + avg_intercept
        cv_predictions[f'Predicted_DeltaDeltaG_{model_name}'] = final_predictions
        cv_predictions['DataSet'] = data_labels

        # Metrics
        mean_rmse = np.mean(rmses)
        std_rmse = np.std(rmses)
        mean_r2 = np.mean(r2s)
        mean_adjusted_r2 = np.mean(adjusted_r2s)
        metrics.append({
            'Model': model_name,
            'Mean RMSE': mean_rmse,
            'Std RMSE': std_rmse,
            'Mean R2': mean_r2,
            'Mean Adjusted R2': mean_adjusted_r2
        })

        # Add Mean RMSE to selected_models if applicable
        if model_name in selected_models['Model'].values:
            selected_models.loc[selected_models['Model'] == model_name, 'Mean RMSE'] = mean_rmse

        # OOS predictions for selected models
        if model_name in selected_models['Model'].values:
            X_oos = oos_data[features]
            X_oos_scaled = pd.DataFrame(
                custom_scaler.transform(X_oos),
                columns=X_oos.columns,
                index=X_oos.index
            )
            oos_preds = np.dot(X_oos_scaled, avg_coeffs) + avg_intercept
            oos_predictions[f'Predicted_DeltaDeltaG_{model_name}'] = oos_preds

            mae = mean_absolute_error(oos_data['DeltaDeltaG'], oos_preds)
            oos_r2 = excel_style_r2(oos_data['DeltaDeltaG'], oos_preds)
            oos_metrics.append({'Model': model_name, 'MAE': mae, 'R2': oos_r2})

        # Print metrics and linear equation
        print(f"\nResults for Model {model_name}:")
        print(f"Mean RMSE: {mean_rmse:.3f}, Std RMSE: {std_rmse:.3f}")
        print(f"Mean R²: {mean_r2:.3f}, Mean Adjusted R²: {mean_adjusted_r2:.3f}")
        print_linear_equation(features, avg_coeffs, avg_intercept)

    # Re-order CrossVal_Predictions with Train first, then Test, and add original indices
    cv_predictions.insert(0, 'Original_Index', cv_predictions.index)
    cv_predictions['DataSet'] = pd.Categorical(cv_predictions['DataSet'], categories=["Train", "Test"], ordered=True)
    cv_predictions = cv_predictions.sort_values(by=['DataSet', 'Original_Index'], ascending=[True, True])

    # Ensure the columns are in the correct order
    predicted_columns = [col for col in cv_predictions.columns if col.startswith('Predicted_DeltaDeltaG_')]
    cv_predictions = cv_predictions[['Original_Index', 'Structure', 'Class', 'ee', 'DeltaDeltaG', 'DataSet'] + predicted_columns]

    # Calculate ensemble predictions and metrics for OOS data
    if not selected_models.empty:
        ensemble_predictions = oos_predictions[['Structure', 'Class', 'ee', 'DeltaDeltaG']].copy()
        prediction_columns = [f'Predicted_DeltaDeltaG_{model}' for model in selected_models['Model']]

        # Create a DataFrame of predictions from selected models
        prediction_df = oos_predictions[prediction_columns]

        # Calculate z-scores for each prediction
        row_means = prediction_df.mean(axis=1)
        row_stds = prediction_df.std(axis=1, ddof=1)
        z_scores = (prediction_df.sub(row_means, axis=0)).div(row_stds, axis=0)

        # Mask predictions with z-scores outside the acceptable range (e.g., ±2)
        valid_preds = prediction_df.mask(z_scores.abs() > 2)

        # Calculate weights based on Mean RMSE
        selected_models['Weight'] = 1 / selected_models['Mean RMSE']
        normalized_weights = selected_models['Weight'] / selected_models['Weight'].sum()

        # Apply weights only to valid predictions and compute the weighted average
        weighted_preds = valid_preds.mul(normalized_weights.values, axis=1).sum(axis=1)

        ensemble_predictions['Ensemble_Predicted_DeltaDeltaG'] = weighted_preds
        ensemble_predictions['Prediction_StdDev'] = valid_preds.std(axis=1)

        # Metrics for the ensemble prediction
        ensemble_mae = mean_absolute_error(
            ensemble_predictions['DeltaDeltaG'], 
            ensemble_predictions['Ensemble_Predicted_DeltaDeltaG']
        )
        ensemble_r2 = excel_style_r2(
            ensemble_predictions['DeltaDeltaG'], 
            ensemble_predictions['Ensemble_Predicted_DeltaDeltaG']
        )
        ensemble_metrics = {
            'MAE': ensemble_mae,
            'R2': ensemble_r2
        }
        print("\nEnsemble Prediction Metrics:")
        print(f"MAE: {ensemble_metrics['MAE']:.3f}")
        print(f"R²: {ensemble_metrics['R2']:.3f}")
    else:
        print("No models were selected for OOS predictions.")
        ensemble_predictions = pd.DataFrame()
        ensemble_metrics = {}

    # Save results to Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        cv_predictions.to_excel(writer, sheet_name='CrossVal_Predictions', index=False)
        pd.DataFrame(metrics).to_excel(writer, sheet_name='CrossVal_Metrics', index=False)
        oos_predictions.to_excel(writer, sheet_name='OOS_Predictions', index=False)
        pd.DataFrame(oos_metrics).to_excel(writer, sheet_name='OOS_Metrics', index=False)
        feature_analysis.to_excel(writer, sheet_name='Feature_Analysis', index=False)
        selected_features_analysis.to_excel(writer, sheet_name='Selected_Feature_Analysis', index=False)
        if not ensemble_predictions.empty:
            ensemble_predictions.to_excel(writer, sheet_name='Ensemble_Predictions', index=False)
            pd.DataFrame([ensemble_metrics]).to_excel(writer, sheet_name='Ensemble_Metrics', index=False)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python ensemble_predictions.py <models_file> <training_file> <oos_file> <output_file>")
        sys.exit(1)

    models_file, training_file, oos_file, output_file = sys.argv[1:]
    main_workflow(models_file, training_file, oos_file, output_file)

