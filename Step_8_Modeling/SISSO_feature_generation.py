#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import re
from sklearn.metrics import r2_score
from sklearn.model_selection import permutation_test_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Helper functions ---

def sanitize_token(token):
    """Sanitize individual components of feature names."""
    token = str(token)
    token = token.encode("ascii", "ignore").decode("ascii")
    token = re.sub(r'[^a-zA-Z0-9]', '', token)  # Remove anything non-alphanumeric
    return token

def make_feature_name(*tokens):
    return '_'.join(sanitize_token(t) for t in tokens)

def is_valid_feature(series):
    return np.all(np.isfinite(series)) and not np.all(series == series.iloc[0])

def target_aware_collinearity_filter(df, y, colin_cut=0.7):
    print(f"[Filtering] Checking collinearity (cutoff: {colin_cut})...")
    corr = df.corr().abs()
    to_drop = set()
    for i, f1 in enumerate(df.columns):
        if f1 in to_drop:
            continue
        for j, f2 in enumerate(df.columns):
            if f1 == f2 or f2 in to_drop:
                continue
            if corr.loc[f1, f2] >= colin_cut:
                x1 = df[f1].values.reshape(-1, 1)
                x2 = df[f2].values.reshape(-1, 1)
                mask1 = np.isfinite(x1.flatten()) & np.isfinite(y)
                mask2 = np.isfinite(x2.flatten()) & np.isfinite(y)
                r2_f1 = LinearRegression().fit(x1[mask1], y[mask1]).score(x1[mask1], y[mask1]) if np.sum(mask1) > 1 else 0
                r2_f2 = LinearRegression().fit(x2[mask2], y[mask2]).score(x2[mask2], y[mask2]) if np.sum(mask2) > 1 else 0
                if r2_f1 >= r2_f2:
                    to_drop.add(f2)
                else:
                    to_drop.add(f1)
                    break
    retained = df.columns.difference(to_drop)
    print(f"[Filtering] Dropping {len(to_drop)} features due to collinearity.")
    print(f"[Filtering] Retaining {len(retained)} features after collinearity filtering.")
    return df.drop(columns=list(to_drop))

def get_pvalue(x_col, y_col, n_permutations):
    mask = np.isfinite(x_col) & np.isfinite(y_col)
    if np.sum(mask) < 2:
        return 1.0
    x = np.clip(x_col[mask], -1e6, 1e6).reshape(-1, 1)
    y = np.clip(y_col[mask], -1e6, 1e6)
    _, _, pval = permutation_test_score(
        LinearRegression(), x, y, n_permutations=n_permutations, n_jobs=-1, random_state=42
    )
    return pval

def relative_permutation_filter(df, parents_dict, y, n_permutations):
    print("[Filtering] Starting permutation testing (relative)...")
    keep_cols = []
    y = pd.Series(y).reset_index(drop=True)
    for i, col in enumerate(df.columns):
        if i % 10 == 0:
            print(f"  Permutation testing {i}/{len(df.columns)}")
        x_col = df[col].reset_index(drop=True).values
        parent_pvals = []
        for parent in parents_dict.get(col, []):
            parent_vals = parent.reset_index(drop=True).values
            parent_pvals.append(get_pvalue(parent_vals, y.values, n_permutations))
        child_pval = get_pvalue(x_col, y.values, n_permutations)
        if len(parent_pvals) == 0 or child_pval <= min(parent_pvals):
            keep_cols.append(col)
    print(f"[Filtering] Features retained after filtering: {len(keep_cols)}")
    return df[keep_cols]

def filter_features(df, y, colin_cut=0.7, n_permutations=1000):
    print("[Filtering] Starting filtering of augmented features from iteration 1...")
    df = df.loc[:, [col for col in df.columns if is_valid_feature(df[col])]]
    df = target_aware_collinearity_filter(df, y, colin_cut)
    parents_dict = {}
    for col in df.columns:
        parts = col.split('_')
        if len(parts) >= 3 and parts[-2] in ['add', 'sub', 'mul', 'div']:
            base1 = '_'.join(parts[:-2])
            base2 = parts[-1]
            if base1 in df.columns and base2 in df.columns:
                parents_dict[col] = [df[base1], df[base2]]
        elif parts[0] in ['sqrt', 'cbrt', 'ln', 'inv', 'exp']:
            base = '_'.join(parts[1:])
            if base in df.columns:
                parents_dict[col] = [df[base]]
        elif parts[-1] in ['squared', 'cubed']:
            base = '_'.join(parts[:-1])
            if base in df.columns:
                parents_dict[col] = [df[base]]
    return relative_permutation_filter(df, parents_dict, y, n_permutations)

def apply_univariate(df, verbose=False):
    new_feats = {}
    for i, col in enumerate(df.columns):
        x = df[col]
        if verbose and i % 10 == 0:
            print(f"  [UNIV] Processing {i}/{len(df.columns)}")
        try: new_feats[make_feature_name("sqrt", col)] = np.sqrt(x)
        except: pass
        try: new_feats[make_feature_name("cbrt", col)] = np.cbrt(x)
        except: pass
        try: new_feats[make_feature_name("ln", col)] = np.log(x)
        except: pass
        try: new_feats[make_feature_name("inv", col)] = 1 / x
        except: pass
        try: new_feats[make_feature_name("exp", col)] = np.exp(x)
        except: pass
        try: new_feats[make_feature_name(col, "squared")] = x ** 2
        except: pass
        try: new_feats[make_feature_name(col, "cubed")] = x ** 3
        except: pass
    return pd.DataFrame({k: v for k, v in new_feats.items() if is_valid_feature(v)})

def apply_bivariate(df, verbose=False):
    new_feats = {}
    cols = df.columns
    total = len(cols)**2
    counter = 0
    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            counter += 1
            if verbose and counter % 100 == 0:
                print(f"  [BIV] Processing {counter}/{total} ({(counter/total)*100:.1f}%)")
            x1 = df[col1]
            x2 = df[col2]
            try: new_feats[make_feature_name(col1, "add", col2)] = x1 + x2
            except: pass
            try: new_feats[make_feature_name(col1, "sub", col2)] = x1 - x2
            except: pass
            try: new_feats[make_feature_name(col1, "mul", col2)] = x1 * x2
            except: pass
            try: new_feats[make_feature_name(col1, "div", col2)] = x1 / x2
            except: pass
    return pd.DataFrame({k: v for k, v in new_feats.items() if is_valid_feature(v)})

# --- Main function ---

def command_center(input_file, colin_cut=0.7, n_permutations=1000, verbose=True):
    print(f"Reading input file: {input_file}")
    df = pd.read_excel(input_file)
    metadata_cols = ["Structure", "Class", "ee", "DeltaDeltaG"]
    meta = df[metadata_cols]
    original_features = df.drop(columns=metadata_cols)
    original_features.columns = [sanitize_token(col) for col in original_features.columns]
    y = meta["DeltaDeltaG"].values

    print("\n=== Iteration 1: Expanding original features ===")
    univ_1 = apply_univariate(original_features, verbose)
    biv_1 = apply_bivariate(original_features, verbose)
    iter1_augmented = pd.concat([univ_1, biv_1], axis=1)

    filtered_iter1 = filter_features(iter1_augmented, y, colin_cut, n_permutations)

    print("\n=== Iteration 2: Expanding filtered features + original features ===")
    iter2_input = pd.concat([original_features, filtered_iter1], axis=1)
    univ_2 = apply_univariate(iter2_input, verbose)
    biv_2 = apply_bivariate(iter2_input, verbose)
    iter2_augmented = pd.concat([univ_2, biv_2], axis=1)

    # NEW: Remove duplicate columns before passing to Boruta
    all_augmented = pd.concat([filtered_iter1, iter2_augmented], axis=1)
    duplicated = all_augmented.columns[all_augmented.columns.duplicated()]
    if len(duplicated) > 0:
        print(f"[WARNING] Found {len(duplicated)} duplicated feature names before Boruta:")
        for name in duplicated.unique():
            count = (all_augmented.columns == name).sum()
            print(f" - {name} appears {count} times")
        all_augmented = all_augmented.loc[:, ~all_augmented.columns.duplicated()]

    print(f"[Boruta] Total features provided to Boruta: {all_augmented.shape[1]}")

    print("\n=== Applying Boruta feature selection ===")
    boruta_df = all_augmented
    boruta_df = boruta_df.loc[:, boruta_df.apply(lambda col: np.all(np.isfinite(col)), axis=0)]
    boruta_df = boruta_df.clip(lower=-1e6, upper=1e6)

    if boruta_df.shape[1] == 0:
        print("[Boruta] No features to select from. Skipping Boruta step.")
        selected_features = []
    else:
        rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=42)
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42, perc=75, max_iter=100)
        feat_selector.fit(boruta_df.values, y)
        selected_features = [boruta_df.columns[i] for i, x in enumerate(feat_selector.support_) if x]
        print(f"\nSelected {len(selected_features)} features from Boruta")

    final_features = pd.concat([original_features, boruta_df[selected_features]], axis=1)
    out_df = pd.concat([meta, final_features], axis=1)
    output_filename = input_file.replace('.xlsx', '_SISSO_Boruta.xlsx')
    out_df.to_excel(output_filename, index=False)
    print(f"\nFeature expansion complete. Output written to: {output_filename}")

# --- Entrypoint ---

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command_center(sys.argv[1])
    else:
        print("Please specify the input Excel file")
        sys.exit(1)

