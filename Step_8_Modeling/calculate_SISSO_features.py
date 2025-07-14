#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import os
import re

UNARY_OPS = ['sqrt', 'cbrt', 'ln', 'inv', 'exp']
BINARY_OPS = ['add', 'sub', 'mul', 'div']

def parse_unary(op, x):
    if op == 'sqrt':
        return np.sqrt(x)
    elif op == 'cbrt':
        return np.cbrt(x)
    elif op == 'ln':
        return np.log(x)
    elif op == 'inv':
        return 1 / x
    elif op == 'exp':
        return np.exp(x)
    else:
        raise ValueError(f"Unknown unary operator: {op}")

def parse_binary(x1, op, x2):
    if op == 'add':
        return x1 + x2
    elif op == 'sub':
        return x1 - x2
    elif op == 'mul':
        return x1 * x2
    elif op == 'div':
        return x1 / x2
    else:
        raise ValueError(f"Unknown binary operator: {op}")

def recursive_evaluate(expr, df):
    # Powers
    if expr.endswith('_squared'):
        return recursive_evaluate(expr[:-8], df) ** 2
    if expr.endswith('_cubed'):
        return recursive_evaluate(expr[:-6], df) ** 3

    # Unary op wrapping full expression: e.g., inv_x1subx1574
    for op in UNARY_OPS:
        if expr.startswith(op + '_'):
            inner_expr = expr[len(op)+1:]
            return parse_unary(op, recursive_evaluate(inner_expr, df))
        if expr.startswith(op + 'x') and re.match(rf'{op}x\d+$', expr):  # expx238
            return parse_unary(op, df[expr[len(op):]])

    # Binary op: x1addx2 or x1_add_x2
    match = re.match(r'^(x\d+)(add|sub|mul|div)(x\d+)$', expr)
    if match:
        x1, op, x2 = match.groups()
        return parse_binary(df[x1], op, df[x2])
    for op in BINARY_OPS:
        if f"_{op}_" in expr:
            parts = expr.split(f"_{op}_")
            if len(parts) == 2:
                return parse_binary(recursive_evaluate(parts[0], df), op, recursive_evaluate(parts[1], df))

    # Fallback to base feature
    if expr in df.columns:
        return df[expr]

    raise ValueError(f"Expression '{expr}' could not be parsed")

# Main logic
def main(sisso_file, new_file):
    print(f"[INFO] Reading SISSO file: {sisso_file}")
    print(f"[INFO] Reading new input file: {new_file}")
    df_sisso = pd.read_excel(sisso_file)
    df_new = pd.read_excel(new_file)

    metadata_cols = ["Structure", "Class", "ee", "DeltaDeltaG"]
    all_cols = df_sisso.columns.tolist()
    original_features = [col for col in all_cols if re.fullmatch(r'x\d+', col)]
    augmented_features = [col for col in all_cols if col not in metadata_cols + original_features]

    print(f"[INFO] Original features: {len(original_features)}")
    print(f"[INFO] Augmented features: {len(augmented_features)}")

    for feat in original_features:
        if feat not in df_new.columns:
            raise ValueError(f"Missing original feature '{feat}' in second input file.")

    df_new_subset = df_new[metadata_cols + original_features].copy()

    print("[VERIFY] Recomputing augmented features from SISSO input file (using only original features)...")
    for fname in augmented_features:
        recomputed = recursive_evaluate(fname, df_sisso[original_features])
        recomputed = np.clip(recomputed, -1e6, 1e6)
        original = df_sisso[fname]
        if not np.allclose(recomputed, original, rtol=1e-5, equal_nan=True):
            raise ValueError(f"Mismatch in recomputed feature: {fname}")
    print("[PASS] All augmented features successfully verified using only base features")

    print("[PROCESS] Computing augmented features for new input file...")
    recomputed_augmented = {
        fname: np.clip(recursive_evaluate(fname, df_new), -1e6, 1e6)
        for fname in augmented_features
    }

    augmented_df = pd.DataFrame(recomputed_augmented, index=df_new.index)
    df_final = pd.concat([df_new_subset, augmented_df], axis=1)

    output_file = os.path.splitext(new_file)[0] + "_SISSO.xlsx"
    df_final.to_excel(output_file, index=False)
    print(f"[DONE] Output written to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculate_SISSO_features.py <sisso_output.xlsx> <new_input.xlsx>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])


