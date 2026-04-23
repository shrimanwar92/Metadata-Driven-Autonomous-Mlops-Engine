import pandas as pd
import numpy as np
import json
import sys
import os
from scipy.stats import skew
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLEANED_DATASET_PATH, GOLD_AUDIT_REPORT, TARGET_COLUMN

def run_gold_audit_supervised(target_col, task='regression'):
    print(f"🌟 Phase 5: Generating Supervised Strategy for [{target_col}]...")
    df = pd.read_csv(CLEANED_DATASET_PATH)
    
    # --- AUTO-DETECTION ---
    if df[target_col].dtype == 'object' or df[target_col].nunique() < 15:
        task = 'classification'

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    num_vars = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_vars = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- TARGET AWARENESS (NEW) ---
    target_skew = skew(y) if task == 'regression' else 0
    apply_log_target = bool(task == "regression" and abs(target_skew) > 1)

    strategy = {
        "metadata": {
            "target": target_col,
            "task": task,
            "apply_log_target": apply_log_target,
            "target_skew": float(target_skew),
            "expected_features": list(X.columns)
        },
        "categorical_encoding": {
            "mean_encoding_cols": cat_vars,
            "rare_label_cols": cat_vars
        },
        "numerical_transformations": {
            "yeo_johnson": [col for col in num_vars if abs(skew(X[col].dropna())) > 0.75],
            "scaling_method": "StandardScaler"
        },
        "feature_selection": {
            "drop_features": []
        }
    }

    # --- MI Calculation ---
    X_audit = X.copy()
    for col in cat_vars:
        X_audit[col] = pd.factorize(X_audit[col])[0]
    
    y_audit = pd.factorize(y)[0] if task == 'classification' else y

    mi_func = mutual_info_classif if task == 'classification' else mutual_info_regression
    mi_scores = mi_func(X_audit, y_audit)
    mi_series = pd.Series(mi_scores, index=X.columns)

    # --- IMPROVED THRESHOLD ---
    strategy["feature_selection"]["drop_features"] = mi_series[mi_series < 0.01].index.tolist()

    with open(GOLD_AUDIT_REPORT, 'w') as f:
        json.dump(strategy, f, indent=4)
    
    print(f"✅ Audit Complete. Target skew: {target_skew:.2f}, Log Transform: {apply_log_target}")

if __name__ == "__main__":
    run_gold_audit_supervised(target_col=TARGET_COLUMN)