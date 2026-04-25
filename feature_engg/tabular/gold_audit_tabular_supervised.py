import pandas as pd
import numpy as np
import json
import sys
import os
from scipy.stats import skew
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import CLEANED_DATASET_PATH, GOLD_AUDIT_REPORT, TARGET_COLUMN, PROFILER_REPORT_PATH


# =========================================================
# 🔍 ENCODING FROM PROFILER
# =========================================================
def encoding_from_profile(profile, df):
    variables = profile.get("variables", {})

    one_hot_cols = []
    mean_encoding_cols = []
    rare_label_cols = []
    drop_features = []

    n_rows = len(df)

    for col, stats in variables.items():

        if col not in df.columns:
            continue

        col_type = stats.get("type", "").lower()

        if col_type not in ["categorical", "text"]:
            continue

        nunique = stats.get("n_distinct", df[col].nunique())
        p_distinct = stats.get("p_distinct", nunique / n_rows)

        # 🔴 ID detection
        if p_distinct > 0.95:
            drop_features.append(col)
            continue

        # 🟢 LOW CARDINALITY → OneHot
        if nunique <= 5:
            one_hot_cols.append(col)

        # 🟡 MID CARDINALITY → Rare + OneHot
        elif nunique <= 20:
            rare_label_cols.append(col)
            one_hot_cols.append(col)

        # 🔵 HIGH CARDINALITY → Mean Encoding
        else:
            rare_label_cols.append(col)
            mean_encoding_cols.append(col)

    return {
        "one_hot_cols": one_hot_cols,
        "mean_encoding_cols": mean_encoding_cols,
        "rare_label_cols": rare_label_cols,
        "drop_features": drop_features
    }


# =========================================================
# 🚀 MAIN AUDIT
# =========================================================
def run_gold_audit_supervised(target_col, task='regression'):
    print(f"🌟 Phase 5: Generating Intelligent Supervised Strategy for [{target_col}]...")

    df = pd.read_csv(CLEANED_DATASET_PATH)

    # --- TASK DETECTION ---
    if df[target_col].dtype == 'object' or df[target_col].nunique() < 15:
        task = 'classification'

    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_vars = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_vars = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # =====================================================
    # 📊 LOAD PROFILER
    # =====================================================
    with open(PROFILER_REPORT_PATH, "r") as f:
        profile = json.load(f)

    encoding_strategy = encoding_from_profile(profile, X)

    one_hot_cols = encoding_strategy["one_hot_cols"]
    mean_encoding_cols = encoding_strategy["mean_encoding_cols"]
    rare_label_cols = encoding_strategy["rare_label_cols"]
    drop_features = encoding_strategy["drop_features"]

    # =====================================================
    # 🧠 TARGET AWARENESS
    # =====================================================
    target_skew = skew(y) if task == 'regression' else 0
    apply_log_target = bool(task == "regression" and abs(target_skew) > 1)

    # =====================================================
    # 🔢 NUMERICAL TRANSFORMATION
    # =====================================================
    yeo_cols = [
        col for col in num_vars
        if abs(skew(X[col].dropna())) > 0.75
    ]

    # =====================================================
    # 🔥 FEATURE SELECTION V2
    # =====================================================
    X_audit = X.copy()

    # Encode categorical for MI
    for col in cat_vars:
        X_audit[col] = pd.factorize(X_audit[col])[0]

    y_audit = pd.factorize(y)[0] if task == 'classification' else y

    mi_func = mutual_info_classif if task == 'classification' else mutual_info_regression
    mi_scores = mi_func(X_audit.fillna(0), y_audit)

    mi_series = pd.Series(mi_scores, index=X.columns)

    # --- Adaptive threshold ---
    mi_threshold = max(0.01, mi_series.quantile(0.15))
    weak_features = mi_series[mi_series < mi_threshold].index.tolist()

    # -----------------------------------------------------
    # 🧠 SAFEGUARDS
    # -----------------------------------------------------

    # Protect categorical used in encoding
    protected_cat = set(one_hot_cols + mean_encoding_cols)

    # Protect ALL numeric features
    protected_num = set(num_vars)

    # Protect moderate MI (avoid instability)
    moderate_features = set(
        mi_series[(mi_series >= mi_threshold) & (mi_series < mi_threshold * 2)].index
    )

    # Missing-aware protection
    missing_ratio = X.isnull().mean()
    missing_important = set(
        col for col in X.columns
        if missing_ratio[col] > 0.2 and mi_series[col] > 0
    )

    # -----------------------------------------------------
    # FINAL DROP LOGIC
    # -----------------------------------------------------
    for col in weak_features:

        if col in protected_num:
            continue

        if col in protected_cat:
            continue

        if col in moderate_features:
            continue

        if col in missing_important:
            continue

        if col not in drop_features:
            drop_features.append(col)

    # =====================================================
    # 📦 FINAL STRATEGY
    # =====================================================
    strategy = {
        "metadata": {
            "target": target_col,
            "task": task,
            "apply_log_target": apply_log_target,
            "target_skew": float(target_skew),
            "expected_features": list(X.columns)
        },
        "categorical_encoding": {
            "one_hot_cols": one_hot_cols,
            "mean_encoding_cols": mean_encoding_cols,
            "rare_label_cols": rare_label_cols
        },
        "numerical_transformations": {
            "yeo_johnson": yeo_cols,
            "scaling_method": "StandardScaler"
        },
        "feature_selection": {
            "drop_features": list(set(drop_features))
        }
    }

    # =====================================================
    # 💾 SAVE
    # =====================================================
    with open(GOLD_AUDIT_REPORT, 'w') as f:
        json.dump(strategy, f, indent=4)

    print("✅ Intelligent Audit Complete")
    print(f"📊 One-hot: {len(one_hot_cols)} | Mean-enc: {len(mean_encoding_cols)}")
    print(f"📉 Dropped: {len(drop_features)} features")
    print(f"🧠 MI Threshold: {round(mi_threshold, 4)}")


# =====================================================
# ENTRY
# =====================================================
if __name__ == "__main__":
    run_gold_audit_supervised(target_col=TARGET_COLUMN)