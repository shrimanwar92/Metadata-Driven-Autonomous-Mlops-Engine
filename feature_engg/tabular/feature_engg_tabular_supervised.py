import pandas as pd
import numpy as np
import json
import joblib
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from feature_engine.encoding import MeanEncoder, RareLabelEncoder
from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.selection import DropFeatures, DropConstantFeatures, SmartCorrelatedSelection
from feature_engine.wrappers import SklearnTransformerWrapper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import (
    CLEANED_DATASET_PATH, GOLD_DATASET_PATH,
    GOLD_AUDIT_REPORT, JOBLIB_PIPELINE_PATH
)


# =========================================================
# 🔧 PIPELINE BUILDER
# =========================================================
def build_dynamic_pipeline(strat, scaler_type="standard"):

    steps = []

    dropped_features = strat["feature_selection"]["drop_features"] or []

    def filter_dropped(cols):
        return [c for c in cols if c not in dropped_features]

    # -----------------------------------------------------
    # 🧹 DROP INITIAL
    # -----------------------------------------------------
    if dropped_features:
        steps.append(('drop_initial', DropFeatures(features_to_drop=dropped_features)))

    # -----------------------------------------------------
    # 🎯 CATEGORICAL HANDLING
    # -----------------------------------------------------
    one_hot_cols = filter_dropped(strat["categorical_encoding"].get("one_hot_cols", []))
    mean_cols = filter_dropped(strat["categorical_encoding"].get("mean_encoding_cols", []))
    rare_cols = filter_dropped(strat["categorical_encoding"].get("rare_label_cols", []))

    # --- Rare Encoding FIRST ---
    if rare_cols:
        steps.append(('rare', RareLabelEncoder(
            tol=0.05,
            n_categories=2,
            replace_with='Rare',
            variables=rare_cols
        )))

    # --- Mean Encoding ---
    if mean_cols:
        steps.append(('mean_enc', MeanEncoder(
            variables=mean_cols,
            smoothing=0.3
        )))

    # --- One Hot Encoding (via ColumnTransformer wrapper) ---
    if one_hot_cols:
        from feature_engine.encoding import OneHotEncoder as FEOneHotEncoder
        steps.append(('one_hot', FEOneHotEncoder(
            variables=one_hot_cols,
            drop_last=False
        )))

    # -----------------------------------------------------
    # 🔢 NUMERICAL
    # -----------------------------------------------------
    yeo_cols = filter_dropped(strat["numerical_transformations"]["yeo_johnson"])
    if yeo_cols:
        steps.append(('yeo', YeoJohnsonTransformer(variables=yeo_cols)))

    # -----------------------------------------------------
    # 🧹 NOISE REMOVAL
    # -----------------------------------------------------
    steps.append(('drop_constant', DropConstantFeatures(tol=0.98)))
    steps.append(('smart_corr', SmartCorrelatedSelection(
        threshold=0.85,
        selection_method="variance"
    )))

    # -----------------------------------------------------
    # 📏 SCALING
    # -----------------------------------------------------
    if scaler_type == "standard":
        steps.append(('scaler', SklearnTransformerWrapper(StandardScaler())))
    elif scaler_type == "robust":
        steps.append(('scaler', SklearnTransformerWrapper(RobustScaler())))

    if not steps:
        steps.append(('identity', FunctionTransformer(lambda x: x)))

    return Pipeline(steps)


# =========================================================
# 🧾 SCHEMA VALIDATION
# =========================================================
def validate_schema(df, expected_cols):
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"❌ Schema Validation Failed. Missing columns: {missing}")


# =========================================================
# 🚀 MAIN EXECUTION
# =========================================================
def run_autonomous_supervised_v2():
    print("🚀 Phase 6: Executing Supervised Feature Engineering v3...")

    df = pd.read_csv(CLEANED_DATASET_PATH)

    with open(GOLD_AUDIT_REPORT, 'r') as f:
        strat = json.load(f)

    target = strat["metadata"]["target"]
    task = strat["metadata"]["task"]

    X = df.drop(columns=[target])
    y = df[target]

    # -----------------------------------------------------
    # 🧾 SCHEMA CONTRACT
    # -----------------------------------------------------
    validate_schema(X, strat["metadata"]["expected_features"])

    # -----------------------------------------------------
    # 🎯 TARGET TRANSFORMATION
    # -----------------------------------------------------
    if task == "regression" and strat["metadata"].get("apply_log_target"):
        print("📉 Applying log1p transformation to target")
        y = np.log1p(y)

    # -----------------------------------------------------
    # 🔠 TARGET ENCODING
    # -----------------------------------------------------
    le = None
    if task == 'classification' and y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # -----------------------------------------------------
    # 🔧 PIPELINE
    # -----------------------------------------------------
    pipe = build_dynamic_pipeline(strat)
    print("🔧 Pipeline Steps:", [name for name, _ in pipe.steps])

    X_gold = pipe.fit_transform(X, y)

    # Convert to DataFrame if needed
    if not isinstance(X_gold, pd.DataFrame):
        X_gold = pd.DataFrame(X_gold)

    # -----------------------------------------------------
    # 🧪 SANITY CHECK
    # -----------------------------------------------------
    if X_gold.shape[1] < 2:
        raise ValueError("❌ Too few features after engineering")

    # -----------------------------------------------------
    # 📊 LINEAGE
    # -----------------------------------------------------
    feature_lineage = {
        "original_features": list(X.columns),
        "gold_features": list(X_gold.columns),
        "removed_during_engg": list(set(X.columns) - set(X_gold.columns)),
        "pipeline_steps": [name for name, _ in pipe.steps]
    }

    # -----------------------------------------------------
    # 💾 SAVE PIPELINE
    # -----------------------------------------------------
    bundle = {
        "pipeline": pipe,
        "label_encoder": le,
        "lineage": feature_lineage,
        "metadata": strat["metadata"]
    }

    joblib.dump(bundle, JOBLIB_PIPELINE_PATH)

    # -----------------------------------------------------
    # 💾 SAVE GOLD DATA
    # -----------------------------------------------------
    X_gold[target] = y
    X_gold.to_csv(GOLD_DATASET_PATH, index=False)

    print(f"✨ Gold Dataset Ready | Features: {X_gold.shape[1]}")
    print(f"📉 Removed Features: {len(feature_lineage['removed_during_engg'])}")


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    run_autonomous_supervised_v2()