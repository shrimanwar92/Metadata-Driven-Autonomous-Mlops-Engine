import pandas as pd
import numpy as np
import json
import os
import sys
import polars as pl

from featurewiz_polars import FeatureWiz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLEANED_DATASET_PATH, GOLD_AUDIT_REPORT, TARGET_COLUMN

MAX_FEATURES = 30

def run_audit():
    print("🌟 Running Feature Intelligence Audit...")

    # 1. Load and Clean names
    df = pd.read_csv(CLEANED_DATASET_PATH)
    df.columns = df.columns.str.strip()
    
    # 2. Identify Target (Strict Case Matching)
    actual_target_col = None
    for col in df.columns:
        if col.lower() == TARGET_COLUMN.lower():
            actual_target_col = col
            break
    
    if not actual_target_col:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset: {df.columns.tolist()}")

    # 3. Determine Problem Type
    unique_targets = df[actual_target_col].nunique()
    is_regression = np.issubdtype(df[actual_target_col].dtype, np.number) and unique_targets > 20
    model_type = "regression" if is_regression else "classification"
    
    print(f"🎯 Target identified as: '{actual_target_col}'")
    print(f"🧠 Problem Mode: {model_type.upper()}")

    # 4. Prepare for FeatureWiz
    # We drop columns that are clearly identifiers to prevent sample size issues
    drop_cols = [col for col in df.columns if 'id' in col.lower() and col != actual_target_col]
    df_clean = df.drop(columns=drop_cols)
    
    # Convert to Polars
    pl_df = pl.from_pandas(df_clean)

    # 5. FeatureWiz Selection
    # Passing the exact string name is critical here
    fw = FeatureWiz(model_type=model_type, corr_limit=0.70, verbose=0)
    
    try:
        # We pass the full Polars DF and the string name of the target
        X_fw, _ = fw.fit_transform(pl_df, actual_target_col)
        
        # Convert back to pandas and ensure target isn't in X
        X_selected = X_fw.to_pandas()
        if actual_target_col in X_selected.columns:
            X_selected = X_selected.drop(columns=[actual_target_col])
            
        selected_features = X_selected.columns.tolist()
        
    except Exception as e:
        print(f"⚠️ FeatureWiz failed: {e}. Falling back to correlation-based selection.")
        # Fallback: simple correlation with target
        correlations = df_clean.select_dtypes(include=[np.number]).corr()[actual_target_col].abs()
        selected_features = correlations.sort_values(ascending=False)[1:MAX_FEATURES+1].index.tolist()

    # 6. Refinement via RFE
    if len(selected_features) > 1:
        X_refine = df_clean[selected_features].fillna(df_clean[selected_features].median(numeric_only=True))
        y_refine = df_clean[actual_target_col]
        
        # Factorize any strings in the selected set for RFE
        for col in X_refine.select_dtypes(include=['object']).columns:
            X_refine[col] = pd.factorize(X_refine[col])[0]

        estimator = RandomForestRegressor(n_estimators=30, n_jobs=-1) if is_regression else RandomForestClassifier(n_estimators=30, n_jobs=-1)
        
        rfe = RFE(estimator, n_features_to_select=min(MAX_FEATURES, len(selected_features)))
        rfe.fit(X_refine, y_refine)
        selected_features = [f for f, s in zip(selected_features, rfe.support_) if s]

    # 7. Finalize Strategy
    strategy = {
        "selected_features": selected_features,
        "base_columns": [col for col in df_clean.columns if col != actual_target_col],
        "target_column": actual_target_col,
        "target_exists": True,
        "model_type": model_type
    }

    with open(GOLD_AUDIT_REPORT, "w") as f:
        json.dump(strategy, f, indent=4)

    print(f"✅ Audit complete. Strategy saved to {GOLD_AUDIT_REPORT}")

if __name__ == "__main__":
    run_audit()