import pandas as pd
import numpy as np
import json
import joblib
import os
import sys

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import (
    CLEANED_DATASET_PATH,
    GOLD_DATASET_PATH,
    JOBLIB_PIPELINE_PATH,
    GOLD_AUDIT_REPORT
)

def generate_features_synchronized(df, base_columns):
    """
    Mirror the logic from the audit script exactly to ensure column consistency.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    # 1. Identify Date and Numeric Columns (Mirroring audit logic)
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    num_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number) and 'date' not in col.lower()]
    cat_cols = [col for col in df.columns if df[col].dtype == 'object' and 'date' not in col.lower()]

    features = pd.DataFrame(index=df.index)

    # 2. Process Dates (dow, month, year)
    for col in date_cols:
        dt = pd.to_datetime(df[col], errors='coerce')
        features[f"{col}_year"] = dt.dt.year
        features[f"{col}_month"] = dt.dt.month
        features[f"{col}_dow"] = dt.dt.dayofweek

    # 3. Temporal Differences
    for i in range(len(date_cols)):
        for j in range(i + 1, len(date_cols)):
            c1, c2 = date_cols[i], date_cols[j]
            d1 = pd.to_datetime(df[c1], errors='coerce')
            d2 = pd.to_datetime(df[c2], errors='coerce')
            features[f"{c2}_minus_{c1}_days"] = (d2 - d1).dt.days

    # 4. Numeric interactions
    for col in num_cols:
        features[col] = df[col]
        
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c1, c2 = num_cols[i], num_cols[j]
            features[f"{c1}_minus_{c2}"] = df[c1] - df[c2]
            features[f"{c1}_ratio_{c2}"] = df[c1] / (df[c2] + 1e-5)

    # 5. Categorical (Factorize)
    for col in cat_cols:
        features[col] = pd.factorize(df[col])[0]

    return features

def execute():
    print("🚀 Running Execution Pipeline...")

    df = pd.read_csv(CLEANED_DATASET_PATH)
    
    with open(GOLD_AUDIT_REPORT, "r") as f:
        strat = json.load(f)

    selected_features = strat["selected_features"]
    base_columns = strat["base_columns"]
    target_col = strat["target_column"]
    target_exists = strat["target_exists"]

    # Generate all possible features
    df_engineered = generate_features_synchronized(df, base_columns)

    # ALIGN: Ensure all columns expected by the audit exist
    for col in base_columns:
        if col not in df_engineered.columns:
            df_engineered[col] = 0 # Fill missing expected columns with 0

    # Subset to base_columns (This ensures X has the exact shape expected)
    X = df_engineered[base_columns].copy()

    # Handle Target
    y = None
    if target_exists and target_col in df.columns:
        y = df[target_col]
        if target_col in X.columns:
            X = X.drop(columns=[target_col])

    # PIPELINE
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    # Fit and transform, maintaining DataFrame structure
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    X = pd.DataFrame(X_scaled, columns=base_columns)

    # FINAL: Filter to the best features selected during Audit
    X_final = X[selected_features].copy()

    if y is not None:
        X_final[target_col] = y.values

    X_final.to_csv(GOLD_DATASET_PATH, index=False)

    joblib.dump({
        "features": selected_features,
        "imputer": imputer,
        "scaler": scaler,
        "base_columns": base_columns
    }, JOBLIB_PIPELINE_PATH)

    print(f"✅ Gold dataset ready with {len(selected_features)} features.")

if __name__ == "__main__":
    execute()