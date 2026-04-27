import numpy as np
import json
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.selection import DropFeatures, DropCorrelatedFeatures

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import DATASET_PATH, PRE_CLEAN_AUDIT_REPORT, CLEANED_DATASET_PATH

def handle_outliers(df, numeric_vars):
    # Ensure we don't use IDs for outlier detection
    numeric_vars = [v for v in numeric_vars if v in df.columns]
    if not numeric_vars: return df
    
    print(f"  🔍 Detecting spatial outliers on: {numeric_vars}")
    iso = IsolationForest(contamination=0.05, random_state=42)
    clean_numeric = df[numeric_vars].fillna(df[numeric_vars].median())
    outlier_preds = iso.fit_predict(clean_numeric)
    return df[outlier_preds == 1].copy()

def run_clustering_cleaning():
    print("🧹 Phase 3: Cleaning Silver Layer (Generic Logic)...")
    df = pd.read_csv(DATASET_PATH)
    with open(PRE_CLEAN_AUDIT_REPORT, 'r') as f:
        audit = json.load(f)

    subject_id = audit['config']['subject_id']
    
    # 1. Pipeline: Pruning & Imputation
    # We drop weak features and garbage IDs, but PROTECT the subject_id
    to_drop = [c for c in (audit['drop_cols'] + audit['weak_features']) 
               if c in df.columns and c != subject_id]
    
    steps = []
    if to_drop:
        steps.append(('noise_pruning', DropFeatures(features_to_drop=to_drop)))
    
    # Identify columns for imputer (excluding subject_id)
    num_vars = df.select_dtypes(include=np.number).columns.tolist()
    final_num = [v for v in num_vars if v not in to_drop and v != subject_id]
    
    cat_vars = df.select_dtypes(exclude=np.number).columns.tolist()
    final_cat = [v for v in cat_vars if v not in to_drop and v != subject_id]

    if final_num:
        steps.append(('num_impute', MeanMedianImputer(variables=final_num)))
        # Multi-collinearity check: Safer alternative to VIF for automated pipelines
        if len(final_num) >= 2:
            steps.append(('corr_filter', DropCorrelatedFeatures(threshold=0.90, variables=final_num)))
    
    if final_cat:
        steps.append(('cat_impute', CategoricalImputer(variables=final_cat, ignore_format=True)))

    # 2. Transform
    df_silver = Pipeline(steps).fit_transform(df)

    # 3. Spatial Outlier Removal (Ignore ID)
    numeric_for_outliers = df_silver.select_dtypes(include=[np.number]).columns.tolist()
    if subject_id in numeric_for_outliers:
        numeric_for_outliers.remove(subject_id)
        
    df_silver = handle_outliers(df_silver, numeric_for_outliers)
    
    # 4. Final Cleanup: Ensure subject_id has no NaNs (Clustering anchors must be solid)
    if subject_id in df_silver.columns:
        df_silver = df_silver.dropna(subset=[subject_id])

    # 5. Save
    os.makedirs(os.path.dirname(CLEANED_DATASET_PATH), exist_ok=True)
    df_silver.to_csv(CLEANED_DATASET_PATH, index=False)
    print(f"✅ Silver Data Saved. Subject Anchor '{subject_id}' preserved.")

if __name__ == "__main__":
    run_clustering_cleaning()