import numpy as np
import json
import pandas as pd
import os
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.selection import DropFeatures, DropCorrelatedFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import (
    DATASET_PATH, 
    YDATA_REPORT_JSON, 
    PRE_CLEAN_AUDIT_REPORT, 
    CLEANED_DATASET_PATH
)

def handle_outliers(df, numeric_vars):
    """
    Critical for Clustering: Removes anomalous points that distort centroids.
    """
    # FIX: Ensure we only attempt to use variables actually present in df
    numeric_vars = [v for v in numeric_vars if v in df.columns]
    
    if not numeric_vars:
        return df
    
    print(f"  🔍 Detecting and removing spatial outliers using: {numeric_vars}")
    iso = IsolationForest(contamination=0.05, random_state=42)
    
    # Rest of the function remains the same...
    clean_numeric = df[numeric_vars].fillna(df[numeric_vars].median())
    outlier_preds = iso.fit_predict(clean_numeric)
    
    initial_len = len(df)
    df = df[outlier_preds == 1].reset_index(drop=True)
    print(f"  ✨ Outlier Filter: Removed {initial_len - len(df)} anomalous records.")
    return df

def handle_duplicates(df, id_cols):
    """Removes duplicate rows that would artificially inflate cluster density."""
    signal_cols = [c for c in df.columns if c not in id_cols]
    initial_len = len(df)
    df = df.drop_duplicates(subset=signal_cols).reset_index(drop=True)
    print(f"  ✨ Deduplication: Removed {initial_len - len(df)} redundant records.")
    return df

def _calculate_vif(df, vars, threshold):
    """Iterative VIF to prevent redundant dimensions from dominating distance math."""
    X = df[vars].fillna(df[vars].median()).loc[:, df[vars].nunique() > 1]
    dropped = []
    
    while X.shape[1] > 1:
        X_c = sm.add_constant(X)
        try:
            vif_s = pd.Series(
                [variance_inflation_factor(X_c.values, i) for i in range(X_c.shape[1])], 
                index=X_c.columns
            ).drop('const')
            
            if vif_s.max() > threshold:
                worst = vif_s.idxmax()
                dropped.append(worst)
                X = X.drop(columns=[worst])
                print(f"  🔥 High VIF (Redundancy) dropped: {worst}")
            else: 
                break
        except: 
            break
            
    return dropped

def autonomous_clustering_cleaning():
    print("\n🛠️ Phase 3: Executing Autonomous Cleaning (Clustering Mode)...")
    df = pd.read_csv(DATASET_PATH)
    
    with open(YDATA_REPORT_JSON, 'r') as f:
        report = json.load(f)
    with open(PRE_CLEAN_AUDIT_REPORT, 'r') as f:
        audit = json.load(f)

    # Use thresholds from clustering audit
    VIF_THRESHOLD = audit['config']['vif_threshold']
    VARIANCE_THRESHOLD = audit['config'].get('variance_threshold', 0.001)
    
    to_drop = list(set(audit['id_cols'] + audit['weak_features']))
    num_vars, cat_vars = [], []
    n_total = report['table']['n']

    # 1. Variable Classification
    for col, stats in report['variables'].items():
        if col in to_drop: continue
        if (stats.get('n_missing', 0) / n_total) > 0.70:
            to_drop.append(col)
        elif stats.get('type') == 'Numeric': 
            num_vars.append(col)
        else: 
            cat_vars.append(col)

    # 2. VIF Filter (No Target required)
    active_num = [c for c in num_vars if c not in to_drop]
    if len(active_num) > 1:
        to_drop.extend(_calculate_vif(df, active_num, VIF_THRESHOLD))

    # 3. Pipeline Execution
    final_num = [c for c in num_vars if c not in to_drop]
    final_cat = [c for c in cat_vars if c not in to_drop]
    
    steps = []

    if to_drop:
        # Filter to_drop to only include columns actually present in the current df
        actual_to_drop = [c for c in to_drop if c in df.columns]
        if actual_to_drop:
            steps.append(('noise_pruning', DropFeatures(features_to_drop=actual_to_drop)))
    
    if final_num:
        steps.append(('num_impute', MeanMedianImputer(variables=final_num)))
        if len(final_num) >= 2:
            # Stricter correlation filter for clustering to ensure distinct dimensions
            steps.append(('corr_filter', DropCorrelatedFeatures(threshold=0.85, variables=final_num)))
    
    if final_cat:
        steps.append(('cat_impute', CategoricalImputer(variables=final_cat, ignore_format=True)))

    df_silver = Pipeline(steps).fit_transform(df)

    # 4. Clustering-Specific Final Logic
    # Outliers must be removed AFTER imputation so Isolation Forest has full data
    surviving_numerics = df_silver.select_dtypes(include=[np.number]).columns.tolist()
    df_silver = handle_outliers(df_silver, surviving_numerics)
    df_silver = handle_duplicates(df_silver, audit['id_cols'])
    
    os.makedirs(os.path.dirname(CLEANED_DATASET_PATH), exist_ok=True)
    df_silver.to_csv(CLEANED_DATASET_PATH, index=False)
    
    print(f"✅ Silver Layer (Clustering) Created: {CLEANED_DATASET_PATH}")
    print(f"📊 Final Shape: {df_silver.shape} | Variables: {df_silver.columns.tolist()}")
    return df_silver

if __name__ == "__main__":
    autonomous_clustering_cleaning()