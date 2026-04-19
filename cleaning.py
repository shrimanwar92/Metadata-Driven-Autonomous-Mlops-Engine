import numpy as np
import json
import pandas as pd
import os
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.selection import DropFeatures, DropCorrelatedFeatures
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from constants import DATASET_PATH, YDATA_REPORT_JSON, PRE_CLEAN_AUDIT_REPORT, CLEANED_DATASET_PATH, TARGET_COLUMN, VIF_THRESHOLD, PPS_THRESHOLD

def get_pps_filter(df, num_vars, target_col, threshold=0.02):
    """
    Identifies weak features while ensuring at least one 'Survivor' remains.
    """
    if not num_vars: return []
    
    X = df[num_vars].fillna(df[num_vars].median())
    y = df[target_col]
    
    is_clf = y.nunique() < 20 or y.dtype == 'object'
    model = RandomForestClassifier(n_estimators=50) if is_clf else RandomForestRegressor(n_estimators=50)
    
    model.fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=3, random_state=42)
    
    # Create a mapping of feature: importance
    importance_map = dict(zip(num_vars, result.importances_mean))
    
    # Sort features: Best performing first
    sorted_features = sorted(importance_map, key=importance_map.get, reverse=True)
    best_feature = sorted_features[0]
    
    # Drop features below threshold, but NEVER drop the best_feature
    weak_features = [f for f in num_vars if importance_map[f] < threshold and f != best_feature]
    
    for col in weak_features:
        print(f"  📉 PPS Filter: Dropping '{col}' (Low predictive signal)")
        
    return weak_features

def handle_duplicates(df, id_cols):
    """Signal-only deduplication to fix 'Data Duplicates' failure."""
    signal_cols = [c for c in df.columns if c not in id_cols]
    initial_len = len(df)
    df = df.drop_duplicates(subset=signal_cols).reset_index(drop=True)
    print(f"  ✨ Removed {initial_len - len(df)} signal-level duplicates.")
    return df

def handle_conflicting_labels(df, target_col):
    """Fixes 'Conflicting Labels' failure (same features, different species)."""
    feature_cols = [c for c in df.columns if c != target_col]
    grouped = df.groupby(feature_cols)[target_col].nunique()
    conflicting_indices = grouped[grouped > 1].index
    
    if conflicting_indices.empty: return df
    
    mask = df.set_index(feature_cols).index.isin(conflicting_indices)
    df = df[~mask].reset_index(drop=True)
    print(f"  ⚠️ Dropped {mask.sum()} rows with conflicting labels.")
    return df

def _calculate_vif(df, vars, threshold):
    """Iterative VIF with an automated safety floor."""
    # Only use columns with variance
    X = df[vars].fillna(df[vars].median()).loc[:, df[vars].nunique() > 1]
    dropped = []
    
    # Safety: If we only have 1 variable left, VIF is irrelevant
    while X.shape[1] > 1:
        X_c = sm.add_constant(X)
        try:
            # Calculate VIF for all columns
            vif_s = pd.Series(
                [variance_inflation_factor(X_c.values, i) for i in range(X_c.shape[1])], 
                index=X_c.columns
            ).drop('const')
            
            if vif_s.max() > threshold:
                # Even if VIF is high, we stop if this is the last available feature
                if X.shape[1] <= 1: 
                    break
                    
                worst = vif_s.idxmax()
                dropped.append(worst)
                X = X.drop(columns=[worst])
                print(f"  🔥 High VIF dropped: {worst}")
            else: 
                break
        except: 
            break
            
    return dropped

def autonomous_cleaning():
    print("\n🛠️ Phase 3: Executing Autonomous Cleaning...")
    df = pd.read_csv(DATASET_PATH)
    
    with open(YDATA_REPORT_JSON, 'r') as f:
        report = json.load(f)
    with open(PRE_CLEAN_AUDIT_REPORT, 'r') as f:
        audit = json.load(f)

    PPS_THRESHOLD = audit['config']['pps_threshold']
    VIF_THRESHOLD = audit['config']['vif_threshold']
    
    to_drop, num_vars, cat_vars = [], [], []
    n_total = report['table']['n']

    # 1. Classification based on ydata-profiling
    for col, stats in report['variables'].items():
        if col == TARGET_COLUMN: continue
        if (stats.get('n_missing', 0) / n_total) > 0.70 or stats.get('is_unique'):
            to_drop.append(col)
        elif stats.get('type') == 'Numeric': num_vars.append(col)
        else: cat_vars.append(col)

    # 2. PPS Importance Filter
    active_num = [c for c in num_vars if c not in to_drop]
    to_drop.extend(get_pps_filter(df, active_num, TARGET_COLUMN, threshold=PPS_THRESHOLD))

    # 3. Multicollinearity (VIF) Filter
    active_num = [c for c in active_num if c not in to_drop]
    to_drop.extend(_calculate_vif(df, active_num, VIF_THRESHOLD))

    # 4. Pipeline Execution with Safeguards
    final_num = [c for c in num_vars if c not in to_drop]
    final_cat = [c for c in cat_vars if c not in to_drop]
    
    steps = [('noise_pruning', DropFeatures(features_to_drop=list(set(to_drop))))]
    
    if final_num:
        steps.append(('num_impute', MeanMedianImputer(variables=final_num)))
        # Conditional check to prevent ValueError with < 2 variables
        if len(final_num) >= 2:
            steps.append(('corr_filter', DropCorrelatedFeatures(threshold=0.8, variables=final_num)))
        else:
            print(f"  ℹ️ Skipping Correlation Filter: Only {len(final_num)} feature(s) left.")
    
    if final_cat:
        steps.append(('cat_impute', CategoricalImputer(variables=final_cat, ignore_format=True)))

    df_silver = Pipeline(steps).fit_transform(df)

    # 5. Final Integrity Logic
    df_silver = handle_duplicates(df_silver, audit['id_cols'])
    df_silver = handle_conflicting_labels(df_silver, TARGET_COLUMN)
    
    os.makedirs(os.path.dirname(CLEANED_DATASET_PATH), exist_ok=True)
    df_silver.to_csv(CLEANED_DATASET_PATH, index=False)
    print(f"✅ Silver Layer Created: {CLEANED_DATASET_PATH} | Final Shape: {df_silver.shape}")
    return df_silver

if __name__ == "__main__":
    autonomous_cleaning()