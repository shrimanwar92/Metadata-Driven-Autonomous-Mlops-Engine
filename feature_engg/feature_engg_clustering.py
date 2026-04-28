import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLEANED_DATASET_PATH, GOLD_DATASET_PATH, GOLD_AUDIT_REPORT, JOBLIB_PIPELINE_PATH

def execute_total_coverage_engg():
    df = pd.read_csv(CLEANED_DATASET_PATH)
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        strat = json.load(f)

    entity = strat["metadata"]["entity_id"]
    metrics = strat["aggregations"]["metrics"]
    
    print(f"🚀 Building Permutation Gold Layer | Metrics: {len(metrics)}")

    # 1. Initial Selection
    X_gold = df.set_index(entity)[metrics].copy()
    
    # 2. PERMUTATION ENGINE: Create All Combinations
    print("  🔄 Generating all pair-wise interactions...")
    base_cols = list(X_gold.columns)
    for i in range(len(base_cols)):
        for j in range(i + 1, len(base_cols)):
            col_a, col_b = base_cols[i], base_cols[j]
            
            # Multiplication (Interaction)
            X_gold[f'{col_a}_x_{col_b}'] = X_gold[col_a] * X_gold[col_b]
            
            # Ratios (Safe Division)
            X_gold[f'{col_a}_per_{col_b}'] = X_gold[col_a] / (X_gold[col_b] + 1e-6)
            X_gold[f'{col_b}_per_{col_a}'] = X_gold[col_b] / (X_gold[col_a] + 1e-6)

    # 3. HIGH-THRESHOLD PRUNING
    # We only drop features that are >98% identical to keep maximum detail
    print(f"  🔍 Feature count pre-pruning: {X_gold.shape[1]}")
    corr_matrix = X_gold.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > strat["selection_reduction"]["correlation_threshold"])]
    X_gold = X_gold.drop(columns=to_drop)
    print(f"  ✂️ Pruned {len(to_drop)} redundant features. Final count: {X_gold.shape[1]}")

    # 4. Total Information Pipeline
    feature_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 5. Fit & Save
    scaled_data = feature_pipeline.fit_transform(X_gold)
    
    # Save the expanded feature set
    pd.DataFrame(scaled_data, columns=X_gold.columns, index=X_gold.index).to_csv(GOLD_DATASET_PATH)
    joblib.dump(feature_pipeline, JOBLIB_PIPELINE_PATH)
    
    print(f"✅ Total Coverage Gold Layer Exported.")

if __name__ == "__main__":
    execute_total_coverage_engg()