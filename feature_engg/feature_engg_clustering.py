import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLEANED_DATASET_PATH, GOLD_DATASET_PATH, GOLD_AUDIT_REPORT, JOBLIB_PIPELINE_PATH

def execute_clustering_engg():
    df = pd.read_csv(CLEANED_DATASET_PATH)
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        strat = json.load(f)

    entity = strat["metadata"]["entity_id"]
    mode = strat["metadata"]["mode"]
    metrics = strat["aggregations"]["metrics"]
    
    print(f"🚀 Building Gold Layer | Entity: {entity} | Mode: {mode}")

    # 1. Enhanced Temporal Features
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_dow'] = df[col].dt.dayofweek
            df[f'{col}_is_weekend'] = df[col].dt.dayofweek >= 5
            
            # Add to metrics for aggregation
            for suffix in ['_hour', '_dow', '_is_weekend']:
                new_col = f'{col}{suffix}'
                if new_col not in metrics: metrics.append(new_col)
        except:
            print(f"⚠️ Failed to parse date column: {col}")

    # 2. Collision-Safe Aggregation
    if mode == "transactional":
        agg_map = {}
        for m in metrics:
            if m != entity and m in df.columns:
                agg_map[m] = ['sum', 'mean', 'std']
        
        # Add categorical diversity
        for dim in strat["metadata"]["segmentation_columns"]:
            if dim != entity and dim in df.columns:
                if dim not in agg_map: agg_map[dim] = []
                agg_map[dim].append('nunique')
        
        X_gold = df.groupby(entity).agg(agg_map)
        X_gold.columns = ['_'.join(col).strip() for col in X_gold.columns.values]
    
    else:
        # 👤 Engineering profile features
        X_gold = df.set_index(entity)[metrics]
    
        # IMPROVEMENT: Only create ratios for features that are never zero
        # and CLIP the results to prevent 1e11 values.
        safe_metrics = [m for m in metrics if df[m].min() > 0]
        for i in range(len(safe_metrics)):
            for j in range(i + 1, len(safe_metrics)):
                col_i, col_j = safe_metrics[i], safe_metrics[j]
                ratio_name = f'{col_i}_to_{col_j}_ratio'
                X_gold[ratio_name] = (X_gold[col_i] / (X_gold[col_j] + 1e-6)).clip(-10, 10)

    # 3. Production Pipeline (Impute -> Scale)
    # Median imputation is safest for skewed clustering data
    feature_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    # 4. Fit & Save Artifacts
    scaled_data = feature_pipeline.fit_transform(X_gold)
    joblib.dump(feature_pipeline, JOBLIB_PIPELINE_PATH)
    
    X_gold_final = pd.DataFrame(
        scaled_data, 
        columns=X_gold.columns,
        index=X_gold.index
    )

    X_gold_final.to_csv(GOLD_DATASET_PATH)
    print(f"✅ Gold Layer saved. Features: {len(X_gold_final.columns)}")
    return X_gold_final

if __name__ == "__main__":
    execute_clustering_engg()