import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import RobustScaler
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLEANED_DATASET_PATH, GOLD_DATASET_PATH, GOLD_AUDIT_REPORT

def execute_clustering_engg():
    df = pd.read_csv(CLEANED_DATASET_PATH)
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        strat = json.load(f)

    entity = strat["metadata"]["entity_id"]
    mode = strat["metadata"]["mode"]
    metrics = strat["aggregations"]["metrics"]
    
    print(f"🚀 Building Gold Layer | Entity: {entity} | Mode: {mode}")

    # 1. Temporal Feature Engineering (Generic for all modes)
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
        df[f'{col}_hour'] = df[col].dt.hour
        if f'{col}_hour' not in metrics: metrics.append(f'{col}_hour')

    # 2. Logic Switch: Transactional vs Profile
    if mode == "transactional":
        print(f"  📦 Aggregating transactions for {entity}...")
        agg_map = {m: ['sum', 'mean', 'std'] for m in metrics}
        for dim in strat["metadata"]["segmentation_columns"]:
            if dim != entity:
                agg_map[dim] = ['nunique']
        
        X_gold = df.groupby(entity).agg(agg_map)
        X_gold.columns = ['_'.join(col).strip() for col in X_gold.columns.values]
    
    else:
        print(f"  👤 Engineering profile features for {entity}...")
        # Use existing numeric rows directly as features
        X_gold = df.set_index(entity)[metrics]
        
        # Add smart ratios for profile signal (Generic)
        # e.g., Income vs Spending, or Balance vs Credit Limit
        if len(metrics) >= 2:
            for i in range(min(len(metrics), 5)): # Limit to avoid explosion
                for j in range(i + 1, min(len(metrics), 5)):
                    col_i, col_j = metrics[i], metrics[j]
                    X_gold[f'{col_i}_to_{col_j}_ratio'] = X_gold[col_i] / (X_gold[col_j] + 1e-9)

    # 3. Scale and Finalize
    # RobustScaler is better for clustering as it handles outliers in behavioral data
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(X_gold)
    
    X_gold_final = pd.DataFrame(
        scaled_data, 
        columns=X_gold.columns, 
        index=X_gold.index
    )

    # 4. Save
    os.makedirs(os.path.dirname(GOLD_DATASET_PATH), exist_ok=True)
    X_gold_final.to_csv(GOLD_DATASET_PATH)
    print(f"✅ Gold Layer Created. Shape: {X_gold_final.shape}")

if __name__ == "__main__":
    execute_clustering_engg()