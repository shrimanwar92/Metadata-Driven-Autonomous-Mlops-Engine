import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import RobustScaler
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

    # 1. Temporal Feature Engineering (Generic)
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
        df[f'{col}_hour'] = df[col].dt.hour
        if f'{col}_hour' not in metrics: metrics.append(f'{col}_hour')

    # 2. Feature Generation
    if mode == "transactional":
        agg_map = {m: ['sum', 'mean', 'std'] for m in metrics if m != entity}
        for dim in strat["metadata"]["segmentation_columns"]:
            if dim != entity:
                agg_map[dim] = ['nunique']
        X_gold = df.groupby(entity).agg(agg_map)
        X_gold.columns = ['_'.join(col).strip() for col in X_gold.columns.values]
    else:
        X_gold = df.set_index(entity)[metrics]
        if len(metrics) >= 2:
            for i in range(min(len(metrics), 5)):
                for j in range(i + 1, min(len(metrics), 5)):
                    col_i, col_j = metrics[i], metrics[j]
                    X_gold[f'{col_i}_to_{col_j}_ratio'] = X_gold[col_i] / (X_gold[col_j] + 1e-9)

    # 3. Create and Save the Pipeline
    # We use a Pipeline object so that the Scaler's state (median/IQR) is preserved
    feature_pipeline = Pipeline([
        ('scaler', RobustScaler())
    ])

    # Fit the pipeline on training data and transform it
    scaled_data = feature_pipeline.fit_transform(X_gold)
    
    # 4. Save Artifacts
    joblib.dump(feature_pipeline, JOBLIB_PIPELINE_PATH)
    print(f"📦 Pipeline saved to {JOBLIB_PIPELINE_PATH}")

    X_gold_final = pd.DataFrame(
        scaled_data, 
        columns=X_gold.columns,
        index=X_gold.index
    )

    X_gold_final.to_csv(GOLD_DATASET_PATH)
    print(f"✅ Gold Layer saved to {GOLD_DATASET_PATH}")
    return X_gold_final

if __name__ == "__main__":
    execute_clustering_engg()