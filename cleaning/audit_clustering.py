import pandas as pd
import numpy as np
import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import DATASET_PATH, YDATA_REPORT_JSON, PRE_CLEAN_AUDIT_REPORT

def run_clustering_audit():
    print("💎 Phase 2: Auditing for Unsupervised Clustering...")
    df = pd.read_csv(DATASET_PATH)
    with open(YDATA_REPORT_JSON, 'r') as f:
        report = json.load(f)

    # 1. Identify Identity Columns (High Cardinality)
    n_rows = len(df)
    id_cols = [col for col, s in report['variables'].items() 
               if s.get('is_unique') or s.get('n_distinct') == n_rows]

    # 2. Variance-Based Signal Detection
    # In clustering, columns with near-zero variance provide no grouping power
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=id_cols, errors='ignore')
    normalized_variance = numeric_df.var() / (numeric_df.mean()**2 + 1e-9)
    
    # Features with variance < 0.001 are considered "constants" or noise
    weak_features = normalized_variance[normalized_variance < 0.001].index.tolist()

    # 3. Dynamic Thresholds for Clustering
    # Clustering is very sensitive to VIF; we want to keep it strict (10.0)
    # unless we have very few features.
    vif_threshold = 10.0 if len(numeric_df.columns) > 10 else 20.0

    audit_results = {
        "config": {
            "vif_threshold": vif_threshold,
            "variance_threshold": 0.001,
            "scaling_required": True # Clustering always requires scaling
        },
        "id_cols": id_cols,
        "weak_features": weak_features
    }

    with open(PRE_CLEAN_AUDIT_REPORT, 'w') as f:
        json.dump(audit_results, f, indent=4)
    
    print(f"✅ Clustering Audit complete. Weak Features: {len(weak_features)}")
    return audit_results

if __name__ == "__main__":
    run_clustering_audit()