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

    variables = report['variables']
    n_rows = len(df)

    # 1. Generic Subject ID Detection (The Anchor)
    # Identify all columns with very high cardinality (>95% unique)
    id_candidates = [col for col, s in variables.items() 
                    if s.get('is_unique') or s.get('p_distinct', 0) > 0.95]
    
    # Priority Heuristic: Pick the anchor based on naming
    subject_id = None
    for col in id_candidates:
        if any(x in col.lower() for x in ['id', 'cust', 'code', 'num', 'invoice']):
            subject_id = col
            break
    
    # Fallback: Just pick the most unique one if no name match
    if not subject_id and id_candidates:
        subject_id = id_candidates[0]

    # 2. Identify Garbage IDs to drop (High cardinality but NOT the anchor)
    drop_cols = [col for col in id_candidates if col != subject_id]

    # 3. Variance-Based Signal Detection
    # Exclude all ID candidates from math checks
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=id_candidates, errors='ignore')
    normalized_variance = numeric_df.var() / (numeric_df.mean()**2 + 1e-9)
    weak_features = normalized_variance[normalized_variance < 0.001].index.tolist()

    # 4. VIF (Multi-collinearity) Check - Stored as Metadata
    # We keep VIF in the audit but let the cleaning script use 
    # 'DropCorrelatedFeatures' as it's safer for preserving signal.
    vif_threshold = 10.0 if len(numeric_df.columns) > 10 else 20.0

    audit_results = {
        "config": {
            "subject_id": subject_id, # THE PROTECTED ANCHOR
            "vif_threshold": vif_threshold,
            "variance_threshold": 0.001,
            "scaling_required": True 
        },
        "id_cols": id_candidates, 
        "drop_cols": drop_cols, # Droppable high-cardinality garbage
        "weak_features": weak_features
    }

    with open(PRE_CLEAN_AUDIT_REPORT, 'w') as f:
        json.dump(audit_results, f, indent=4)
    
    print(f"✅ Audit complete. Subject Anchor: {subject_id}")
    return audit_results

if __name__ == "__main__":
    run_clustering_audit()