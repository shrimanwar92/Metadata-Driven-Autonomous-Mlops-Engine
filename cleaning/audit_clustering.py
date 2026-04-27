import pandas as pd
import numpy as np
import json
import sys
import os
from itertools import combinations

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import DATASET_PATH, YDATA_REPORT_JSON, PRE_CLEAN_AUDIT_REPORT

def identify_logical_anchor(df, variables):
    n_rows = len(df)
    scores = {}
    
    # 1. Identify all mathematically valid Single-Column Keys
    # An anchor MUST be 100% unique and have 0 missing values
    id_candidates = [col for col, s in variables.items() 
                    if (s.get('is_unique') or s.get('n_distinct') == n_rows) 
                    and s.get('n_missing', 0) == 0]

    if id_candidates:
        primary_keywords = ['customer', 'user', 'member', 'subject', 'patient', 'client']
        secondary_keywords = ['id', 'code', 'num', 'key', 'pk', 'invoice']
        
        for col in id_candidates:
            score = 0
            col_lower = col.lower()
            if any(x in col_lower for x in primary_keywords): score += 50
            if any(x in col_lower for x in secondary_keywords): score += 30
            if df.columns.get_loc(col) == 0: score += 20
            scores[col] = score
            
        subject_id = max(scores, key=scores.get)
        return subject_id, id_candidates

    # Fallback to composite keys if no single column works
    for combo in combinations(df.columns, 2):
        if not df.duplicated(subset=list(combo)).any() and df[list(combo)].isnull().sum().sum() == 0:
            return list(combo), list(combo)

    return None, []

def run_clustering_audit():
    df = pd.read_csv(DATASET_PATH)
    with open(YDATA_REPORT_JSON, 'r') as f:
        report = json.load(f)

    variables = report.get('variables', {})
    subject_id, id_candidates = identify_logical_anchor(df, variables)

    # 2. SEPARATION OF CONCERNS: The "Drop List"
    # anchor_list ensures we don't drop our selected index
    anchor_list = [subject_id] if isinstance(subject_id, str) else subject_id
    
    # Drop all other high-cardinality ID columns (Garbage IDs)
    drop_cols = [col for col in id_candidates if col not in (anchor_list or [])]

    # Add numeric columns with near-zero variance to the drop list
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=id_candidates, errors='ignore')
    normalized_variance = numeric_df.var() / (numeric_df.mean()**2 + 1e-9)
    weak_features = normalized_variance[normalized_variance < 0.001].index.tolist()
    
    # Add weak features to drop_cols so the cleaner actually removes them
    drop_cols.extend(weak_features)

    audit_results = {
        "config": {
            "subject_id": subject_id,
            "vif_threshold": 10.0 if len(numeric_df.columns) > 10 else 20.0,
            "variance_threshold": 0.001,
            "scaling_required": True 
        },
        "id_cols": id_candidates, 
        "drop_cols": list(set(drop_cols)), # Unique list of columns to prune
        "weak_features": weak_features
    }

    with open(PRE_CLEAN_AUDIT_REPORT, 'w') as f:
        json.dump(audit_results, f, indent=4)
    
    return audit_results

if __name__ == "__main__":
    run_clustering_audit()