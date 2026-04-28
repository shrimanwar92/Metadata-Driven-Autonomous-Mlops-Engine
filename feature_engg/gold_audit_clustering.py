import pandas as pd
import numpy as np
import json
from scipy.stats import skew
import sys
import os
from ydata_profiling import ProfileReport

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLEANED_DATASET_PATH, GOLD_AUDIT_REPORT, PROFILER_CLEAN_REPORT_PATH, PRE_CLEAN_AUDIT_REPORT

def run_gold_audit_total_coverage():
    print("🌟 Phase 5: Enhanced Total Coverage Audit")
    df = pd.read_csv(CLEANED_DATASET_PATH)
    
    # 1. Load Subject ID from previous layers if available
    try:
        with open(PRE_CLEAN_AUDIT_REPORT, 'r') as f:
            silver_audit = json.load(f)
        subject_id = silver_audit['config'].get('subject_id')
    except (FileNotFoundError, KeyError):
        subject_id = None

    # 2. Profile the data to understand cardinality
    profile = ProfileReport(df, minimal=True, explorative=True)
    profiler_data = json.loads(profile.to_json())
    variables = profiler_data.get('variables', {})

    # 3. Enhanced Entity Selection Logic (INTEGRATED)
    if not subject_id:
        potential_entities = []
        for col, s in variables.items():
            # Criteria: High cardinality but not a technical unique row index
            if 0.001 < s.get('p_distinct', 0) < 0.99 and s.get('n_distinct', 0) > 1:
                potential_entities.append(col)
        
        # Scoring: Strong business keywords get top priority
        keywords = ['customer', 'user', 'client', 'account', 'member', 'patient', 'cust', 'id']
        priority = [e for e in potential_entities if any(k in e.lower() for k in keywords)]
        
        if priority:
            selected_entity = priority[0]
        else:
            # Fallback to general IDs or the first column
            id_fallback = [e for e in potential_entities if 'id' in e.lower()]
            selected_entity = id_fallback[0] if id_fallback else df.columns[0]
    else:
        selected_entity = subject_id

    print(f"🎯 Selected Entity for Persona Clustering: {selected_entity}")

    # 4. Metric Identification
    num_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    if selected_entity in num_vars: 
        num_vars.remove(selected_entity)

    # 5. Define Strategy
    strategy = {
        "metadata": {
            "entity_id": selected_entity,
            "is_clusterable": True,
            "total_base_metrics": len(num_vars)
        },
        "numerical_transformations": {
            "scaling_method": "RobustScaler" # Robust to outliers in permutations
        },
        "selection_reduction": {
            "correlation_threshold": 0.98, # High threshold to keep maximum detail
            "apply_pca": True,
            "target_variance": 0.99
        },
        "aggregations": {
            "metrics": num_vars
        }
    }

    with open(GOLD_AUDIT_REPORT, 'w') as f:
        json.dump(strategy, f, indent=4)
    
    print(f"✅ Audit Complete. Strategy exported for {len(num_vars)} base metrics.")

if __name__ == "__main__":
    run_gold_audit_total_coverage()