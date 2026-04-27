import pandas as pd
import json
import os
import sys
from ydata_profiling import ProfileReport

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLEANED_DATASET_PATH, GOLD_AUDIT_REPORT, PROFILER_CLEAN_REPORT_PATH, PRE_CLEAN_AUDIT_REPORT

def run_gold_audit_clustering():
    print("🌟 Phase 4: Robust Entity & Mode Discovery...")

    df = pd.read_csv(CLEANED_DATASET_PATH)
    
    # 1. Load Anchor from Silver Audit (Phase 2) if available
    try:
        with open(PRE_CLEAN_AUDIT_REPORT, 'r') as f:
            silver_audit = json.load(f)
        subject_id = silver_audit['config'].get('subject_id')
    except (FileNotFoundError, KeyError):
        subject_id = None

    print("🔍 Profiling Clean Data...")
    profile = ProfileReport(df, minimal=True, explorative=True)
    profile.to_file(PROFILER_CLEAN_REPORT_PATH)

    with open(PROFILER_CLEAN_REPORT_PATH, 'r') as f:
        profiler = json.load(f)
    variables = profiler.get('variables', {})

    # 2. Enhanced Entity Selection Logic
    if not subject_id:
        potential_entities = []
        for col, s in variables.items():
            # Criteria: High cardinality but not a technical row index
            if 0.001 < s.get('p_distinct', 0) < 0.99 and s.get('n_distinct', 0) > 1:
                potential_entities.append(col)
        
        # Scoring: Strong business keywords get top priority
        keywords = ['customer', 'user', 'client', 'account', 'member', 'patient']
        priority = [e for e in potential_entities if any(k in e.lower() for k in keywords)]
        
        if priority:
            selected_entity = priority[0]
        else:
            # Fallback to general IDs
            id_fallback = [e for e in potential_entities if 'id' in e.lower()]
            selected_entity = id_fallback[0] if id_fallback else potential_entities[0]
    else:
        selected_entity = subject_id

    # 3. Robust Mode Detection
    # If the selected entity has duplicates, it's transactional
    distinct_ratio = variables.get(selected_entity, {}).get('p_distinct', 1.0)
    mode = "profile" if distinct_ratio > 0.98 else "transactional"

    # 4. Strategy Definition
    # Increased threshold for categories to capture more signal
    clusterable_dimensions = [col for col, s in variables.items() 
                             if 1 < s.get('n_distinct', 0) < 500]

    strategy = {
        "metadata": {
            "entity_id": selected_entity,
            "mode": mode,
            "segmentation_columns": clusterable_dimensions,
            "row_count": len(df)
        },
        "aggregations": {
            "metrics": [c for c, s in variables.items() if s.get('type') == 'Numeric' and c != selected_entity],
            "methods": ["identity"] if mode == "profile" else ["sum", "mean", "std", "nunique"]
        }
    }

    with open(GOLD_AUDIT_REPORT, 'w') as f:
        json.dump(strategy, f, indent=4)
    
    print(f"✅ Strategy Saved: {selected_entity} ({mode} mode)")

if __name__ == "__main__":
    run_gold_audit_clustering()