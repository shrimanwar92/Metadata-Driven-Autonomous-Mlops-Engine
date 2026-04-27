import pandas as pd
import json
import os
import sys
from ydata_profiling import ProfileReport

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLEANED_DATASET_PATH, GOLD_AUDIT_REPORT, PROFILER_CLEAN_REPORT_PATH, PRE_CLEAN_AUDIT_REPORT

def run_gold_audit_clustering():
    print("🌟 Phase 4: Generic Entity & Mode Discovery...")

    df = pd.read_csv(CLEANED_DATASET_PATH)
    
    # 1. Load Anchor from Silver Audit (Phase 2)
    # This ensures we respect the CustomerID found earlier
    try:
        with open(PRE_CLEAN_AUDIT_REPORT, 'r') as f:
            silver_audit = json.load(f)
        subject_id = silver_audit['config'].get('subject_id')
    except FileNotFoundError:
        subject_id = None

    print("🔍 Profiling clean Data...")
    profile = ProfileReport(df, minimal=True, explorative=True)
    profile.to_file(PROFILER_CLEAN_REPORT_PATH)

    # 2. Extract Profiler Metadata
    with open(PROFILER_CLEAN_REPORT_PATH, 'r') as f:
        profiler = json.load(f)
    variables = profiler.get('variables', {})

    # 3. Determine Entity
    # If a subject_id was identified in cleaning, it MUST be the entity
    if subject_id and subject_id in df.columns:
        selected_entity = subject_id
    else:
        # Fallback: Find high-cardinality candidates
        potential_entities = [col for col, s in variables.items() 
                             if s.get('p_distinct', 0) > 0.001 and s.get('n_distinct', 0) > 1]
        priority = [e for e in potential_entities if any(x in e.lower() for x in ['customer', 'user', 'id'])]
        selected_entity = priority[0] if priority else potential_entities[0]

    # 4. Mode Detection: Profile vs Transactional
    # If the entity is unique, we don't aggregate; we process as a profile.
    is_unique = variables.get(selected_entity, {}).get('is_unique', False)
    mode = "profile" if is_unique else "transactional"

    # 5. Define Strategy
    clusterable_dimensions = [col for col, s in variables.items() 
                             if 1 < s.get('n_distinct', 0) < 100]

    strategy = {
        "metadata": {
            "entity_id": selected_entity,
            "mode": mode,
            "segmentation_columns": clusterable_dimensions
        },
        "aggregations": {
            "metrics": [c for c, s in variables.items() if s.get('type') == 'Numeric' and c != selected_entity],
            "methods": ["identity"] if mode == "profile" else ["sum", "mean", "std"]
        }
    }

    with open(GOLD_AUDIT_REPORT, 'w') as f:
        json.dump(strategy, f, indent=4)
    
    print(f"✅ Gold Audit complete. Entity: {selected_entity} | Mode: {mode.upper()}")

if __name__ == "__main__":
    run_gold_audit_clustering()