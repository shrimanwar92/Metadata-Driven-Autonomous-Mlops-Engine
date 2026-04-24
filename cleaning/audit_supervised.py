import pandas as pd
import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
import json
import os
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import DataDuplicates, FeatureLabelCorrelation

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import DATASET_PATH, TARGET_COLUMN, YDATA_REPORT_JSON, PRE_CLEAN_AUDIT_REPORT

def run_pre_clean_audit():
    print("⚖️ Phase 2: Auditing Signal Integrity (Dynamic Mode)...")
    df = pd.read_csv(DATASET_PATH)
    
    with open(YDATA_REPORT_JSON, 'r') as f:
        report = json.load(f)

    # 1. Dimensionality Context
    n_vars = report['table']['n_var']
    n_rows = report['table']['n']
    
    # 2. Identify Identity Columns
    id_cols = [col for col, s in report['variables'].items() 
               if s.get('is_unique') or s.get('n_distinct') == n_rows]
    
    # 3. Initial Dynamic Strategy
    if n_vars > 50:
        pps_threshold, vif_threshold = 0.05, 5.0
    elif n_vars < 15:
        pps_threshold, vif_threshold = 0.005, 20.0
    else:
        pps_threshold, vif_threshold = 0.02, 10.0

    # 4. Preliminary Deepchecks Audit
    audit_df = df.drop(columns=id_cols)
    
    # Address the Deepchecks warning: explicitly pass categorical features
    cat_features = audit_df.select_dtypes(include=['object']).columns.tolist()
    if TARGET_COLUMN in cat_features: cat_features.remove(TARGET_COLUMN)
    
    ds = Dataset(audit_df, label=TARGET_COLUMN, cat_features=cat_features)
    pps_check = FeatureLabelCorrelation().run(ds)
    dup_check = DataDuplicates().run(ds)
    
    weak_candidates = [f for f, pps in pps_check.value.items() if pps < pps_threshold]

    high_signal_count = len([f for f, pps in pps_check.value.items() if pps > 0.1])
    # TRIGGER: If we have high signal but the VIF threshold is strict (10.0), 
    # we risk 'over-cleaning' a high-quality medical dataset.
    if high_signal_count > 15 and vif_threshold < 20.0:
        print(f"  🚨 High-Density Signal Detected ({high_signal_count} strong features).")
        print("  🔄 Raising VIF Floor to 25.0 to preserve multi-dimensional signal.")
        vif_threshold = 25.0
        pps_threshold = 0.001 # Also lower PPS to be safe

    # 6. Final Results Compilation
    audit_results = {
        "config": {
            "pps_threshold": pps_threshold,
            "vif_threshold": vif_threshold
        },
        "id_cols": id_cols,
        "duplicate_ratio": float(dup_check.value),
        "weak_features": weak_candidates
    }

    with open(PRE_CLEAN_AUDIT_REPORT, 'w') as f:
        json.dump(audit_results, f, indent=4)
    
    print(f"✅ Audit complete. Resulting PPS: {pps_threshold}, VIF: {vif_threshold}")
    return audit_results

if __name__ == "__main__":
    run_pre_clean_audit()