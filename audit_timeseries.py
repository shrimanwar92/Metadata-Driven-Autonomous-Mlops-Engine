import pandas as pd
import numpy as np
import json
import os
from constants import DATASET_PATH, YDATA_REPORT_JSON, PRE_CLEAN_AUDIT_REPORT

def run_timeseries_audit():
    print("⏳ Phase 2: Auditing for Sequential Time Series...")
    df = pd.read_csv(DATASET_PATH)
    
    # In Time Series, we check for 'Temporal Gaps' instead of PPS
    # We assume the index or a 'date' column defines the sequence
    
    # 1. Check for Monotonicity (Is the data already sorted?)
    is_sorted = df.index.is_monotonic_increasing
    
    # 2. Identify Potential ID columns that would break sequences
    with open(YDATA_REPORT_JSON, 'r') as f:
        report = json.load(f)
    
    id_cols = [col for col, s in report['variables'].items() if s.get('is_unique')]

    # 3. Detect Missing Steps
    # If the index is a DatetimeIndex, we check for frequency consistency
    has_date_col = any(col in df.columns.lower() for col in ['date', 'time', 'timestamp'])

    audit_results = {
        "config": {
            "imputation_strategy": "forward_fill", # ffill is safe for TS
            "interpolation": "linear",            # Optional for numeric gaps
            "preserve_order": True,
            "stationarity_test_required": True
        },
        "id_cols": id_cols,
        "temporal_consistency": {
            "is_sorted": bool(is_sorted),
            "has_date_column": has_date_col
        }
    }

    os.makedirs(os.path.dirname(PRE_CLEAN_AUDIT_REPORT), exist_ok=True)
    with open(PRE_CLEAN_AUDIT_REPORT, 'w') as f:
        json.dump(audit_results, f, indent=4)
    
    print(f"✅ Time Series Audit complete. Preserving order: {is_sorted}")
    return audit_results

if __name__ == "__main__":
    run_timeseries_audit()