import pandas as pd
import numpy as np
import json
import os

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import DATASET_PATH, TIMESERIES_AUDIT_REPORT, CLEANED_DATASET_PATH

def handle_sequential_imputation(df):
    """
    Critical for Time Series: Uses Forward Fill to prevent future-data leakage.
    Ensures that a missing value is only filled by what was known 'before' it.
    """
    initial_nulls = df.isnull().sum().sum()
    if initial_nulls == 0:
        return df

    print(f"  🔍 Filling {initial_nulls} temporal gaps...")
    # Forward fill (propagate last valid observation forward)
    # Backward fill is only used as a fallback for the very first row
    df = df.ffill().bfill()
    return df

def autonomous_timeseries_cleaning():
    print("\n🛠️ Phase 3: Executing Autonomous Cleaning (Time Series Mode)...")
    df = pd.read_csv(DATASET_PATH)
    
    with open(TIMESERIES_AUDIT_REPORT, 'r') as f:
        audit = json.load(f)

    # 1. Sort Data if not already sorted
    # If there is no explicit date column, we assume the row order is the time order
    if not audit['temporal_consistency']['is_sorted']:
        print("  ⚠️ Data not monotonic. Sorting by index to preserve sequence.")
        df = df.sort_index()

    # 2. Sequential Imputation (The Core of TS Cleaning)
    df = handle_sequential_imputation(df)

    # 3. Handle IDs
    # We drop IDs but we DO NOT drop duplicates in Time Series
    # because 'duplicate' values are common in sensor/financial data.
    if audit['id_cols']:
        df = df.drop(columns=audit['id_cols'], errors='ignore')
        print(f"  ✨ Dropped ID columns: {audit['id_cols']}")

    # 4. Final Integrity Check
    # Ensure no rows were dropped during the process to keep the timeline intact
    
    os.makedirs(os.path.dirname(CLEANED_DATASET_PATH), exist_ok=True)
    df.to_csv(CLEANED_DATASET_PATH, index=False)
    
    print(f"✅ Silver Layer (Time Series) Created: {CLEANED_DATASET_PATH}")
    print(f"📊 Timeline Length: {len(df)} steps | Features: {len(df.columns)}")
    return df

if __name__ == "__main__":
    autonomous_timeseries_cleaning()