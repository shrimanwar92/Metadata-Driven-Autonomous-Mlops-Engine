import pandas as pd
import numpy as np
import os
# Fix for older deepchecks versions
np.Inf = np.inf
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import ARTIFACTS_PATH, CLEANED_DATASET_PATH

def validate_clustering_silver_layer():
    print("\n✅ Phase 4: Validating Silver Layer (Clustering Mode)...")
    
    if not os.path.exists(CLEANED_DATASET_PATH):
        print(f"❌ Error: Cleaned dataset not found at {CLEANED_DATASET_PATH}")
        return
        
    df = pd.read_csv(CLEANED_DATASET_PATH)
    
    # 1. Initialize Dataset WITHOUT a label
    # For clustering, we treat all columns as features.
    ds = Dataset(df, label=None)
    
    # 2. Run Data Integrity Suite
    # This suite is ideal for clustering as it checks for:
    # - Feature-Feature Correlation (to avoid redundant distance weights)
    # - Isolate Outliers (to ensure Isolation Forest worked)
    # - Single Value Columns
    suite = data_integrity()
    results = suite.run(ds)
    
    # 3. Save Results
    output_file = os.path.join(ARTIFACTS_PATH, "silver_clustering_integrity_report.html")
    results.save_as_html(output_file, as_widget=False)
    
    print(f"📊 Clustering validation report saved: {output_file}")
    
    # 4. Critical Logic Check
    # If the suite finds features with 0 variance or 100% correlation, we should be alerted.
    if not results.passed():
        print("⚠️ Warning: Data Integrity Suite flagged potential issues in the Silver Layer.")
    
    return results

if __name__ == "__main__":
    validate_clustering_silver_layer()