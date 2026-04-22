import pandas as pd
import numpy as np
np.Inf = np.inf
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
from ..constants import TARGET_COLUMN, ARTIFACTS_PATH, CLEANED_DATASET_PATH

def validate_silver_layer():
    print("\n✅ Phase 4: Validating Silver Layer (Deepchecks)...")
    df = pd.read_csv(CLEANED_DATASET_PATH)
    
    # Dataset auto-detects categorical if not passed, 
    # but since our cleaner already fixed dtypes, it's very reliable.
    ds = Dataset(df, label=TARGET_COLUMN)
    
    suite = data_integrity()
    results = suite.run(ds)
    
    results.save_as_html(f"{ARTIFACTS_PATH}/silver_integrity_report.html", as_widget=False)
    print(f"📊 Final validation report: {ARTIFACTS_PATH}/silver_integrity_report.html")
    return results

if __name__ == "__main__":
    validate_silver_layer()