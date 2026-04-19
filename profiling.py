from ydata_profiling import ProfileReport
import pandas as pd
import os
from constants import DATASET_PATH, ARTIFACTS_PATH, YDATA_REPORT_JSON

# Ensure directories exist
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

data = pd.read_csv(DATASET_PATH)

print("🔍 Phase 1: Profiling Raw Data...")
profile = ProfileReport(data, minimal=True, explorative=True)

# Generate reports for downstream scripts
profile.to_file(f"{ARTIFACTS_PATH}/report.html")
profile.to_file(YDATA_REPORT_JSON)
print(f"✅ Profiling complete. Metadata saved to {YDATA_REPORT_JSON}")