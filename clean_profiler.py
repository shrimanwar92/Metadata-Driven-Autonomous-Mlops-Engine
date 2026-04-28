from ydata_profiling import ProfileReport
import pandas as pd
import json
from constants import CLEANED_DATASET_PATH, PROFILER_CLEAN_REPORT_PATH

data = pd.read_csv(CLEANED_DATASET_PATH, encoding="latin1")

print("🔍 Phase 1: Profiling Clean Data...")
profile = ProfileReport(data, 
                        minimal=True, 
                        explorative=True,
                        correlations=None,
                        interactions=None,
                        missing_diagrams=None,
                        samples=None,
                        duplicates=None
                        )

profile.to_file(PROFILER_CLEAN_REPORT_PATH)
print(f"✅ Clean data profiling complete.")