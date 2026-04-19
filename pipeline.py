import pandas as pd
import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
import json
from ydata_profiling import ProfileReport
from deepchecks.tabular.checks import DataDuplicates, FeatureLabelCorrelation
from deepchecks.tabular import Dataset
from cleaning import autonomous_cleaning # Importing your existing logic
from constants import DATASET_PATH, ARTIFACTS_PATH, TARGET_COLUMN, CLEANED_DATASET_PATH

class DataStrategyEngine:
    def __init__(self, data_path, target_col):
        self.df = pd.read_csv(data_path)
        self.target = target_col
        self.results_path = ARTIFACTS_PATH

    def run_profiling(self):
        """Phase 1: Statistical Observation"""
        print("🔍 Phase 1: Running ydata-profiling...")
        profile = ProfileReport(self.df, minimal=True, explorative=True)
        profile.to_file(f"{self.results_path}/pre_clean_report.json")
        return f"{self.results_path}/pre_clean_report.json"

    def run_pre_clean_checks(self):
        """Phase 2: Data Integrity Audit"""
        print("⚖️ Phase 2: Running Pre-Clean Deepchecks...")
        ds = Dataset(self.df, label=self.target)
        
        # Check for Duplicates
        dup_res = DataDuplicates().run(ds)
        has_duplicates = dup_res.value > 0.05 # Threshold 5%
        
        # Check for PPS (Predictive Power)
        pps_res = FeatureLabelCorrelation().run(ds)
        # Extract features with low PPS from the deepchecks result object
        weak_features = [f for f, pps in pps_res.value.items() if pps < 0.05]
        
        return {
            "has_duplicates": has_duplicates,
            "weak_features": weak_features
        }

    def execute_pipeline(self):
        # 1. Profile the data
        report_path = self.run_profiling()
        
        # 2. Audit the data
        audit_results = self.run_pre_clean_checks()

        print(audit_results)
        
        # 3. Clean the data (Using your refined cleaning.py logic)
        # We pass the audit results so cleaning.py knows what to prioritize
        # print("🛠️ Phase 3: Executing Autonomous Cleaning...")
        # df_silver = autonomous_cleaning(
        #     data_path=DATASET_PATH,
        #     report_json_path=report_path,
        #     target_col=self.target,
        #     # We can now dynamically adjust thresholds based on audit
        #     weak_predictor_threshold=0.01 if not audit_results['weak_features'] else 0.05
        # )
        
        # # 4. Final Validation
        # self.validate_silver_layer(df_silver)

    def validate_silver_layer(self, df_cleaned):
        """Phase 4: Post-Clean Validation"""
        print("✅ Phase 4: Final Silver Layer Validation...")
        ds_clean = Dataset(df_cleaned, label=self.target)
        
        # Run a suite to ensure cleaning worked
        from deepchecks.tabular.suites import data_integrity
        suite = data_integrity()
        result = suite.run(ds_clean)
        result.save_as_html(f"{self.results_path}/silver_validation_report.html")
        print(f"Validation Report Saved to {self.results_path}/silver_validation_report.html")

if __name__ == "__main__":
    engine = DataStrategyEngine(DATASET_PATH, TARGET_COLUMN)
    engine.execute_pipeline()