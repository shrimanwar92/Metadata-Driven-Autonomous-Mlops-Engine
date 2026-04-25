import featuretools as ft
import pandas as pd
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import CLEANED_DATASET_PATH, GOLD_AUDIT_REPORT, TARGET_COLUMN, PROFILER_REPORT_PATH

def run_gold_audit_supervised(target_col):
    with open(PROFILER_REPORT_PATH, 'r') as f:
        profile = json.load(f)
    
    df = pd.read_csv(CLEANED_DATASET_PATH)
    es = ft.EntitySet(id="autonomous_data")
    es.add_dataframe(dataframe_name="base", dataframe=df, index="id", make_index=True)

    # Use profiler metadata to find 'Parent' candidates
    # We look for variables the profiler flagged as 'Categorical' with high associations
    for col, stats in profile['variables'].items():
        if stats.get('type') == 'Categorical':
            # Dynamic check: if it's an ID or aGrouper, normalize it
            if 1 < df[col].nunique() < (len(df) * 0.8):
                es.normalize_dataframe(base_dataframe_name="base", 
                                       new_dataframe_name=f"parent_{col}", 
                                       index=col)

    # Save features identified using the profiler's structural insights
    feature_defs = ft.dfs(entityset=es, target_dataframe_name="base", features_only=True)
    feature_json_string = ft.save_features(feature_defs)
    with open(GOLD_AUDIT_REPORT, "w") as f:
        json.dump(json.loads(feature_json_string), f, indent=4)
    print(f"✅ Audit Complete. Identified {len(feature_defs)} potential features.")

if __name__ == "__main__":
    run_gold_audit_supervised(TARGET_COLUMN)