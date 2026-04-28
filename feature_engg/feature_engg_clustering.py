import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.preprocessing import RobustScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLEANED_DATASET_PATH, GOLD_DATASET_PATH, GOLD_AUDIT_REPORT

def run_gold_engineering():
    print("🚀 Phase 5: Gold Engine (Semantic Construction)")
    
    df = pd.read_csv(CLEANED_DATASET_PATH)
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        plan = json.load(f)

    # 1. Set Anchor
    if plan["subject_id"] in df.columns:
        df = df.set_index(plan["subject_id"])

    X_gold = df.copy()

    # 2. Semantic Construction (What the LLM suggested)
    for inter in plan["priority_interactions"]:
        col1, col2 = inter["pair"]
        if col1 in X_gold.columns and col2 in X_gold.columns:
            if inter["logic"] == "multiplication":
                X_gold[inter["name"]] = X_gold[col1] * X_gold[col2]
            elif inter["logic"] == "ratio":
                X_gold[inter["name"]] = X_gold[col1] / (X_gold[col2] + 1e-6)
            print(f"💎 Created Semantic Feature: {inter['name']}")

    # 3. Numeric Only & Scaling
    # Filter to only numeric for clustering math
    X_gold = X_gold.select_dtypes(include=[np.number])
    
    # 4. Pruning (Generic Correlation Drop)
    corr_matrix = X_gold.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Don't drop features the LLM specifically told us to create
    to_drop = [col for col in upper.columns 
               if any(upper[col] > plan["selection_params"]["correlation_threshold"]) 
               and col not in plan["selection_params"]["force_keep"]]
    
    X_gold.drop(columns=to_drop, inplace=True)
    print(f"✂️ Pruned {len(to_drop)} redundant features. Keeping semantic priorities.")

    # 5. Final Robust Scaling
    scaler = RobustScaler()
    X_final_scaled = pd.DataFrame(
        scaler.fit_transform(X_gold),
        columns=X_gold.columns,
        index=X_gold.index
    )

    X_final_scaled.to_csv(GOLD_DATASET_PATH)
    print(f"🏆 Gold Dataset saved. Final Features: {list(X_gold.columns)}")

if __name__ == "__main__":
    run_gold_engineering()