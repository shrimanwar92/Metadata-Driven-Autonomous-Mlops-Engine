import json
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import PROFILER_CLEAN_REPORT_PATH, GOLD_AUDIT_REPORT, DOMAIN_POLICY_PATH, FEATURE_ENGG_STRATEGY_PATH


def run_gold_audit():
    print("🧠 Phase 4: Gold Audit (Strategy Sync)")
    
    # Load all context files
    with open(PROFILER_CLEAN_REPORT_PATH, 'r') as f:
        report = json.load(f)
    with open(DOMAIN_POLICY_PATH, 'r') as f:
        policy = json.load(f)
    with open(FEATURE_ENGG_STRATEGY_PATH, 'r') as f:
        strategy = json.load(f)

    variables = report.get('variables', {})
    
    # 1. Resolve Subject ID: Use strategy choice first, fallback to policy
    final_subject_id = strategy.get("subject_id") or policy.get("subject_id")

    # 2. Filter Numeric Metrics while excluding the resolved subject_id
    metrics = [
        col for col, stats in variables.items() 
        if stats.get('type') == 'Numeric' and col != final_subject_id
    ]

    # 3. Extract thresholds from the JSON strategy
    f_selection = strategy.get("feature_selection", {})
    prep_meta = strategy.get("preprocessing_metadata", {})

    # NEW: Sync algorithm, scaling, and specific selection thresholds from LLM
    gold_plan = {
        "subject_id": final_subject_id,
        "recommended_algorithm": strategy.get("recommended_algorithm", "KMeans"),
        "priority_interactions": strategy.get("interaction_priorities", []),
        "base_metrics": metrics,
        "aggregation_strategy": strategy.get("aggregation_strategy", {}), # Injected for downstream groupbys
        "selection_params": {
            "correlation_threshold": f_selection.get("max_correlation_threshold", 0.85),
            "variance_threshold": f_selection.get("variance_threshold", 0.01),
            "force_keep": [i['name'] for i in strategy.get("interaction_priorities", [])]
        },
        "preprocessing": {
            "scaler": prep_meta.get("recommended_scaler", "PowerTransformer"),
            "imputation": prep_meta.get("imputation_strategy", "median"),
            "encoding": prep_meta.get("categorical_encoding", "one-hot")
        }
    }

    # Write the enriched plan
    with open(GOLD_AUDIT_REPORT, 'w') as f:
        json.dump(gold_plan, f, indent=4)
        
    print(f"✅ Gold Plan updated.")
    print(f"   - Algorithm: {gold_plan['recommended_algorithm']}")
    print(f"   - Scaler: {gold_plan['preprocessing']['scaler']}")
    print(f"   - Subject ID: {gold_plan['subject_id']}")

if __name__ == "__main__":
    run_gold_audit()