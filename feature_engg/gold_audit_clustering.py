import json
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import PROFILER_CLEAN_REPORT_PATH, GOLD_AUDIT_REPORT, DOMAIN_POLICY_PATH, FEATURE_ENGG_STRATEGY_PATH

def run_gold_audit():
    print("🧠 Phase 4: Gold Audit (Semantic Strategy)")
    
    with open(PROFILER_CLEAN_REPORT_PATH, 'r') as f:
        report = json.load(f)
    with open(DOMAIN_POLICY_PATH, 'r') as f:
        policy = json.load(f)
    with open(FEATURE_ENGG_STRATEGY_PATH, 'r') as f:
        strategy = json.load(f)

    variables = report.get('variables', {})
    metrics = [col for col, stats in variables.items() if stats.get('type') == 'Numeric' and col != policy.get("subject_id")]

    # Create the execution plan for the engine
    gold_plan = {
        "subject_id": policy.get("subject_id"),
        "protected_features": policy.get("protected_features", []),
        "priority_interactions": strategy.get("interaction_priorities", []),
        "base_metrics": metrics,
        "selection_params": {
            "correlation_threshold": 0.95,
            "force_keep": [i['name'] for i in strategy.get("interaction_priorities", [])]
        },
        "scaling": "RobustScaler"
    }

    with open(GOLD_AUDIT_REPORT, 'w') as f:
        json.dump(gold_plan, f, indent=4)
    
    print(f"✅ Gold Plan generated. Prioritizing {len(gold_plan['priority_interactions'])} semantic features.")

if __name__ == "__main__":
    run_gold_audit()