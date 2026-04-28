import json
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import PROFILER_REPORT_PATH, PRE_CLEAN_AUDIT_REPORT, DOMAIN_POLICY_PATH

def run_clustering_audit():
    print("🧠 Phase 2: Audit Brain (Strategic Planning)")
    
    with open(PROFILER_REPORT_PATH, 'r') as f:
        report = json.load(f)
    with open(DOMAIN_POLICY_PATH, 'r') as f:
        policy = json.load(f)

    variables = report.get('variables', {})
    protected = policy.get("protected_features", [])
    garbage = policy.get("technical_garbage", [])
    subject_id = policy.get("subject_id")

    contract = {
        "drop_features": [],
        "subject_id": subject_id,
        "imputation": {"mean_median": [], "categorical": []},
        "encoding": {"one_hot": [], "rare_label": []},
        "transformations": {"log": [], "yeo_johnson": []},
        "scaling": "RobustScaler" 
    }

    # 1. MANDATORY DROPS (Technical Garbage)
    # We populate this first from the Domain Policy
    contract["drop_features"] = [g for g in garbage if g != subject_id]

    for col, stats in variables.items():
        print(col)
        if col in garbage or col == subject_id:
            continue
        col_type = stats.get('type')

        # 2. IDENTIFICATION LOGIC
        # If it's the subject_id, we don't transform it, but we keep it in the contract 
        # so the engine knows to set it as the index later.
        if col == subject_id:
            continue

        # 3. IMPUTATION LOGIC
        n_missing = stats.get('n_missing', 0)
        if n_missing > 0:
            if col_type == 'Numeric':
                contract["imputation"]["mean_median"].append(col)
            elif col_type == 'Text':
                contract["imputation"]["categorical"].append(col)

        # 4. ENCODING LOGIC
        if col_type == 'Text':
            n_distinct = stats.get('n_distinct', 0)
            if n_distinct <= 10:
                contract["encoding"]["one_hot"].append(col)
            else:
                contract["encoding"]["rare_label"].append(col)

        # 5. TRANSFORMATION LOGIC (Skewness)
        # Apply ONLY to Numeric and NON-PROTECTED features
        if col_type == 'Numeric' and col not in protected:
            skewness = stats.get('skewness', 0)
            if abs(skewness) > 0.75:
                # Log for strictly positive, Yeo-Johnson for others
                if stats.get('min', 0) > 0:
                    contract["transformations"]["log"].append(col)
                else:
                    contract["transformations"]["yeo_johnson"].append(col)

    with open(PRE_CLEAN_AUDIT_REPORT, 'w') as f:
        json.dump(contract, f, indent=4)
    
    print(f"✅ Audit Brain Complete.")
    print(f"📊 Plan: {json.dumps(contract)}")

if __name__ == "__main__":
    run_clustering_audit()