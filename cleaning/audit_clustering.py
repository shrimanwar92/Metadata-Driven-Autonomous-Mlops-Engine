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
        "imputation": {"mean": [], "median": [], "categorical": []},
        "encoding": {"one_hot": [], "rare_label": []},
        "transformations": {"log": [], "yeo_johnson": [], "outlier_clipping": []},
        "scaling": {"robust": [], "standard": []} 
    }

    # 1. MANDATORY DROPS (Technical Garbage)
    contract["drop_features"] = [g for g in garbage if g != subject_id]

    for col, stats in variables.items():
        if col in garbage or col == subject_id:
            continue
        
        col_type = stats.get('type')
        p_missing = stats.get('p_missing', 0.0)
        n_distinct = stats.get('n_distinct', 0)

        # 2. QUALITY GATE: High Missingness or Zero Variance
        if p_missing > 0.6 or n_distinct <= 1:
            contract["drop_features"].append(col)
            continue

        # 3. ADVANCED IMPUTATION LOGIC
        if stats.get('n_missing', 0) > 0:
            if col_type == 'Numeric':
                # Use Median for skewed data, Mean for symmetric
                skew = abs(stats.get('skewness', 0.0))
                if skew > 1.0:
                    contract["imputation"]["median"].append(col)
                else:
                    contract["imputation"]["mean"].append(col)
            elif col_type == 'Text':
                contract["imputation"]["categorical"].append(col)

        # 4. ENCODING LOGIC
        if col_type == 'Text':
            if n_distinct <= 10:
                contract["encoding"]["one_hot"].append(col)
            else:
                contract["encoding"]["rare_label"].append(col)

        # 5. TRANSFORMATION & OUTLIER LOGIC
        if col_type == 'Numeric' and col not in protected:
            skewness = stats.get('skewness', 0.0)
            iqr = stats.get('iqr', 0.0)
            q75 = stats.get('75%', 0.0)
            max_val = stats.get('max', 0.0)

            # Check for extreme outliers using Tukey's Fence
            if iqr > 0 and max_val > (q75 + 1.5 * iqr):
                contract["transformations"]["outlier_clipping"].append(col)
            
            # Skewness transformations
            if abs(skewness) > 0.75:
                if stats.get('min', 0) > 0:
                    contract["transformations"]["log"].append(col)
                else:
                    contract["transformations"]["yeo_johnson"].append(col)

        # 6. SMARTER SCALING LOGIC
        if col_type == 'Numeric':
            # Coefficient of Variation (cv) > 0.5 indicates high dispersion
            cv = stats.get('cv', 0.0)
            if cv > 0.5:
                contract["scaling"]["robust"].append(col)
            else:
                contract["scaling"]["standard"].append(col)

    with open(PRE_CLEAN_AUDIT_REPORT, 'w') as f:
        json.dump(contract, f, indent=4)
    
    print(f"✅ Audit Brain Complete. Plan generated with distribution awareness.")

if __name__ == "__main__":
    run_clustering_audit()