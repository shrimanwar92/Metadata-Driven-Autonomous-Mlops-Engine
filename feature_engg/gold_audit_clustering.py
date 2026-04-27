import pandas as pd
import numpy as np
import json
import os
import re
from scipy.stats import skew
from pyclustertend import hopkins
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLEANED_DATASET_PATH, GOLD_AUDIT_REPORT

def is_probable_id(col_name, series, threshold=0.90):
    """
    Heuristic using naming conventions (suffixes) and uniqueness ratios.
    Matches: ID, No, Number, Code, _id, _no, etc.
    """
    # Pattern for standard ID suffixes
    pattern = r'.*(id|no|number|code|_no|_id|_number|_code)$'
    is_id_pattern = bool(re.match(pattern, col_name.lower()))

    if series.nunique() == 0: return False
    uniqueness_ratio = series.nunique() / len(series)
    
    # If it looks like an ID name and has some uniqueness, it's an ID
    if is_id_pattern and uniqueness_ratio > 0.005:
        return True
    
    # Fallback for high-cardinality columns without standard names
    return uniqueness_ratio > threshold

def run_gold_audit_clustering():
    print("🌟 Phase 5: Generating Full-Spectrum Clustering Strategy...")
    df = pd.read_csv(CLEANED_DATASET_PATH)

    date_vars = [col for col in df.columns if 'date' in col.lower()]
    num_vars = [c for c in df.select_dtypes(include=[np.number]).columns if c not in date_vars]
    cat_vars = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in date_vars]

    strategy = {
        "metadata": {
            "hopkins_statistic": None,
            "is_clusterable": False,
            "entity_id": None, 
            "suggested_algo": "KMeans"
        },
        "behavioral_features": {
            "enabled": False,
            "normalize_numeric": True,
            "agg_methods": ["sum", "mean", "std"]
        },
        "temporal_features": {
            "date_cols": date_vars,
            "extract_features": ["month", "day_of_week", "hour"],
            "calculate_diffs": len(date_vars) > 1
        },
        "categorical_encoding": {
            "rare_label_cols": [],
            "one_hot_cols": [],
            "frequency_cols": [],
            "drop_features": []
        },
        "numerical_transformations": {
            "log_transform": [],
            "yeo_johnson": [],
            "scaling_method": "StandardScaler"
        },
        "selection_reduction": {
            "correlation_threshold": 0.90,
            "apply_pca": False,
            "pca_variance": 0.95
        }
    }

    # --- ID IDENTIFICATION & PRIORITIZATION ---
    potential_ids = [col for col in df.columns if is_probable_id(col, df[col])]
    
    # Business logic: We want to cluster Customers/Users over Invoices/Codes
    priority_keywords = ['customer', 'user', 'member', 'account', 'client', 'partner']
    selected_id = None

    if potential_ids:
        for keyword in priority_keywords:
            for pid in potential_ids:
                if keyword in pid.lower():
                    selected_id = pid
                    break
            if selected_id: break
        
        if not selected_id:
            selected_id = potential_ids[0]

    strategy["metadata"]["entity_id"] = selected_id
    
    # Remove the ID from feature processing lists
    if selected_id in cat_vars: cat_vars.remove(selected_id)
    if selected_id in num_vars: num_vars.remove(selected_id)

    # --- CLUSTERABILITY & BEHAVIORAL ---
    try:
        sample_size = min(1000, len(df))
        sample_df = df[num_vars].sample(sample_size).fillna(0)
        h_stat = hopkins(sample_df, sample_size)
        strategy["metadata"]["hopkins_statistic"] = round(h_stat, 4)
        strategy["metadata"]["is_clusterable"] = bool(h_stat < 0.3)
    except:
        strategy["metadata"]["is_clusterable"] = True

    if len(date_vars) > 0 and len(num_vars) > 0:
        strategy["behavioral_features"]["enabled"] = True

    # --- CATEGORICAL & NUMERICAL STRATEGY ---
    for col in cat_vars:
        nunique = df[col].nunique()
        if nunique <= 10:
            strategy["categorical_encoding"]["one_hot_cols"].append(col)
        elif nunique <= 100:
            strategy["categorical_encoding"]["frequency_cols"].append(col)
        else:
            strategy["categorical_encoding"]["drop_features"].append(col)

    for col in num_vars:
        s = skew(df[col].dropna())
        if abs(s) > 0.75:
            if (df[col] > 0).all():
                strategy["numerical_transformations"]["log_transform"].append(col)
            else:
                strategy["numerical_transformations"]["yeo_johnson"].append(col)

    # PCA Check (Aggregation triples features)
    if (len(num_vars) * 3) > 20:
        strategy["selection_reduction"]["apply_pca"] = True

    with open(GOLD_AUDIT_REPORT, 'w') as f:
        json.dump(strategy, f, indent=4)

    print(f"✅ Gold audit Complete. Hopkins: {strategy['metadata']['hopkins_statistic']}")
    print(f"🔑 Entity ID identified: {strategy['metadata']['entity_id']}")
    return strategy

if __name__ == "__main__":
    run_gold_audit_clustering()