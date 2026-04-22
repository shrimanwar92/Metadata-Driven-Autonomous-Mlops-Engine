import pandas as pd
import numpy as np
import json
import os
from scipy.stats import skew
from pyclustertend import hopkins

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLEANED_DATASET_PATH, GOLD_AUDIT_REPORT

def is_probable_id(series, threshold=0.95):
    return series.nunique() / len(series) > threshold

def run_gold_audit_clustering():
    print("🌟 Phase 5: Generating Full-Spectrum Clustering Strategy...")
    df = pd.read_csv(CLEANED_DATASET_PATH)
    
    num_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_vars = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- THE COMPREHENSIVE CONTRACT SCHEMA ---
    strategy = {
        "metadata": {
            "hopkins_statistic": None,
            "is_clusterable": False,
            "suggested_algo": "KMeans"
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
            "correlation_threshold": 0.85,
            "apply_pca": False,
            "pca_variance": 0.95
        }
    }

    # 1. Spatial Validation (Hopkins)
    # < 0.3 is the professional threshold for 'Good' clusterability
    try:
        sample_df = df[num_vars].sample(min(1000, len(df)))
        h_stat = hopkins(sample_df, len(sample_df))
        strategy["metadata"]["hopkins_statistic"] = round(h_stat, 4)
        strategy["metadata"]["is_clusterable"] = bool(h_stat < 0.3)
    except:
        strategy["metadata"]["is_clusterable"] = True

    # 2. Categorical Analysis
    for col in cat_vars:
        nunique = df[col].nunique()
        if is_probable_id(df[col]):
            strategy["categorical_encoding"]["drop_features"].append(col)
        elif nunique <= 10:
            strategy["categorical_encoding"]["one_hot_cols"].append(col)
            strategy["categorical_encoding"]["rare_label_cols"].append(col)
        elif nunique <= 50:
            strategy["categorical_encoding"]["frequency_cols"].append(col)
        else:
            strategy["categorical_encoding"]["drop_features"].append(col)

    # 3. Numerical Analysis
    outlier_count = 0
    for col in num_vars:
        s = skew(df[col].dropna())
        if abs(s) > 0.75:
            if (df[col] > 0).all():
                strategy["numerical_transformations"]["log_transform"].append(col)
            else:
                strategy["numerical_transformations"]["yeo_johnson"].append(col)
        
        q1, q3 = df[col].quantile([0.25, 0.75])
        if ((df[col] > (q3 + 1.5 * (q3-q1))) | (df[col] < (q1 - 1.5 * (q3-q1)))).any():
            outlier_count += 1
    
    if outlier_count > (len(num_vars) * 0.3):
        strategy["numerical_transformations"]["scaling_method"] = "RobustScaler"

    if len(num_vars) > 15:
        strategy["selection_reduction"]["apply_pca"] = True

    # Save Strategy
    with open(GOLD_AUDIT_REPORT, 'w') as f:
        json.dump(strategy, f, indent=4)
    
    print(f"✅ Gold audit Complete. Hopkins: {strategy['metadata']['hopkins_statistic']}")
    return strategy

if __name__ == "__main__":
    run_gold_audit_clustering()