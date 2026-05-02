import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLEANED_DATASET_PATH, GOLD_DATASET_PATH, GOLD_AUDIT_REPORT

def run_gold_engineering():
    print("🚀 Phase 5: Gold Engine (Executing based on Gold Audit)")
    
    # Load Data and Audit Plan
    df = pd.read_csv(CLEANED_DATASET_PATH)
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        audit_plan = json.load(f)

    # --- STEP 1: KEY RESOLUTION ---
    # Direct mapping to the Gold Audit JSON schema
    interactions = audit_plan.get("priority_interactions", [])
    selection = audit_plan.get("selection_params", {})
    prep = audit_plan.get("preprocessing", {})
    pca_cfg = prep.get("pca", {})
    subject_id = audit_plan.get("subject_id")

    # --- STEP 2: SEMANTIC CONSTRUCTION (Interactions) ---
    for inter in interactions:
        col1, col2 = inter["pair"]
        if col1 in df.columns and col2 in df.columns:
            new_name = inter["name"]
            
            if inter["logic"] == "ratio":
                # Handle division by zero based on strategy
                denom = df[col2].replace(0, np.nan) if inter.get("error_handling") == "zero_fill" else df[col2]
                df[new_name] = df[col1] / (denom + 1e-6)
                if inter.get("error_handling") == "zero_fill":
                    df[new_name] = df[new_name].fillna(0)
            
            elif inter["logic"] == "multiplication":
                df[new_name] = df[col1] * df[col2]

            # Outlier Clipping
            pct = inter.get("outlier_clipping_percentile", 99) / 100
            upper_limit = df[new_name].quantile(pct)
            df[new_name] = df[new_name].clip(upper=upper_limit)
            print(f"✨ Created interaction: {new_name}")

    # --- STEP 3: FEATURE SELECTION (Pruning) ---
    X = df.select_dtypes(include=[np.number])
    
    var_threshold = selection.get("variance_threshold", 0.01)
    corr_threshold = selection.get("correlation_threshold", 0.85)
    force_keep = selection.get("force_keep", [])

    # Variance Filter
    X = X.drop(columns=[c for c in X.columns if X[c].var() < var_threshold and c not in force_keep])
    
    # Correlation Filter
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold) and col not in force_keep]
    X.drop(columns=to_drop, inplace=True)
    print(f"✂️ Selection complete. Features retained: {X.shape[1]}")

    # --- STEP 4: IMPUTATION ---
    impute_method = prep.get("imputation", "median")
    imputer = SimpleImputer(strategy=impute_method)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    # --- STEP 5: NORMALIZATION (Scaling) ---
    scaler_type = prep.get("scaler", "PowerTransformer")
    scaler = PowerTransformer() if scaler_type == "PowerTransformer" else RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns, index=X_imputed.index)
    print(f"⚖️ Scaling complete using {scaler_type}.")

    # --- STEP 6: DIMENSIONALITY REDUCTION (PCA) ---
    if pca_cfg.get("method") == "PCA":
        threshold = pca_cfg.get("variance_retention_threshold", 0.95)
        print(f"📉 Triggering PCA with {threshold*100}% variance retention...")
        
        pca = PCA(n_components=threshold)
        pca_results = pca.fit_transform(X_scaled)
        
        # Re-index for consistent output
        X_final = pd.DataFrame(
            pca_results, 
            columns=[f"PC{i+1}" for i in range(pca_results.shape[1])],
            index=X_scaled.index
        )
    else:
        X_final = X_scaled
        print("ℹ️ PCA not requested. Skipping.")

    # Export Final Gold Dataset
    X_final.to_csv(GOLD_DATASET_PATH, index=False)
    print(f"🏆 Gold Dataset finalized: {X_final.shape[1]} features.")

if __name__ == "__main__":
    run_gold_engineering()