import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLEANED_DATASET_PATH, GOLD_DATASET_PATH, GOLD_AUDIT_REPORT

def run_gold_engineering():
    print("🚀 Phase 5: Gold Engine (Construction & Regularization)")
    
    df = pd.read_csv(CLEANED_DATASET_PATH)
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        plan = json.load(f)

    # 1. Dynamic Aggregation Strategy
    subject_id = plan.get("subject_id")
    agg_strat = plan.get("aggregation_strategy", {})

    if subject_id and subject_id in df.columns:
        print(f"📦 Aggregating data by {subject_id}...")
        
        # Build the aggregation map based on Gold Audit instructions
        agg_map = {}
        for col in plan.get("base_metrics", []):
            if col in agg_strat.get("columns_to_sum", []):
                agg_map[col] = 'sum'
            elif col in agg_strat.get("columns_to_mode", []):
                agg_map[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            else:
                # Default to mean for everything else numeric
                agg_map[col] = 'mean'
        
        if agg_map:
            df = df.groupby(subject_id).agg(agg_map)
    else:
        print("ℹ️ No subject_id found. Proceeding with row-level clustering.")

    # 2. Semantic Construction (Interactions)
    for inter in plan["priority_interactions"]:
        col1, col2 = inter["pair"]
        if col1 in df.columns and col2 in df.columns:
            new_name = inter["name"]
            if inter["logic"] == "multiplication":
                df[new_name] = df[col1] * df[col2]
            elif inter["logic"] == "ratio":
                # Handle division by zero based on strategy error_handling
                denom = df[col2].replace(0, np.nan) if inter.get("error_handling") == "zero_fill" else df[col2]
                df[new_name] = df[col1] / (denom + 1e-6)
                if inter.get("error_handling") == "zero_fill":
                    df[new_name] = df[new_name].fillna(0)

            # CLIP outliers based on the specific percentile in the strategy
            pct = inter.get("outlier_clipping_percentile", 99) / 100
            upper_limit = df[new_name].quantile(pct)
            df[new_name] = df[new_name].clip(upper=upper_limit)
            print(f"✨ Created interaction: {new_name}")

    # 3. Imputation (Sync with Strategy)
    # Strategy provides 'median' or 'mean'
    impute_logic = plan.get("preprocessing", {}).get("imputation", "median")
    imputer = SimpleImputer(strategy=impute_logic)
    
    # Select only numeric for the final matrix
    X_gold = df.select_dtypes(include=[np.number])
    X_imputed = pd.DataFrame(imputer.fit_transform(X_gold), columns=X_gold.columns, index=X_gold.index)

    # 4. Pruning & Selection
    corr_threshold = plan["selection_params"].get("correlation_threshold", 0.85)
    var_threshold = plan["selection_params"].get("variance_threshold", 0.01)
    force_keep = plan["selection_params"].get("force_keep", [])

    # Variance Filter
    variances = X_imputed.var()
    low_var_cols = variances[variances < var_threshold].index
    to_drop_var = [c for c in low_var_cols if c not in force_keep]
    X_imputed.drop(columns=to_drop_var, inplace=True)

    # Correlation Filter
    corr_matrix = X_imputed.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper.columns 
                    if any(upper[col] > corr_threshold) 
                    and col not in force_keep]
    
    X_imputed.drop(columns=to_drop_corr, inplace=True)

    # 5. Dynamic Scaling
    prep = plan.get("preprocessing", {})
    scaler_type = prep.get("scaler", "PowerTransformer")
    scaler = PowerTransformer() if scaler_type == "PowerTransformer" else RobustScaler()
    
    print(f"⚖️ Applying {scaler_type} and {impute_logic} imputation...")
    X_final_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=X_imputed.columns,
        index=X_imputed.index
    )

    X_final_scaled.to_csv(GOLD_DATASET_PATH, index=False)
    print(f"🏆 Gold Dataset finalized: {X_final_scaled.shape[1]} features, {X_final_scaled.shape[0]} samples.")

if __name__ == "__main__":
    run_gold_engineering()