import pandas as pd
import json
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import numpy as np

from feature_engine.encoding import OneHotEncoder, RareLabelEncoder, CountFrequencyEncoder
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer
from feature_engine.selection import SmartCorrelatedSelection, DropFeatures, DropConstantFeatures
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import RelativeFeatures

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLEANED_DATASET_PATH, GOLD_DATASET_PATH, GOLD_AUDIT_REPORT, JOBLIB_PIPELINE_PATH

def build_pipeline(strat, use_pca_override=None):
    """
    Factory to build pipeline. 
    Redundant/irrelevant features are dropped via SmartCorrelatedSelection.
    """
    steps = []
    
    # 0. Temporal Expansion (Mimicking FeatureWiz)
    temp_strat = strat.get("temporal_features", {})
    date_cols = temp_strat.get("date_cols", [])
    
    if date_cols:
        # Extract features (Year, Month, etc.)
        steps.append(('date_extract', DatetimeFeatures(
            variables=date_cols,
            features_to_extract=temp_strat.get("extract_features", ["year", "month", "day_of_week"]),
            drop_original=False 
        )))
        
        # Calculate Differences (Target - Reference)
        if temp_strat.get("calculate_diffs") and len(date_cols) > 1:
            steps.append(('date_diff', RelativeFeatures(
                variables=[date_cols[-1]], 
                reference=[date_cols[0]],
                func=["sub"]
            )))
        
        # Drop original raw datetime objects
        steps.append(('drop_raw_dates', DropFeatures(features_to_drop=date_cols)))

    # 1. Pruning High-Cardinality/ID columns
    if strat["categorical_encoding"]["drop_features"]:
        steps.append(('drop_hc', DropFeatures(features_to_drop=strat["categorical_encoding"]["drop_features"])))

    # 2. Categorical Encoding
    if strat["categorical_encoding"]["rare_label_cols"]:
        steps.append(('rare', RareLabelEncoder(tol=0.05, n_categories=2, variables=strat["categorical_encoding"]["rare_label_cols"])))
    if strat["categorical_encoding"]["one_hot_cols"]:
        steps.append(('ohe', OneHotEncoder(variables=strat["categorical_encoding"]["one_hot_cols"], drop_last=True)))
    if strat["categorical_encoding"]["frequency_cols"]:
        steps.append(('freq', CountFrequencyEncoder(variables=strat["categorical_encoding"]["frequency_cols"])))

    # 3. Numeric Transformations
    if strat["numerical_transformations"]["log_transform"]:
        steps.append(('log', LogTransformer(variables=strat["numerical_transformations"]["log_transform"])))
    if strat["numerical_transformations"]["yeo_johnson"]:
        steps.append(('yeo', YeoJohnsonTransformer(variables=strat["numerical_transformations"]["yeo_johnson"])))

    # 4. Selection & Pruning (The "FeatureWiz" step)
    # This step will drop the temporal features if they are constant or highly correlated
    steps.append(('const', DropConstantFeatures(tol=0.99)))
    steps.append(('smart_corr', SmartCorrelatedSelection(
        threshold=strat["selection_reduction"]["correlation_threshold"],
        selection_method="variance", # Keep feature with higher variance
        estimator=None
    )))
    
    # 5. Scaling
    scaler = RobustScaler() if strat["numerical_transformations"]["scaling_method"] == "RobustScaler" else StandardScaler()
    steps.append(('scaler', SklearnTransformerWrapper(transformer=scaler)))

    # 6. Dimensionality Reduction
    use_pca = strat["selection_reduction"]["apply_pca"] if use_pca_override is None else use_pca_override
    if use_pca:
        steps.append(('pca', PCA(n_components=strat["selection_reduction"]["pca_variance"])))

    return Pipeline(steps), use_pca

def execute_clustering_engg():
    print("\n🚀 Phase 6: Executing Autonomous Feature Engineering...")
    df = pd.read_csv(CLEANED_DATASET_PATH)
    
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        strat = json.load(f)
    
    # Ensure date columns are actual datetime objects for Feature-Engine
    for col in strat.get("temporal_features", {}).get("date_cols", []):
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Fit and Transform
    pipe, used_pca = build_pipeline(strat)
    #X_gold = pipe.fit_transform(df)

    # --- SPEED CHANGE 1: Fit heavy selectors on a 50k sample instead of 500k ---
    # This prevents O(n^2) correlation matrix overhead on the full dataset
    sample_size = min(100000, len(df))
    print(f"📉 Learning feature selection logic from {sample_size} rows...")
    pipe.fit(df.sample(n=sample_size, random_state=42))

    # --- SPEED CHANGE 2: Transform is O(n), run this on the full dataset ---
    print(f"⚙️ Transforming full dataset ({len(df)} rows)...")
    X_gold = pipe.transform(df)
    
    # --- EVALUATION FEEDBACK LOOP ---
    # test_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_gold)
    # score = silhouette_score(X_gold, test_labels)
    # print(f"📊 Initial Silhouette Score: {score:.4f}")

    # --- SPEED CHANGE 3: Use MiniBatchKMeans for the evaluation loop ---
    print("🧪 Running fast evaluation...")
    mbk = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=2048, n_init=3)
    test_labels = mbk.fit_predict(X_gold)

    # --- SPEED CHANGE 4: Sample Silhouette Score ---
    # Calculating Silhouette on 500k rows is O(n^2). 10k is plenty for a feedback loop.
    idx = np.random.choice(X_gold.shape[0], 100000, replace=False)
    X_eval = X_gold.iloc[idx] if hasattr(X_gold, 'iloc') else X_gold[idx]
    score = silhouette_score(X_eval, test_labels[idx])
    
    print(f"📊 Sampled Silhouette Score: {score:.4f}")

    # Retry without PCA if structure is too compressed
    if score < 0.25 and used_pca:
        print("⚠️ Poor separation. Retrying without PCA...")
        pipe, used_pca = build_pipeline(strat, use_pca_override=False)
        X_gold = pipe.fit_transform(df)
        score = silhouette_score(X_gold, test_labels)
        print(f"📊 Revised Silhouette Score: {score:.4f}")

    # --- PERSISTENCE ---
    joblib.dump(pipe, JOBLIB_PIPELINE_PATH)
    
    if used_pca:
        X_gold_df = pd.DataFrame(X_gold, columns=[f"PC{i+1}" for i in range(X_gold.shape[1])])
    else:
        # get_feature_names_out() will reflect only the features that SURVIVED selection
        X_gold_df = pd.DataFrame(X_gold, columns=pipe.get_feature_names_out())

    X_gold_df.to_csv(GOLD_DATASET_PATH, index=False)
    print(f"✅ Gold Medallion Layer Created with {X_gold_df.shape[1]} features.")

if __name__ == "__main__":
    execute_clustering_engg()