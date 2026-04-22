import pandas as pd
import json
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from feature_engine.encoding import OneHotEncoder, RareLabelEncoder, CountFrequencyEncoder
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer
from feature_engine.selection import SmartCorrelatedSelection, DropFeatures, DropConstantFeatures
from feature_engine.wrappers import SklearnTransformerWrapper

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLEANED_DATASET_PATH, GOLD_DATASET_PATH, GOLD_AUDIT_REPORT, JOBLIB_SAVE_PATH

def build_pipeline(strat, use_pca_override=None):
    """Factory to build pipeline based on Strategy JSON."""
    steps = []
    
    # 1. Pruning
    if strat["categorical_encoding"]["drop_features"]:
        steps.append(('drop_hc', DropFeatures(features_to_drop=strat["categorical_encoding"]["drop_features"])))

    # 2. Categorical
    if strat["categorical_encoding"]["rare_label_cols"]:
        steps.append(('rare', RareLabelEncoder(tol=0.05, n_categories=2, variables=strat["categorical_encoding"]["rare_label_cols"])))
    if strat["categorical_encoding"]["one_hot_cols"]:
        steps.append(('ohe', OneHotEncoder(variables=strat["categorical_encoding"]["one_hot_cols"], drop_last=True)))
    if strat["categorical_encoding"]["frequency_cols"]:
        steps.append(('freq', CountFrequencyEncoder(variables=strat["categorical_encoding"]["frequency_cols"])))

    # 3. Numeric
    if strat["numerical_transformations"]["log_transform"]:
        steps.append(('log', LogTransformer(variables=strat["numerical_transformations"]["log_transform"])))
    if strat["numerical_transformations"]["yeo_johnson"]:
        steps.append(('yeo', YeoJohnsonTransformer(variables=strat["numerical_transformations"]["yeo_johnson"])))

    # 4. Selection & Scaling
    steps.append(('const', DropConstantFeatures(tol=0.98)))
    steps.append(('smart_corr', SmartCorrelatedSelection(threshold=strat["selection_reduction"]["correlation_threshold"])))
    
    scaler = RobustScaler() if strat["numerical_transformations"]["scaling_method"] == "RobustScaler" else StandardScaler()
    steps.append(('scaler', SklearnTransformerWrapper(transformer=scaler)))

    # 5. Conditional PCA
    use_pca = strat["selection_reduction"]["apply_pca"] if use_pca_override is None else use_pca_override
    if use_pca:
        steps.append(('pca', PCA(n_components=strat["selection_reduction"]["pca_variance"])))

    return Pipeline(steps), use_pca

def execute_clustering_engg():
    print("\n🚀 Phase 6: Executing Autonomous Feature Engineering...")
    df = pd.read_csv(CLEANED_DATASET_PATH)
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        strat = json.load(f)

    # LOOP 1: Initial Attempt
    pipe, used_pca = build_pipeline(strat)
    X_gold = pipe.fit_transform(df)
    
    # --- EVALUATION FEEDBACK LOOP ---
    # Quick KMeans test to check spatial separation
    test_labels = KMeans(n_clusters=3, random_state=42).fit_predict(X_gold)
    score = silhouette_score(X_gold, test_labels)
    print(f"📊 Initial Silhouette Score: {score:.4f}")

    # If PCA destroyed the cluster density, retry without it
    if score < 0.25 and used_pca:
        print("⚠️ Poor separation. Retrying without PCA...")
        pipe, used_pca = build_pipeline(strat, use_pca_override=False)
        X_gold = pipe.fit_transform(df)
        score = silhouette_score(X_gold, test_labels)
        print(f"📊 Revised Silhouette Score: {score:.4f}")

    # --- PERSISTENCE ---
    joblib.dump(pipe, JOBLIB_SAVE_PATH)
    
    # Convert to DF for Gold Storage
    if used_pca:
        X_gold_df = pd.DataFrame(X_gold, columns=[f"PC{i+1}" for i in range(X_gold.shape[1])])
    else:
        X_gold_df = pd.DataFrame(X_gold, columns=pipe.get_feature_names_out())

    X_gold_df.to_csv(GOLD_DATASET_PATH, index=False)
    print(f"✅ Gold Medallion Layer Created. Final Score: {score:.4f}")

if __name__ == "__main__":
    execute_clustering_engg()