import pandas as pd
import json
import os
import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

from feature_engine.encoding import OneHotEncoder, RareLabelEncoder, CountFrequencyEncoder
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer
from feature_engine.selection import SmartCorrelatedSelection, DropFeatures, DropConstantFeatures
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.datetime import DatetimeFeatures

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLEANED_DATASET_PATH, GOLD_DATASET_PATH, GOLD_AUDIT_REPORT

class BehavioralFeatureGenerator:
    """Generates cross-feature interactions before aggregation."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = X.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in numeric_cols:
            max_val = df[col].max()
            if max_val > 0: df[f"{col}_norm"] = df[col] / max_val
        if len(numeric_cols) >= 2:
            for i in range(min(2, len(numeric_cols) - 1)):
                df[f"{numeric_cols[i]}_x_{numeric_cols[i+1]}"] = df[numeric_cols[i]] * df[numeric_cols[i+1]]
        return df

def build_pipeline(strat):
    steps = []
    
    # 0. Temporal Features
    temp_strat = strat.get("temporal_features", {})
    if temp_strat.get("date_cols"):
        steps.append(('date_extract', DatetimeFeatures(
            variables=temp_strat["date_cols"],
            features_to_extract=temp_strat.get("extract_features", ["month", "day_of_week", "hour"]),
            drop_original=True
        )))

    # 1. Pruning & Behavioral
    if strat["categorical_encoding"]["drop_features"]:
        steps.append(('drop_hc', DropFeatures(features_to_drop=strat["categorical_encoding"]["drop_features"])))

    if strat.get("behavioral_features", {}).get("enabled", False):
        steps.append(('behavior', BehavioralFeatureGenerator()))
    
    # 2. Encoding
    if strat["categorical_encoding"]["one_hot_cols"]:
        steps.append(('ohe', OneHotEncoder(variables=strat["categorical_encoding"]["one_hot_cols"], drop_last=True)))
    if strat["categorical_encoding"]["frequency_cols"]:
        steps.append(('freq', CountFrequencyEncoder(variables=strat["categorical_encoding"]["frequency_cols"])))

    # 3. Numeric Transformation & Scaling
    if strat["numerical_transformations"]["log_transform"]:
        steps.append(('log', LogTransformer(variables=strat["numerical_transformations"]["log_transform"])))
    
    scaler = RobustScaler() if strat["numerical_transformations"]["scaling_method"] == "RobustScaler" else StandardScaler()
    steps.append(('scaler', SklearnTransformerWrapper(transformer=scaler)))

    return Pipeline(steps)

def execute_clustering_engg():
    print("\n🚀 Phase 6: Executing Generic Feature Engineering...")
    df = pd.read_csv(CLEANED_DATASET_PATH)
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        strat = json.load(f)

    # 1. Target the ID
    id_col = strat["metadata"].get("entity_id")
    if not id_col or id_col not in df.columns:
        print("⚠️ ID not found in Audit. Using fallback.")
        id_col = df.columns[0]
    print(f"🔑 Aggregating data by: {id_col}")

    # Safety: Remove ID from pipeline drop lists
    if id_col in strat["categorical_encoding"]["drop_features"]:
        strat["categorical_encoding"]["drop_features"].remove(id_col)

    # 2. Isolate Entity ID
    y_id = df[id_col].copy()
    X_features = df.drop(columns=[id_col])

    # Convert dates
    for col in strat.get("temporal_features", {}).get("date_cols", []):
        if col in X_features.columns:
            X_features[col] = pd.to_datetime(X_features[col], errors='coerce')

    # 3. Transactional Transformation
    pipe = build_pipeline(strat)
    X_transformed = pipe.fit_transform(X_features)

    # 4. Behavioral Aggregation (Entity Profiling)
    X_transformed[id_col] = y_id.values
    numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
    if id_col in numeric_cols: numeric_cols.remove(id_col)

    print(f"📊 Consolidating {len(df)} transactions into {y_id.nunique()} behavioral profiles...")
    X_grouped = X_transformed.groupby(id_col)[numeric_cols].agg(['sum', 'mean', 'std']).fillna(0)
    X_grouped.columns = [f"{col[0]}_{col[1]}" for col in X_grouped.columns]

    # 5. Final Selection & Redundancy Check
    final_steps = [
        ('const', DropConstantFeatures(tol=0.99)),
        ('smart_corr', SmartCorrelatedSelection(threshold=0.90, selection_method="variance")),
        ('final_scaler', SklearnTransformerWrapper(transformer=RobustScaler()))
    ]
    
    if strat["selection_reduction"].get("apply_pca"):
        final_steps.append(('pca', PCA(n_components=strat["selection_reduction"]["pca_variance"])))
        
    selector = Pipeline(final_steps)
    X_gold = selector.fit_transform(X_grouped)

    # 6. Save Gold Dataset
    X_gold.to_csv(GOLD_DATASET_PATH) 
    print(f"✅ Gold Medallion Layer Created: {X_gold.shape}")

if __name__ == "__main__":
    execute_clustering_engg()