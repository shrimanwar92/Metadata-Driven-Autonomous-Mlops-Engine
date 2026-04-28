import pandas as pd
import json
import os
import sys
from sklearn.pipeline import Pipeline
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer
from feature_engine.selection import DropFeatures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import DATASET_PATH, PRE_CLEAN_AUDIT_REPORT, CLEANED_DATASET_PATH

def run_clustering_cleaning():
    print("⚙️ Phase 3: Cleaning Engine (Execution)")
    df = pd.read_csv(DATASET_PATH, encoding="latin1")
    
    with open(PRE_CLEAN_AUDIT_REPORT, 'r') as f:
        contract = json.load(f)

    steps = []

    # 1. Execution: Drop Technical Garbage
    if contract["drop_features"]:
        # Only drop if the column actually exists in the current dataframe
        valid_drops = [c for c in contract["drop_features"] if c in df.columns]
        steps.append(('drop_garbage', DropFeatures(features_to_drop=valid_drops)))

    # 2. Execution: Imputation
    if contract["imputation"]["mean_median"]:
        steps.append(('num_imputer', MeanMedianImputer(variables=contract["imputation"]["mean_median"])))
    if contract["imputation"]["categorical"]:
        steps.append(('cat_imputer', CategoricalImputer(variables=contract["imputation"]["categorical"])))

    # 3. Execution: Encoding
    if contract["encoding"]["rare_label"]:
        steps.append(('rare_enc', RareLabelEncoder(variables=contract["encoding"]["rare_label"])))
    if contract["encoding"]["one_hot"]:
        steps.append(('ohe_enc', OneHotEncoder(variables=contract["encoding"]["one_hot"])))

    # 4. Execution: Transformations
    if contract["transformations"]["log"]:
        steps.append(('log_trans', LogTransformer(variables=contract["transformations"]["log"])))
    if contract["transformations"]["yeo_johnson"]:
        steps.append(('yeo_trans', YeoJohnsonTransformer(variables=contract["transformations"]["yeo_johnson"])))

    # Run Pipeline
    cleaning_pipe = Pipeline(steps)
    df_silver = cleaning_pipe.fit_transform(df)

    # 5. Execution: Anchor ID to Index
    subject_id = contract.get("subject_id")
    if subject_id and subject_id in df_silver.columns:
        df_silver = df_silver.set_index(subject_id)
        print(f"⚓ Anchor Set: {subject_id}")
    elif subject_id is None:
        df_silver.index.name = "observation_id"
        print("⚓ No Subject ID found. Using Observation Index.")

    # Save Silver Layer
    df_silver.to_csv(CLEANED_DATASET_PATH)
    print(f"✅ Silver Layer saved to {CLEANED_DATASET_PATH}. Shape: {df_silver.shape}")

if __name__ == "__main__":
    run_clustering_cleaning()