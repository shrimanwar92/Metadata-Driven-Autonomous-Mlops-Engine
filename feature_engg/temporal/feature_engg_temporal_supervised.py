import pandas as pd
import numpy as np
import featuretools as ft
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import CLEANED_DATASET_PATH, TARGET_COLUMN, GOLD_DATASET_PATH, JOBLIB_PIPELINE_PATH


# =====================================================
# 🔥 SAFE GENERIC FEATURE ENGINE (NO LEAKAGE)
# =====================================================
def generate_generic_features(df):

    print("🧠 Generating SAFE relationship features...")

    df = df.copy()

    # ---------------------------------------
    # 1. Detect datetime columns
    # ---------------------------------------
    date_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().mean() > 0.8:
                    df[col] = parsed
                    date_cols.append(col)
            except:
                continue
        elif np.issubdtype(df[col].dtype, np.datetime64):
            date_cols.append(col)

    # ---------------------------------------
    # 2. Temporal features (FIRST ORDER ONLY)
    # ---------------------------------------
    temporal_features = []

    for i in range(len(date_cols)):
        for j in range(i + 1, len(date_cols)):
            c1, c2 = date_cols[i], date_cols[j]

            new_col = f"{c2}_minus_{c1}_days"
            df[new_col] = (df[c2] - df[c1]).dt.days

            temporal_features.append(new_col)

    print(f"⏱️ Temporal features: {len(temporal_features)}")

    # ---------------------------------------
    # 3. Numeric interactions (SAFE)
    # ---------------------------------------
    # 🔥 IMPORTANT: exclude target
    original_num_cols = [
        col for col in df.select_dtypes(include=np.number).columns
        if col != TARGET_COLUMN
    ]

    interaction_features = []

    for i in range(len(original_num_cols)):
        for j in range(i + 1, len(original_num_cols)):
            c1, c2 = original_num_cols[i], original_num_cols[j]

            # FIRST ORDER ONLY (no chaining)
            diff_col = f"{c1}_minus_{c2}"
            ratio_col = f"{c1}_div_{c2}"

            df[diff_col] = df[c1] - df[c2]
            df[ratio_col] = df[c1] / (df[c2] + 1e-5)

            interaction_features.extend([diff_col, ratio_col])

    print(f"🔢 Numeric interaction features: {len(interaction_features)}")

    return df


# =====================================================
# 🚀 MAIN EXECUTION
# =====================================================
def execute_autonomous_feature_engg():

    print("🚀 Starting Autonomous Feature Engineering vFinal...")

    # ---------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------
    df = pd.read_csv(CLEANED_DATASET_PATH)

    # ---------------------------------------
    # 2. GENERATE SAFE FEATURES
    # ---------------------------------------
    df = generate_generic_features(df)

    # ---------------------------------------
    # 3. FEATURETOOLS ENTITYSET
    # ---------------------------------------
    es = ft.EntitySet(id="autonomous_data")
    es.add_dataframe(dataframe_name="base", dataframe=df, index="id", make_index=True)

    # dynamic normalization (same as your logic)
    for col in df.columns:
        if df[col].dtype == 'object' and col != TARGET_COLUMN:
            if 1 < df[col].nunique() < (len(df) * 0.8):
                es.normalize_dataframe(
                    base_dataframe_name="base",
                    new_dataframe_name=f"parent_{col}",
                    index=col
                )

    # ---------------------------------------
    # 4. DFS FEATURE GENERATION
    # ---------------------------------------
    print("⚙️ Running Deep Feature Synthesis...")

    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="base",
        agg_primitives=["mean", "max", "count"],
        trans_primitives=["day", "month", "weekday"],
        max_depth=2
    )

    feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)

    # ---------------------------------------
    # 5. SPLIT
    # ---------------------------------------
    y = feature_matrix[TARGET_COLUMN].astype(int)
    X = feature_matrix.drop(columns=[TARGET_COLUMN])

    # ---------------------------------------
    # 6. PIPELINE
    # ---------------------------------------
    imputer = SimpleImputer(strategy='median')
    selector = VarianceThreshold(threshold=0)

    X_imputed = imputer.fit_transform(X)
    X_variance = selector.fit_transform(X_imputed)

    surviving_cols = X.columns[selector.get_support()]

    # limit features (prevents explosion)
    actual_k = min(25, X_variance.shape[1])

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('selection', SelectKBest(score_func=mutual_info_classif, k=actual_k))
    ])

    print(f"🛠 Selecting top {actual_k} features...")

    X_final = pipeline.fit_transform(X_variance, y)

    final_features = list(
        surviving_cols[pipeline.named_steps['selection'].get_support()]
    )

    # ---------------------------------------
    # 7. SAVE PIPELINE
    # ---------------------------------------
    pipeline_metadata = {
        'imputer': imputer,
        'variance_selector': selector,
        'feature_pipeline': pipeline,
        'input_columns': list(X.columns),
        'final_features': final_features
    }

    joblib.dump(pipeline_metadata, JOBLIB_PIPELINE_PATH)
    print(f"💾 Pipeline saved: {JOBLIB_PIPELINE_PATH}")

    # ---------------------------------------
    # 8. SAVE GOLD DATA
    # ---------------------------------------
    gold_df = pd.DataFrame(X_final, columns=final_features)
    gold_df[TARGET_COLUMN] = y.values

    gold_df.to_csv(GOLD_DATASET_PATH, index=False)

    print("✅ Gold dataset created")
    print(f"📊 Final features: {len(final_features)}")


# =====================================================
# ENTRY
# =====================================================
if __name__ == "__main__":
    execute_autonomous_feature_engg()