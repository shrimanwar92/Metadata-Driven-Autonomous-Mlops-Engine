import pandas as pd
import numpy as np
import json
import joblib
import sys
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, FunctionTransformer
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from feature_engine.encoding import MeanEncoder, RareLabelEncoder
from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.selection import DropConstantFeatures, SmartCorrelatedSelection
from feature_engine.datetime import DatetimeFeatures
from feature_engine.wrappers import SklearnTransformerWrapper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import (
    CLEANED_DATASET_PATH, GOLD_DATASET_PATH,
    GOLD_AUDIT_REPORT, JOBLIB_PIPELINE_PATH
)

# ==============================
# Helpers
# ==============================

def convert_datetime_columns(X, dt_vars):
    X = X.copy()
    for col in dt_vars:
        if col in X.columns:
            X[col] = pd.to_datetime(X[col], errors='coerce')
    return X


class SafeFeatureDropper:
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.cols if c in X.columns], errors='ignore')


# ==============================
# 🔥 Temporal Feature Generator
# ==============================

class TemporalFeatureGenerator:
    def __init__(self, dt_vars):
        self.dt_vars = dt_vars
        self.generated_features_ = []

    def fit(self, X, y=None):
        self.generated_features_ = []

        if not self.dt_vars or len(self.dt_vars) < 2:
            return self

        for i in range(len(self.dt_vars)):
            for j in range(i + 1, len(self.dt_vars)):
                c1 = self.dt_vars[i]
                c2 = self.dt_vars[j]
                if c1 in X.columns and c2 in X.columns:
                    self.generated_features_.append(f"{c2}_minus_{c1}_days")

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.dt_vars:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col], errors='coerce')

        for i in range(len(self.dt_vars)):
            for j in range(i + 1, len(self.dt_vars)):
                c1 = self.dt_vars[i]
                c2 = self.dt_vars[j]

                if c1 in X.columns and c2 in X.columns:
                    new_col = f"{c2}_minus_{c1}_days"
                    X[new_col] = (X[c2] - X[c1]).dt.days

        print(f"⏱️ Temporal features: {self.generated_features_}")
        return X


# ==============================
# Datetime Feature Selector
# ==============================

class DatetimeFeatureSelector:
    def __init__(self, dt_vars, task):
        self.dt_vars = dt_vars
        self.task = task
        self.selected_cols_ = []

    def fit(self, X, y=None):
        if y is None or not self.dt_vars:
            return self

        dt_cols = []
        for base in self.dt_vars:
            dt_cols.extend([c for c in X.columns if c.startswith(base + "_")])

        if not dt_cols:
            return self

        try:
            mi_func = mutual_info_classif if self.task == "classification" else mutual_info_regression
            scores = mi_func(X[dt_cols].fillna(0), y)
            mi_series = pd.Series(scores, index=dt_cols)

            threshold = max(0.001, mi_series.median() * 0.5)
            self.selected_cols_ = mi_series[mi_series > threshold].index.tolist()

            print(f"🧠 Selected datetime features: {self.selected_cols_}")

        except Exception as e:
            print(f"⚠️ Datetime selection failed: {e}")
            self.selected_cols_ = dt_cols

        return self

    def transform(self, X):
        dt_cols = []
        for base in self.dt_vars:
            dt_cols.extend([c for c in X.columns if c.startswith(base + "_")])

        drop_cols = [c for c in dt_cols if c not in self.selected_cols_]
        return X.drop(columns=drop_cols, errors='ignore')


# ==============================
# Pipeline Builder
# ==============================

def build_dynamic_pipeline(strat, task, scaler_type="standard"):

    steps = []

    dropped = strat["feature_selection"]["drop_features"] or []
    dt_vars = strat.get("datetime_transformations", {}).get("dt_vars", [])

    def filter_cols(cols):
        return [c for c in cols if c not in dropped and c not in dt_vars]

    # DATETIME BLOCK
    if dt_vars:
        steps.append(("to_datetime", FunctionTransformer(
            convert_datetime_columns,
            kw_args={"dt_vars": dt_vars}
        )))

        steps.append(("temporal", TemporalFeatureGenerator(dt_vars)))

        steps.append(("dt_features", DatetimeFeatures(
            variables=dt_vars,
            features_to_extract=["year", "month", "day_of_week", "weekend"]
        )))

        steps.append(("dt_select", DatetimeFeatureSelector(dt_vars, task)))

        steps.append(("drop_dates", SafeFeatureDropper(dt_vars)))

    # DROP INITIAL
    if dropped:
        steps.append(("drop_initial", SafeFeatureDropper(dropped)))

    # CATEGORICAL
    rare_cols = filter_cols(strat["categorical_encoding"].get("rare_label_cols", []))
    if rare_cols:
        steps.append(("rare", RareLabelEncoder(tol=0.05, n_categories=2, variables=rare_cols)))

    mean_cols = filter_cols(strat["categorical_encoding"].get("mean_encoding_cols", []))
    if mean_cols:
        steps.append(("mean", MeanEncoder(variables=mean_cols, smoothing=0.3)))

    # NUMERICAL
    yeo_cols = filter_cols(strat["numerical_transformations"].get("yeo_johnson", []))
    if yeo_cols:
        steps.append(("yeo", YeoJohnsonTransformer(variables=yeo_cols)))

    # CLEANUP
    steps.append(("const", DropConstantFeatures(tol=0.98)))
    steps.append(("corr", SmartCorrelatedSelection(threshold=0.85, selection_method="variance")))

    # SCALING
    scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
    steps.append(("scale", SklearnTransformerWrapper(scaler)))

    return Pipeline(steps)


# ==============================
# Execution
# ==============================

def run_autonomous_supervised_v2():

    print("🚀 Feature Engineering vFinal")

    df = pd.read_csv(CLEANED_DATASET_PATH)

    with open(GOLD_AUDIT_REPORT, 'r') as f:
        strat = json.load(f)

    target = strat["metadata"]["target"]
    task = strat["metadata"]["task"]

    X = df.drop(columns=[target])
    y = df[target]

    # target transform
    if task == "regression" and strat["metadata"].get("apply_log_target"):
        y = np.log1p(y)

    # label encoding
    le = None
    if task == "classification" and y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    pipe = build_dynamic_pipeline(strat, task)

    X_gold = pipe.fit_transform(X, y)

    # 🔥 DO NOT USE get_feature_names_out
    X_gold_df = pd.DataFrame(X_gold)

    # persistence
    bundle = {
        "pipeline": pipe,
        "label_encoder": le,
        "metadata": strat["metadata"]
    }

    joblib.dump(bundle, JOBLIB_PIPELINE_PATH)

    X_gold_df[target] = y
    X_gold_df.to_csv(GOLD_DATASET_PATH, index=False)

    print(f"✅ Gold dataset created | Shape: {X_gold_df.shape}")


if __name__ == "__main__":
    run_autonomous_supervised_v2()