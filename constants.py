from pathlib import Path

DATASET = "online_retail"
TARGET_COLUMN = None

artifacts_path = Path(f"artifacts/{DATASET}")
cleaned_path = Path(f"dataset/cleaned/{DATASET}")

artifacts_path.mkdir(parents=True, exist_ok=True)
cleaned_path.mkdir(parents=True, exist_ok=True)

DATASET_PATH = f"dataset/{DATASET}.csv"
ARTIFACTS_PATH = artifacts_path
PROFILER_REPORT_PATH = f"{artifacts_path}/report.json"
PROFILER_CLEAN_REPORT_PATH = f"{artifacts_path}/report-clean.json"
CLEANED_DATASET_PATH = f"{cleaned_path}/silver_cleaned_data.csv"
GOLD_DATASET_PATH = f"{cleaned_path}/gold_feature_engineered_data.csv"
JOBLIB_PIPELINE_PATH = f"{cleaned_path}/{DATASET}.joblib"
BEST_MODEL_PATH = f"{cleaned_path}/{DATASET}_model.joblib"
CLUSTER_PERSONAS_PATH = f"{cleaned_path}/cluster_personas.json"

# Added metadata paths
PRE_CLEAN_AUDIT_REPORT = f"{ARTIFACTS_PATH}/silver_audit.json"
YDATA_REPORT_JSON = f"{ARTIFACTS_PATH}/report.json"
GOLD_AUDIT_REPORT = f"{ARTIFACTS_PATH}/gold_audit.json"
MODEL_METRICS_REPORT = f"{ARTIFACTS_PATH}/metrics.json"
DOMAIN_POLICY_PATH = f"{ARTIFACTS_PATH}/domain_policy.json"
FEATURE_ENGG_STRATEGY_PATH = f"{ARTIFACTS_PATH}/feature_engg_strategy.json"