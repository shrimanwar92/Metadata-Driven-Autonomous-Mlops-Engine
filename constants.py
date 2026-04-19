from pathlib import Path

DATASET = "iris"
artifacts_path = Path(f"artifacts/{DATASET}")
cleaned_path = Path(f"dataset/cleaned/{DATASET}")

artifacts_path.mkdir(parents=True, exist_ok=True)
cleaned_path.mkdir(parents=True, exist_ok=True)

DATASET_PATH = f"dataset/{DATASET}.csv"
ARTIFACTS_PATH = artifacts_path
CLEANED_DATASET_PATH = f"{cleaned_path}/silver_cleaned_data.csv"
TARGET_COLUMN = "Species"

# Added metadata paths
PRE_CLEAN_AUDIT_REPORT = f"{ARTIFACTS_PATH}/pre_clean_audit.json"
YDATA_REPORT_JSON = f"{ARTIFACTS_PATH}/report.json"

PPS_THRESHOLD = 0.005
VIF_THRESHOLD = 20.0