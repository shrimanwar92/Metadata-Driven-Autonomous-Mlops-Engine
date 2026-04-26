import json
import subprocess
import sys
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import PROFILER_REPORT_PATH


# =========================================================
# LOAD PROFILER REPORT
# =========================================================
def load_profile(path):
    with open(path, "r") as f:
        return json.load(f)


# =========================================================
# DECIDE ENGINE (PURELY FROM PROFILER)
# =========================================================
def decide_pipeline(profile):

    table = profile.get("table", {})
    types = table.get("types", {})

    n_var = table.get("n_var", 0)
    text_cols = types.get("Text", 0)
    num_cols = types.get("Numeric", 0)

    time_index = profile.get("time_index_analysis")

    # 🔥 CASE 1: TRUE TEMPORAL
    if time_index and time_index != "None":
        return "temporal"

    # 🔥 CASE 2: EVENT / TRANSACTIONAL DATA
    # (like your customer dataset)
    if n_var <= 10 and text_cols >= 3:
        return "temporal"

    # 🔥 CASE 3: TABULAR (Ames, Iris, etc.)
    if n_var > 20:
        return "tabular"

    # default
    return "tabular"


# =========================================================
# EXECUTION ROUTER
# =========================================================
def run_pipeline():

    print(f"🧠 Running pipeline")

    # ---- AUDIT ----
    subprocess.run([
        sys.executable,
        "feature_engg/gold_audit_supervised.py"
    ], check=True)

    # ---- FEATURE ENGINEERING ----
    subprocess.run([
        sys.executable,
        "feature_engg/feature_engg_supervised.py"
    ], check=True)


# =========================================================
# MAIN ROUTER
# =========================================================
def run_router():
    print("🚀 Phase 4: Routing using Profiler Report")
    #profile = load_profile(PROFILER_REPORT_PATH)
    #pipeline_type = decide_pipeline(profile)
    #print(f"✅ Selected Pipeline: {pipeline_type}")
    run_pipeline()


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    run_router()