import json
import subprocess
import sys
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_pipeline():

    # ---- AUDIT ----
    subprocess.run([
        sys.executable,
        "feature_engg/gold_audit_clustering.py"
    ], check=True)

    # ---- FEATURE ENGINEERING ----
    subprocess.run([
        sys.executable,
        "feature_engg/feature_engg_clustering.py"
    ], check=True)


if __name__ == "__main__":
    print("🚀 Running clustering pipeline...")
    run_pipeline()