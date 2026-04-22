import pandas as pd
import joblib
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import BEST_MODEL_PATH, JOBLIB_SAVE_PATH


def run_inference(new_data_df):
    # 1. Load all artifacts
    pipeline = joblib.load(JOBLIB_SAVE_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    
    with open('cluster_personas.json', 'r') as f:
        personas = json.load(f)
    persona_map = {p['Cluster']: p['Persona_Name'] for p in personas}

    # 2. Preprocess (Scale/Encode)
    # The pipeline knows how to handle 'Gender', 'Age', etc.
    X_processed = pipeline.transform(new_data_df)
    
    # 3. Predict Cluster
    cluster_ids = model.predict(X_processed)
    
    # 4. Map to Human Names
    results = new_data_df.copy()
    results['Cluster_ID'] = cluster_ids
    results['Persona'] = [persona_map.get(cid, "Unknown") for cid in cluster_ids]
    
    return results

# --- GENERATING TEST DATA ---
test_data = pd.DataFrame([
    {"Gender": "Male", "Age": 22, "Annual Income (k$)": 15, "Spending Score (1-100)": 81}, # Expected: Young Trend-Setter
    {"Gender": "Female", "Age": 55, "Annual Income (k$)": 120, "Spending Score (1-100)": 15}, # Expected: High-Income Senior
    {"Gender": "Female", "Age": 30, "Annual Income (k$)": 90, "Spending Score (1-100)": 95}, # Expected: VIP
    {"Gender": "Male", "Age": 45, "Annual Income (k$)": 50, "Spending Score (1-100)": 50}   # Expected: Middle-Class
])

if __name__ == "__main__":
    print("🚀 Running Inference on New Customers...")
    inference_output = run_inference(test_data)
    print(inference_output[['Gender', 'Age', 'Persona']])