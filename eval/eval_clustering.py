import pandas as pd
import numpy as np
import joblib
import json
import sys

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import BEST_MODEL_PATH, JOBLIB_SAVE_PATH, CLEANED_DATASET_PATH, GOLD_DATASET_PATH


def generate_cluster_profiles():
    # 1. Load Artifacts
    model = joblib.load(BEST_MODEL_PATH)
    pipeline = joblib.load(JOBLIB_SAVE_PATH)
    silver_df = pd.read_csv(CLEANED_DATASET_PATH)
    gold_df = pd.read_csv(GOLD_DATASET_PATH)

    # 2. Assign Clusters to the Gold Data
    labels = model.predict(gold_df)
    silver_df['Cluster'] = labels

    # 3. Calculate Cluster Means in Original Units
    profiles = silver_df.groupby('Cluster').agg({
        'Age': 'mean',
        'Annual Income (k$)': 'mean',
        'Spending Score (1-100)': 'mean',
        'Gender': lambda x: x.mode()[0]
    }).reset_index()

    # 4. The Naming Engine (Heuristics)
    def name_cluster(row):
        inc = row['Annual Income (k$)']
        spd = row['Spending Score (1-100)']
        age = row['Age']
        
        label = ""
        if inc > 70 and spd > 70: label = "VIP / Top Spenders"
        elif inc > 70 and spd < 40: label = "High-Income / Frugal"
        elif inc < 40 and spd > 70: label = "Young / Trend-Setters"
        elif inc < 40 and spd < 40: label = "Budget Conscious"
        elif 40 <= inc <= 70: label = "Middle-Class / Average"
        
        if age < 30: label = f"Young {label}"
        elif age > 50: label = f"Senior {label}"
        
        return label

    profiles['Persona_Name'] = profiles.apply(name_cluster, axis=1)
    
    # Save Profiles for the Dashboard
    profiles.to_json('cluster_personas.json', orient='records', indent=4)
    print("✅ Cluster Personas Generated:")
    print(profiles[['Cluster', 'Persona_Name', 'Annual Income (k$)', 'Spending Score (1-100)']])

if __name__ == "__main__":
    generate_cluster_profiles()