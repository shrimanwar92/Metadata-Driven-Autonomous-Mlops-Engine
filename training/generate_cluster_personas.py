import pandas as pd
import numpy as np
import json
import joblib
import os
import sys

# Add parent directory to path to import constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import GOLD_DATASET_PATH, BEST_MODEL_PATH, GOLD_AUDIT_REPORT, CLUSTER_PERSONAS_PATH

def generate_agnostic_personas(output_path=CLUSTER_PERSONAS_PATH):
    print("💎 Phase 8: Generating Data-Agnostic Cluster Personas...")

    # 1. Load Artifacts
    df = pd.read_csv(GOLD_DATASET_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        audit = json.load(f)
    
    entity_col = audit["metadata"].get("entity_id", "id")

    # 2. Isolate Features
    # We move the ID to the index so it's not part of the statistical math
    if entity_col in df.columns:
        df.set_index(entity_col, inplace=True)

    # 3. Get Predictions (Agnostic Handling)
    # AgglomerativeClustering doesn't have .predict(), it only has .labels_ 
    # from its training fit.
    if hasattr(model, 'labels_'):
        print(f"  🔗 Model uses internal labels (Transductive/Agglomerative).")
        cluster_labels = model.labels_
        
        # Safety check: ensure the data length matches the labels length
        if len(cluster_labels) != len(df):
            print("  ❌ Error: Label length mismatch. For Agglomerative, you must use the same data as training.")
            return
    else:
        print(f"  🔮 Model using .predict() (Inductive/KMeans/GMM).")
        cluster_labels = model.predict(df)

    df['cluster'] = cluster_labels

    # 4. Statistical Baseline (Global vs. Cluster)
    # We calculate the mean and standard deviation for the whole population
    global_mean = df.drop(columns=['cluster']).mean()
    global_std = df.drop(columns=['cluster']).std() + 1e-9  # Avoid division by zero
    
    cluster_profiles = df.groupby('cluster').mean()

    # 5. Calculate Z-Scores (The "Agnostic" Logic)
    # Z = (Cluster Mean - Global Mean) / Global Std
    # This tells us how many standard deviations away a cluster is from the 'norm'
    z_scores = (cluster_profiles - global_mean) / global_std

    personas = []

    for cluster_id in cluster_profiles.index:
        # Find the most deviating traits (Top 3 highest absolute Z-scores)
        # This identifies what is "weird" or "unique" about this specific cluster
        cluster_z = z_scores.loc[cluster_id]
        top_traits = cluster_z.abs().sort_values(ascending=False).head(3).index
        
        characteristics = []
        for trait in top_traits:
            z_val = cluster_z[trait]
            # Convert math to human terms
            intensity = "Significantly High" if z_val > 1.5 else "High" if z_val > 0 else \
                        "Significantly Low" if z_val < -1.5 else "Low"
            characteristics.append(f"{intensity} {trait}")

        # Construct JSON Object
        persona = {
            "cluster_id": int(cluster_id),
            "internal_name": f"Segment_{cluster_id}",
            "population_share": f"{(len(df[df['cluster'] == cluster_id]) / len(df)) * 100:.1f}%",
            "defining_traits": characteristics,
            "description": f"This group is primarily defined by having {characteristics[0]} and {characteristics[1]} relative to the average data point."
        }
        personas.append(persona)

    # 6. Save Report
    with open(output_path, 'w') as f:
        json.dump(personas, f, indent=4)
    
    print(f"✅ Created {len(personas)} personas in {output_path}")
    for p in personas:
        print(f"  - {p['internal_name']} ({p['population_share']}): {', '.join(p['defining_traits'])}")

if __name__ == "__main__":
    generate_agnostic_personas()