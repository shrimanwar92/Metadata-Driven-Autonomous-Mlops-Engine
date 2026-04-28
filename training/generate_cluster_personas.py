import pandas as pd
import numpy as np
import json
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import GOLD_DATASET_PATH, BEST_MODEL_PATH, GOLD_AUDIT_REPORT, CLUSTER_PERSONAS_PATH

def generate_total_coverage_personas():
    print("💎 Phase 8: Generating LLM-Ready Micro-Personas")

    # 1. Load Data
    df = pd.read_csv(GOLD_DATASET_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        audit = json.load(f)
    
    entity_col = audit["metadata"].get("entity_id")
    if entity_col in df.columns:
        df.set_index(entity_col, inplace=True)

    # 2. Assign Clusters
    if hasattr(model, 'labels_'):
        df['cluster'] = model.labels_
    else:
        df['cluster'] = model.predict(df)

    # 3. Calculate Z-Scores for every Engineered Feature
    cluster_means = df.groupby('cluster').mean()
    global_mean = df.drop(columns=['cluster']).mean()
    global_std = df.drop(columns=['cluster']).std()
    z_scores = (cluster_means - global_mean) / (global_std + 1e-6)

    personas = []
    
    # 4. Feature interpretation for the LLM
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        
        # Identify top 5 traits (increased from 3 for better LLM context)
        cluster_z = z_scores.loc[cluster_id]
        top_traits = cluster_z.abs().sort_values(ascending=False).head(5).index
        
        characteristics = []
        for trait in top_traits:
            z_val = cluster_z[trait]
            intensity = "Extreme High" if z_val > 2.0 else "High" if z_val > 0.5 else \
                        "Extreme Low" if z_val < -2.0 else "Low"
            characteristics.append(f"{intensity} {trait}")

        # Build persona with specific focus on coverage
        persona = {
            "persona_id": int(cluster_id),
            "population_share": f"{(len(cluster_data) / len(df)) * 100:.2f}%",
            "defining_traits": characteristics,
            "logical_summary": f"Group {cluster_id} captures users defined primarily by {characteristics[0]}."
        }
        personas.append(persona)

    # 5. Save the "LLM Map"
    with open(CLUSTER_PERSONAS_PATH, 'w') as f:
        json.dump(personas, f, indent=4)
    
    print(f"✅ Created {len(personas)} Micro-Personas for LLM mapping.")

if __name__ == "__main__":
    generate_total_coverage_personas()