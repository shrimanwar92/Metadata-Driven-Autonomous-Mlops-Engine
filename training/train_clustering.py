import pandas as pd
import numpy as np
import json
import os
import sys
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

# Add parent directory to path to import constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import (
    GOLD_DATASET_PATH, 
    MODEL_METRICS_REPORT, 
    GOLD_AUDIT_REPORT, 
    BEST_MODEL_PATH, 
    CLUSTER_PERSONAS_PATH
)

def run_training_pipeline():
    print("🧠 Phase 6: Model Audit (Brain)")
    
    # 1. Load Data & Context
    df = pd.read_csv(GOLD_DATASET_PATH)
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        gold_plan = json.load(f)

    # Prepare features
    sid = gold_plan.get("subject_id")
    if sid and sid in df.columns:
        df = df.set_index(sid)
    
    X = df.select_dtypes(include=[np.number]).dropna()
    
    # 2. Iterative Search for Optimal K (2 to 10)
    distortions = []
    silhouette_avg = []
    k_range = range(2, 11)

    print(f"🔬 Testing K-range {list(k_range)} to find the mathematical 'sweet spot'...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        distortions.append(kmeans.inertia_)
        
        # FIX: Stratified Sampling for Silhouette Score
        # We need at least 2 labels in the sample to avoid the ValueError
        if len(X) > 10000:
            # Create a temporary dataframe to sample from each cluster
            temp_df = pd.DataFrame({'idx': range(len(labels)), 'label': labels})
            
            # Sample 10k rows, ensuring representation from all clusters
            sample_size = 10000
            # Calculate fraction to sample
            frac = sample_size / len(X)
            
            sample_indices = temp_df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(frac=frac) if len(x) > 1 else x
            )['idx'].values
            
            # Final safety check: if sampling resulted in < 2 unique labels, 
            # fall back to a simple random sample until 2 labels are found
            if len(np.unique(labels[sample_indices])) < 2:
                sample_indices = np.random.choice(len(X), min(len(X), 10000), replace=False)

            score = silhouette_score(X.iloc[sample_indices], labels[sample_indices])
        else:
            score = silhouette_score(X, labels)
            
        silhouette_avg.append(float(score))
        print(f"   - K={k}: Distortion={kmeans.inertia_:.2f}, Silhouette={score:.4f}")

    # 3. Determine Optimal K
    kn = KneeLocator(k_range, distortions, curve='convex', direction='decreasing')
    suggested_k = int(kn.knee) if kn.knee else int(np.argmax(silhouette_avg) + 2)
    
    model_metrics = {
        "suggested_k": suggested_k,
        "metrics": {
            "elbow_k": int(kn.knee) if kn.knee else None,
            "best_silhouette_k": int(np.argmax(silhouette_avg) + 2),
            "silhouette_scores": dict(zip(map(str, k_range), silhouette_avg))
        },
        "features_used": list(X.columns)
    }
    with open(MODEL_METRICS_REPORT, 'w') as f:
        json.dump(model_metrics, f, indent=4)
    
    print(f"\n✅ Audit Complete. Best K identified: {suggested_k}")

    print("\n🚀 Phase 7: Model Engine (Execution)")
    
    # 4. Final Training
    print(f"🧬 Fitting final KMeans model with {suggested_k} clusters...")
    final_model = KMeans(n_clusters=suggested_k, random_state=42, n_init=10)
    final_labels = final_model.fit_predict(X)

    # 5. Persona Profiling (Centroid Analysis)
    X_results = X.copy()
    X_results['Cluster'] = final_labels
    
    # Calculate the mean of each feature per cluster to define "Personas"
    personas = X_results.groupby('Cluster').mean().to_dict(orient='index')

    # 6. Save Final Artifacts
    joblib.dump(final_model, BEST_MODEL_PATH)
    
    with open(CLUSTER_PERSONAS_PATH, 'w') as f:
        json.dump(personas, f, indent=4)

    # Save the labeled dataset
    df_labeled = df.copy()
    df_labeled['Cluster'] = pd.Series(final_labels, index=X.index)
    output_path = GOLD_DATASET_PATH.replace(".csv", "_labeled.csv")
    df_labeled.to_csv(output_path)
    
    print(f"🏆 Training Complete!")
    print(f"📂 Labeled Data: {output_path}")

    # Visual summary
    for cluster_id, traits in personas.items():
        top_trait = max(traits, key=traits.get)
        print(f"   📍 Cluster {cluster_id}: Dominant Signal -> {top_trait}")

if __name__ == "__main__":
    run_training_pipeline()