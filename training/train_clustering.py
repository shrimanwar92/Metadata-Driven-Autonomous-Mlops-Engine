import pandas as pd
import numpy as np
import json
import os
import sys
import joblib
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import (
    GOLD_DATASET_PATH, 
    MODEL_METRICS_REPORT, 
    GOLD_AUDIT_REPORT, 
    BEST_MODEL_PATH, 
    CLUSTER_PERSONAS_PATH
)

def generate_cluster_plot(X, labels, output_path, algo_name):
    """Generates a PCA-based 2D visualization of the clusters."""
    print(f"🎨 Generating {algo_name} visualization...")
    pca = PCA(n_components=2)
    components = pca.fit_transform(X) # Already scaled from previous step
    
    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='viridis', data=pca_df, s=80, alpha=0.6)
    
    plt.title(f'Cluster Distribution ({algo_name} via PCA)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_file = output_path.replace('_labeled.csv', '_viz.png')
    plt.savefig(plot_file)
    plt.close()

def run_training_pipeline():
    print("🧠 Phase 6: Model Audit (Brain)")
    
    # 1. Load Data & Strategy Context
    df = pd.read_csv(GOLD_DATASET_PATH)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    with open(GOLD_AUDIT_REPORT, 'r') as f:
        gold_plan = json.load(f)

    subject_id = gold_plan.get("subject_id")
    target_algo = gold_plan.get("recommended_algorithm", "KMeans")

    if subject_id and subject_id in df.columns:
        df = df.set_index(subject_id)
    
    X = df.select_dtypes(include=[np.number]).dropna()
    
    # 2. Hyperparameter Optimization (K-Selection)
    distortions = []
    k_range = range(2, 11)

    print(f"🔬 Optimizing for {target_algo}...")
    for k in k_range:
        if target_algo == "GMM":
            model = GaussianMixture(n_components=k, random_state=42)
            model.fit(X)
            distortions.append(model.bic(X)) # GMM uses BIC (Bayesian Information Criterion)
        else:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            model.fit(X)
            distortions.append(model.inertia_)

    # Identify optimal K
    if target_algo == "GMM":
        suggested_k = k_range[np.argmin(distortions)] # Best GMM K minimizes BIC
    else:
        kn = KneeLocator(k_range, distortions, curve='convex', direction='decreasing')
        suggested_k = int(kn.knee) if kn.knee else 3

    print(f"✅ Best K identified: {suggested_k}")

    # 3. Final Model Training
    if target_algo == "GMM":
        final_model = GaussianMixture(n_components=suggested_k, random_state=42)
        final_labels = final_model.fit_predict(X)
        # For GMM, we can extract the probability/certainty of the cluster assignment
        probs = final_model.predict_proba(X).max(axis=1)
    else:
        final_model = KMeans(n_clusters=suggested_k, random_state=42, n_init=10)
        final_labels = final_model.fit_predict(X)
        probs = [1.0] * len(X) # KMeans is "hard" clustering (100% assignment)

    # 4. Persona Profiling (Z-Score Logic)
    X_results = X.copy()
    X_results['Cluster'] = final_labels
    X_results['Certainty'] = probs
    
    # Identify traits via Z-scores
    scaler = StandardScaler()
    X_scaled_df = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    X_scaled_df['Cluster'] = final_labels
    cluster_profiles = X_scaled_df.groupby('Cluster').mean()
    
    personas = {}
    for cluster_id, row in cluster_profiles.iterrows():
        traits = {feat: ("High" if val > 0.5 else "Low" if val < -0.5 else "Average") 
                  for feat, val in row.items()}
        
        personas[str(cluster_id)] = {
            "dominant_traits": traits,
            "sample_size": int((final_labels == cluster_id).sum()),
            "mean_certainty": float(X_results[X_results['Cluster'] == cluster_id]['Certainty'].mean())
        }

    # 5. Save Artifacts
    joblib.dump(final_model, BEST_MODEL_PATH)
    with open(CLUSTER_PERSONAS_PATH, 'w') as f:
        json.dump(personas, f, indent=4)

    output_path = GOLD_DATASET_PATH.replace(".csv", "_labeled.csv")
    X_results.to_csv(output_path)
    
    print(f"🏆 {target_algo} Training Complete! Labeled data saved.")
    generate_cluster_plot(X, final_labels, output_path, target_algo)

if __name__ == "__main__":
    run_training_pipeline()