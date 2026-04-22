import pandas as pd
import numpy as np
import json
import joblib
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import GOLD_DATASET_PATH, GOLD_AUDIT_REPORT, MODEL_METRICS_REPORT, BEST_MODEL_PATH

def evaluate_algorithm(X, algo_name, k_range=[2, 3, 4, 5, 6, 7, 8]):
    """Evaluates an algorithm across a range of K and finds the best."""
    best_score = -1
    best_results = {}
    
    for k in k_range:
        if algo_name == 'KMeans':
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)
        elif algo_name == 'Agglomerative':
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X)
        elif algo_name == 'GMM':
            model = GaussianMixture(n_components=k, random_state=42)
            labels = model.fit_predict(X)
        
        # Calculate Metrics
        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        
        if sil > best_score:
            best_score = sil
            best_results = {
                "k": k,
                "silhouette": round(sil, 4),
                "calinski_harabasz": round(ch, 4),
                "davies_bouldin": round(db, 4),
                "model_instance": model
            }
            
    return best_results

def run_clustering_training():
    print("🧠 Phase 7: Training & Evaluating Clustering Models...")
    X = pd.read_csv(GOLD_DATASET_PATH)
    
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        audit = json.load(f)
    
    all_metrics = {}
    best_overall_sil = -1
    best_overall_model = None
    best_algo_name = ""

    # Algorithms to compete
    algorithms = ['KMeans', 'Agglomerative', 'GMM']
    
    for algo in algorithms:
        print(f"Testing {algo}...")
        res = evaluate_algorithm(X, algo)
        
        # Store metrics without the model object
        all_metrics[algo] = {
            "optimal_k": res["k"],
            "silhouette": res["silhouette"],
            "calinski_harabasz": res["calinski_harabasz"],
            "davies_bouldin": res["davies_bouldin"]
        }
        
        if res["silhouette"] > best_overall_sil:
            best_overall_sil = res["silhouette"]
            best_overall_model = res["model_instance"]
            best_algo_name = algo

    # Save Metrics JSON
    with open(MODEL_METRICS_REPORT, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    # Save the Champion Model
    joblib.dump(best_overall_model, BEST_MODEL_PATH)
    
    print(f"✅ Training Complete. Champion: {best_algo_name} (K={all_metrics[best_algo_name]['optimal_k']})")
    print(f"📄 Metrics stored in {MODEL_METRICS_REPORT}")
    return all_metrics

if __name__ == "__main__":
    run_clustering_training()