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
            # Maintained reg_covar to prevent LinAlgError
            model = GaussianMixture(n_components=k, random_state=42, reg_covar=1e-3)
            labels = model.fit_predict(X)
        
        score = silhouette_score(X, labels)
        
        if score > best_score:
            best_score = score
            best_results = {
                "k": k,
                "silhouette": score,
                "calinski_harabasz": calinski_harabasz_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
                "model_instance": model
            }
            
    return best_results

def run_clustering_training():
    print("🧠 Phase 7: Training & Evaluating Clustering Models...")
    
    # 1. Load Data
    X = pd.read_csv(GOLD_DATASET_PATH)

    # 2. Load the audit to find the dynamic ID name
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        audit = json.load(f)
    entity_col = audit["metadata"].get("entity_id") # e.g., "patientId" or "id"

    # 3. Dynamic Isolation
    if entity_col in X.columns:
        X.set_index(entity_col, inplace=True)
        print(f"  ✅ Entity '{entity_col}' isolated as index.")
    else:
        # Fallback: check for common ID substrings if the audit is missing
        id_candidates = [c for c in X.columns if any(x in c.lower() for x in ['id', 'uuid', 'pk'])]
        if id_candidates:
            X.set_index(id_candidates[0], inplace=True)
            print(f"  ⚠️ Warning: Exact entity not found, using {id_candidates[0]} as index.")
    
    all_metrics = {}
    best_overall_sil = -1
    best_overall_model = None
    best_algo_name = ""

    # Algorithms to compete
    algorithms = ['KMeans', 'Agglomerative', 'GMM']
    
    for algo in algorithms:
        print(f"Testing {algo}...")
        res = evaluate_algorithm(X, algo)
        
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
    
    # Save the best model
    if best_overall_model:
        joblib.dump(best_overall_model, BEST_MODEL_PATH)
        print(f"🏆 Best Model: {best_algo_name} with Silhouette {best_overall_sil:.4f}")

if __name__ == "__main__":
    run_clustering_training()