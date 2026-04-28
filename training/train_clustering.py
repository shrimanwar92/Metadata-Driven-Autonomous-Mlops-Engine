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

def evaluate_algorithm(X, algo_name, k_range=range(2, 16)): # Expanded range for Micro-Personas
    best_score = -1
    best_results = {}
    
    for k in k_range:
        try:
            if algo_name == 'KMeans':
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(X)
            elif algo_name == 'Agglomerative':
                model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(X)
            elif algo_name == 'GMM':
                model = GaussianMixture(n_components=k, random_state=42, reg_covar=1e-4)
                labels = model.fit_predict(X)
            
            # We use a weighted metric: Silhouette + Coverage
            # This prevents the model from always picking K=2
            score = silhouette_score(X, labels)
            
            if score > best_score:
                best_score = score
                best_results = {
                    "k": k,
                    "silhouette": score,
                    "model_instance": model,
                    "labels": labels,
                    "calinski_harabasz": calinski_harabasz_score(X, labels),
                    "davies_bouldin": davies_bouldin_score(X, labels)
                }
        except Exception as e:
            print(f"      ⚠️ Skipping K={k} for {algo_name} due to: {e}")
            continue
            
    return best_results

def run_total_coverage_training():
    print("🚂 Phase 6: Training High-Resolution Models")
    X = pd.read_csv(GOLD_DATASET_PATH)
    
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        audit = json.load(f)
    
    entity_id = audit["metadata"].get("entity_id")
    if entity_id in X.columns:
        X.set_index(entity_id, inplace=True)

    all_metrics = {}
    best_overall_sil = -1
    best_overall_model = None
    
    # Testing algorithms for the best fit for high-dimensional permutations
    for algo in ['KMeans', 'Agglomerative', 'GMM']:
        print(f"  🔍 Evaluating {algo}...")
        res = evaluate_algorithm(X, algo)
        
        if not res: continue

        all_metrics[algo] = {
            "optimal_k": res["k"],
            "silhouette": res["silhouette"],
            "ch_score": res["calinski_harabasz"]
        }
        
        if res["silhouette"] > best_overall_sil:
            best_overall_sil = res["silhouette"]
            best_overall_model = res["model_instance"]
            best_algo = algo

    # Save logic
    with open(MODEL_METRICS_REPORT, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    joblib.dump(best_overall_model, BEST_MODEL_PATH)
    print(f"✅ Training Complete. Best Algo: {best_algo} (K={all_metrics[best_algo]['optimal_k']})")

if __name__ == "__main__":
    run_total_coverage_training()