import pandas as pd
import numpy as np
import json
import os
import sys

INPUT_PATH = "data/processed/energy_features.csv"
OUTPUT_CONTEXT = "data/processed/anomalies_context.json"

PRECISION_WEIGHT = 0.7
RECALL_WEIGHT = 0.3

def find_optimal_threshold(scores, y_true=None):
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    best_threshold = None
    best_f1 = -1

    for t in thresholds:
        preds = (scores > t).astype(int)

        if y_true is not None:
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

            # Applying Business Bias
            f1 = (PRECISION_WEIGHT * precision + RECALL_WEIGHT * recall) / (PRECISION_WEIGHT + RECALL_WEIGHT + 1e-8)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

    return best_threshold

def extract_anomaly_context(df, anomaly_indices):
    context = []
    for idx in anomaly_indices:
        row = df.iloc[idx]
        context.append({
            "timestamp": row.name,
            "appliances": float(row["Appliances"]),
            "rolling_mean": float(row["rolling_mean_24h"]),
            "hour": int(row["hour"]),
            "deviation": float(row["Appliances"] - row["rolling_mean_24h"])
        })
    return context

def evaluate():
    print("Loading processed data...")
    df = pd.read_csv(INPUT_PATH)
    df = df.set_index('date')
    
    print("Simulating ground truth labels for threshold optimization...")
    # Simulate y_true: top 5% energy consumption spikes as true anomalies
    y_true = (df['Appliances'] > df['Appliances'].quantile(0.95)).astype(int)
    
    print("Computing anomaly scores...")
    from sklearn.ensemble import IsolationForest
    # Higher score = more anomalous
    clf = IsolationForest(n_estimators=100, random_state=42)
    clf.fit(df.values)
    scores = -clf.decision_function(df.values) 
    
    print("Finding optimal threshold based on Precision-Recall Trade-off...")
    best_t = find_optimal_threshold(scores, y_true.values)
    print(f"Optimal threshold found: {best_t:.4f}")
    
    preds = (scores > best_t).astype(int)
    anomaly_indices = np.where(preds == 1)[0]
    
    print(f"Total anomalies filtered: {len(anomaly_indices)}")
    
    # Cap to top 15 anomalies to speed up zero-shot NLP classification for demonstration
    anomaly_indices = anomaly_indices[:15] 
    
    context = extract_anomaly_context(df, anomaly_indices)
    
    with open(OUTPUT_CONTEXT, "w") as f:
        json.dump(context, f)
    print(f"Anomaly contexts mapped and saved to {OUTPUT_CONTEXT}")

if __name__ == "__main__":
    evaluate()
