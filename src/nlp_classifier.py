import json
import pandas as pd
from transformers import pipeline
import os
import sys

# Ensure imports work for db_logger
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from db_logger import init_db, log_anomaly, get_severity

CONTEXT_PATH = "data/processed/anomalies_context.json"
OUTPUT_PATH = "data/processed/anomaly_alerts.csv"

CANDIDATE_LABELS = [
    "HVAC Overload",
    "Late-Night Energy Spike",
    "Equipment Malfunction",
    "Unexpected Load Increase",
    "Sensor Anomaly",
    "Normal Variation"
]

def build_prompt(context):
    return (
        f"At {context['timestamp']}, energy consumption was "
        f"{context['appliances']}W which deviates from normal "
        f"({context['rolling_mean']:.1f}W). Hour: {context['hour']}."
    )

def classify_anomaly(classifier, context):
    text = build_prompt(context)
    result = classifier(text, candidate_labels=CANDIDATE_LABELS)
    return {
        "label": result["labels"][0],
        "confidence": result["scores"][0]
    }

def generate_alerts(context_list):
    print("Initializing NLP Classification Layer...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    alerts = []
    
    # Init DB
    print("Initializing SQLite Database...")
    init_db()
    
    print("Classifying and logging anomalies...")
    for i, ctx in enumerate(context_list):
        classification = classify_anomaly(classifier, ctx)
        
        alert_info = {
            "timestamp": ctx["timestamp"],
            "alert_type": classification["label"],
            "confidence": round(classification["confidence"], 4),
            "deviation": f"{ctx['deviation']:+.1f}W",
            "hour": ctx["hour"],
            "appliances": ctx["appliances"]
        }
        alerts.append(alert_info)
        
        # Log to SQLite
        severity = get_severity(classification['confidence'], 0.5, 0.8)
        log_anomaly(
            timestamp=str(ctx["timestamp"]),
            energy_reading=float(ctx["appliances"]),
            severity=severity,
            explanation=classification["label"]
        )
        print(f" [{i+1}/{len(context_list)}] {ctx['timestamp']} -> {classification['label']} (Conf: {classification['confidence']:.2f})")
        
    return alerts

def main():
    if not os.path.exists(CONTEXT_PATH):
        print(f"Error: {CONTEXT_PATH} not found. Run evaluate.py first.")
        return
        
    with open(CONTEXT_PATH, "r") as f:
        context_list = json.load(f)
        
    alerts = generate_alerts(context_list)
    df_alerts = pd.DataFrame(alerts)
    df_alerts.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\nSuccessfully generated and logged {len(alerts)} actionable alerts!")

if __name__ == "__main__":
    main()
