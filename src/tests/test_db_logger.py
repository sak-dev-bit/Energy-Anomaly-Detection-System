import sys
import os

# Add parent directory to path so db_logger can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_logger import init_db, log_anomaly, fetch_open_alerts, mark_resolved

def test_db():
    print("--- Running DB Logger Tests ---")
    
    # Initialize DB
    init_db()
    
    # Insert 3 mock anomalies
    print("\nInserting 3 mock anomalies...")
    log_anomaly("2024-01-01 03:00", 450.0, "HIGH", "HVAC Overload")
    log_anomaly("2024-01-01 14:00", 220.0, "MEDIUM", "Equipment Malfunction")
    log_anomaly("2024-01-01 22:00", 110.0, "LOW", "Normal Variation")
    
    # Fetch open alerts
    print("\nFetching open alerts...")
    open_alerts = fetch_open_alerts()
    print(f"Open alerts fetched: {len(open_alerts)}")
    for alert in open_alerts:
        print(alert)
    assert len(open_alerts) >= 3, "Failed to insert/fetch anomalies"
    
    # Mark one as resolved
    target_id = open_alerts[0][0] # ID is first column
    print(f"\nMarking alert ID {target_id} as RESOLVED...")
    mark_resolved(target_id)
    
    # Re-query and verify
    open_alerts_after = fetch_open_alerts()
    print(f"Open alerts fetched after resolution: {len(open_alerts_after)}")
    assert len(open_alerts_after) == len(open_alerts) - 1, "Failed to mark anomaly as resolved"
    
    print("\n--- All Tests Passed Successfully! ---")

if __name__ == "__main__":
    test_db()
