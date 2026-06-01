import sqlite3
import os

DB_PATH = "data/anomalies.db"

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS anomaly_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        energy_reading FLOAT NOT NULL,
        severity TEXT CHECK(severity IN ('LOW','MEDIUM','HIGH')),
        nlp_explanation TEXT,
        resolution_status TEXT DEFAULT 'OPEN',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()
    print("Database initialized.")

def get_severity(score, low_thresh=0.5, high_thresh=0.8):
    if score < low_thresh:
        return "LOW"
    elif score < high_thresh:
        return "MEDIUM"
    else:
        return "HIGH"

def log_anomaly(timestamp, energy_reading, severity, explanation):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO anomaly_events (
        timestamp,
        energy_reading,
        severity,
        nlp_explanation,
        resolution_status
    ) VALUES (?, ?, ?, ?, 'OPEN')
    """, (timestamp, energy_reading, severity, explanation))

    conn.commit()
    conn.close()

    print(f"Logged anomaly at {timestamp} with severity {severity}")

def fetch_open_alerts():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM anomaly_events
    WHERE resolution_status = 'OPEN'
    ORDER BY timestamp DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows

def mark_resolved(event_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE anomaly_events
    SET resolution_status = 'RESOLVED'
    WHERE id = ?
    """, (event_id,))

    conn.commit()
    conn.close()
