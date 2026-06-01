import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide")

def load_energy_data():
    return pd.read_csv("data/processed/energy_features.csv", parse_dates=["date"], index_col="date")

def load_anomalies():
    if not os.path.exists("data/anomalies.db"):
        return pd.DataFrame(columns=["timestamp", "energy_reading", "severity", "nlp_explanation", "resolution_status"])
    conn = sqlite3.connect("data/anomalies.db")
    df = pd.read_sql("SELECT * FROM anomaly_events", conn)
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def plot_energy(df, anomalies):
    fig = go.Figure()

    # Energy line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Appliances"],
        mode="lines",
        name="Energy Consumption"
    ))

    # Anomalies
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies["energy_reading"],
            mode="markers",
            marker=dict(color="red", size=8),
            name="Anomalies"
        ))

    fig.update_layout(
        title="Energy Consumption with Anomalies",
        xaxis_title="Time",
        yaxis_title="Energy"
    )

    return fig

def sidebar_filters(anomalies):
    st.sidebar.header("Filters")

    if anomalies.empty:
        st.sidebar.warning("No anomalies found in database.")
        return anomalies

    min_date = anomalies["timestamp"].min().date()
    max_date = anomalies["timestamp"].max().date()
    
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)

    severity = st.sidebar.multiselect(
        "Severity",
        ["LOW", "MEDIUM", "HIGH"],
        default=["LOW", "MEDIUM", "HIGH"]
    )

    filtered = anomalies[
        (anomalies["timestamp"].dt.date >= start_date) &
        (anomalies["timestamp"].dt.date <= end_date) &
        (anomalies["severity"].isin(severity))
    ]

    return filtered

def color_severity(val):
    if val == "HIGH":
        return "background-color: #ffcccc"
    elif val == "MEDIUM":
        return "background-color: #ffe5cc"
    return "background-color: #ccffcc"

def display_insights(anomalies):
    st.subheader("🚨 Actionable Insights")

    if anomalies.empty:
        st.write("No alerts to display for this criteria.")
        return

    display_df = anomalies.sort_values(by="timestamp", ascending=False)
    
    # Apply styler to map color_severity based on 'severity'
    styled_df = display_df[[
        "timestamp",
        "energy_reading",
        "severity",
        "nlp_explanation",
        "resolution_status"
    ]].style.applymap(color_severity, subset=["severity"])

    st.dataframe(styled_df, use_container_width=True)

st.title("⚡ Energy Anomaly Detection Dashboard")

# Load data
try:
    energy_df = load_energy_data()
    anomalies_df = load_anomalies()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# KPIs
col_kpi1, col_kpi2 = st.columns(2)
with col_kpi1:
    st.metric("Total Anomalies Logged", len(anomalies_df))
with col_kpi2:
    if not anomalies_df.empty:
        st.metric("High Severity Anomalies", (anomalies_df["severity"] == "HIGH").sum())
    else:
        st.metric("High Severity Anomalies", 0)

# Filters
filtered_anomalies = sidebar_filters(anomalies_df)

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    fig = plot_energy(energy_df, filtered_anomalies)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    display_insights(filtered_anomalies)
