import mlflow
import mlflow.sklearn
import mlflow.pytorch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from anomaly_models import IsolationForestModel, LOFModel, AutoencoderModel
import sys
import os

# Add parent dir to path so we can import anomaly_models directly if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_data():
    df = pd.read_csv("data/processed/energy_features.csv", index_col='date')
    df = df.dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    return X, df.columns.tolist()

def train_isolation_forest(X):
    with mlflow.start_run(run_name="IsolationForest"):
        n_estimators = 100
        contamination = 0.05
        model = IsolationForestModel(n_estimators=n_estimators, contamination=contamination)
        model.fit(X)
        preds = model.predict(X)
        anomaly_rate = preds.mean()

        mlflow.log_param("model", "IsolationForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("contamination", contamination)
        mlflow.log_metric("anomaly_rate", anomaly_rate)
        mlflow.sklearn.log_model(model.model, "model")
        print(f"Isolation Forest Anomaly Rate: {anomaly_rate:.4f}")

def train_lof(X):
    with mlflow.start_run(run_name="LOF"):
        n_neighbors = 20
        contamination = 0.05
        model = LOFModel(n_neighbors=n_neighbors, contamination=contamination)
        model.fit(X)
        preds = model.predict(X)
        anomaly_rate = preds.mean()

        mlflow.log_param("model", "LOF")
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("contamination", contamination)
        mlflow.log_metric("anomaly_rate", anomaly_rate)
        mlflow.sklearn.log_model(model.model, "model")
        print(f"LOF Anomaly Rate: {anomaly_rate:.4f}")

def train_autoencoder(X):
    with mlflow.start_run(run_name="Autoencoder"):
        input_dim = X.shape[1]
        latent_dim = 8
        epochs = 20
        batch_size = 32
        model = AutoencoderModel(input_dim=input_dim, latent_dim=latent_dim)
        model.fit(X, epochs=epochs, batch_size=batch_size)
        preds, error = model.predict(X)
        anomaly_rate = preds.mean()

        mlflow.log_param("model", "Autoencoder")
        mlflow.log_param("latent_dim", latent_dim)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_metric("anomaly_rate", anomaly_rate)
        mlflow.log_metric("recon_error_mean", error.mean())
        mlflow.log_metric("recon_error_std", error.std())
        mlflow.pytorch.log_model(model.model, "model")
        print(f"Autoencoder Anomaly Rate: {anomaly_rate:.4f}")

if __name__ == "__main__":
    mlflow.set_experiment("energy_anomaly_detection")
    X, feature_names = load_data()
    print(f"Data loaded. Shape: {X.shape}")
    
    train_isolation_forest(X)
    train_lof(X)
    train_autoencoder(X)
    print("All models trained and logged to MLflow successfully!")
