# ⚡ Anti-Gravity Energy Anomaly Detection System

An enterprise-grade Machine Learning pipeline designed for detecting and classifying anomalies in energy consumption time-series data. The system is built with industrial reliability in mind (e.g., detecting UPS failures, HVAC inefficiencies, unauthorized loads) and bridges the gap between raw data and actionable operational insights.

## 🌟 Key Features

1. **Automated Data Ingestion & EDA**: Robust fetching of the UCI Appliances Energy Prediction dataset with automated profiling.
2. **Feature Engineering**: Creation of lag, rolling, and temporal domain features to capture operational cycles without data leakage.
3. **Multi-Model Anomaly Detection**: Implementation of Unsupervised models (`Isolation Forest`, `Local Outlier Factor`, `PyTorch Autoencoder`).
4. **MLflow Experiment Tracking**: Systematic tracking of hyperparameters, anomaly rates, and reconstruction errors.
5. **Threshold Optimization**: Business-biased tuning of detection thresholds balancing precision and recall.
6. **NLP Classification Layer**: HuggingFace Zero-Shot Classification (`facebook/bart-large-mnli`) translates numerical anomalies into human-readable alerts (e.g., "HVAC Overload").
7. **Enterprise SQLite Logging**: ACID-compliant persistence of anomalies into a relational database to support maintenance workflows.
8. **Interactive Streamlit Dashboard**: Real-time monitoring, visualization with Plotly, operational filters, and severity-coded insights.

## 📂 Project Structure

```text
anti-gravity-energy-anomaly/
│
├── app/
│   └── dashboard.py               # Streamlit interactive dashboard
├── configs/
├── data/
│   ├── processed/                 # Engineered features & anomaly contexts
│   ├── raw/                       # Raw downloaded datasets
│   └── anomalies.db               # SQLite database for persistent logging
├── notebooks/
├── reports/                       # EDA profiling outputs
├── src/
│   ├── data_ingestion/
│   │   ├── download_data.py       # Data acquisition script
│   │   └── validate_data.py       # Data integrity validation
│   ├── eda/
│   │   └── generate_profile.py    # Automated profiling
│   ├── models/
│   │   ├── anomaly_models.py      # IF, LOF, and Autoencoder implementations
│   │   └── train.py               # MLflow training pipeline
│   ├── tests/
│   │   └── test_db_logger.py      # End-to-end DB logger tests
│   ├── evaluate.py                # Threshold optimization & context extraction
│   ├── nlp_classifier.py          # Zero-shot classification layer
│   └── preprocess.py              # Feature engineering & temporal resampling
├── mlruns/                        # MLflow tracking artifacts
├── requirements.txt
└── README.md
```

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sak-dev-bit/Energy-Anomaly-Detection-System.git
   cd Energy-Anomaly-Detection-System
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Linux/Mac
   .\venv\Scripts\activate        # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage Guide

### 1. Data Ingestion & Preprocessing
Download the raw dataset, validate it, run EDA, and generate engineered features.
```bash
python src/data_ingestion/download_data.py
python src/data_ingestion/validate_data.py
python src/eda/generate_profile.py
python src/preprocess.py
```

### 2. Model Training & MLflow Tracking
Train the Isolation Forest, LOF, and Autoencoder models. Track the runs via MLflow.
```bash
python src/models/train.py
```
*To view the MLflow UI, run `mlflow ui` and visit `http://127.0.0.1:5000`.*

### 3. Threshold Tuning & NLP Classification
Extract anomalies based on an optimized precision-recall threshold and run them through the NLP classification layer.
```bash
python src/evaluate.py
python src/nlp_classifier.py
```
*Classified alerts are simultaneously logged to the SQLite database.*

### 4. Streamlit Dashboard
Launch the operational monitoring dashboard.
```bash
streamlit run app/dashboard.py
```

## 💼 Strategic Value (Eaton Alignment)
This system provides operational teams with actionable intelligence, reducing alert fatigue through precision optimization and plain-language categorization. By seamlessly bridging complex unsupervised ML models with an accessible operational UI, it ensures real-time facility monitoring and robust maintenance traceability.
