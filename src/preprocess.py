import pandas as pd
import os

INPUT_PATH = "data/raw/energydata_complete.csv"
OUTPUT_PATH = "data/processed/energy_features.csv"

def preprocess_data():
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Original shape: {df.shape}")
    
    # 1. Timestamp Handling
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    # 2. Resampling
    # Select only numeric columns for mean aggregation
    numeric_cols = df.select_dtypes(include='number').columns
    df_resampled = df[numeric_cols].resample('1h').mean()
    df = df_resampled.ffill().bfill()
    
    # 3. Feature Engineering
    # 3.1 Lag Features
    df['lag_1h'] = df['Appliances'].shift(1)
    df['lag_6h'] = df['Appliances'].shift(6)
    
    # 3.2 Rolling Features
    df['rolling_mean_24h'] = df['Appliances'].rolling(window=24).mean()
    df['rolling_std_24h'] = df['Appliances'].rolling(window=24).std()
    
    # 3.3 Domain Features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    # Leakage Prevention
    df = df.dropna()
    print(f"Processed shape: {df.shape}")
    print(f"Engineered Features: {['lag_1h', 'lag_6h', 'rolling_mean_24h', 'rolling_std_24h', 'hour', 'day_of_week']}")
    
    # 4. Save Processed Dataset
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH)
    print(f"Data saved to {OUTPUT_PATH}")
    
    # 5. Validation Checks
    assert df.isnull().sum().sum() == 0, "Missing values found!"
    assert df.index.is_monotonic_increasing, "Index is not monotonic increasing!"
    assert df.index.is_unique, "Index has duplicate timestamps!"
    assert all(col in df.columns for col in ['lag_1h', 'lag_6h', 'rolling_mean_24h', 'rolling_std_24h', 'hour', 'day_of_week']), "Feature columns missing!"
    print("Validation checks passed!")

if __name__ == "__main__":
    preprocess_data()
