import pandas as pd

FILE_PATH = "data/raw/energydata_complete.csv"

def validate_data():
    df = pd.read_csv(FILE_PATH)

    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)

    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicates:", df.duplicated().sum())

    assert "Appliances" in df.columns, "Target column missing!"

    with open("data/raw/data_summary.txt", "w") as f:
        f.write(str(df.describe()))

    print("Validation complete.")

if __name__ == "__main__":
    validate_data()
