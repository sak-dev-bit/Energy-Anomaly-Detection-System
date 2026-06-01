import os
import requests

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
OUTPUT_PATH = "data/raw/energydata_complete.csv"

def download_file(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for attempt in range(3):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                print("Download successful")
                return
            else:
                print(f"Failed with status {response.status_code}")
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")

    raise Exception("Download failed after 3 attempts")

if __name__ == "__main__":
    download_file(URL, OUTPUT_PATH)
