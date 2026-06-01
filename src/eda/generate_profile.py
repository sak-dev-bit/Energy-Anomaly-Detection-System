import pandas as pd
try:
    from ydata_profiling import ProfileReport
except ImportError:
    ProfileReport = None

INPUT_PATH = "data/raw/energydata_complete.csv"
OUTPUT_PATH = "reports/energy_profile.html"

def generate_profile():
    df = pd.read_csv(INPUT_PATH)
    if ProfileReport:
        profile = ProfileReport(
            df,
            title="Energy Dataset Profiling Report",
            minimal=True
        )
        profile.to_file(OUTPUT_PATH)
    else:
        # Mock for Python 3.14 incompatibility
        with open(OUTPUT_PATH, "w") as f:
            f.write("<html><body><h1>Mock Profiling Report</h1><p>ydata-profiling is incompatible with Python 3.14</p></body></html>")
    print(f"Profile report saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_profile()
