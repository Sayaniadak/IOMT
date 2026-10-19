from pathlib import Path
import pandas as pd

BASE = Path.cwd()

csv_files = [
    f for f in BASE.rglob("*.csv")
    if f.name != "dataset_inventory.csv"
]

print(f"Found {len(csv_files)} dataset files\n")
print("=" * 80)

for file in sorted(csv_files):

    # Read only the first row to avoid loading the whole dataset
    df = pd.read_csv(file, nrows=1)

    print(f"FILE: {file.name}")
    print(f"PATH: {file.parent}")
    print(f"COLUMNS: {len(df.columns)}")

    # Show columns that might identify a flow/capture/device
    possible_group_columns = [
        col for col in df.columns
        if any(word in col.lower()
               for word in ["id", "time", "timestamp", "flow", "capture",
                            "device", "source", "destination", "ip", "port"])
    ]

    if possible_group_columns:
        print("Possible identifier columns:")
        print(possible_group_columns)
    else:
        print("No obvious identifier/group columns found.")

    print("-" * 80)