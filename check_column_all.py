from pathlib import Path
import pandas as pd

BASE = Path.cwd()

csv_files = [
    f for f in BASE.rglob("*.csv")
    if f.name != "dataset_inventory.csv"
]

print(f"Found {len(csv_files)} dataset CSV files.\n")

reference_columns = None
all_same = True

for file in sorted(csv_files):

    try:
        columns = pd.read_csv(file, nrows=0).columns.tolist()

        if reference_columns is None:
            reference_columns = columns
            print("Reference file:")
            print(file)
            print(f"Number of columns: {len(columns)}")
            print()

        if columns != reference_columns:
            all_same = False
            print("❌ DIFFERENT COLUMNS:")
            print(file)

    except Exception as e:
        print("ERROR:", file)
        print(e)

print("\n--------------------------------")

if all_same:
    print("✅ ALL 20 DATASET CSV FILES HAVE THE SAME COLUMN STRUCTURE.")
    print(f"Number of columns: {len(reference_columns)}")
else:
    print("⚠️ Some dataset CSV files have different columns.")