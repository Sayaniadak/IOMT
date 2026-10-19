
from pathlib import Path
import pandas as pd

# Folder containing our downloaded CSV files
BASE = Path(r"C:\Users\SAYANI\OneDrive\Desktop\IOMT RESEARCH\datasets")

rows = []

for split in ["train", "test"]:
    folder = BASE / split

    for file in sorted(folder.glob("*.csv")):
        try:
            # Read only the CSV header first
            columns = pd.read_csv(file, nrows=0).columns.tolist()

            # Count rows without loading the entire file into memory
            with file.open("r", encoding="utf-8-sig", errors="replace") as f:
                row_count = sum(1 for _ in f) - 1

            rows.append({
                "split": split,
                "file_name": file.name,
                "rows": row_count,
                "columns": len(columns),
                "column_names": ", ".join(columns),
                "status": "OK"
            })

        except Exception as e:
            rows.append({
                "split": split,
                "file_name": file.name,
                "rows": None,
                "columns": None,
                "column_names": "",
                "status": str(e)
            })

result = pd.DataFrame(rows)

print("\nDATASET INVENTORY\n")
print(result[["split", "file_name", "rows", "columns", "status"]]
      .to_string(index=False))

output = Path(r"C:\Users\SAYANI\OneDrive\Desktop\IOMT RESEARCH\dataset_inventory.csv")
result.to_csv(output, index=False)

print(f"\nInventory saved to: {output}")
