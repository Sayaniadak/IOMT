from pathlib import Path
import pandas as pd

# Start from the current project folder
BASE = Path.cwd()

print("Current folder:")
print(BASE)

print("\nSearching for ARP_Spoofing_train.pcap.csv ...")

matches = list(BASE.rglob("ARP_Spoofing_train.pcap.csv"))

if not matches:
    print("\n❌ File was not found inside this project folder.")
    print("Check where you saved the downloaded CSV.")
else:
    file = matches[0]

    print("\n✅ File found:")
    print(file)

    df = pd.read_csv(file, nrows=5)

    print("\nNUMBER OF COLUMNS:")
    print(len(df.columns))

    print("\nCOLUMN NAMES:")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i}. {col}")

    print("\nFIRST 5 ROWS:")
    print(df.to_string(index=False))