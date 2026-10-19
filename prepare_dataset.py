from pathlib import Path
import pandas as pd

BASE = Path.cwd()

TRAIN_DIR = BASE / "datasets" / "train"
OUTPUT_DIR = BASE / "research_data"

OUTPUT_DIR.mkdir(exist_ok=True)

selected_files = [
    "Benign_train.pcap.csv",
    "ARP_Spoofing_train.pcap.csv",
    "MQTT-DDoS-Connect_Flood_train.pcap.csv",
    "MQTT-DDoS-Publish_Flood_train.pcap.csv",
    "MQTT-DoS-Connect_Flood_train.pcap.csv",
    "MQTT-DoS-Publish_Flood_train.pcap.csv",
]

MAX_ROWS_PER_CLASS = 50000

all_data = []

for filename in selected_files:

    file_path = TRAIN_DIR / filename

    print(f"\nReading: {filename}")

    df = pd.read_csv(file_path)

    # Create label from filename
    label = filename.replace("_train.pcap.csv", "")

    # Limit very large classes
    if len(df) > MAX_ROWS_PER_CLASS:
        df = df.sample(
            n=MAX_ROWS_PER_CLASS,
            random_state=42
        )

    df["Label"] = label

    print(f"Label: {label}")
    print(f"Rows used: {len(df)}")

    all_data.append(df)


combined = pd.concat(
    all_data,
    ignore_index=True
)

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print(f"Total rows: {len(combined)}")
print(f"Total columns: {len(combined.columns)}")

print("\nClass distribution:")
print(combined["Label"].value_counts())

output_file = OUTPUT_DIR / "research_train_6class.csv"

combined.to_csv(
    output_file,
    index=False
)

print("\nSaved to:")
print(output_file)