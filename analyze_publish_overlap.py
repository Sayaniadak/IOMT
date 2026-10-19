
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE = Path(__file__).resolve().parent

TRAIN_FILE = (
    BASE / "research_data" / "research_train_6class.csv"
)

TEST_DIR = BASE / "datasets" / "test"
RESULTS_DIR = BASE / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_DDOS = "MQTT-DDoS-Publish_Flood"
CLASS_DOS = "MQTT-DoS-Publish_Flood"

FEATURES = [
    "ack_count",
    "ack_flag_number",
    "fin_count",
    "rst_count",
]


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

print("=" * 65)
print("MQTT PUBLISH-FLOOD FEATURE OVERLAP ANALYSIS")
print("=" * 65)

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training dataset not found: {TRAIN_FILE}"
    )

train_df = pd.read_csv(TRAIN_FILE)

required_columns = FEATURES + ["Label"]

missing = set(required_columns) - set(train_df.columns)

if missing:
    raise ValueError(
        f"Missing columns in training data: {sorted(missing)}"
    )

train_df = train_df[
    train_df["Label"].isin([CLASS_DDOS, CLASS_DOS])
].copy()

print(f"\nSelected training rows: {len(train_df):,}")


# ============================================================
# 3. LOAD THE CORRESPONDING TEST FILES
# ============================================================

test_files = {
    CLASS_DDOS: TEST_DIR / "MQTT-DDoS-Publish_Flood_test.pcap.csv",
    CLASS_DOS: TEST_DIR / "MQTT-DoS-Publish_Flood_test.pcap.csv",
}

test_frames = []

for label, file_path in test_files.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"Test file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    missing_features = set(FEATURES) - set(df.columns)

    if missing_features:
        raise ValueError(
            f"{file_path.name} is missing: "
            f"{sorted(missing_features)}"
        )

    df = df[FEATURES].copy()
    df["Label"] = label

    test_frames.append(df)

test_df = pd.concat(test_frames, ignore_index=True)

print(f"Selected test rows: {len(test_df):,}")


# ============================================================
# 4. CALCULATE FEATURE STATISTICS
# ============================================================

def calculate_statistics(data, split_name, label, feature):

    values = pd.to_numeric(
        data[feature], errors="raise"
    ).to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"Non-finite values found in {split_name}, "
            f"{label}, feature {feature}"
        )

    return {
        "Dataset": split_name,
        "Class": label,
        "Feature": feature,
        "Count": len(values),
        "Mean": np.mean(values),
        "Median": np.median(values),
        "Standard_Deviation": np.std(values, ddof=1),
        "Q1": np.percentile(values, 25),
        "Q3": np.percentile(values, 75),
        "Min": np.min(values),
        "Max": np.max(values),
    }


rows = []

for split_name, data in [
    ("Training", train_df),
    ("Official_Test", test_df),
]:

    for label in [CLASS_DDOS, CLASS_DOS]:

        class_data = data[data["Label"] == label]

        if class_data.empty:
            raise ValueError(
                f"No {split_name} rows found for {label}"
            )

        for feature in FEATURES:

            rows.append(
                calculate_statistics(
                    class_data,
                    split_name,
                    label,
                    feature,
                )
            )

stats_df = pd.DataFrame(rows)


# ============================================================
# 5. COMPARE THE TWO CLASSES WITHIN EACH DATASET
# ============================================================

overlap_rows = []

for split_name in ["Training", "Official_Test"]:

    for feature in FEATURES:

        ddos_values = (
            train_df if split_name == "Training" else test_df
        )

        ddos_values = ddos_values.loc[
            ddos_values["Label"] == CLASS_DDOS, feature
        ].to_numpy(dtype=float)

        dos_values = (
            train_df if split_name == "Training" else test_df
        )

        dos_values = dos_values.loc[
            dos_values["Label"] == CLASS_DOS, feature
        ].to_numpy(dtype=float)

        # Range overlap: intersection divided by union.
        intersection = max(
            0.0,
            min(np.max(ddos_values), np.max(dos_values))
            - max(np.min(ddos_values), np.min(dos_values)),
        )

        union = (
            max(np.max(ddos_values), np.max(dos_values))
            - min(np.min(ddos_values), np.min(dos_values))
        )

        if union == 0:
            range_overlap = (
                1.0
                if np.min(ddos_values) == np.min(dos_values)
                else 0.0
            )
        else:
            range_overlap = intersection / union

        overlap_rows.append({
            "Dataset": split_name,
            "Feature": feature,
            "DDoS_Mean": np.mean(ddos_values),
            "DoS_Mean": np.mean(dos_values),
            "DDoS_Median": np.median(ddos_values),
            "DoS_Median": np.median(dos_values),
            "DDoS_Min": np.min(ddos_values),
            "DDoS_Max": np.max(ddos_values),
            "DoS_Min": np.min(dos_values),
            "DoS_Max": np.max(dos_values),
            "Range_Overlap_Ratio": range_overlap,
        })

overlap_df = pd.DataFrame(overlap_rows)


# ============================================================
# 6. SAVE RESULTS
# ============================================================

stats_path = RESULTS_DIR / "publish_overlap_statistics.csv"
overlap_path = RESULTS_DIR / "publish_overlap_comparison.csv"

stats_df.to_csv(stats_path, index=False)
overlap_df.to_csv(overlap_path, index=False)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("FEATURE STATISTICS")
print("=" * 65)

print(
    stats_df[
        [
            "Dataset",
            "Class",
            "Feature",
            "Count",
            "Mean",
            "Median",
            "Q1",
            "Q3",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\n" + "=" * 65)
print("RANGE OVERLAP COMPARISON")
print("=" * 65)

print(
    overlap_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\n" + "=" * 65)
print("ANALYSIS COMPLETED")
print("=" * 65)

print(f"Statistics saved to: {stats_path}")
print(f"Overlap comparison saved to: {overlap_path}")

print(
    "\nInterpretation note: Range overlap describes overlap "
    "between minimum-to-maximum ranges only. It does not measure "
    "full distribution overlap or prove the cause of model errors."
)
