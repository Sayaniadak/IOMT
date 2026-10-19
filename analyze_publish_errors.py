
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 1. Project paths and class names
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "research_data"
RESULTS_DIR = ROOT / "results"

TRAIN_FILE = DATA_DIR / "research_train_6class.csv"
TEST_DIR = ROOT / "datasets" / "test"
CONFUSION_FILE = RESULTS_DIR / "experiment_B_confusion_matrix.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_A = "MQTT-DDoS-Publish_Flood"
CLASS_B = "MQTT-DoS-Publish_Flood"

FEATURE_COLUMNS = None


# --------------------------------------------------
# 2. Load the prepared training dataset
# --------------------------------------------------

print("=" * 65)
print("TARGETED MQTT PUBLISH-FLOOD ERROR ANALYSIS")
print("=" * 65)

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Training dataset not found: {TRAIN_FILE}")

train_df = pd.read_csv(TRAIN_FILE)

if "Label" not in train_df.columns:
    raise ValueError("The training dataset must contain a 'Label' column.")

FEATURE_COLUMNS = [
    col for col in train_df.columns if col != "Label"
]

print(f"\nTraining rows: {len(train_df):,}")
print(f"Number of features: {len(FEATURE_COLUMNS)}")


# --------------------------------------------------
# 3. Load the two corresponding official test files
# --------------------------------------------------

test_files = {
    CLASS_A: TEST_DIR / "MQTT-DDoS-Publish_Flood_test.pcap.csv",
    CLASS_B: TEST_DIR / "MQTT-DoS-Publish_Flood_test.pcap.csv",
}

test_frames = []

for label, path in test_files.items():
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    df = pd.read_csv(path)

    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing features in {path.name}: {sorted(missing)}"
        )

    df = df[FEATURE_COLUMNS].copy()
    df["Label"] = label
    test_frames.append(df)

test_df = pd.concat(test_frames, ignore_index=True)

print(f"Targeted test rows: {len(test_df):,}")


# --------------------------------------------------
# 4. Compare training and test distributions
#    using standardized mean differences
# --------------------------------------------------

def standardized_mean_difference(train_values, test_values):
    """Calculate an absolute standardized difference in means."""

    train_values = np.asarray(train_values, dtype=float)
    test_values = np.asarray(test_values, dtype=float)

    train_mean = np.mean(train_values)
    test_mean = np.mean(test_values)

    train_var = np.var(train_values, ddof=1)
    test_var = np.var(test_values, ddof=1)

    pooled_sd = np.sqrt((train_var + test_var) / 2)

    if pooled_sd == 0:
        return 0.0 if train_mean == test_mean else np.inf

    return abs(test_mean - train_mean) / pooled_sd


comparison_rows = []

for label in [CLASS_A, CLASS_B]:
    train_class = train_df.loc[
        train_df["Label"] == label, FEATURE_COLUMNS
    ]

    test_class = test_df.loc[
        test_df["Label"] == label, FEATURE_COLUMNS
    ]

    if train_class.empty:
        raise ValueError(f"No training rows found for {label}.")

    if test_class.empty:
        raise ValueError(f"No test rows found for {label}.")

    for feature in FEATURE_COLUMNS:
        train_values = train_class[feature].to_numpy(dtype=float)
        test_values = test_class[feature].to_numpy(dtype=float)

        comparison_rows.append({
            "Class": label,
            "Feature": feature,
            "Train_Mean": np.mean(train_values),
            "Test_Mean": np.mean(test_values),
            "Train_Std": np.std(train_values, ddof=1),
            "Test_Std": np.std(test_values, ddof=1),
            "Absolute_Standardized_Mean_Difference":
                standardized_mean_difference(train_values, test_values),
        })

comparison_df = pd.DataFrame(comparison_rows)

comparison_df = comparison_df.sort_values(
    ["Class", "Absolute_Standardized_Mean_Difference"],
    ascending=[True, False],
)

comparison_path = RESULTS_DIR / "publish_error_feature_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)

print("\nTop 10 distribution differences per class:")

for label in [CLASS_A, CLASS_B]:
    print(f"\n{label}")

    display_df = comparison_df[comparison_df["Class"] == label].head(10)

    print(
        display_df[
            [
                "Feature",
                "Train_Mean",
                "Test_Mean",
                "Absolute_Standardized_Mean_Difference",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


# --------------------------------------------------
# 5. Read the existing Experiment B confusion matrix
# --------------------------------------------------

print("\n" + "=" * 65)
print("EXISTING EXPERIMENT B CONFUSION MATRIX")
print("=" * 65)

if CONFUSION_FILE.exists():
    cm_df = pd.read_csv(CONFUSION_FILE, index_col=0)

    print("\nSaved confusion matrix:")
    print(cm_df.to_string())

    if CLASS_A in cm_df.index and CLASS_B in cm_df.columns:
        value = cm_df.loc[CLASS_A, CLASS_B]
        print(
            f"\nActual {CLASS_A} predicted as {CLASS_B}: "
            f"{int(value):,}"
        )
else:
    print(
        "Confusion-matrix CSV not found. "
        "The distribution comparison was still saved."
    )


# --------------------------------------------------
# 6. Save a concise text report
# --------------------------------------------------

report_path = RESULTS_DIR / "publish_error_analysis_report.txt"

with open(report_path, "w", encoding="utf-8") as report:
    report.write("TARGETED MQTT PUBLISH-FLOOD ERROR ANALYSIS\n")
    report.write("=" * 55 + "\n\n")
    report.write(f"Training dataset: {TRAIN_FILE.name}\n")
    report.write(f"Targeted test rows: {len(test_df):,}\n")
    report.write(f"Number of features: {len(FEATURE_COLUMNS)}\n\n")

    for label in [CLASS_A, CLASS_B]:
        report.write(f"\nClass: {label}\n")
        report.write("-" * 45 + "\n")

        top_features = comparison_df[
            comparison_df["Class"] == label
        ].head(10)

        report.write(
            top_features[
                [
                    "Feature",
                    "Train_Mean",
                    "Test_Mean",
                    "Absolute_Standardized_Mean_Difference",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
        report.write("\n\n")

    if CONFUSION_FILE.exists():
        report.write("\nEXPERIMENT B CONFUSION MATRIX\n")
        report.write(cm_df.to_string())
        report.write("\n")

print("\n" + "=" * 65)
print("ANALYSIS COMPLETED")
print("=" * 65)
print(f"Feature comparison: {comparison_path}")
print(f"Text report:        {report_path}")
print("\nNo original dataset or experiment result was modified.")
