from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

BASE = Path.cwd()

TRAIN_FILE = (
    BASE
    / "research_data"
    / "research_train_6class.csv"
)

TEST_DIR = (
    BASE
    / "datasets"
    / "test"
)

RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("EXPERIMENT B: INDEPENDENT TEST-FILE EVALUATION")
print("=" * 70)

print("\nLoading training data...")

train_df = pd.read_csv(TRAIN_FILE)

X_train = train_df.drop(columns=["Label"])
y_train = train_df["Label"]

print(f"Training rows: {len(train_df)}")
print(f"Training features: {len(X_train.columns)}")


# ============================================================
# 3. DEFINE MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 4. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# 5. TEST FILES
# ============================================================

test_files = [
    "Benign_test.pcap.csv",
    "ARP_Spoofing_test.pcap.csv",
    "MQTT-DDoS-Connect_Flood_test.pcap.csv",
    "MQTT-DDoS-Publish_Flood_test.pcap.csv",
    "MQTT-DoS-Connect_Flood_test.pcap.csv",
    "MQTT-DoS-Publish_Flood_test.pcap.csv",
]


# ============================================================
# 6. LOAD AND COMBINE TEST DATA
# ============================================================

test_data = []

for filename in test_files:

    file_path = TEST_DIR / filename

    print(f"\nReading: {filename}")

    df = pd.read_csv(file_path)

    label = filename.replace("_test.pcap.csv", "")

    df["Label"] = label

    print(f"Rows: {len(df)}")

    test_data.append(df)


test_df = pd.concat(
    test_data,
    ignore_index=True
)


# ============================================================
# 7. SEPARATE FEATURES AND LABEL
# ============================================================

X_test = test_df.drop(columns=["Label"])
y_test = test_df["Label"]


# ============================================================
# 8. PREDICTION
# ============================================================

print("\nRunning predictions...")

y_pred = model.predict(X_test)

print("Prediction complete.")


# ============================================================
# 9. OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# 10. PRINT OVERALL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT B RESULTS")
print("=" * 70)

print(f"Test rows:       {len(test_df)}")
print(f"Accuracy:        {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")


# ============================================================
# 11. PER-CLASS RESULTS
# ============================================================

class_names = sorted(y_test.unique())

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    labels=class_names,
    zero_division=0
)

print(report)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_names
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm_df)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

cm_df.to_csv(
    RESULTS_DIR / "experiment_B_confusion_matrix.csv"
)

with open(
    RESULTS_DIR / "experiment_B_summary.txt",
    "w"
) as f:

    f.write(
        "Experiment B: Independent Test-File Evaluation\n\n"
    )

    f.write(f"Test rows: {len(test_df)}\n")
    f.write(f"Accuracy: {accuracy:.6f}\n")
    f.write(
        f"Macro Precision: "
        f"{macro_precision:.6f}\n"
    )
    f.write(
        f"Macro Recall: "
        f"{macro_recall:.6f}\n"
    )
    f.write(
        f"Macro F1: "
        f"{macro_f1:.6f}\n"
    )

    f.write("\nClassification Report:\n")
    f.write(report)


print("\nResults saved in:")
print(RESULTS_DIR)

print("\nExperiment B complete.")