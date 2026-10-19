from pathlib import Path
import pandas as pd
import numpy as np

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
# EXPERIMENT C: TWO-STAGE RANDOM FOREST CLASSIFIER
# ============================================================

BASE_DIR = Path.cwd()
TRAIN_FILE = BASE_DIR / "research_data" / "research_train_6class.csv"
TEST_DIR = BASE_DIR / "datasets" / "test"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# The six classes used in Experiments A and B
CLASSES = [
    "ARP_Spoofing",
    "Benign",
    "MQTT-DDoS-Connect_Flood",
    "MQTT-DDoS-Publish_Flood",
    "MQTT-DoS-Connect_Flood",
    "MQTT-DoS-Publish_Flood"
]

ATTACK_CLASSES = [label for label in CLASSES if label != "Benign"]

# ============================================================
# STEP 1: LOAD TRAINING DATA
# ============================================================

print("Loading training dataset...")

train_df = pd.read_csv(TRAIN_FILE)

if "Label" not in train_df.columns:
    raise ValueError("The training dataset must contain a 'Label' column.")

X_train = train_df.drop(columns=["Label"])
y_train = train_df["Label"]

if set(y_train.unique()) != set(CLASSES):
    raise ValueError(
        "The training labels do not match the expected six classes."
    )

print(f"Training rows: {len(train_df):,}")
print(f"Features: {X_train.shape[1]}")

# ============================================================
# STEP 2: LOAD THE SAME OFFICIAL TEST FILES
# ============================================================

print("\nLoading official test files...")

test_frames = []

for label in CLASSES:
    test_file = TEST_DIR / f"{label}_test.pcap.csv"

    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    df = pd.read_csv(test_file)

    # Assign labels using the test filename, as in Experiment B
    df["Label"] = label
    test_frames.append(df)

test_df = pd.concat(test_frames, ignore_index=True)

X_test = test_df.drop(columns=["Label"])
y_test = test_df["Label"]

# Make sure training and test features have matching columns
if set(X_train.columns) != set(X_test.columns):
    raise ValueError(
        "Training and test feature columns do not match."
    )

# Preserve the exact training feature order
X_test = X_test[X_train.columns]

print(f"Test rows: {len(test_df):,}")
print("Test class distribution:")
print(y_test.value_counts().to_string())

# ============================================================
# STAGE 1: BENIGN VS ATTACK
# ============================================================

print("\nTraining Stage 1: Benign vs Attack...")

y_train_binary = y_train.apply(
    lambda label: "Benign" if label == "Benign" else "Attack"
)

stage1_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

stage1_model.fit(X_train, y_train_binary)

# ============================================================
# STAGE 2: IDENTIFY THE ATTACK TYPE
# ============================================================

print("Training Stage 2: Attack type classification...")

# Train Stage 2 only on attack samples
attack_mask = y_train != "Benign"

X_train_attack = X_train.loc[attack_mask]
y_train_attack = y_train.loc[attack_mask]

stage2_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

stage2_model.fit(X_train_attack, y_train_attack)

# ============================================================
# STEP 3: PREDICT ON THE OFFICIAL TEST SET
# ============================================================

print("\nPredicting test samples...")

# Stage 1 predicts whether each test row is benign or an attack
stage1_predictions = stage1_model.predict(X_test)

# Start with Stage 1's predictions
final_predictions = np.full(
    len(X_test),
    "Benign",
    dtype=object
)

# Only send rows predicted as attacks to Stage 2
predicted_attack_mask = stage1_predictions == "Attack"

if predicted_attack_mask.any():
    attack_predictions = stage2_model.predict(
        X_test.loc[predicted_attack_mask]
    )

    final_predictions[predicted_attack_mask] = attack_predictions

# ============================================================
# STEP 4: EVALUATE THE SIX-CLASS RESULTS
# ============================================================

accuracy = accuracy_score(y_test, final_predictions)

macro_precision = precision_score(
    y_test,
    final_predictions,
    labels=CLASSES,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test,
    final_predictions,
    labels=CLASSES,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    final_predictions,
    labels=CLASSES,
    average="macro",
    zero_division=0
)

report = classification_report(
    y_test,
    final_predictions,
    labels=CLASSES,
    target_names=CLASSES,
    digits=4,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    final_predictions,
    labels=CLASSES
)

# ============================================================
# STEP 5: SAVE RESULTS WITHOUT OVERWRITING A OR B
# ============================================================

summary_file = RESULTS_DIR / "experiment_C_two_stage_summary.txt"
matrix_file = RESULTS_DIR / "experiment_C_two_stage_confusion_matrix.csv"
predictions_file = RESULTS_DIR / "experiment_C_two_stage_predictions.csv"

summary = f"""
EXPERIMENT C: TWO-STAGE RANDOM FOREST
=====================================

Stage 1: Benign vs Attack
Stage 2: Five-class attack classification

Model: RandomForestClassifier
Number of trees per stage: 100
Random state: 42

Training rows: {len(train_df)}
Test rows: {len(test_df)}
Number of features: {X_train.shape[1]}

Accuracy: {accuracy:.4f}
Macro Precision: {macro_precision:.4f}
Macro Recall: {macro_recall:.4f}
Macro F1: {macro_f1:.4f}

CLASSIFICATION REPORT
---------------------
{report}

Confusion matrix class order:
{CLASSES}
"""

summary_file.write_text(summary, encoding="utf-8")

matrix_df = pd.DataFrame(
    matrix,
    index=CLASSES,
    columns=CLASSES
)

matrix_df.to_csv(matrix_file)

prediction_df = pd.DataFrame({
    "Actual_Label": y_test.to_numpy(),
    "Predicted_Label": final_predictions
})

prediction_df.to_csv(predictions_file, index=False)

print("\n" + "=" * 60)
print("EXPERIMENT C RESULTS")
print("=" * 60)
print(f"Accuracy:        {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")

print("\nClassification Report:")
print(report)

print("\nResults saved to:")
print(summary_file)
print(matrix_file)
print(predictions_file)

print("\nExperiment C completed successfully.")
