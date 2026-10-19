
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

TRAIN_FILE = ROOT / "research_data" / "research_train_6class.csv"
TEST_DIR = ROOT / "datasets" / "test"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

CLASSES = [
    "ARP_Spoofing",
    "Benign",
    "MQTT-DDoS-Connect_Flood",
    "MQTT-DDoS-Publish_Flood",
    "MQTT-DoS-Connect_Flood",
    "MQTT-DoS-Publish_Flood",
]

ATTACK_CLASSES = [c for c in CLASSES if c != "Benign"]

DDOS_PUBLISH = "MQTT-DDoS-Publish_Flood"
DOS_PUBLISH = "MQTT-DoS-Publish_Flood"

STAGE1_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
PUBLISH_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


# --------------------------------------------------
# 2. LOAD THE TRAINING DATA
# --------------------------------------------------

print("Loading prepared training data...")

train_df = pd.read_csv(TRAIN_FILE)

if "Label" not in train_df.columns:
    raise ValueError("The training CSV does not contain a Label column.")

X = train_df.drop(columns=["Label"])
y = train_df["Label"].astype(str)

if set(y.unique()) != set(CLASSES):
    raise ValueError(
        f"Unexpected training labels: {sorted(y.unique())}"
    )

feature_columns = X.columns.tolist()

print(f"Training rows: {len(train_df):,}")
print(f"Features: {len(feature_columns)}")


# --------------------------------------------------
# 3. CREATE AN INTERNAL VALIDATION SPLIT
# --------------------------------------------------

print("\nCreating an 80/20 training-validation split...")

X_fit, X_val, y_fit, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Internal training rows: {len(X_fit):,}")
print(f"Validation rows: {len(X_val):,}")


# --------------------------------------------------
# 4. TRAIN THE TWO MODELS ON INTERNAL TRAINING DATA
# --------------------------------------------------

print("\nTraining Stage 1: benign vs attack...")

y_fit_binary = np.where(y_fit == "Benign", "Benign", "Attack")

stage1 = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

stage1.fit(X_fit, y_fit_binary)

print("Training Stage 2: attack-type classifier...")

attack_mask = y_fit != "Benign"

stage2 = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

stage2.fit(
    X_fit.loc[attack_mask],
    y_fit.loc[attack_mask],
)

# Validation probabilities
stage1_classes = list(stage1.classes_)
attack_column = stage1_classes.index("Attack")

val_attack_probability = stage1.predict_proba(X_val)[:, attack_column]

stage2_probabilities = stage2.predict_proba(X_val)
stage2_classes = list(stage2.classes_)

ddos_column = stage2_classes.index(DDOS_PUBLISH)
dos_column = stage2_classes.index(DOS_PUBLISH)

# Base Stage 2 prediction for every validation row
base_stage2_indices = np.argmax(stage2_probabilities, axis=1)
base_stage2_predictions = np.array(stage2_classes)[base_stage2_indices]


# --------------------------------------------------
# 5. DEFINE THE TWO-STAGE PREDICTION RULE
# --------------------------------------------------

def make_predictions(
    attack_probabilities,
    attack_type_probabilities,
    stage1_threshold,
    publish_threshold,
):
    """
    Stage 1 decides benign vs attack.
    Stage 2 identifies the attack type.

    For predictions between DDoS Publish and DoS Publish,
    publish_threshold adjusts the decision using:

    P(DDoS) / (P(DDoS) + P(DoS))
    """

    attack_type_classes = list(stage2.classes_)

    ddos_idx = attack_type_classes.index(DDOS_PUBLISH)
    dos_idx = attack_type_classes.index(DOS_PUBLISH)

    predictions = []

    for i, attack_probability in enumerate(attack_probabilities):

        if attack_probability < stage1_threshold:
            predictions.append("Benign")
            continue

        probabilities = attack_type_probabilities[i]
        predicted_index = int(np.argmax(probabilities))
        predicted_label = attack_type_classes[predicted_index]

        # Adjust only decisions involving the two publish classes.
        if predicted_label in (DDOS_PUBLISH, DOS_PUBLISH):
            ddos_probability = probabilities[ddos_idx]
            dos_probability = probabilities[dos_idx]

            denominator = ddos_probability + dos_probability

            if denominator > 0:
                ddos_share = ddos_probability / denominator

                if ddos_share >= publish_threshold:
                    predicted_label = DDOS_PUBLISH
                else:
                    predicted_label = DOS_PUBLISH

        predictions.append(predicted_label)

    return np.asarray(predictions)


# --------------------------------------------------
# 6. SELECT THRESHOLDS USING VALIDATION MACRO-F1
# --------------------------------------------------

print("\nSearching thresholds using validation data only...")

best_score = -1
best_stage1_threshold = None
best_publish_threshold = None
validation_results = []

for threshold1 in STAGE1_THRESHOLDS:
    for threshold_publish in PUBLISH_THRESHOLDS:

        val_predictions = make_predictions(
            val_attack_probability,
            stage2_probabilities,
            threshold1,
            threshold_publish,
        )

        macro_f1 = f1_score(
            y_val,
            val_predictions,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        )

        validation_results.append({
            "stage1_attack_threshold": threshold1,
            "publish_ddos_threshold": threshold_publish,
            "validation_macro_f1": macro_f1,
        })

        if macro_f1 > best_score:
            best_score = macro_f1
            best_stage1_threshold = threshold1
            best_publish_threshold = threshold_publish

validation_results = pd.DataFrame(validation_results)

validation_results.to_csv(
    RESULTS_DIR / "experiment_D_validation_thresholds.csv",
    index=False,
)

print(f"\nBest validation macro-F1: {best_score:.4f}")
print(f"Selected Stage 1 threshold: {best_stage1_threshold}")
print(f"Selected publish threshold: {best_publish_threshold}")


# --------------------------------------------------
# 7. RETRAIN BOTH MODELS ON ALL PREPARED TRAINING DATA
# --------------------------------------------------

print("\nRetraining both models on all prepared training data...")

full_binary_labels = np.where(y == "Benign", "Benign", "Attack")

final_stage1 = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

final_stage1.fit(X, full_binary_labels)

full_attack_mask = y != "Benign"

final_stage2 = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

final_stage2.fit(
    X.loc[full_attack_mask],
    y.loc[full_attack_mask],
)


# --------------------------------------------------
# 8. LOAD THE OFFICIAL TEST FILES
# --------------------------------------------------

print("\nLoading official test files...")

test_frames = []

for label in CLASSES:
    matching_files = sorted(
        TEST_DIR.glob(f"{label}_test.pcap.csv")
    )

    if len(matching_files) != 1:
        raise FileNotFoundError(
            f"Expected one test file for {label}, "
            f"found: {[p.name for p in matching_files]}"
        )

    file_path = matching_files[0]
    frame = pd.read_csv(file_path)

    # The raw CSV files are labelled using their filenames.
    frame["Label"] = label
    test_frames.append(frame)

test_df = pd.concat(test_frames, ignore_index=True)

y_test = test_df["Label"].astype(str)
X_test = test_df.drop(columns=["Label"])

# Ensure the official test features match the training feature order.
missing_features = set(feature_columns) - set(X_test.columns)
extra_features = set(X_test.columns) - set(feature_columns)

if missing_features or extra_features:
    raise ValueError(
        f"Feature mismatch. Missing: {missing_features}; "
        f"Extra: {extra_features}"
    )

X_test = X_test[feature_columns]

print(f"Official test rows: {len(test_df):,}")


# --------------------------------------------------
# 9. EVALUATE ON THE OFFICIAL TEST SET ONCE
# --------------------------------------------------

print("\nEvaluating Experiment D on official test files...")

test_stage1_classes = list(final_stage1.classes_)
test_attack_column = test_stage1_classes.index("Attack")

test_attack_probability = final_stage1.predict_proba(X_test)[
    :, test_attack_column
]

test_stage2_probabilities = final_stage2.predict_proba(X_test)

test_predictions = make_predictions(
    test_attack_probability,
    test_stage2_probabilities,
    best_stage1_threshold,
    best_publish_threshold,
)

accuracy = accuracy_score(y_test, test_predictions)

macro_precision = precision_score(
    y_test,
    test_predictions,
    labels=CLASSES,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    y_test,
    test_predictions,
    labels=CLASSES,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    y_test,
    test_predictions,
    labels=CLASSES,
    average="macro",
    zero_division=0,
)

report = classification_report(
    y_test,
    test_predictions,
    labels=CLASSES,
    target_names=CLASSES,
    zero_division=0,
)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=CLASSES,
)

summary_path = RESULTS_DIR / "experiment_D_threshold_tuning_summary.txt"
cm_path = RESULTS_DIR / "experiment_D_threshold_tuning_confusion_matrix.csv"
predictions_path = RESULTS_DIR / "experiment_D_threshold_tuning_predictions.csv"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("EXPERIMENT D: TWO-STAGE RANDOM FOREST WITH THRESHOLD TUNING\n")
    f.write("=" * 65 + "\n")
    f.write(f"Training rows: {len(train_df)}\n")
    f.write(f"Official test rows: {len(test_df)}\n")
    f.write(f"Features: {len(feature_columns)}\n")
    f.write(f"Selected Stage 1 threshold: {best_stage1_threshold}\n")
    f.write(f"Selected publish threshold: {best_publish_threshold}\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro precision: {macro_precision:.4f}\n")
    f.write(f"Macro recall: {macro_recall:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n\n")
    f.write("Classification report:\n")
    f.write(report)

pd.DataFrame(
    cm,
    index=CLASSES,
    columns=CLASSES,
).to_csv(cm_path)

pd.DataFrame({
    "Actual_Label": y_test.to_numpy(),
    "Predicted_Label": test_predictions,
}).to_csv(predictions_path, index=False)

print("\nEXPERIMENT D RESULTS")
print("=" * 40)
print(f"Accuracy:       {accuracy:.4f}")
print(f"Macro precision:{macro_precision:.4f}")
print(f"Macro recall:   {macro_recall:.4f}")
print(f"Macro F1:       {macro_f1:.4f}")

print("\nClassification report:")
print(report)

print("\nSaved new files:")
print(summary_path)
print(cm_path)
print(predictions_path)
print(RESULTS_DIR / "experiment_D_validation_thresholds.csv")
