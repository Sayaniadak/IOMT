from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

BASE = Path.cwd()

DATA_FILE = (
    BASE
    / "research_data"
    / "research_train_6class.csv"
)

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("EXPERIMENT A: RANDOM ROW-LEVEL 5-FOLD CROSS-VALIDATION")
print("=" * 70)

print(f"Rows: {len(df)}")
print(f"Features: {len(df.columns) - 1}")
print(f"Classes: {df['Label'].nunique()}")


# ============================================================
# 2. SEPARATE FEATURES AND LABEL
# ============================================================

X = df.drop(columns=["Label"])
y = df["Label"]


# ============================================================
# 3. DEFINE MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 4. DEFINE STRATIFIED 5-FOLD CV
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 5. STORAGE FOR RESULTS
# ============================================================

fold_results = []

class_names = sorted(y.unique())

per_class_f1 = {
    class_name: []
    for class_name in class_names
}


# ============================================================
# 6. RUN 5 FOLDS
# ============================================================

for fold, (train_index, test_index) in enumerate(
    skf.split(X, y),
    start=1
):

    print("\n" + "-" * 70)
    print(f"FOLD {fold}")
    print("-" * 70)

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
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

    class_f1 = f1_score(
        y_test,
        y_pred,
        labels=class_names,
        average=None,
        zero_division=0
    )

    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")

    fold_results.append({
        "Fold": fold,
        "Accuracy": accuracy,
        "Macro_Precision": precision,
        "Macro_Recall": recall,
        "Macro_F1": macro_f1
    })

    for class_name, score in zip(class_names, class_f1):
        per_class_f1[class_name].append(score)

        print(
            f"F1 - {class_name}: {score:.4f}"
        )


# ============================================================
# 7. OVERALL RESULTS
# ============================================================

results_df = pd.DataFrame(fold_results)

print("\n")
print("=" * 70)
print("EXPERIMENT A RESULTS")
print("=" * 70)

for column in [
    "Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1"
]:

    mean = results_df[column].mean()
    std = results_df[column].std()

    print(
        f"{column}: "
        f"{mean:.4f} ± {std:.4f}"
    )


# ============================================================
# 8. PER-CLASS RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("PER-CLASS F1")
print("=" * 70)

for class_name in class_names:

    scores = np.array(per_class_f1[class_name])

    print(
        f"{class_name}: "
        f"{scores.mean():.4f} ± {scores.std():.4f}"
    )


# ============================================================
# 9. SAVE RESULTS
# ============================================================

RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

results_df.to_csv(
    RESULTS_DIR / "experiment_A_fold_results.csv",
    index=False
)

print("\nResults saved to:")

print(
    RESULTS_DIR
    / "experiment_A_fold_results.csv"
)

print("\nExperiment A complete.")