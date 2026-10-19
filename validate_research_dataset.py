
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path.cwd()
DATA_FILE = BASE / "research_data" / "research_train_6class.csv"

EXPECTED_CLASSES = {
    "Benign",
    "ARP_Spoofing",
    "MQTT-DDoS-Connect_Flood",
    "MQTT-DDoS-Publish_Flood",
    "MQTT-DoS-Connect_Flood",
    "MQTT-DoS-Publish_Flood",
}

EXPECTED_FEATURE_COUNT = 45
MAX_ROWS_PER_CLASS = 50000


def main():
    print("=" * 65)
    print("RESEARCH DATASET VALIDATION")
    print("=" * 65)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run prepare_dataset.py first."
        )

    df = pd.read_csv(DATA_FILE)

    # 1. Basic structure
    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    if "Label" not in df.columns:
        raise ValueError("Required column 'Label' is missing.")

    X = df.drop(columns=["Label"])
    y = df["Label"]

    if X.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"but found {X.shape[1]}."
        )

    print(f"Feature count: {X.shape[1]}")

    # 2. Labels and class distribution
    print("\nCLASS DISTRIBUTION")
    print(y.value_counts(dropna=False).to_string())

    if y.isna().any():
        raise ValueError("Label column contains missing values.")

    actual_classes = set(y.unique())

    if actual_classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected classes.\n"
            f"Missing: {sorted(EXPECTED_CLASSES - actual_classes)}\n"
            f"Unexpected: {sorted(actual_classes - EXPECTED_CLASSES)}"
        )

    counts = y.value_counts()

    if (counts > MAX_ROWS_PER_CLASS).any():
        print(
            "\nWARNING: One or more classes exceed "
            f"{MAX_ROWS_PER_CLASS:,} rows."
        )

    # 3. Duplicate column names
    if not df.columns.is_unique:
        raise ValueError("Duplicate column names were found.")

    # 4. Numeric features
    non_numeric = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"Non-numeric features found: {non_numeric}"
        )

    print("\nAll features are numeric.")

    # 5. Missing values
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        print("\nMissing values:")
        print(missing.to_string())
        raise ValueError("Missing values must be investigated.")

    print("No missing values.")

    # 6. Infinite values
    numeric_values = X.to_numpy(dtype=float)
    infinite_count = int(np.isinf(numeric_values).sum())

    print(f"Infinite feature values: {infinite_count}")

    if infinite_count:
        raise ValueError("Infinite feature values were found.")

    # 7. Duplicate rows
    duplicate_count = int(df.duplicated().sum())
    print(f"Duplicate rows: {duplicate_count}")

    # Duplicates are reported, not automatically removed.
    if duplicate_count:
        print(
            "NOTE: Investigate duplicates before deciding "
            "whether they should be removed."
        )

    print("\n" + "=" * 65)
    print("VALIDATION PASSED")
    print("=" * 65)
    print("Dataset structure and required checks are valid.")
    print("No dataset values were changed.")


if __name__ == "__main__":
    main()
