
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE = Path(__file__).resolve().parent

TRAIN_FILE = BASE / "research_data" / "research_train_6class.csv"
TEST_DIR = BASE / "datasets" / "test"
RESULTS_DIR = BASE / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "MQTT-DDoS-Publish_Flood",
    "MQTT-DoS-Publish_Flood",
]

TEST_FILES = {
    "MQTT-DDoS-Publish_Flood":
        "MQTT-DDoS-Publish_Flood_test.pcap.csv",
    "MQTT-DoS-Publish_Flood":
        "MQTT-DoS-Publish_Flood_test.pcap.csv",
}


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("TRAIN-TEST DISTRIBUTION TESTS")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)

if "Label" not in train_df.columns:
    raise ValueError("Training dataset must contain a Label column.")

FEATURES = [
    col for col in train_df.columns if col != "Label"
]

if not FEATURES:
    raise ValueError("No feature columns were found.")

if not all(pd.api.types.is_numeric_dtype(train_df[c]) for c in FEATURES):
    raise ValueError("All training features must be numeric.")

print(f"Training rows: {len(train_df):,}")
print(f"Features tested per class: {len(FEATURES)}")


# ============================================================
# 3. LOAD TEST DATA FOR THE TWO CLASSES
# ============================================================

test_frames = {}

for label, filename in TEST_FILES.items():

    path = TEST_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    df = pd.read_csv(path)

    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(
            f"{filename} is missing features: {sorted(missing)}"
        )

    df = df[FEATURES].copy()

    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in FEATURES):
        raise ValueError(f"Non-numeric feature found in {filename}")

    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError(f"Missing or infinite values found in {filename}")

    test_frames[label] = df

print("Official test files loaded successfully.")


# ============================================================
# 4. RUN KS TESTS AND EFFECT-SIZE CALCULATIONS
# ============================================================

results = []

for label in CLASSES:

    train_class = train_df.loc[
        train_df["Label"] == label, FEATURES
    ]

    test_class = test_frames[label]

    if train_class.empty:
        raise ValueError(f"No training rows found for {label}")

    for feature in FEATURES:

        train_values = train_class[feature].to_numpy(dtype=float)
        test_values = test_class[feature].to_numpy(dtype=float)

        # Two-sample Kolmogorov-Smirnov test.
        ks_result = ks_2samp(
            train_values,
            test_values,
            alternative="two-sided",
            method="auto",
        )

        # Absolute standardized difference in means.
        train_mean = np.mean(train_values)
        test_mean = np.mean(test_values)

        train_var = np.var(train_values, ddof=1)
        test_var = np.var(test_values, ddof=1)

        pooled_sd = np.sqrt((train_var + test_var) / 2)

        if pooled_sd == 0:
            effect_size = (
                0.0 if train_mean == test_mean else np.inf
            )
        else:
            effect_size = abs(test_mean - train_mean) / pooled_sd

        results.append({
            "Class": label,
            "Feature": feature,
            "Train_Count": len(train_values),
            "Test_Count": len(test_values),
            "Train_Mean": train_mean,
            "Test_Mean": test_mean,
            "Train_Median": np.median(train_values),
            "Test_Median": np.median(test_values),
            "KS_Statistic": ks_result.statistic,
            "KS_P_Value": ks_result.pvalue,
            "Absolute_Standardized_Mean_Difference": effect_size,
        })

results_df = pd.DataFrame(results)


# ============================================================
# 5. CORRECT FOR MULTIPLE TESTING
# ============================================================

# Correct all feature-test p-values together using the
# Benjamini-Hochberg false-discovery-rate procedure.

# Benjamini-Hochberg false-discovery-rate correction.
# This avoids needing the statsmodels package.

p_values = results_df["KS_P_Value"].to_numpy(dtype=float)

number_of_tests = len(p_values)
sorted_indices = np.argsort(p_values)
sorted_p_values = p_values[sorted_indices]

adjusted_sorted = np.empty(number_of_tests, dtype=float)

if number_of_tests > 0:
    adjusted_sorted = (
        sorted_p_values
        * number_of_tests
        / np.arange(1, number_of_tests + 1)
    )

    # Enforce monotonic adjusted p-values.
    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]

    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

adjusted_p_values = np.empty(number_of_tests, dtype=float)
adjusted_p_values[sorted_indices] = adjusted_sorted

results_df["BH_Adjusted_P_Value"] = adjusted_p_values


results_df["Significant_After_BH_0.05"] = (
    results_df["BH_Adjusted_P_Value"] < 0.05
)

results_df = results_df.sort_values(
    ["Class", "KS_Statistic"],
    ascending=[True, False],
)


# ============================================================
# 6. SAVE COMPLETE RESULTS
# ============================================================

output_file = RESULTS_DIR / "distribution_ks_test_results.csv"

results_df.to_csv(output_file, index=False)


# ============================================================
# 7. PRINT TOP RESULTS
# ============================================================

for label in CLASSES:

    subset = results_df[results_df["Class"] == label]

    print("\n" + "=" * 70)
    print(f"CLASS: {label}")
    print("=" * 70)

    print(
        subset.head(10)[
            [
                "Feature",
                "KS_Statistic",
                "KS_P_Value",
                "BH_Adjusted_P_Value",
                "Absolute_Standardized_Mean_Difference",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    significant_count = int(
        subset["Significant_After_BH_0.05"].sum()
    )

    print(
        f"\nFeatures significant after BH correction: "
        f"{significant_count} / {len(FEATURES)}"
    )

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)
print(f"Results saved to: {output_file}")

print(
    "\nInterpretation: small p-values indicate evidence of a "
    "distribution difference, while the KS statistic describes "
    "its magnitude. Statistical significance alone does not "
    "prove that distribution shift caused model errors."
)
