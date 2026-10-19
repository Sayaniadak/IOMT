
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path.cwd()
TRAIN_FILE = BASE / "research_data" / "research_train_6class.csv"
TEST_DIR = BASE / "datasets" / "test"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SELECTED_CLASSES = [
    "Benign",
    "ARP_Spoofing",
    "MQTT-DDoS-Connect_Flood",
    "MQTT-DDoS-Publish_Flood",
    "MQTT-DoS-Connect_Flood",
    "MQTT-DoS-Publish_Flood",
]


def main():
    print("=" * 70)
    print("TRAIN-TEST DISTRIBUTION SHIFT ANALYSIS")
    print("=" * 70)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training dataset not found: {TRAIN_FILE}")

    train_df = pd.read_csv(TRAIN_FILE)

    if "Label" not in train_df.columns:
        raise ValueError("Training dataset has no Label column.")

    feature_columns = [
        col for col in train_df.columns if col != "Label"
    ]

    if len(feature_columns) != 45:
        raise ValueError(
            f"Expected 45 features; found {len(feature_columns)}."
        )

    if set(train_df["Label"].unique()) != set(SELECTED_CLASSES):
        raise ValueError("Training dataset classes do not match expected classes.")

    all_results = []
    summary_results = []

    for class_name in SELECTED_CLASSES:
        print(f"\nAnalyzing: {class_name}")

        train_class = train_df.loc[
            train_df["Label"] == class_name, feature_columns
        ]

        test_file = TEST_DIR / f"{class_name}_test.pcap.csv"

        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")

        test_class = pd.read_csv(test_file)

        # Verify test features match the prepared training schema.
        if set(test_class.columns) != set(feature_columns):
            missing = sorted(set(feature_columns) - set(test_class.columns))
            extra = sorted(set(test_class.columns) - set(feature_columns))
            raise ValueError(
                f"Feature mismatch for {class_name}.\n"
                f"Missing: {missing}\nExtra: {extra}"
            )

        test_class = test_class[feature_columns]

        # Ensure all features are numeric and finite.
        for data_name, data in [
            ("training", train_class),
            ("test", test_class),
        ]:
            if not all(pd.api.types.is_numeric_dtype(data[c]) for c in data.columns):
                raise ValueError(
                    f"Non-numeric feature found in {data_name} data for {class_name}."
                )

            values = data.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Missing or infinite values in {data_name} data for {class_name}."
                )

        class_effects = []

        for feature in feature_columns:
            train_values = train_class[feature]
            test_values = test_class[feature]

            train_mean = train_values.mean()
            test_mean = test_values.mean()
            train_std = train_values.std()
            test_std = test_values.std()

            pooled_std = np.sqrt(
                (train_std ** 2 + test_std ** 2) / 2
            )

            if pooled_std == 0:
                effect_size = (
                    0.0 if train_mean == test_mean else np.nan
                )
            else:
                effect_size = abs(train_mean - test_mean) / pooled_std

            row = {
                "Class": class_name,
                "Feature": feature,
                "Train_Rows": len(train_class),
                "Test_Rows": len(test_class),
                "Train_Mean": train_mean,
                "Test_Mean": test_mean,
                "Train_Std": train_std,
                "Test_Std": test_std,
                "Effect_Size": effect_size,
            }

            all_results.append(row)
            class_effects.append(effect_size)

        valid_effects = pd.Series(class_effects).dropna()

        summary_results.append({
            "Class": class_name,
            "Train_Rows": len(train_class),
            "Test_Rows": len(test_class),
            "Features_Compared": len(feature_columns),
            "Mean_Effect_Size": valid_effects.mean(),
            "Median_Effect_Size": valid_effects.median(),
            "Max_Effect_Size": valid_effects.max(),
            "Features_Effect_Above_0_5": int((valid_effects > 0.5).sum()),
            "Features_Effect_Above_1_0": int((valid_effects > 1.0).sum()),
        })

        print(f"Training rows: {len(train_class):,}")
        print(f"Test rows:     {len(test_class):,}")
        print(f"Mean effect size: {valid_effects.mean():.4f}")

    details_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame(summary_results)

    details_path = RESULTS_DIR / "distribution_shift_by_feature.csv"
    summary_path = RESULTS_DIR / "distribution_shift_summary.csv"

    details_df.to_csv(details_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY BY CLASS")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    print("\nLargest feature differences by class:")
    for class_name in SELECTED_CLASSES:
        subset = details_df[details_df["Class"] == class_name]
        subset = subset.sort_values("Effect_Size", ascending=False)

        print(f"\n{class_name}")
        print(
            subset[
                ["Feature", "Effect_Size"]
            ].head(5).to_string(index=False)
        )

    print("\nSaved:")
    print(details_path)
    print(summary_path)
    print("\nDistribution analysis complete.")


if __name__ == "__main__":
    main()
