
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Paths
BASE = Path.cwd()
TRAIN_FILE = BASE / "research_data" / "research_train_6class.csv"
TEST_DIR = BASE / "datasets" / "test"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 2. Load training data
train_df = pd.read_csv(TRAIN_FILE)

X_train = train_df.drop(columns=["Label"])
y_train = train_df["Label"]

# 3. Train the same model as Experiment B
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 4. Load the two publish-flood test files
ddos_file = TEST_DIR / "MQTT-DDoS-Publish_Flood_test.pcap.csv"
dos_file = TEST_DIR / "MQTT-DoS-Publish_Flood_test.pcap.csv"

ddos_df = pd.read_csv(ddos_file)
dos_df = pd.read_csv(dos_file)

ddos_df["Label"] = "MQTT-DDoS-Publish_Flood"
dos_df["Label"] = "MQTT-DoS-Publish_Flood"

# 5. Predict DDoS Publish samples
X_ddos = ddos_df.drop(columns=["Label"])
y_pred = model.predict(X_ddos)


# Prediction confidence for DDoS Publish samples
probabilities = model.predict_proba(X_ddos)
class_names = list(model.classes_)

ddos_index = class_names.index("MQTT-DDoS-Publish_Flood")
dos_index = class_names.index("MQTT-DoS-Publish_Flood")

ddos_probability = probabilities[:, ddos_index]
dos_probability = probabilities[:, dos_index]

results = pd.DataFrame({
    "actual_label": ddos_df["Label"].values,
    "predicted_label": y_pred,
    "ddos_probability": ddos_probability,
    "dos_probability": dos_probability,
})

results["correct"] = (
    results["actual_label"] == results["predicted_label"]
)

print("\nPrediction confidence:")
print(
    results.groupby("correct")[
        ["ddos_probability", "dos_probability"]
    ].mean().round(4)
)

print("\nConfidence of misclassified samples:")
print(
    results.loc[~results["correct"], [
        "ddos_probability", "dos_probability"
    ]].describe().round(4)
)

results.to_csv(
    RESULTS_DIR / "ddos_prediction_confidence.csv",
    index=False
)

print("\nSaved prediction confidence results.")


# 6. Separate correct and misclassified DDoS samples
misclassified = ddos_df.loc[
    y_pred != ddos_df["Label"]
].copy()

correct = ddos_df.loc[
    y_pred == ddos_df["Label"]
].copy()

print("=" * 65)
print("DDoS PUBLISH MISCLASSIFICATION ANALYSIS")
print("=" * 65)
print(f"Total DDoS test samples: {len(ddos_df)}")
print(f"Correctly classified: {len(correct)}")
print(f"Misclassified: {len(misclassified)}")

print("\nPredicted labels for misclassified samples:")
print(pd.Series(y_pred, index=ddos_df.index)[
    y_pred != ddos_df["Label"]
].value_counts())

# 7. Compare feature means for correct vs misclassified samples
feature_cols = X_train.columns.tolist()

comparison = pd.DataFrame({
    "correct_mean": correct[feature_cols].mean(),
    "misclassified_mean": misclassified[feature_cols].mean(),
})

comparison["absolute_mean_difference"] = (
    comparison["misclassified_mean"] -
    comparison["correct_mean"]
).abs()

comparison = comparison.sort_values(
    "absolute_mean_difference",
    ascending=False
)

# 8. Save results without overwriting existing files
output_file = RESULTS_DIR / "ddos_misclassified_feature_comparison.csv"
comparison.to_csv(output_file)

print("\nTop 15 features differing between correct and")
print("misclassified DDoS Publish samples:")
print(comparison.head(15).to_string())

print(f"\nSaved feature comparison to:\n{output_file}")
print("\nAnalysis complete.")


# 9. Standardized mean differences
correct_means = correct[feature_cols].mean()
wrong_means = misclassified[feature_cols].mean()

correct_vars = correct[feature_cols].var()
wrong_vars = misclassified[feature_cols].var()

pooled_std = (
    (correct_vars + wrong_vars) / 2
) ** 0.5

comparison["standardized_mean_difference"] = (
    wrong_means - correct_means
) / pooled_std.replace(0, float("nan"))

comparison["absolute_smd"] = (
    comparison["standardized_mean_difference"].abs()
)

comparison = comparison.sort_values(
    "absolute_smd",
    ascending=False
)

comparison.to_csv(
    RESULTS_DIR / "ddos_misclassified_feature_comparison.csv"
)

print("\nTop 15 features by absolute standardized mean difference:")
print(
    comparison[
        [
            "correct_mean",
            "misclassified_mean",
            "standardized_mean_difference",
            "absolute_smd",
        ]
    ].head(15).to_string()
)
