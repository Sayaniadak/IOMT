
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE = Path.cwd()
TRAIN_FILE = BASE / "research_data" / "research_train_6class.csv"
TEST_FILE = (
    BASE / "datasets" / "test" /
    "MQTT-DDoS-Publish_Flood_test.pcap.csv"
)

RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("Loading data...", flush=True)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

X_train = train_df.drop(columns=["Label"])
y_train = train_df["Label"]
X_test = test_df[X_train.columns]

actual_label = "MQTT-DDoS-Publish_Flood"
test_labels = pd.Series(actual_label, index=test_df.index)

print("Training model...", flush=True)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

print("Calculating probabilities...", flush=True)

probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test)
classes = list(model.classes_)

ddos_idx = classes.index("MQTT-DDoS-Publish_Flood")
dos_idx = classes.index("MQTT-DoS-Publish_Flood")

results = pd.DataFrame({
    "actual_label": test_labels.to_numpy(),
    "predicted_label": predictions,
    "ddos_probability": probabilities[:, ddos_idx],
    "dos_probability": probabilities[:, dos_idx],
})

results["correct"] = (
    results["predicted_label"] == results["actual_label"]
)

print("\n=== PREDICTION CONFIDENCE ===", flush=True)
print("Rows by correct/incorrect prediction:", flush=True)
print(results["correct"].value_counts().to_string(), flush=True)

print("\nMean probabilities by prediction correctness:", flush=True)
print(
    results.groupby("correct")[
        ["ddos_probability", "dos_probability"]
    ].mean().round(4).to_string(),
    flush=True
)

print("\nMisclassified samples' probability summary:", flush=True)
print(
    results.loc[~results["correct"], [
        "ddos_probability", "dos_probability"
    ]].describe().round(4).to_string(),
    flush=True
)

output = RESULTS_DIR / "ddos_prediction_confidence.csv"
results.to_csv(output, index=False)

print(f"\nSaved results to: {output}", flush=True)
print("Confidence analysis complete.", flush=True)
