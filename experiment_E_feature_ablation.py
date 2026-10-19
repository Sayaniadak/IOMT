"""
Experiment E: Feature-Ablation Study (Removing Top Drifting Shortcut Features)

This experiment evaluates the causal hypothesis:
If the model's collapse on MQTT-DDoS-Publish_Flood in Experiment B is driven by 
severe distribution shift in low-level TCP/packet shortcut features, does removing
the top drifting features restore or alter performance on the independent test set?

Features identified with extreme standardized mean difference (> 1.0) on DDoS Publish:
1. ack_count (diff: 2.091)
2. ack_flag_number (diff: 1.920)
3. fin_count (diff: 1.803)
4. rst_flag_number (diff: 1.774)
5. rst_count (diff: 1.400)
6. Header_Length (diff: 1.207)
7. Covariance (diff: 1.094)
8. syn_flag_number (diff: 1.058)

We evaluate:
Setting E1: Drop Top-5 drifting features (ack_count, ack_flag_number, fin_count, rst_flag_number, rst_count)
Setting E2: Drop Top-8 drifting features (Top-5 + Header_Length, Covariance, syn_flag_number)
"""



from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

BASE = Path(__file__).resolve().parent
TRAIN_FILE = BASE / "research_data" / "research_train_6class.csv"
TEST_DIR = BASE / "datasets" / "test"
RESULTS_DIR = BASE / "results"

TEST_FILES = [
    "Benign_test.pcap.csv",
    "ARP_Spoofing_test.pcap.csv",
    "MQTT-DDoS-Connect_Flood_test.pcap.csv",
    "MQTT-DDoS-Publish_Flood_test.pcap.csv",
    "MQTT-DoS-Connect_Flood_test.pcap.csv",
    "MQTT-DoS-Publish_Flood_test.pcap.csv",
]

TOP_5_DRIFT = [
    "ack_count",
    "ack_flag_number",
    "fin_count",
    "rst_flag_number",
    "rst_count"
]

TOP_8_DRIFT = TOP_5_DRIFT + [
    "Header_Length",
    "Covariance",
    "syn_flag_number"
]

def load_data():
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_FILE)
    
    print("Loading test data...")
    test_data = []
    for filename in TEST_FILES:
        file_path = TEST_DIR / filename
        df = pd.read_csv(file_path)
        label = filename.replace("_test.pcap.csv", "")
        df["Label"] = label
        test_data.append(df)
    test_df = pd.concat(test_data, ignore_index=True)
    return train_df, test_df

def run_experiment(train_df, test_df, drop_cols, name):
    print("\n" + "=" * 70)
    print(f"RUNNING {name}: Dropping {len(drop_cols)} features")
    print(f"Dropped features: {drop_cols}")
    print("=" * 70)
    
    X_train = train_df.drop(columns=["Label"] + drop_cols)
    y_train = train_df["Label"]
    
    X_test = test_df.drop(columns=["Label"] + drop_cols)
    y_test = test_df["Label"]
    
    print(f"Training features remaining: {X_train.shape[1]}")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
    cm_df = pd.DataFrame(cm, index=sorted(y_test.unique()), columns=sorted(y_test.unique()))
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm_df)
    
    return {
        "name": name,
        "dropped_features": drop_cols,
        "acc": acc,
        "macro_f1": macro_f1,
        "report": report,
        "cm_df": cm_df
    }

def main():
    train_df, test_df = load_data()
    
    res_e1 = run_experiment(train_df, test_df, TOP_5_DRIFT, "EXPERIMENT E1 (Drop Top-5 Drift Features)")
    res_e2 = run_experiment(train_df, test_df, TOP_8_DRIFT, "EXPERIMENT E2 (Drop Top-8 Drift Features)")
    
    summary_path = RESULTS_DIR / "experiment_E_feature_ablation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("EXPERIMENT E: FEATURE ABLATION (REMOVING SHORTCUT / DRIFTING FEATURES)\n")
        f.write("=" * 75 + "\n\n")
        
        for res in [res_e1, res_e2]:
            f.write(f"{res['name']}\n")
            f.write(f"Dropped features: {res['dropped_features']}\n")
            f.write(f"Accuracy: {res['acc']:.6f}\n")
            f.write(f"Macro F1: {res['macro_f1']:.6f}\n\n")
            f.write("Classification Report:\n")
            f.write(res["report"])
            f.write("\nConfusion Matrix:\n")
            f.write(res["cm_df"].to_string())
            f.write("\n\n" + "-" * 75 + "\n\n")
            
            cm_save_name = "experiment_E1_confusion_matrix.csv" if "E1" in res["name"] else "experiment_E2_confusion_matrix.csv"
            res["cm_df"].to_csv(RESULTS_DIR / cm_save_name)
            
    print(f"\nAll results saved to {summary_path}")

if __name__ == "__main__":
    main()
