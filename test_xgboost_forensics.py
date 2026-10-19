"""
XGBoost Deep Forensics & Stress Test: Is XGBoost Truly Robust or Hiding Something?

This script performs a rigorous, multi-angle stress test on XGBoost:
1. Full Per-Class Classification Report & Confusion Matrix (Check where errors lie).
2. Prediction Confidence & Entropy on MQTT-DDoS-Publish_Flood (Are predictions confident or fragile?).
3. Top Feature Importances (Gain & Weight) — Did XGBoost use the drifting TCP flags, or did it ignore them?
4. Hyperparameter Sensitivity / Depth Stress-Test (Does a deeper, unregularized XGBoost also collapse like RF?).
5. XGBoost with Feature Ablation (Drop Top-5 drift features — does it improve or hold steady?).
"""

from pathlib import Path
import warnings
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
TRAIN_FILE = BASE / "research_data" / "research_train_6class.csv"
TEST_DIR = BASE / "datasets" / "test"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TEST_FILES = [
    "Benign_test.pcap.csv",
    "ARP_Spoofing_test.pcap.csv",
    "MQTT-DDoS-Connect_Flood_test.pcap.csv",
    "MQTT-DDoS-Publish_Flood_test.pcap.csv",
    "MQTT-DoS-Connect_Flood_test.pcap.csv",
    "MQTT-DoS-Publish_Flood_test.pcap.csv",
]

TOP_5_DRIFT = ["ack_count", "ack_flag_number", "fin_count", "rst_flag_number", "rst_count"]

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

def main():
    train_df, test_df = load_data()
    
    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"]
    
    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"]
    
    unique_labels = sorted(y_train.unique())
    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    num_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    y_train_num = y_train.map(label_to_num)
    y_test_num = y_test.map(label_to_num)
    
    print("\n" + "=" * 75)
    print("TEST 1: BASELINE XGBOOST ON OFFICIAL TEST FILES (ALL 45 FEATURES)")
    print("=" * 75)
    
    xgb_default = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss"
    )
    xgb_default.fit(X_train, y_train_num)
    
    preds_num = xgb_default.predict(X_test)
    probs = xgb_default.predict_proba(X_test)
    preds_str = [num_to_label[p] for p in preds_num]
    
    print(f"Accuracy: {accuracy_score(y_test_num, preds_num):.4f}")
    print(f"Macro F1: {f1_score(y_test_num, preds_num, average='macro'):.4f}")
    print("\nFull Classification Report:")
    print(classification_report(y_test, preds_str, digits=4, zero_division=0))
    
    cm = confusion_matrix(y_test, preds_str, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    print("\nConfusion Matrix:")
    print(cm_df)
    
    # -------------------------------------------------------------
    # TEST 2: Inspect DDoS-Publish predictions and error breakdown
    # -------------------------------------------------------------
    ddos_mask = (y_test == "MQTT-DDoS-Publish_Flood")
    ddos_preds = np.array(preds_str)[ddos_mask]
    ddos_probs = probs[ddos_mask]
    ddos_idx = label_to_num["MQTT-DDoS-Publish_Flood"]
    
    print("\n" + "=" * 75)
    print("TEST 2: FORENSICS ON MQTT-DDoS-Publish_Flood (8,416 Total Rows)")
    print("=" * 75)
    print(f"Correctly predicted as DDoS Publish: {np.sum(ddos_preds == 'MQTT-DDoS-Publish_Flood')} / 8416")
    print(f"Misclassified as DoS Publish:        {np.sum(ddos_preds == 'MQTT-DoS-Publish_Flood')}")
    print(f"Misclassified as other:              {np.sum(~np.isin(ddos_preds, ['MQTT-DDoS-Publish_Flood', 'MQTT-DoS-Publish_Flood']))}")
    print(f"Average probability assigned to DDoS class: {np.mean(ddos_probs[:, ddos_idx]):.4f}")
    
    # -------------------------------------------------------------
    # TEST 3: Feature Importances (Gain) - Did XGBoost rely on drifting features?
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("TEST 3: TOP 10 MOST IMPORTANT FEATURES FOR XGBOOST (BY GAIN)")
    print("=" * 75)
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance_Gain": xgb_default.feature_importances_
    }).sort_values(by="Importance_Gain", ascending=False)
    
    print(importance_df.head(10).to_string(index=False))
    
    # Check rank of the top drifting features in XGBoost
    print("\nRank & Importance of Top Drifting Features in XGBoost:")
    for f in TOP_5_DRIFT:
        row = importance_df[importance_df["Feature"] == f]
        rank = importance_df.index.get_loc(row.index[0]) + 1
        gain = row["Importance_Gain"].values[0]
        print(f" - {f:<18}: Rank #{rank:<2} | Gain: {gain:.5f}")
        
    # -------------------------------------------------------------
    # TEST 4: Stress-Test: Deep Trees (max_depth=15, simulating RF overfitting)
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("TEST 4: STRESS-TEST: DEEP UNCONSTRAINED XGBOOST (max_depth=15, min_child_weight=1)")
    print("=" * 75)
    xgb_deep = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=15,
        min_child_weight=1,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss"
    )
    xgb_deep.fit(X_train, y_train_num)
    preds_deep = xgb_deep.predict(X_test)
    preds_deep_str = [num_to_label[p] for p in preds_deep]
    
    print(f"Deep XGBoost Accuracy: {accuracy_score(y_test_num, preds_deep):.4f}")
    print(f"Deep XGBoost Macro F1: {f1_score(y_test_num, preds_deep, average='macro'):.4f}")
    rep_deep = classification_report(y_test, preds_deep_str, output_dict=True, zero_division=0)
    print(f"Deep XGBoost DDoS-Publish F1: {rep_deep['MQTT-DDoS-Publish_Flood']['f1-score']:.4f}")
    
    # -------------------------------------------------------------
    # TEST 5: XGBoost with Feature Ablation (Drop Top-5 drift features)
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("TEST 5: XGBOOST WITH TOP-5 DRIFTING FEATURES REMOVED")
    print("=" * 75)
    X_train_abl = X_train.drop(columns=TOP_5_DRIFT)
    X_test_abl = X_test.drop(columns=TOP_5_DRIFT)
    
    xgb_abl = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss"
    )
    xgb_abl.fit(X_train_abl, y_train_num)
    preds_abl = xgb_abl.predict(X_test_abl)
    preds_abl_str = [num_to_label[p] for p in preds_abl]
    
    print(f"Ablated XGBoost Accuracy: {accuracy_score(y_test_num, preds_abl):.4f}")
    print(f"Ablated XGBoost Macro F1: {f1_score(y_test_num, preds_abl, average='macro'):.4f}")
    rep_abl = classification_report(y_test, preds_abl_str, output_dict=True, zero_division=0)
    print(f"Ablated XGBoost DDoS-Publish F1: {rep_abl['MQTT-DDoS-Publish_Flood']['f1-score']:.4f}")
    
    # Save results
    with open(RESULTS_DIR / "xgboost_forensic_stress_test.txt", "w") as f:
        f.write("XGBOOST DEEP FORENSIC & STRESS TEST RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. Baseline XGBoost (45 features):\n")
        f.write(f"Accuracy: {accuracy_score(y_test_num, preds_num):.6f}\n")
        f.write(f"Macro F1: {f1_score(y_test_num, preds_num, average='macro'):.6f}\n")
        f.write(f"DDoS Publish F1: {rep_deep['MQTT-DDoS-Publish_Flood']['f1-score']:.6f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(cm_df.to_string())
        f.write("\n\nTop 10 Important Features (Gain):\n")
        f.write(importance_df.head(10).to_string(index=False))
        
    print(f"\nAll forensic tests complete! Saved to results/xgboost_forensic_stress_test.txt")

if __name__ == "__main__":
    main()
