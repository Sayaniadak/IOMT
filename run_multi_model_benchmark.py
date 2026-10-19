"""
Experiment Multi-Model: Cross-Architecture Evaluation (RF, XGBoost, MLP, Logistic Regression)

This script evaluates whether the Experiment A -> B performance collapse and the 
catastrophic collapse on MQTT-DDoS-Publish_Flood is unique to Random Forest or an inherent
property of shortcut learning and distribution shift affecting multiple distinct model families.

Models Evaluated:
1. Random Forest (Tree Ensemble - Bagging)
2. XGBoost (Gradient Boosted Decision Trees - Boosting)
3. Multi-Layer Perceptron (MLP - Neural Network)
4. Logistic Regression (Linear Baseline)

Protocols:
- Protocol A: 5-Fold Stratified Cross-Validation on Research Train Set
- Protocol B: Training on Research Train Set, Testing on Official Test Files
"""

from pathlib import Path
import warnings
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Try importing XGBoost; if not present, provide a clear fallback warning
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

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

def load_data():
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_FILE)
    
    print("Loading official test data...")
    test_data = []
    for filename in TEST_FILES:
        file_path = TEST_DIR / filename
        df = pd.read_csv(file_path)
        label = filename.replace("_test.pcap.csv", "")
        df["Label"] = label
        test_data.append(df)
    test_df = pd.concat(test_data, ignore_index=True)
    return train_df, test_df

def get_models(label_mapping=None):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, random_state=42))
        ]),
        "MLP (Neural Net)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=40, random_state=42, early_stopping=True))
        ])
    }
    
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss"
        )
    else:
        print("\n[NOTE] xgboost package not installed in environment; skipping XGBoost.")
        
    return models

def run_cross_validation(X, y, models, label_to_num, num_to_label):
    print("\n" + "=" * 70)
    print("PROTOCOL A: 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 70)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary = {}
    
    for name, model in models.items():
        print(f"\nEvaluating CV for: {name}...")
        y_encoded = y.map(label_to_num).to_numpy()
        X_arr = X.to_numpy()
        
        macro_f1s = []
        accs = []
        ddos_f1s = []
        ddos_idx = label_to_num["MQTT-DDoS-Publish_Flood"]
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_encoded), 1):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_encoded[train_idx], y_encoded[val_idx]
            
            # For linear/MLP, training on 50k subsample per fold speeds up CV without loss of statistical significance
            if name in ["MLP (Neural Net)", "Logistic Regression"] and len(X_tr) > 60000:
                sub_idx = np.random.RandomState(42 + fold).choice(len(X_tr), 60000, replace=False)
                X_tr, y_tr = X_tr[sub_idx], y_tr[sub_idx]
                
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            
            acc = accuracy_score(y_val, preds)
            mf1 = f1_score(y_val, preds, average="macro", zero_division=0)
            
            # Class-specific DDoS Publish F1
            rep = classification_report(y_val, preds, output_dict=True, zero_division=0)
            ddos_f1 = rep.get(str(ddos_idx), {}).get("f1-score", 0.0)
            
            accs.append(acc)
            macro_f1s.append(mf1)
            ddos_f1s.append(ddos_f1)
            
        cv_summary[name] = {
            "Accuracy": np.mean(accs),
            "Macro_F1": np.mean(macro_f1s),
            "DDoS_Publish_F1": np.mean(ddos_f1s)
        }
        print(f" -> {name} CV Results | Acc: {np.mean(accs):.4f} | Macro-F1: {np.mean(macro_f1s):.4f} | DDoS-Publish F1: {np.mean(ddos_f1s):.4f}")
        
    return cv_summary

def run_independent_test(X_train, y_train, X_test, y_test, models, label_to_num, num_to_label):
    print("\n" + "=" * 70)
    print("PROTOCOL B: OFFICIAL INDEPENDENT TEST FILES EVALUATION")
    print("=" * 70)
    
    test_summary = {}
    detailed_reports = {}
    
    y_tr_enc = y_train.map(label_to_num).to_numpy()
    y_te_enc = y_test.map(label_to_num).to_numpy()
    X_tr_arr = X_train.to_numpy()
    X_te_arr = X_test.to_numpy()
    
    ddos_idx = label_to_num["MQTT-DDoS-Publish_Flood"]
    
    for name, model in models.items():
        print(f"\nTraining and Testing: {name} on Official Split...")
        model.fit(X_tr_arr, y_tr_enc)
        preds = model.predict(X_te_arr)
        
        acc = accuracy_score(y_te_enc, preds)
        mf1 = f1_score(y_te_enc, preds, average="macro", zero_division=0)
        
        # Get full classification report with original string labels
        preds_str = [num_to_label[p] for p in preds]
        rep = classification_report(y_test, preds_str, digits=4, zero_division=0)
        rep_dict = classification_report(y_test, preds_str, output_dict=True, zero_division=0)
        
        ddos_f1 = rep_dict.get("MQTT-DDoS-Publish_Flood", {}).get("f1-score", 0.0)
        
        test_summary[name] = {
            "Accuracy": acc,
            "Macro_F1": mf1,
            "DDoS_Publish_F1": ddos_f1
        }
        detailed_reports[name] = rep
        print(f" -> {name} Test Results | Acc: {acc:.4f} | Macro-F1: {mf1:.4f} | DDoS-Publish F1: {ddos_f1:.4f}")
        
    return test_summary, detailed_reports

def main():
    train_df, test_df = load_data()
    
    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"]
    
    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"]
    
    unique_labels = sorted(y_train.unique())
    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    num_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    models = get_models(label_to_num)
    
    cv_summary = run_cross_validation(X_train, y_train, models, label_to_num, num_to_label)
    test_summary, detailed_reports = run_independent_test(X_train, y_train, X_test, y_test, models, label_to_num, num_to_label)
    
    # Save formatted comparison results
    output_path = RESULTS_DIR / "multi_model_comparison_results.txt"
    with open(output_path, "w") as f:
        f.write("CROSS-ARCHITECTURE BENCHMARK COMPARISON (EXPERIMENT A vs EXPERIMENT B)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Model':<22} | {'Exp A (CV) Macro-F1':<20} | {'Exp B (Test) Macro-F1':<22} | {'Drop (pp)':<10} | {'DDoS F1 (Test)':<15}\n")
        f.write("-" * 98 + "\n")
        
        for name in models.keys():
            cv_f1 = cv_summary[name]["Macro_F1"] * 100
            test_f1 = test_summary[name]["Macro_F1"] * 100
            drop = cv_f1 - test_f1
            ddos_f1 = test_summary[name]["DDoS_Publish_F1"] * 100
            f.write(f"{name:<22} | {cv_f1:>18.2f}% | {test_f1:>20.2f}% | {drop:>8.2f} pp | {ddos_f1:>13.2f}%\n")
            
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("DETAILED CLASSIFICATION REPORTS (EXPERIMENT B)\n")
        f.write("=" * 80 + "\n\n")
        for name, rep in detailed_reports.items():
            f.write(f"MODEL: {name}\n")
            f.write("-" * 40 + "\n")
            f.write(rep + "\n\n")
            
    print(f"\nMulti-model benchmark complete! Saved to {output_path}")

if __name__ == "__main__":
    main()
