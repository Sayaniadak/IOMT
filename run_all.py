"""
run_all.py  --  ONE script, ONE configuration, every number in the CICIoMT2024 paper.

Usage:
    python run_all.py research_train_6class.csv research_test_6class.csv [--label Label] [--out results_final]

Needs: pandas numpy scipy scikit-learn xgboost
Writes (in --out): report.txt (all tables, paper order), plus one CSV per table.

Fixed configuration (state it once in the paper):
    Random Forest : 100 trees, other parameters default, random_state = seed
    Logistic Reg. : StandardScaler + LogisticRegression(max_iter=1000)
    MLP           : StandardScaler + MLPClassifier((64,32), max_iter=200, random_state=seed)
    XGBoost       : 200 estimators, other parameters default, random_state = seed
    Seeds         : 0..4  (Protocol B and ablations); Protocol A uses seed 0 (KFold shuffle, random_state=0)

Sections (each is wrapped so one failure does not stop the others):
    1 drift            Cohen's d, KS, Wasserstein for every class x feature       -> Tables 4, 5, 10  (item 4)
    2 protocol B       5 seeds x 4 models, per-class report, confusion matrices   -> Tables 3, 6, 8   (item 2)
    3 protocol A       random 5-fold CV, 4 models                                 -> Table 6
    4 blocked CV       within-class contiguous blocks + row-order diagnostics     -> Section 9.3
    5 ablation         oracle E1/E2 and THREE train-only rankings, 5 seeds        -> Tables 7, 9      (item 5)
    6 adversarial      train-vs-test classifier AUC per class + within-train AUC  -> new, item 5
    7 reverse          train on test, test on train                               -> new, item 5
    8 xgboost          gain ranks, Rate/IAT/Srate ablation, depth-15              -> Sections 7.1, 7.2
    9 mitigations      Experiment C (two-stage), Experiment D (thresholds)        -> Section 8
"""
import argparse, os, sys, time, traceback, warnings
import numpy as np, pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
DEFAULT_TRAIN = "IOMT RESEARCH/research_data/research_train_6class.csv"
DEFAULT_TEST = "IOMT RESEARCH/research_data/research_test_6class.csv"

ap = argparse.ArgumentParser()
ap.add_argument("train", nargs="?", default=DEFAULT_TRAIN, help=f"Path to train CSV (default: {DEFAULT_TRAIN})")
ap.add_argument("test", nargs="?", default=DEFAULT_TEST, help=f"Path to test CSV (default: {DEFAULT_TEST})")
ap.add_argument("--label", default="Label")
ap.add_argument("--out", default="results_final")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--trees", type=int, default=100)
ap.add_argument("--xgb-est", type=int, default=200)
ap.add_argument("--mlp-iter", type=int, default=200)
ap.add_argument("--adv-max", type=int, default=20000, help="max rows per side in adversarial validation")
ap.add_argument("--ordered", default="ARP_Spoofing,MQTT-DDoS-Publish_Flood,MQTT-DoS-Connect_Flood,MQTT-DoS-Publish_Flood",
                help="classes kept in ORIGINAL file order (not randomly downsampled)")
ap.add_argument("--fast", action="store_true", help="smoke test only: tiny config, NOT for paper numbers")
args = ap.parse_args()
if args.fast:
    args.seeds, args.trees, args.xgb_est, args.mlp_iter, args.adv_max = 2, 30, 50, 40, 4000
os.makedirs(args.out, exist_ok=True)

tr = pd.read_csv(args.train); te = pd.read_csv(args.test)
L = args.label
if L not in tr.columns:
    sys.exit(f"label column '{L}' not found; columns end with {list(tr.columns[-3:])}; use --label")
feats = [c for c in tr.columns if c != L]
assert list(te.columns) == list(tr.columns) or set(feats) <= set(te.columns), "train/test columns differ"
Xtr, Xte = tr[feats].values.astype(float), te[feats].values.astype(float)
classes = sorted(set(tr[L]) | set(te[L])); K = len(classes); le = {c: i for i, c in enumerate(classes)}
ytr = tr[L].map(le).values; yte = te[L].map(le).values
DDOS, DOSP, ARP, BEN = le["MQTT-DDoS-Publish_Flood"], le["MQTT-DoS-Publish_Flood"], le["ARP_Spoofing"], le["Benign"]
ORDERED = [c for c in args.ordered.split(",") if c in le]
print(f"train {Xtr.shape}, test {Xte.shape}, classes {classes}\nordered classes: {ORDERED}")

REPORT = []
def say(s=""):
    print(s); REPORT.append(str(s))
    open(os.path.join(args.out, "report.txt"), "w", encoding="utf8").write("\n".join(REPORT))
def tab(df, name, nd=4):
    df.to_csv(os.path.join(args.out, name + ".csv"))
    say(f"\n[{name}]"); say(df.round(nd).to_string())
def section(title, fn):
    t = time.time(); say(f"\n{'='*78}\n{title}\n{'='*78}")
    try: fn()
    except Exception as e:
        say(f"!! section failed: {e}"); traceback.print_exc()
    say(f"(took {time.time()-t:.0f}s)")

# ---------------------------------------------------------------- helpers
def make_models(seed):
    return {"Random Forest": RandomForestClassifier(args.trees, n_jobs=-1, random_state=seed),
            "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "MLP": make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=args.mlp_iter, random_state=seed)),
            "XGBoost": XGBClassifier(n_estimators=args.xgb_est, n_jobs=-1, random_state=seed, eval_metric="mlogloss")}
def rf(seed=0, **kw): return RandomForestClassifier(args.trees, n_jobs=-1, random_state=seed, **kw)
def xgb(seed=0, **kw): return XGBClassifier(n_estimators=args.xgb_est, n_jobs=-1, random_state=seed, eval_metric="mlogloss", **kw)
def mf1(y, p): return f1_score(y, p, average="macro", labels=list(range(K)))
def cf1(y, p, c): return f1_score(y, p, labels=[c], average=None)[0]
def binf1(y, p): return f1_score(y != BEN, p != BEN)
def d2b(y, p): return int(((y == DDOS) & (p == BEN)).sum())
def d2dos(y, p): return int(((y == DDOS) & (p == DOSP)).sum())
def idx(names): return [feats.index(n) for n in names]
def fit_eval_rf(cols, seed):
    keep = [i for i in range(len(feats)) if feats[i] not in cols]
    p = rf(seed).fit(Xtr[:, keep], ytr).predict(Xte[:, keep])
    return dict(macroF1=mf1(yte, p), acc=accuracy_score(yte, p), ddosF1=cf1(yte, p, DDOS), arpF1=cf1(yte, p, ARP), ddos_to_dos=d2dos(yte, p))
STATE = {}

# ---------------------------------------------------------------- 1 drift
def s1():
    rows = []
    for c in classes:
        a = tr[tr[L] == c][feats]; b = te[te[L] == c][feats]
        for f in feats:
            s = np.sqrt((a[f].var() + b[f].var()) / 2); s = s if s and s > 0 else 1e-12
            rows.append(dict(cls=c, feature=f, train_mean=a[f].mean(), test_mean=b[f].mean(), train_std=a[f].std(), test_std=b[f].std(),
                             cohen_d=abs(a[f].mean() - b[f].mean()) / s, ks=ks_2samp(a[f], b[f]).statistic,
                             wasserstein=wasserstein_distance(a[f], b[f])))
    D = pd.DataFrame(rows); D.to_csv(os.path.join(args.out, "drift_all.csv"), index=False); STATE["drift"] = D
    t4 = D.groupby("cls").cohen_d.agg(mean="mean", median="median", max="max")
    t4["d>0.5"] = D[D.cohen_d > .5].groupby("cls").size(); t4["d>1.0"] = D[D.cohen_d > 1].groupby("cls").size()
    tab(t4.fillna(0), "table4_drift_summary")
    dd = D[D.cls == "MQTT-DDoS-Publish_Flood"].sort_values("cohen_d", ascending=False)
    tab(dd.head(10).set_index("feature")[["train_mean", "test_mean", "train_std", "test_std", "cohen_d", "ks", "wasserstein"]], "table5_ddos_publish_top10")
    say("Rate / IAT / Srate for DDoS-Publish (for the Section 7.1 claim 'd <= 0.86'):")
    say(dd[dd.feature.isin(["Rate", "IAT", "Srate"])][["feature", "cohen_d", "ks", "wasserstein"]].round(4).to_string(index=False))
    ar = D[D.cls == "ARP_Spoofing"].sort_values("cohen_d", ascending=False).head(10)
    tab(ar.set_index("feature")[["cohen_d", "ks", "wasserstein"]], "table10_arp_top10")
    STATE["E1"] = list(dd.feature.head(5)); STATE["E2"] = list(dd[dd.cohen_d > 1.0].feature)
    say(f"oracle E1 (top-5 by d): {STATE['E1']}\noracle E2 (d>1.0, {len(STATE['E2'])} feats): {STATE['E2']}")
section("1. Drift: Cohen's d, KS, Wasserstein (all classes)", s1)

# ---------------------------------------------------------------- 2 protocol B
def s2():
    rows, cms, rep = [], {}, None
    for seed in range(args.seeds):
        for n, m in make_models(seed).items():
            t = time.time(); m.fit(Xtr, ytr); p = m.predict(Xte)
            pc = precision_recall_fscore_support(yte, p, labels=list(range(K)), zero_division=0)
            r = dict(model=n, seed=seed, macroF1=mf1(yte, p), acc=accuracy_score(yte, p), ddosF1=cf1(yte, p, DDOS),
                     attack_vs_benign_F1=binf1(yte, p), ddos_to_benign=d2b(yte, p), ddos_to_dos=d2dos(yte, p))
            for i, c in enumerate(classes): r["F1_" + c] = pc[2][i]
            rows.append(r)
            if seed == 0:
                cms[n] = confusion_matrix(yte, p, labels=list(range(K)))
                if n == "Random Forest":
                    rep = pd.DataFrame({"precision": pc[0], "recall": pc[1], "F1": pc[2], "support": pc[3]}, index=classes)
                    rep.loc["accuracy"] = [np.nan, np.nan, accuracy_score(yte, p), len(yte)]
                    rep.loc["macro avg"] = [pc[0].mean(), pc[1].mean(), pc[2].mean(), len(yte)]
                    rep.loc["weighted avg"] = [np.average(pc[0], weights=pc[3]), np.average(pc[1], weights=pc[3]), np.average(pc[2], weights=pc[3]), len(yte)]
            say(f"  seed {seed} {n}: macroF1={r['macroF1']:.4f} DDoS F1={r['ddosF1']:.4f} ({time.time()-t:.0f}s)")
    R = pd.DataFrame(rows); R.to_csv(os.path.join(args.out, "protocolB_all_runs.csv"), index=False); STATE["B"] = R
    g = R.groupby("model")[["macroF1", "acc", "ddosF1", "attack_vs_benign_F1", "ddos_to_benign", "ddos_to_dos"]].agg(["mean", "std"])
    out = pd.DataFrame(index=g.index)
    for c in ["macroF1", "acc", "ddosF1", "attack_vs_benign_F1"]:
        out[c] = [f"{g.loc[i,(c,'mean')]*100:.2f} +- {g.loc[i,(c,'std')]*100:.2f}" for i in g.index]
    out["ddos_to_benign"] = [f"{g.loc[i,('ddos_to_benign','mean')]:.1f} +- {g.loc[i,('ddos_to_benign','std')]:.1f}" for i in g.index]
    tab(out, "table8_protocolB_mean_std")
    say("NOTE: models with std 0.00 are deterministic here; their std is not evidence of stability.")
    tab(rep, "table3_RF_seed0_per_class")
    for n, cm in cms.items(): tab(pd.DataFrame(cm, index=classes, columns=classes), "confusion_seed0_" + n.replace(" ", "_"), nd=0)
section("2. Protocol B: official test files, 5 seeds x 4 models (reference configuration)", s2)

# ---------------------------------------------------------------- 3 protocol A
def s3():
    rows = []; cv = KFold(5, shuffle=True, random_state=0)
    for n, m in make_models(0).items():
        t = time.time()
        if n == "Random Forest":
            P = cross_val_predict(m, Xtr, ytr, cv=cv, method="predict_proba"); p = P.argmax(1); STATE["rf_oof_proba"] = P
        else: p = cross_val_predict(m, Xtr, ytr, cv=cv)
        rows.append(dict(model=n, cv_macroF1=mf1(ytr, p), cv_acc=accuracy_score(ytr, p), cv_ddosF1=cf1(ytr, p, DDOS)))
        say(f"  {n}: CV macroF1={rows[-1]['cv_macroF1']:.4f} DDoS F1={rows[-1]['cv_ddosF1']:.4f} ({time.time()-t:.0f}s)")
    A = pd.DataFrame(rows).set_index("model")
    if "B" in STATE:
        b = STATE["B"].groupby("model")[["macroF1", "ddosF1"]].mean()
        A["test_macroF1_5seed_mean"] = b.macroF1; A["test_ddosF1_5seed_mean"] = b.ddosF1
        A["drop_pp"] = (A.test_macroF1_5seed_mean - A.cv_macroF1) * 100
    tab(A, "table6_protocolA_vs_B")
section("3. Protocol A: random 5-fold CV (seed 0) vs Protocol B", s3)

# ---------------------------------------------------------------- 4 blocked CV
def s4():
    fold = np.zeros(len(ytr), int)
    for c in range(K):
        ix = np.where(ytr == c)[0]; fold[ix] = (np.arange(len(ix)) * 5) // len(ix)
    p = np.empty_like(ytr)
    for k in range(5): p[fold == k] = rf(0).fit(Xtr[fold != k], ytr[fold != k]).predict(Xtr[fold == k])
    rnd = cross_val_predict(rf(0), Xtr, ytr, cv=KFold(5, shuffle=True, random_state=0))
    t = pd.DataFrame({"macroF1": [mf1(ytr, rnd), mf1(ytr, p)], "ddosF1": [cf1(ytr, rnd, DDOS), cf1(ytr, p, DDOS)],
                      "arpF1": [cf1(ytr, rnd, ARP), cf1(ytr, p, ARP)]}, index=["random 5-fold", "within-class blocked 5-fold"])
    tab(t, "blocked_cv"); say(f"shuffling effect = {(t.macroF1.iloc[0]-t.macroF1.iloc[1])*100:.2f} pp macro-F1")
    # row-order diagnostics: is row order informative (autocorrelation / trend)?  ~0 means rows look shuffled
    rows = []
    for c in range(K):
        X = Xtr[ytr == c][:50000]; sd = X.std(0); ok = sd > 0; X = X[:, ok]
        a, b = X[:-1], X[1:]; ac = [abs(np.corrcoef(a[:, j], b[:, j])[0, 1]) for j in range(X.shape[1])]
        tr_ = [abs(np.corrcoef(np.arange(len(X)), X[:, j])[0, 1]) for j in range(X.shape[1])]
        rows.append(dict(cls=classes[c], mean_abs_lag1_autocorr=np.nanmean(ac), mean_abs_corr_with_row_index=np.nanmean(tr_),
                         declared_original_order=classes[c] in ORDERED))
    tab(pd.DataFrame(rows).set_index("cls"), "row_order_diagnostics")
    say("Reading: for classes in original capture order, lag-1 autocorrelation should be clearly > 0; ~0.00 means rows look shuffled "
        "and 'blocked' folds are effectively random for that class.")
section("4. Blocked CV (within class) + row-order diagnostics", s4)

# ---------------------------------------------------------------- train-only rankings (used by 5)
def rankings():
    ordered_ix = [le[c] for c in ORDERED]; A_imp, A4_imp, B_ks = [], [], []
    for c in range(K):
        X = Xtr[ytr == c]; h = len(X) // 2
        y = np.r_[np.zeros(h), np.ones(len(X) - h)]
        im = RandomForestClassifier(200 if not args.fast else 40, n_jobs=-1, random_state=0).fit(X, y).feature_importances_
        A_imp.append(im)
        if c in ordered_ix:
            A4_imp.append(im)
            B_ks.append([ks_2samp(X[:h, j], X[h:, j]).statistic for j in range(X.shape[1])])
    mk = lambda v: pd.Series(np.mean(v, 0), feats).sort_values(ascending=False)
    return {"trainonly_RFhalf_all6": mk(A_imp), "trainonly_RFhalf_ordered4": mk(A4_imp), "trainonly_KS_ordered4": mk(B_ks)}

# ---------------------------------------------------------------- 5 ablation
def s5():
    if "E1" not in STATE: raise RuntimeError("section 1 failed; oracle sets missing")
    R = rankings(); STATE["rank"] = R
    pd.DataFrame(R).to_csv(os.path.join(args.out, "trainonly_rankings.csv"))
    for k, v in R.items(): say(f"{k} top10: " + ", ".join(f"{i}({s:.3f})" for i, s in v.head(10).items()))
    cfgs = {"none (baseline)": []}
    cfgs["ORACLE E1 (test-guided, top5)"] = STATE["E1"]; cfgs["ORACLE E2 (test-guided, d>1)"] = STATE["E2"]
    for k, v in R.items():
        cfgs[f"{k} top5"] = list(v.index[:5]); cfgs[f"{k} top8"] = list(v.index[:8])
    rows = []
    for name, cols in cfgs.items():
        for seed in range(args.seeds):
            r = fit_eval_rf(cols, seed); r.update(config=name, seed=seed); rows.append(r)
        say(f"  done: {name}")
    Z = pd.DataFrame(rows); Z.to_csv(os.path.join(args.out, "ablation_all_runs.csv"), index=False)
    g = Z.groupby("config", sort=False)[["macroF1", "ddosF1", "arpF1", "ddos_to_dos"]].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]; g["removed"] = [", ".join(cfgs[i]) for i in g.index]
    tab(g, "table7_9_ablation_5seed")
    for k, v in R.items():
        ov = set(v.index[:5]) & set(STATE["E1"]); say(f"overlap of {k} top5 with oracle E1: {sorted(ov) if ov else 'none'}")
section("5. Ablation: oracle (test-guided) vs three train-only rankings, 5 seeds", s5)

# ---------------------------------------------------------------- 6 adversarial validation
def s6():
    rng = np.random.RandomState(0); rows = []
    for c in range(K):
        A = Xtr[ytr == c]; B = Xte[yte == c]
        A = A[rng.choice(len(A), min(len(A), args.adv_max), replace=False)]; B = B[rng.choice(len(B), min(len(B), args.adv_max), replace=False)]
        X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
        P = cross_val_predict(RandomForestClassifier(100 if not args.fast else 30, n_jobs=-1, random_state=0), X, y, cv=KFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        imp = pd.Series(RandomForestClassifier(100 if not args.fast else 30, n_jobs=-1, random_state=0).fit(X, y).feature_importances_, feats).sort_values(ascending=False)
        Ct = Xtr[ytr == c]; h = len(Ct) // 2; Xh = np.vstack([Ct[:h], Ct[h:]]); yh = np.r_[np.zeros(h), np.ones(len(Ct) - h)]
        sel = rng.choice(len(Xh), min(len(Xh), 2 * args.adv_max), replace=False)
        Ph = cross_val_predict(RandomForestClassifier(100 if not args.fast else 30, n_jobs=-1, random_state=0), Xh[sel], yh[sel], cv=KFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        rows.append(dict(cls=classes[c], AUC_train_vs_test=roc_auc_score(y, P), AUC_within_train_halves=roc_auc_score(yh[sel], Ph),
                         top_features_train_vs_test=", ".join(imp.index[:5])))
    tab(pd.DataFrame(rows).set_index("cls"), "adversarial_validation")
    say("AUC 0.5 = captures indistinguishable; ~1.0 = strongly separable. Test-guided: a DIAGNOSTIC, not a deployable selector.")
section("6. Adversarial validation per class (train vs test; and first vs second half of train)", s6)

# ---------------------------------------------------------------- 7 reverse direction
def s7():
    fw = rf(0).fit(Xtr, ytr).predict(Xte); bw = rf(0).fit(Xte, yte).predict(Xtr)
    t = pd.DataFrame({"macroF1": [mf1(yte, fw), mf1(ytr, bw)], "ddosF1": [cf1(yte, fw, DDOS), cf1(ytr, bw, DDOS)], "arpF1": [cf1(yte, fw, ARP), cf1(ytr, bw, ARP)]},
                     index=["train->test (official)", "test->train (reverse)"])
    tab(t, "reverse_direction")
    say("If the reverse direction also fails on DDoS-Publish, the problem is a train/test capture mismatch, not something specific to the official training file.")
section("7. Reverse direction (RF seed 0)", s7)

# ---------------------------------------------------------------- 8 xgboost
def s8():
    m = xgb(0).fit(Xtr, ytr); imp = pd.Series(m.feature_importances_, feats).sort_values(ascending=False)
    rk = pd.DataFrame({"gain": imp, "rank": range(1, len(imp) + 1)}); tab(rk.head(15), "xgb_gain_top15")
    say("ranks: " + ", ".join(f"{f}=#{int(rk.loc[f,'rank'])}" for f in ["Rate", "IAT", "Srate", "rst_count", "ack_count"] if f in rk.index))
    rows = []
    for name, cols in {"none": [], "oracle E1": STATE.get("E1", []), "Rate+IAT+Srate": ["Rate", "IAT", "Srate"]}.items():
        keep = [i for i, f in enumerate(feats) if f not in cols]
        px = xgb(0).fit(Xtr[:, keep], ytr).predict(Xte[:, keep]); pr = rf(0).fit(Xtr[:, keep], ytr).predict(Xte[:, keep])
        rows.append(dict(removed=name, XGB_macroF1=mf1(yte, px), XGB_ddosF1=cf1(yte, px, DDOS), RF_macroF1=mf1(yte, pr), RF_ddosF1=cf1(yte, pr, DDOS)))
    tab(pd.DataFrame(rows).set_index("removed"), "xgb_rf_feature_removal")
    p = xgb(0, max_depth=15, min_child_weight=1).fit(Xtr, ytr).predict(Xte)
    say(f"XGBoost max_depth=15, min_child_weight=1: macroF1={mf1(yte,p):.4f}, DDoS F1={cf1(yte,p,DDOS):.4f}")
section("8. XGBoost: gain ranks, feature-removal tests, depth-15", s8)

# ---------------------------------------------------------------- 9 mitigations
def s9():
    # Experiment C: two-stage (benign vs attack, then attack type)
    s1m = rf(0).fit(Xtr, (ytr != BEN).astype(int)); atk = ytr != BEN
    s2m = rf(0).fit(Xtr[atk], ytr[atk]); isatk = s1m.predict(Xte).astype(bool)
    p = np.full(len(yte), BEN); p[isatk] = s2m.predict(Xte[isatk])
    say(f"Experiment C two-stage RF: macroF1={mf1(yte,p):.4f}, DDoS F1={cf1(yte,p,DDOS):.4f}, DDoS recall={(p[yte==DDOS]==DDOS).mean():.4f}, "
        f"attacks misclassified={(p[yte!=BEN]!=yte[yte!=BEN]).sum()}")
    # Experiment D: class weights tuned on training OOF probabilities only, applied to test
    P = STATE.get("rf_oof_proba")
    if P is None: P = cross_val_predict(rf(0), Xtr, ytr, cv=KFold(5, shuffle=True, random_state=0), method="predict_proba")
    w = np.ones(K); best = mf1(ytr, P.argmax(1))
    for _ in range(2):
        for c in range(K):
            for g in [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]:
                w2 = w.copy(); w2[c] = g; s = mf1(ytr, (P * w2).argmax(1))
                if s > best + 1e-9: best, w = s, w2
    pt = (rf(0).fit(Xtr, ytr).predict_proba(Xte) * w).argmax(1)
    say(f"Experiment D tuned class weights {{c: float(v) for c, v in zip(classes, w)}}: train-OOF macroF1={best:.4f}; TEST macroF1={mf1(yte,pt):.4f}, DDoS F1={cf1(yte,pt,DDOS):.4f}")
    say("NOTE: C and D are re-implemented from the paper's one-line descriptions; if your original code differed, say which version the paper reports.")
section("9. Mitigations C (two-stage) and D (validation-tuned thresholds)", s9)

say("\nALL DONE. Replace Tables 3, 6, 7, 8, 9 and every number in the abstract/conclusion with the values above, from this single run.")