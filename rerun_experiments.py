"""Tier-2 reruns for the CICIoMT2024 six-class audit.
Usage: python rerun_experiments.py train.csv test.csv [label_column]
Expects numeric feature columns + a label column (default 'label'); train/test = your 200,819 / 101,319 row files.
Needs: pandas numpy scikit-learn xgboost scipy.  Writes results_*.csv in the current folder.
"""
import sys, numpy as np, pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import os

if len(sys.argv) >= 3:
    train_path, test_path = sys.argv[1], sys.argv[2]
    L = sys.argv[3] if len(sys.argv) > 3 else None
else:
    default_tr = "IOMT RESEARCH/research_data/research_train_6class.csv"
    default_te = "IOMT RESEARCH/research_data/research_test_6class.csv"
    if os.path.exists(default_tr) and os.path.exists(default_te):
        print(f"No arguments provided. Using default datasets:\n  Train: {default_tr}\n  Test:  {default_te}")
        train_path, test_path = default_tr, default_te
        L = None
    else:
        print("Usage: python rerun_experiments.py <train.csv> <test.csv> [label_column]")
        sys.exit(1)

print(f"Loading datasets...\n  Train: {train_path}\n  Test:  {test_path}")
tr = pd.read_csv(train_path)
te = pd.read_csv(test_path)
if L is None:
    L = 'Label' if 'Label' in tr.columns else ('label' if 'label' in tr.columns else tr.columns[-1])
print(f"Using label column: '{L}' (Train rows: {len(tr):,}, Test rows: {len(te):,})")
feats=[c for c in tr.columns if c!=L]
Xtr,ytr,Xte,yte=tr[feats].values,tr[L].values,te[feats].values,te[L].values
classes=sorted(set(ytr)); le={c:i for i,c in enumerate(classes)}
ytr_i=np.array([le[c] for c in ytr]); yte_i=np.array([le[c] for c in yte])
DDOS=le['MQTT-DDoS-Publish_Flood']; BEN=le['Benign']
def models(seed):
    return {'RF':RandomForestClassifier(100,n_jobs=-1,random_state=seed),
     'LR':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),
     'MLP':make_pipeline(StandardScaler(),MLPClassifier((64,32),max_iter=200,random_state=seed)),
     'XGB':XGBClassifier(n_estimators=200,n_jobs=-1,random_state=seed,eval_metric='mlogloss')}
def mf1(y,p): return f1_score(y,p,average='macro')
def ddosf1(y,p): return f1_score(y,p,labels=[DDOS],average=None)[0]
def binf1(y,p): return f1_score(y!=BEN,p!=BEN)   # attack-vs-benign

# 1) multi-seed protocol B + binary view + CI
rows=[]
for seed in range(5):
    for n,m in models(seed).items():
        m.fit(Xtr,ytr_i); p=m.predict(Xte)
        rows.append(dict(model=n,seed=seed,macroF1=mf1(yte_i,p),acc=accuracy_score(yte_i,p),ddosF1=ddosf1(yte_i,p),
                         attack_vs_benign_F1=binf1(yte_i,p),ddos_to_benign=int(((yte_i==DDOS)&(p==BEN)).sum())))
r=pd.DataFrame(rows); r.to_csv('results_protocolB_seeds.csv',index=False)
g=r.groupby('model').agg(['mean','std']); g.to_csv('results_protocolB_summary.csv'); print(g.round(4))

# 2) random CV vs within-class BLOCKED CV (5 contiguous blocks inside each class)
p_rnd = cross_val_predict(RandomForestClassifier(100, n_jobs=-1, random_state=0), Xtr, ytr_i, cv=KFold(5, shuffle=True, random_state=0))
print(f'RF random 5-fold CV: macroF1 = {mf1(ytr_i, p_rnd):.4f}, DDoS F1 = {ddosf1(ytr_i, p_rnd):.4f}')

fold = np.zeros(len(ytr_i), int)
for c in range(len(classes)):
    idx = np.where(ytr_i == c)[0]
    fold[idx] = (np.arange(len(idx)) * 5) // len(idx)
p = np.empty_like(ytr_i)
for k in range(5):
    m = RandomForestClassifier(100, n_jobs=-1, random_state=0).fit(Xtr[fold!=k], ytr_i[fold!=k])
    p[fold==k] = m.predict(Xtr[fold==k])
print('within-class blocked: macroF1', mf1(ytr_i, p), 'DDoS F1', ddosf1(ytr_i, p))

# 3) drift metrics (KS, Wasserstein, Cohen's d) for each class
out=[]
for c in classes:
    a=tr[tr[L]==c][feats]; b=te[te[L]==c][feats]
    for f in feats:
        s=np.sqrt((a[f].var()+b[f].var())/2) or 1e-12
        out.append(dict(cls=c,feature=f,cohen_d=abs(a[f].mean()-b[f].mean())/s,
                        ks=ks_2samp(a[f],b[f]).statistic,wasserstein=wasserstein_distance(a[f],b[f])))
pd.DataFrame(out).to_csv('results_drift.csv',index=False)

# 4) within-class adversarial validation: first vs second half of each class
imps = []
for c in range(len(classes)):
    idx = np.where(ytr_i == c)[0]; h = len(idx)//2
    Xc = Xtr[idx]; yc = np.r_[np.zeros(h), np.ones(len(idx)-h)]
    imps.append(RandomForestClassifier(200, n_jobs=-1, random_state=0).fit(Xc, yc).feature_importances_)
imp = pd.Series(np.mean(imps, axis=0), feats).sort_values(ascending=False)
imp.to_csv('results_adversarial_importance.csv'); print(imp.head(10))

for k in (5, 8):
    drop = list(imp.index[:k]); keep = [i for i, f in enumerate(feats) if f not in drop]
    m = RandomForestClassifier(100, n_jobs=-1, random_state=0).fit(Xtr[:, keep], ytr_i); p = m.predict(Xte[:, keep])
    print(f'train-only ablation drop {k}: macroF1={mf1(yte_i, p):.4f} DDoS F1={ddosf1(yte_i, p):.4f}', drop)

# 5) ARP_Spoofing analysis: confusion matrix + top drifting ARP features
m=RandomForestClassifier(100,n_jobs=-1,random_state=0).fit(Xtr,ytr_i); p=m.predict(Xte)
pd.DataFrame(confusion_matrix(yte_i,p),index=classes,columns=classes).to_csv('results_confusion_baseline.csv')
print(pd.DataFrame(out).query("cls=='ARP_Spoofing'").sort_values('cohen_d',ascending=False).head(10))
print(tr[L].ne(tr[L].shift()).cumsum().nunique())  # number of label runs
print(pd.crosstab(np.arange(len(tr))//(len(tr)//5), tr[L]))