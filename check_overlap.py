"""Exact-row overlap between the prepared training file and the official test files.
Usage: python check_overlap.py research_train_6class.csv research_test_6class.csv [--label Label]
Reports the number of identical feature rows shared by train and test (overall and per class pair)."""
import os, sys, pandas as pd

default_tr = "IOMT RESEARCH/research_data/research_train_6class.csv"
default_te = "IOMT RESEARCH/research_data/research_test_6class.csv"

train_file = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else default_tr
test_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else default_te
L = sys.argv[sys.argv.index('--label')+1] if '--label' in sys.argv else 'Label'

print(f"Checking overlap between:\n  Train: {train_file}\n  Test:  {test_file}")
tr = pd.read_csv(train_file)
te = pd.read_csv(test_file)
if L not in tr.columns and 'label' in tr.columns:
    L = 'label'
f = [c for c in tr.columns if c != L]
a = tr[f].drop_duplicates(); b = te[f].drop_duplicates()
m = a.merge(b, on=f, how='inner')
print(f"unique train rows {len(a):,}; unique test rows {len(b):,}; exact shared rows {len(m):,}")
tl = tr.drop_duplicates(subset=f)[f+[L]]; el = te.drop_duplicates(subset=f)[f+[L]]
mm = tl.merge(el, on=f, suffixes=('_train','_test'))
print(mm.groupby([L+'_train', L+'_test']).size().to_string() if len(mm) else "no shared rows")