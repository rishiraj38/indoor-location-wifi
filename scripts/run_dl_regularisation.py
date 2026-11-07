"""Phase 2: input scaling and regularisation for the neural network.

Owner: Yash Yadav.

Compares, with user-grouped cross-validation:
  - input scaling: min-max vs standardise
  - dropout: 0, 0.2, 0.3, 0.5
  - weight decay: 0, 1e-5, 1e-4, 1e-3
Writes results/dl_regularisation.csv and results/dl_best_regularisation.json.

To do (Yash Yadav):
  - run it, read the table, pick the best setting and explain why in
    docs/dl_regularisation.md (one page: what each setting does, what won, why)
  - add the "powed" representation as a third scaling option and test it
  - check early-stopping patience (10, 15, 30) and report the effect
"""

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from indoorloc.data import ROOT, load_splits, split_xy
from indoorloc.deep import MultitaskModel
from indoorloc.validation import cv_eval, user_folds

RES = ROOT / "results"
train, _, _ = load_splits()
Xtr, ytr, mtr = split_xy(train)
groups = mtr["USERID"].to_numpy()
folds = user_folds(groups)

rows = []
for scaler, dropout, wd in itertools.product(["minmax", "standard"], [0.0, 0.2, 0.3, 0.5], [0.0, 1e-5, 1e-4, 1e-3]):
    cv = cv_eval(lambda: MultitaskModel(scaler=scaler, dropout=dropout, weight_decay=wd), Xtr, ytr, groups, folds)
    rows.append({"scaler": scaler, "dropout": dropout, "weight_decay": wd, **cv})
    print(f"{scaler:9s} dropout={dropout:.1f} wd={wd:g}  floor {cv['floor_acc']:.3f}  err {cv['mean_error_m']:.2f} m", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RES / "dl_regularisation.csv", index=False)
best = df.loc[df.mean_error_m.idxmin()]
json.dump({"scaler": best.scaler, "dropout": float(best.dropout), "weight_decay": float(best.weight_decay)},
          open(RES / "dl_best_regularisation.json", "w"), indent=2)
print("best:", best.to_dict())
