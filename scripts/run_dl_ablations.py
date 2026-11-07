"""Phase 2: ablation studies for the multitask network.

Owner: Aditya Samadhiya.

Each ablation removes or changes one part of the model and measures the effect
with user-grouped cross-validation:
  1. multitask (one network, three heads) vs three single-task networks
  2. loss weights for building / floor / coordinates
  3. trunk depth and width
Writes results/dl_ablations.csv and results/figures/dl_ablations.png.

To do (Aditya):
  - run it after Rishi has folded Yash Yadav's regularisation result into
    results/dl_best.json; every Phase 2 script starts from that one file
  - write up each ablation in docs/dl_ablations.md: what was removed, what
    happened to floor accuracy and position error, and what that tells us
  - add any ablation you think is missing (for example no early stopping)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from indoorloc.data import ROOT, load_splits, split_xy
from indoorloc.deep import MultitaskModel
from indoorloc.metrics import summarize
from indoorloc.plotting import SERIES, save, setup
from indoorloc.validation import cv_eval, user_folds

RES = ROOT / "results"
setup()
train, _, _ = load_splits()
Xtr, ytr, mtr = split_xy(train)
groups = mtr["USERID"].to_numpy()
folds = user_folds(groups)

best_file = RES / "dl_best.json"
base = json.load(open(best_file)) if best_file.exists() else {}
print("base settings:", base or "defaults")


def model(**kw):
    return MultitaskModel(**{**base, **kw})


rows = []

# 1. multitask vs single-task: three separate networks, one per target
cv = cv_eval(lambda: model(), Xtr, ytr, groups, folds)
rows.append({"ablation": "tasks", "setting": "multitask (3 heads)", **cv})
print("multitask", round(cv["floor_acc"], 3), round(cv["mean_error_m"], 2), flush=True)


class SingleTaskEnsemble:
    """Three single-task networks, combined into one prediction."""

    def fit(self, X, y, groups=None):
        self.models_ = {t: model(tasks=(t,)).fit(X, y, groups=groups) for t in ["building", "floor", "coords"]}
        return self

    def predict(self, X):
        pb = self.models_["building"].predict(X)
        pf = self.models_["floor"].predict(X)
        pc = self.models_["coords"].predict(X)
        return {"BUILDINGID": pb["BUILDINGID"], "FLOOR": pf["FLOOR"], "LONGITUDE": pc["LONGITUDE"], "LATITUDE": pc["LATITUDE"]}


cv = cv_eval(lambda: SingleTaskEnsemble(), Xtr, ytr, groups, folds)
rows.append({"ablation": "tasks", "setting": "three single-task networks", **cv})
print("single-task", round(cv["floor_acc"], 3), round(cv["mean_error_m"], 2), flush=True)

# 2. loss weights (building, floor, coords)
for w in [(1, 1, 1), (1, 2, 1), (1, 1, 3), (1, 3, 3), (0.5, 1, 5)]:
    cv = cv_eval(lambda w=w: model(loss_weights=w), Xtr, ytr, groups, folds)
    rows.append({"ablation": "loss_weights", "setting": str(w), **cv})
    print("weights", w, round(cv["floor_acc"], 3), round(cv["mean_error_m"], 2), flush=True)

# 3. trunk size
for h in [(256,), (512, 256), (512, 256, 128), (1024, 512, 256)]:
    cv = cv_eval(lambda h=h: model(hidden=h), Xtr, ytr, groups, folds)
    rows.append({"ablation": "trunk", "setting": str(h), **cv})
    print("trunk", h, round(cv["floor_acc"], 3), round(cv["mean_error_m"], 2), flush=True)

df = pd.DataFrame(rows)
df.to_csv(RES / "dl_ablations.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
labels = df.ablation + ": " + df.setting
axes[0].barh(labels, df.floor_acc * 100, color=SERIES[0])
axes[0].set_xlabel("CV floor accuracy (%)")
axes[0].invert_yaxis()
axes[1].barh(labels, df.mean_error_m, color=SERIES[0])
axes[1].set_xlabel("CV mean position error (m)")
axes[1].invert_yaxis()
axes[1].set_yticklabels([])
fig.suptitle("Ablations (5-fold CV by user)")
save(fig, "dl_ablations")
print(df[["ablation", "setting", "floor_acc", "mean_error_m"]].round(3).to_string(index=False))
