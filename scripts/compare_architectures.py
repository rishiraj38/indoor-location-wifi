"""Phase 2: compare architecture choices for the multitask network.

Owner: Rishi Raj.

Starting from the tuned loss balance (coordinate weight 20, patience 30),
each variant changes one thing. All with 5-fold cross-validation by user.
Writes results/dl_architecture.csv.
"""

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

BASE = dict(loss_weights=(1, 1, 20), patience=30, epochs=300)
variants = {
    "baseline: 512-256, mse": {},
    "batch normalisation": dict(batch_norm=True),
    "huber coordinate loss (delta 0.1)": dict(coord_loss="huber"),
    "wider trunk: 1024-512-256": dict(hidden=(1024, 512, 256)),
    "smaller trunk: 256": dict(hidden=(256,)),
    "more dropout: 0.5": dict(dropout=0.5),
    "wider trunk + huber (delta 0.1)": dict(hidden=(1024, 512, 256), coord_loss="huber"),
}

rows = []
for name, kw in variants.items():
    cv = cv_eval(lambda kw=kw: MultitaskModel(**BASE, **kw), Xtr, ytr, groups, folds)
    rows.append({"variant": name, **kw, **cv})
    print(f"{name:28s} building {cv['building_acc']:.3f}  floor {cv['floor_acc']:.3f}  err {cv['mean_error_m']:.2f} m", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RES / "dl_architecture.csv", index=False)
