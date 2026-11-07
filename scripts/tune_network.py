"""Phase 2: tune the multitask network's loss balance and early stopping.

Owner: Rishi Raj.

With default settings the network already beats Phase 1 on floor but is far
worse on position, because the coordinate loss is small next to the two
cross-entropies and early stopping fires before the coordinate head is
trained. This sweeps the coordinate loss weight and the patience with
user-grouped cross-validation and writes results/dl_tuning.csv.
"""

import itertools
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
# weights 30 and 50 (patience 30 only) were added after the first sweep to check that 20 is not just the edge of the grid
settings = list(itertools.product([1, 3, 5, 10, 20], [15, 30])) + [(30, 30), (50, 30)]
for w_xy, patience in settings:
    cv = cv_eval(lambda: MultitaskModel(loss_weights=(1, 1, w_xy), patience=patience, epochs=300), Xtr, ytr, groups, folds)
    rows.append({"w_xy": w_xy, "patience": patience, **cv})
    print(f"w_xy={w_xy:2d} patience={patience:2d}  building {cv['building_acc']:.3f}  floor {cv['floor_acc']:.3f}  err {cv['mean_error_m']:.2f} m", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RES / "dl_tuning.csv", index=False)
print(df[["w_xy", "patience", "floor_acc", "mean_error_m"]].round(3).to_string(index=False))
