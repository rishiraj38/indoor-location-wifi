"""Phase 2: input scaling and regularisation for the neural network.

Compares, with user-grouped cross-validation:
  - input scaling: min-max, standard, powed
  - dropout: 0.0, 0.3, 0.5
  - weight decay: 0.0, 1e-4
Starts from the settings in results/dl_best.json and changes one thing at a time.
Writes results/dl_regularisation.csv and results/dl_best_regularisation.json.
The full grid of 18 combinations takes about 85 minutes on a laptop CPU.
"""

import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from indoorloc.data import ROOT, load_splits, split_xy
from indoorloc.deep import MultitaskModel
from indoorloc.validation import cv_eval, user_folds

RES = ROOT / "results"
best_file = RES / "dl_best.json"
base = json.load(open(best_file)) if best_file.exists() else {}
print("base settings:", base or "defaults")

train, _, _ = load_splits()
Xtr, ytr, mtr = split_xy(train)
groups = mtr["USERID"].to_numpy()
folds = user_folds(groups)

GRID = list(itertools.product(["minmax", "standard", "powed"], [0.0, 0.3, 0.5], [0.0, 1e-4]))

# rows already in the CSV are skipped, so a stopped run can be started again;
# delete the CSV first if you want a fresh sweep
csv_file = RES / "dl_regularisation.csv"
finished = set()
if csv_file.exists():
    df_done = pd.read_csv(csv_file)
    for _, r in df_done.iterrows():
        finished.add((str(r["scaler"]), float(r["dropout"]), float(r["weight_decay"])))

for scaler, dropout, wd in GRID:
    if (scaler, float(dropout), float(wd)) in finished:
        print(f"skip (already done): {scaler} dropout={dropout:.1f} wd={wd:g}")
        continue
    t0 = time.time()
    cv = cv_eval(lambda: MultitaskModel(**{**base, "scaler": scaler, "dropout": dropout, "weight_decay": wd}), Xtr, ytr, groups, folds)
    elapsed = time.time() - t0
    row = {"scaler": scaler, "dropout": dropout, "weight_decay": wd, **cv, "elapsed_s": round(elapsed, 1)}
    header = not csv_file.exists()
    pd.DataFrame([row]).to_csv(csv_file, mode="a", index=False, header=header)
    finished.add((scaler, float(dropout), float(wd)))
    print(f"{scaler:9s} dropout={dropout:.1f} wd={wd:g}  floor {cv['floor_acc']:.3f}  err {cv['mean_error_m']:.2f} m ({elapsed:.1f}s)", flush=True)

df = pd.read_csv(csv_file)
# only rows from the current grid take part in the choice
in_grid = [(str(r.scaler), float(r.dropout), float(r.weight_decay)) in set(GRID) for r in df.itertuples()]
df = df[in_grid].reset_index(drop=True)
best = df.loc[df.mean_error_m.idxmin()]
json.dump(
    {
        "scaler": best.scaler,
        "dropout": float(best.dropout),
        "weight_decay": float(best.weight_decay),
        "floor_acc": float(best.floor_acc),
        "mean_error_m": float(best.mean_error_m),
    },
    open(RES / "dl_best_regularisation.json", "w"),
    indent=2,
)
print("best:", best.to_dict())
