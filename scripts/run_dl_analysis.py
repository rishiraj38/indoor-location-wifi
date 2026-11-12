"""Phase 2: final test results and error analysis for the neural network,
and the comparison with the Phase 1 models.

Owner: Yash Raj.

Trains the final multitask network once on all training data, predicts the
official test file once, and writes:
  results/dl_test_metrics.csv, results/dl_breakdown_*.csv
  results/figures/dl_confusion_floor.png, dl_error_map.png, dl_training_curve.png,
  results/figures/aml_vs_dl.png and results/aml_vs_dl.csv
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from indoorloc.data import ROOT, load_splits, split_xy
from indoorloc.deep import MultitaskModel
from indoorloc.metrics import hierarchical_breakdown, per_group_error, positioning_errors, summarize
from indoorloc.plotting import INK_2, SEQ, SERIES, save, setup

RES = ROOT / "results"
setup()
train, test, _ = load_splits()
Xtr, ytr, mtr = split_xy(train)
Xte, yte, mte = split_xy(test)

best_file = RES / "dl_best.json"
settings = json.load(open(best_file)) if best_file.exists() else {}
print("settings:", settings or "defaults")

model = MultitaskModel(**settings).fit(Xtr, ytr, groups=mtr["USERID"].to_numpy())
pred = model.predict(Xte)
metrics = summarize(yte, pred)
pd.DataFrame([{"model": "Multitask MLP", **metrics, "epochs": model.epochs_run_}]).to_csv(RES / "dl_test_metrics.csv", index=False)
print({k: round(v, 3) for k, v in metrics.items()})

hierarchical_breakdown(yte, pred).to_csv(RES / "dl_breakdown_hierarchical.csv", index=False)
per_group_error(yte, pred, mte["PHONEID"]).to_csv(RES / "dl_breakdown_phone.csv", index=False)

# training curve
h = pd.DataFrame(model.history_)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(h.epoch, h.train_loss, label="train", color=SERIES[0])
ax.plot(h.epoch, h.val_loss, label="validation (held-out users)", color=SERIES[1])
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("Training curve")
ax.legend()
save(fig, "dl_training_curve")

# floor confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, b in zip(axes, range(3)):
    m = yte["BUILDINGID"].to_numpy() == b
    floors = sorted(yte.loc[m, "FLOOR"].unique())
    cm = confusion_matrix(yte.loc[m, "FLOOR"], pred["FLOOR"][m], labels=floors)
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap=SEQ, vmin=0, vmax=1)
    for i in range(len(floors)):
        for j in range(len(floors)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm[i].sum() * 0.55 else "#0b0b0b")
    ax.set_xticks(range(len(floors)), floors)
    ax.set_yticks(range(len(floors)), floors)
    ax.set_xlabel("Predicted floor")
    ax.set_ylabel("True floor")
    ax.set_title(f"Building {b}: {np.trace(cm) / cm.sum() * 100:.1f}% correct")
    ax.grid(False)
fig.suptitle("Floor confusion matrices, multitask MLP (test set)", fontsize=11, fontweight="bold")
save(fig, "dl_confusion_floor")

# error map
X0, Y0 = train.LONGITUDE.min(), train.LATITUDE.min()
err = positioning_errors(yte, pred)
fig, ax = plt.subplots(figsize=(9, 5.5))
tx, ty = yte.LONGITUDE - X0, yte.LATITUDE - Y0
px, py = pred["LONGITUDE"] - X0, pred["LATITUDE"] - Y0
for i in np.flatnonzero(err > np.percentile(err, 90)):
    ax.plot([tx.iloc[i], px[i]], [ty.iloc[i], py[i]], color="#9a9893", linewidth=0.6, zorder=1)
sc = ax.scatter(tx, ty, c=np.clip(err, 0, 30), cmap=SEQ, s=14, linewidths=0, zorder=2)
fig.colorbar(sc, ax=ax, label="Position error (m), capped at 30")
ax.set_aspect("equal")
ax.set_xlabel("Metres east")
ax.set_ylabel("Metres north")
ax.set_title("Error map, multitask MLP (grey lines join the worst 10% to their prediction)")
save(fig, "dl_error_map")

# comparison with Phase 1
p1 = pd.read_csv(RES / "test_metrics.csv")
cmp = pd.concat([p1, pd.DataFrame([{"model": "Multitask MLP", **metrics}])], ignore_index=True)
cmp.to_csv(RES / "aml_vs_dl.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].barh(cmp.model, cmp.building_floor_acc * 100, color=SERIES[0], height=0.6)
axes[0].set_xlim(80, 100)
axes[0].set_xlabel("Building + floor correct (%)")
axes[0].invert_yaxis()
axes[1].barh(cmp.model, cmp.mean_error_m, color=SERIES[0], height=0.6)
axes[1].set_xlabel("Mean position error (m)")
axes[1].invert_yaxis()
axes[1].set_yticklabels([])
fig.suptitle("Phase 1 models vs the multitask network (test set)")
save(fig, "aml_vs_dl")
print(cmp[["model", "building_acc", "floor_acc", "mean_error_m", "median_error_m"]].round(3).to_string(index=False))
