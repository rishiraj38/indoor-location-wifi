"""Phase 1 experiments: feature comparison, cross-validated tuning, final test
evaluation, and every figure/table used in the report.

How models are chosen: only the training set is used, with 5-fold GroupKFold
by USERID (the users in a validation fold never appear in its training fold).
The official validation file is used once, at the end, as the test set.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from indoorloc.data import NOT_DETECTED, ROOT, load_splits, split_xy
from indoorloc.features import FingerprintFeatures
from indoorloc.metrics import hierarchical_breakdown, per_group_error, positioning_errors, summarize
from indoorloc.models import CentroidBaseline, HierarchicalModel, IndependentModel
from indoorloc.plotting import INK_2, SEQ, SERIES, save, setup

RES = ROOT / "results"
N_FOLDS = 5

setup()
train, test, clean_report = load_splits()
Xtr, ytr, mtr = split_xy(train)
Xte, yte, mte = split_xy(test)
folds = list(GroupKFold(n_splits=N_FOLDS).split(Xtr, groups=mtr["USERID"]))


def log(*a):
    print(*a, flush=True)


def cv_eval(make_model):
    """Average of each metric over the user-grouped folds."""
    rows = []
    for tr_idx, va_idx in folds:
        m = make_model().fit(Xtr[tr_idx], ytr.iloc[tr_idx].reset_index(drop=True))
        rows.append(summarize(ytr.iloc[va_idx].reset_index(drop=True), m.predict(Xtr[va_idx])))
    return pd.DataFrame(rows).mean().to_dict()


def test_eval(make_model):
    t0 = time.time()
    m = make_model().fit(Xtr, ytr)
    fit_s = time.time() - t0
    t0 = time.time()
    pred = m.predict(Xte)
    ms = (time.time() - t0) / len(Xte) * 1000
    return m, pred, {**summarize(yte, pred), "fit_seconds": fit_s, "predict_ms_per_scan": ms}


F = FingerprintFeatures

# =========================================================== 1. which features?
log("== feature comparison (CV picks, test shown for reference)")
feature_configs = {
    "raw dBm, not-heard = -105": dict(representation="raw", add_indicators=False, add_summary=False),
    "positive (shifted so not-heard = 0)": dict(representation="positive", add_indicators=False, add_summary=False),
    "powed (compress weak signals)": dict(representation="powed", add_indicators=False, add_summary=False),
    "raw + heard/not-heard flags": dict(representation="raw", add_indicators=True, add_summary=False),
    "raw + flags + scan summary": dict(representation="raw", add_indicators=True, add_summary=True),
}
rows = []
for model_name, make_for in [
    ("Random Forest (per-building)", lambda cfg: HierarchicalModel("rf", features=F(**cfg), n_estimators=200)),
    ("k-NN (k=5)", lambda cfg: IndependentModel("knn", features=F(**cfg), k=5)),
]:
    for feat_name, cfg in feature_configs.items():
        if model_name.startswith("k-NN") and cfg["add_summary"]:
            continue  # summary columns (AP ids, counts) are not meaningful inside a distance
        make = lambda cfg=cfg, mf=make_for: mf(cfg)
        cv = cv_eval(make)
        _, _, te = test_eval(make)
        rows.append({"model": model_name, "features": feat_name,
                     **{f"cv_{k}": v for k, v in cv.items()}, **{f"test_{k}": v for k, v in te.items()}})
        log(f"  {model_name:28s} {feat_name:38s} cv ipin {cv['ipin_score_m']:.2f} | test ipin {te['ipin_score_m']:.2f}")
abl = pd.DataFrame(rows)
abl.to_csv(RES / "ablation_features.csv", index=False)


def cv_pick(model_name):
    sub = abl[abl.model == model_name]
    return feature_configs[sub.loc[sub.cv_ipin_score_m.idxmin(), "features"]]


rf_feats, knn_feats = cv_pick("Random Forest (per-building)"), cv_pick("k-NN (k=5)")
lin_feats = feature_configs["raw dBm, not-heard = -105"]
log("  chosen: RF", rf_feats, "| k-NN", knn_feats)

# =========================================================== 2. tuning
log("== tuning (user-grouped CV)")
tuning = []
for k in [1, 3, 5, 7, 11]:
    cv = cv_eval(lambda k=k: IndependentModel("knn", features=F(**knn_feats), k=k))
    tuning.append({"model": "k-NN", "params": json.dumps({"k": k}), **cv})
    log(f"  k-NN k={k}: cv ipin {cv['ipin_score_m']:.2f}")
for mf in ["sqrt", 0.1, 0.3]:
    p = {"max_features": mf, "n_estimators": 300}
    cv = cv_eval(lambda p=p: HierarchicalModel("rf", features=F(**rf_feats), **p))
    tuning.append({"model": "Random Forest", "params": json.dumps(p), **cv})
    log(f"  RF {p}: cv ipin {cv['ipin_score_m']:.2f}")
tuning = pd.DataFrame(tuning)
tuning.to_csv(RES / "tuning_cv.csv", index=False)
best = {m: json.loads(tuning.loc[tuning[tuning.model == m].ipin_score_m.idxmin(), "params"])
        for m in ["k-NN", "Random Forest"]}
(RES / "best_params.json").write_text(json.dumps(best, indent=2))
log("  best:", best)

# How many trees? One held-out-user fold, growing the forest.
tr_idx, va_idx = folds[0]
trees = []
for n in [1, 5, 10, 25, 50, 100, 200, 300]:
    m = HierarchicalModel("rf", features=F(**rf_feats), **{**best["Random Forest"], "n_estimators": n})
    m.fit(Xtr[tr_idx], ytr.iloc[tr_idx].reset_index(drop=True))
    s = summarize(ytr.iloc[va_idx].reset_index(drop=True), m.predict(Xtr[va_idx]))
    trees.append({"n_trees": n, **s})
    log(f"  trees={n}: floor acc {s['building_floor_acc']:.3f}, mean err {s['mean_error_m']:.2f}")
trees = pd.DataFrame(trees)
trees.to_csv(RES / "rf_number_of_trees.csv", index=False)

# =========================================================== 3. final comparison on the test set
log("== final test evaluation")
MODELS = {
    "Centroid (guess the average)": lambda: CentroidBaseline(),
    "Linear + Logistic Regression": lambda: IndependentModel("linear", features=F(**lin_feats)),
    "k-NN": lambda: IndependentModel("knn", features=F(**knn_feats), **best["k-NN"]),
    "Random Forest": lambda: IndependentModel("rf", features=F(**rf_feats), **best["Random Forest"]),
    "Random Forest (per-building)": lambda: HierarchicalModel("rf", features=F(**rf_feats), **best["Random Forest"]),
}
SHORT = {"Centroid (guess the average)": "Centroid", "Linear + Logistic Regression": "Linear/Logistic",
         "k-NN": "k-NN", "Random Forest": "RF", "Random Forest (per-building)": "RF per-building"}
final, preds, models = [], {}, {}
for name, make in MODELS.items():
    m, p, te = test_eval(make)
    models[name], preds[name] = m, p
    final.append({"model": name, **te})
    log(f"  {name:30s} building {te['building_acc']:.4f} floor {te['floor_acc']:.4f} "
        f"mean err {te['mean_error_m']:.2f} m  ipin {te['ipin_score_m']:.2f}")
final = pd.DataFrame(final)
final.to_csv(RES / "test_metrics.csv", index=False)

MAIN = "Random Forest (per-building)"  # the model the brief asks for; analysed in depth below

# =========================================================== 4. does removing phone-19 outliers help?
log("== outlier check")
strong = ((Xtr != NOT_DETECTED) & (Xtr > -30)).any(axis=1)
rows = []
for label, mask in [("all training scans", np.ones(len(Xtr), bool)), ("without scans having RSSI > -30 dBm", ~strong)]:
    m = HierarchicalModel("rf", features=F(**rf_feats), **best["Random Forest"])
    m.fit(Xtr[mask], ytr[mask].reset_index(drop=True))
    rows.append({"training_set": label, "n_train": int(mask.sum()), **summarize(yte, m.predict(Xte))})
pd.DataFrame(rows).to_csv(RES / "ablation_outliers.csv", index=False)
log(pd.DataFrame(rows)[["training_set", "n_train", "building_floor_acc", "mean_error_m"]].to_string())

# =========================================================== 5. breakdowns
SLUG = {"k-NN": "knn", MAIN: "rf"}
for name, slug in SLUG.items():
    hierarchical_breakdown(yte, preds[name]).to_csv(RES / f"breakdown_hierarchical_{slug}.csv", index=False)
    per_group_error(yte, preds[name], mte["PHONEID"]).to_csv(RES / f"breakdown_phone_{slug}.csv", index=False)
    bf = (yte["BUILDINGID"].astype(str) + "-" + yte["FLOOR"].astype(str)).rename("building_floor")
    per_group_error(yte, preds[name], bf).to_csv(RES / f"breakdown_building_floor_{slug}.csv", index=False)

# =========================================================== 6. figures
# 6a. model comparison (centroid left out: its 140 m would squash the other bars)
f = final[final.model != "Centroid (guess the average)"]
labels = [SHORT[m] for m in f.model]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
axes[0].barh(labels, f.building_floor_acc * 100, color=SERIES[0], height=0.6)
for i, v in enumerate(f.building_floor_acc * 100):
    axes[0].text(v + 0.2, i, f"{v:.1f}%", va="center", color=INK_2, fontsize=9)
axes[0].set_xlim(80, 100)
axes[0].set_xlabel("Building + floor correct (%)")
axes[0].invert_yaxis()
axes[0].set_title("Floor accuracy (test set)")
axes[1].barh(labels, f.mean_error_m, color=SERIES[0], height=0.6)
for i, v in enumerate(f.mean_error_m):
    axes[1].text(v + 0.5, i, f"{v:.1f} m", va="center", color=INK_2, fontsize=9)
axes[1].set_xlabel("Mean position error (m), lower is better")
axes[1].invert_yaxis()
axes[1].set_title("Position error (test set)")
save(fig, "model_comparison")

# 6b. error CDF
fig, ax = plt.subplots(figsize=(7, 4))
for i, name in enumerate([n for n in MODELS if n != "Centroid (guess the average)"]):
    e = np.sort(positioning_errors(yte, preds[name]))
    ax.plot(e, np.arange(1, len(e) + 1) / len(e) * 100, color=SERIES[i], label=SHORT[name])
ax.set_xlim(0, 60)
ax.set_xlabel("Position error (m)")
ax.set_ylabel("% of test scans with error ≤ x")
ax.set_title("Cumulative error distribution")
ax.legend(loc="lower right")
save(fig, "error_cdf")

# 6c. floor confusion matrices
for name, slug in SLUG.items():
    p = preds[name]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, b in zip(axes, range(3)):
        m = yte["BUILDINGID"].to_numpy() == b
        floors = sorted(yte.loc[m, "FLOOR"].unique())
        cm = confusion_matrix(yte.loc[m, "FLOOR"], p["FLOOR"][m], labels=floors)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(floors)):
            for j in range(len(floors)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if cmn[i, j] > 0.55 else "#0b0b0b")
        ax.set_xticks(range(len(floors)), floors)
        ax.set_yticks(range(len(floors)), floors)
        ax.set_xlabel("Predicted floor")
        ax.set_ylabel("True floor")
        ax.set_title(f"Building {b}: {np.trace(cm) / cm.sum() * 100:.1f}% correct")
        ax.grid(False)
    fig.suptitle(f"Floor confusion matrices — {name} (test set)", fontsize=11, fontweight="bold")
    save(fig, f"confusion_floor_{slug}")

# 6d. error maps
X0, Y0 = train.LONGITUDE.min(), train.LATITUDE.min()
for name, slug in SLUG.items():
    p = preds[name]
    err = positioning_errors(yte, p)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    tx, ty = yte.LONGITUDE - X0, yte.LATITUDE - Y0
    px, py = p["LONGITUDE"] - X0, p["LATITUDE"] - Y0
    for i in np.flatnonzero(err > np.percentile(err, 90)):
        ax.plot([tx.iloc[i], px[i]], [ty.iloc[i], py[i]], color="#9a9893", linewidth=0.6, zorder=1)
    sc = ax.scatter(tx, ty, c=np.clip(err, 0, 30), cmap=SEQ, s=14, linewidths=0, zorder=2)
    fig.colorbar(sc, ax=ax, label="Position error (m), capped at 30")
    ax.set_aspect("equal")
    ax.set_xlabel("Metres east")
    ax.set_ylabel("Metres north")
    ax.set_title(f"Error map — {name}\n(dot = true position; grey line joins the worst 10% to their prediction)")
    save(fig, f"error_map_{slug}")

# 6e. hierarchical breakdown
hb = hierarchical_breakdown(yte, preds[MAIN])
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.barh(hb["case"], hb["mean"], color=SERIES[0], height=0.55)
for i, (v, n) in enumerate(zip(hb["mean"], hb["count"])):
    ax.text(v + 0.3, i, f"{v:.1f} m  (n={n})", va="center", color=INK_2, fontsize=9)
ax.set_xlabel("Mean position error (m)")
ax.set_title(f"Error split by building/floor correctness — {MAIN}")
save(fig, "hierarchical_breakdown")

# 6f. per phone
fig, ax = plt.subplots(figsize=(8, 3.4))
for i, name in enumerate(SLUG):
    g = per_group_error(yte, preds[name], mte["PHONEID"])
    x = np.arange(len(g))
    ax.bar(x + (i - 0.5) * 0.4, g.mean_error_m, 0.38, color=SERIES[i], label=SHORT[name])
ax.set_xticks(x, [f"{int(pid)}\n(n={n})" for pid, n in zip(g.PHONEID, g.n)], fontsize=8)
ax.set_xlabel("Test phone ID")
ax.set_ylabel("Mean position error (m)")
ax.set_title("Error by phone model")
ax.legend()
save(fig, "per_phone_error")

# 6g. feature comparison
sub = abl[abl.model == "Random Forest (per-building)"]
fig, ax = plt.subplots(figsize=(8, 3.2))
y = np.arange(len(sub))
ax.barh(y - 0.2, sub.cv_ipin_score_m, 0.38, color=SERIES[0], label="CV (unseen users)")
ax.barh(y + 0.2, sub.test_ipin_score_m, 0.38, color=SERIES[1], label="Test")
ax.set_yticks(y, sub.features)
ax.invert_yaxis()
ax.set_xlabel("IPIN score (m), lower is better")
ax.set_title("Feature comparison — Random Forest (per-building)")
ax.legend(loc="lower right")
save(fig, "ablation_features")

# 6h. choosing k, and choosing the number of trees
fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
t = tuning[tuning.model == "k-NN"]
axes[0].plot([json.loads(p)["k"] for p in t.params], t.mean_error_m, marker="o", color=SERIES[0])
axes[0].set_xlabel("k (number of neighbours)")
axes[0].set_ylabel("CV mean error (m)")
axes[0].set_title("Choosing k for k-NN")
axes[1].plot(trees.n_trees, trees.mean_error_m, marker="o", color=SERIES[0])
axes[1].set_xscale("log")
axes[1].set_xlabel("Number of trees (log scale)")
axes[1].set_ylabel("Mean error on held-out users (m)")
axes[1].set_title("More trees help, then level off")
save(fig, "tuning_curves")

# 6i. feature importance
rf = models[MAIN]
names_ = rf.fe_.get_feature_names_out()
imp = pd.Series(rf.building_.model_.feature_importances_, index=names_).sort_values(ascending=False)
imp_b2 = pd.Series(rf.per_building_[2][0].model_.feature_importances_, index=names_).sort_values(ascending=False)
imp.head(30).to_csv(RES / "feature_importance_building.csv")
imp_b2.head(30).to_csv(RES / "feature_importance_floor_b2.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, s, title in [(axes[0], imp.head(15), "building classifier"), (axes[1], imp_b2.head(15), "floor classifier, building 2")]:
    ax.barh(s.index[::-1], s.values[::-1], color=SERIES[0], height=0.6)
    ax.set_xlabel("Importance (mean decrease in Gini impurity)")
    ax.set_title(f"Most useful access points — {title}")
    ax.tick_params(axis="y", labelsize=8)
save(fig, "feature_importance")

# 6j. error vs APs heard
n_heard = (Xte != NOT_DETECTED).sum(1)
err = positioning_errors(yte, preds[MAIN])
g = pd.DataFrame({"bin": pd.cut(n_heard, [0, 5, 10, 15, 20, 25, 30, 60]), "err": err}) \
    .groupby("bin", observed=True).err.agg(["mean", "size"])
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar([str(i) for i in g.index], g["mean"], color=SERIES[0])
for i, (v, n) in enumerate(zip(g["mean"], g["size"])):
    ax.text(i, v, f"n={n}", ha="center", va="bottom", fontsize=8, color=INK_2)
ax.set_xlabel("Access points heard in the scan")
ax.set_ylabel("Mean position error (m)")
ax.set_title("Fewer access points heard → bigger error")
save(fig, "error_vs_aps_heard")

summary = {"cleaning": clean_report, "n_folds": N_FOLDS, "best_params": best,
           "rf_features": rf_feats, "knn_features": knn_feats, "main_model": MAIN,
           "test_metrics": final.set_index("model").round(4).to_dict(orient="index")}
(RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
log("done")
