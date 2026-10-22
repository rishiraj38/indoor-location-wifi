"""Generate notebooks/phase1_ml_baseline.ipynb (run it afterwards with nbconvert --execute)."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Indoor Location Prediction from Wi-Fi Signals: Phase 1 (Machine Learning)

**Task.** From one Wi-Fi scan (RSSI of up to 520 access points), predict the **building** (3 classes),
**floor** (up to 5) and **(x, y) coordinates** in metres.
**Data.** UJIIndoorLoc: 19,937 training scans and 1,111 validation scans (used as the held-out test set).

This notebook walks through the full Phase 1 pipeline. The reusable code is in `src/indoorloc/`; the
full experiment run (feature comparison, cross-validated tuning) is in `scripts/run_experiments.py`, and this
notebook reuses the settings it chose (`results/best_params.json`).

**Models:** Linear + Logistic Regression (baseline), k-NN (classic fingerprinting), Random Forest (main model)."""),
    code("""import sys, json
sys.path.insert(0, "../src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display

from indoorloc.data import load_raw, load_splits, split_xy, WAP_COLS, NOT_DETECTED
from indoorloc.features import FingerprintFeatures
from indoorloc.models import IndependentModel, HierarchicalModel, CentroidBaseline
from indoorloc.metrics import summarize, hierarchical_breakdown, per_group_error
pd.set_option("display.precision", 3)"""),
    md("## 1. Data and cleaning"),
    code("""raw = load_raw("train")
train, test, report = load_splits()
print(report)
print("train", train.shape, "| test", test.shape)
train.iloc[:3, list(range(5)) + list(range(520, 529))]"""),
    md("""The value **+100 means "AP not detected"**. Real readings run from −104 to 0 dBm. About 96% of all
entries are +100. Used as a plain number, +100 would tell a model that undetected APs are the *strongest*."""),
    code("""W = train[WAP_COLS].to_numpy()
heard = W != NOT_DETECTED
print(f"missing fraction: {1 - heard.mean():.3f}")
print(f"APs heard per scan: mean {heard.sum(1).mean():.1f}, median {np.median(heard.sum(1)):.0f}")
print(f"APs never heard in training: {(~heard.any(0)).sum()}")"""),
    md("## 2. Exploratory data analysis\nFigures are produced by `scripts/run_eda.py`."),
    *[code(f'display(Image("../results/figures/{f}.png"))') for f in [
        "eda_campus_map", "eda_class_balance", "eda_rssi_distribution", "eda_sparsity_heatmap",
        "eda_ap_locality", "eda_floors_per_building", "eda_users_phones", "eda_suspicious_readings", "eda_pca"]],
    md("""**Takeaways.** (1) Buildings are physically and radio-separated: most APs are heard in one building only,
so building classification is easy. (2) Floors share the same footprint, so floor must come from the signal.
(3) The test set is skewed to building 0, was collected 4 months later, and 64% of its scans come from phones
not present in training. (4) Phone 19 produces implausibly strong readings (> −30 dBm)."""),
    md("""## 3. Feature engineering
The main step is replacing **+100 ("not heard") with −105 dBm**, just below the weakest real reading. We also tried
other versions (shifted, compressed, heard/not-heard flags, per-scan summary). Cross-validation on unseen users chose
between them."""),
    code("""Xtr, ytr, mtr = split_xy(train)
Xte, yte, mte = split_xy(test)
fe = FingerprintFeatures(representation="raw", add_indicators=False, add_summary=False).fit(Xtr)
Z = fe.transform(Xtr)
print("APs kept (heard at least once in training):", len(fe.keep_))
print("value used for 'not heard':", fe.floor_)
print("example scan, first 8 APs:", Z[0, :8])"""),
    code("""abl = pd.read_csv("../results/ablation_features.csv")
abl[["model", "features", "cv_ipin_score_m", "test_ipin_score_m", "test_building_floor_acc", "test_mean_error_m"]]"""),
    code('display(Image("../results/figures/ablation_features.png"))'),
    md("""**Raw and shifted give exactly the same result.** A tree only asks "is this AP above a threshold?", and shifting
every value by the same amount keeps their order. k-NN uses distances, and shifting both scans by the same amount
does not change the distance between them. The extra features did not help, so we keep the simple version."""),
    md("## 4. Choosing hyper-parameters with cross-validation (5 folds, each holding out whole users)"),
    code("""tuning = pd.read_csv("../results/tuning_cv.csv")
best = json.load(open("../results/best_params.json"))
summary = json.load(open("../results/summary.json"))
print("chosen:", best)
tuning[["model", "params", "building_floor_acc", "mean_error_m", "ipin_score_m"]]"""),
    code('display(Image("../results/figures/tuning_curves.png"))'),
    md("## 5. Train the final models on all training data and test on the validation file"),
    code("""F = FingerprintFeatures
rf_f, knn_f = summary["rf_features"], summary["knn_features"]
raw = dict(representation="raw", add_indicators=False, add_summary=False)
models = {
    "Centroid": CentroidBaseline(),
    "Linear + Logistic Regression": IndependentModel("linear", features=F(**raw)),
    "k-NN": IndependentModel("knn", features=F(**knn_f), **best["k-NN"]),
    "Random Forest": IndependentModel("rf", features=F(**rf_f), **best["Random Forest"]),
    "Random Forest (per-building)": HierarchicalModel("rf", features=F(**rf_f), **best["Random Forest"]),
}
preds, rows = {}, []
for name, m in models.items():
    m.fit(Xtr, ytr)
    preds[name] = m.predict(Xte)
    rows.append({"model": name, **summarize(yte, preds[name])})
results = pd.DataFrame(rows).set_index("model")
results[["building_acc", "floor_acc", "building_floor_acc", "mean_error_m", "median_error_m", "ipin_score_m"]]"""),
    code('display(Image("../results/figures/model_comparison.png"))\ndisplay(Image("../results/figures/error_cdf.png"))'),
    md("## 6. Error analysis of the main model (Random Forest, per-building)"),
    code("""p = preds["Random Forest (per-building)"]
hierarchical_breakdown(yte, p)"""),
    code('display(Image("../results/figures/hierarchical_breakdown.png"))'),
    code("""for f in ["confusion_floor_rf", "error_map_rf", "confusion_floor_knn", "error_map_knn"]:
    display(Image(f"../results/figures/{f}.png"))"""),
    code("""per_group_error(yte, p, mte["PHONEID"])"""),
    code('display(Image("../results/figures/per_phone_error.png"))\ndisplay(Image("../results/figures/error_vs_aps_heard.png"))\ndisplay(Image("../results/figures/feature_importance.png"))'),
    code("""pd.read_csv("../results/ablation_outliers.csv")[["training_set", "n_train", "building_floor_acc", "mean_error_m", "ipin_score_m"]]"""),
    md("""## 7. Gap identified for Phase 2

Our models treat building, floor and position as **separate problems**. The per-building forest links them only
through a hand-made "building first" step, and a wrong building sends the scan to the wrong sub-model. None of them
learns a representation of the ~465 interacting APs that all three tasks could share. Phase 2 trains a
**multitask MLP**: a shared trunk with two classification heads and one regression head, compared with
these baselines on exactly the same split and metrics."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = ROOT / "notebooks" / "phase1_ml_baseline.ipynb"
nbf.write(nb, out)
print(out)
