"""EDA: saves figures to results/figures and numbers to results/eda_summary.json.

Exploratory analysis authored by: Yash Raj (2401010514)
  - Campus map, class-balance, RSSI distribution, AP-locality, floor-layout,
    user/phone breakdown, suspicious-readings, PCA, and sparsity-heatmap figures.
  - Summary statistics written to results/eda_summary.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from indoorloc.data import NOT_DETECTED, ROOT, WAP_COLS, clean_train, load_raw
from indoorloc.features import FingerprintFeatures
from indoorloc.plotting import (BUILDING_COLORS, FLOOR_COLORS, INK_2, SEQ, SERIES, save, setup)

setup()
raw_train, test = load_raw("train"), load_raw("test")
train, clean_report = clean_train(raw_train)
summary = {"cleaning": clean_report}

W_tr = train[WAP_COLS].to_numpy()
W_te = test[WAP_COLS].to_numpy()
heard_tr, heard_te = W_tr != NOT_DETECTED, W_te != NOT_DETECTED

# basic numbers
summary["shape"] = {"train": list(train.shape), "test": list(test.shape)}
summary["sparsity"] = {"train_missing_frac": float(1 - heard_tr.mean()),
                       "test_missing_frac": float(1 - heard_te.mean())}
summary["aps_per_scan"] = {
    s: {"mean": float(h.sum(1).mean()), "median": float(np.median(h.sum(1))), "max": int(h.sum(1).max())}
    for s, h in [("train", heard_tr), ("test", heard_te)]}
summary["aps"] = {
    "never_heard_train": int((~heard_tr.any(0)).sum()),
    "never_heard_test": int((~heard_te.any(0)).sum()),
    "heard_in_test_but_not_train": int((heard_te.any(0) & ~heard_tr.any(0)).sum()),
}
summary["rssi_range"] = {"train": [int(W_tr[heard_tr].min()), int(W_tr[heard_tr].max())],
                         "test": [int(W_te[heard_te].min()), int(W_te[heard_te].max())]}
strong = heard_tr & (W_tr > -30)
summary["suspicious_strong_readings"] = {
    "values_above_-30dBm": int(strong.sum()), "scans_affected": int(strong.any(1).sum()),
    "by_user_phone": {f"user{u}_phone{p}": int(n) for (u, p), n in
                      train.loc[strong.any(1), ["USERID", "PHONEID"]].value_counts().items()},
}
summary["users"] = {"train": int(train.USERID.nunique()), "test_userid_values": sorted(test.USERID.unique().tolist())}
summary["phones"] = {"train": sorted(train.PHONEID.unique().tolist()), "test": sorted(test.PHONEID.unique().tolist()),
                     "test_only": sorted(set(test.PHONEID) - set(train.PHONEID))}
summary["time"] = {s: [str(pd.to_datetime(d.TIMESTAMP.min(), unit="s").date()),
                       str(pd.to_datetime(d.TIMESTAMP.max(), unit="s").date())] for s, d in [("train", train), ("test", test)]}

# Count how many buildings each AP is heard in (most are only in one).
ap_buildings = np.array([train.loc[heard_tr[:, j], "BUILDINGID"].nunique() for j in range(len(WAP_COLS))])
summary["aps_by_number_of_buildings_heard_in"] = {str(k): int((ap_buildings == k).sum()) for k in range(4)}

# figure: campus map
# Plot in metres from the south-west corner so the axis numbers are readable.
X0, Y0 = train.LONGITUDE.min(), train.LATITUDE.min()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
for ax, (name, d) in zip(axes, [("Training (cleaned)", train), ("Validation / test", test)]):
    for b, c in BUILDING_COLORS.items():
        s = d[d.BUILDINGID == b]
        ax.scatter(s.LONGITUDE - X0, s.LATITUDE - Y0, s=6, color=c, alpha=0.6, linewidths=0, label=f"Building {b}")
    ax.set_title(f"{name}: {len(d):,} scans")
    ax.set_xlabel("Metres east")
    ax.set_aspect("equal")
axes[0].set_ylabel("Metres north")
axes[0].legend(markerscale=3, loc="upper right")
save(fig, "eda_campus_map")

# figure: class balance
counts = pd.DataFrame({
    "Train": train.groupby(["BUILDINGID", "FLOOR"]).size(),
    "Test": test.groupby(["BUILDINGID", "FLOOR"]).size(),
}).fillna(0)
summary["scans_per_building_floor"] = {f"B{b}F{f}": {k: int(v) for k, v in r.items()} for (b, f), r in counts.iterrows()}
fig, ax = plt.subplots(figsize=(9, 3.6))
x = np.arange(len(counts))
share = counts / counts.sum()
ax.bar(x - 0.2, share["Train"] * 100, width=0.38, color=SERIES[0], label="Train")
ax.bar(x + 0.2, share["Test"] * 100, width=0.38, color=SERIES[1], label="Test")
ax.set_xticks(x, [f"B{b}-F{f}" for b, f in counts.index])
ax.set_ylabel("Share of scans (%)")
ax.set_title("Building-floor class balance: test is skewed towards building 0")
ax.legend()
save(fig, "eda_class_balance")

# figure: RSSI distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
bins = np.arange(-105, 1, 2)
for d, h, c, lab in [(W_tr, heard_tr, SERIES[0], "Train"), (W_te, heard_te, SERIES[1], "Test")]:
    axes[0].hist(d[h], bins=bins, density=True, histtype="step", color=c, linewidth=2, label=lab)
axes[0].set_xlabel("RSSI of detected access points (dBm)")
axes[0].set_ylabel("Density")
axes[0].set_title("Detected signals are weak: most below -70 dBm")
axes[0].legend()
bins = np.arange(0, 60, 2)
for h, c, lab in [(heard_tr, SERIES[0], "Train"), (heard_te, SERIES[1], "Test")]:
    axes[1].hist(h.sum(1), bins=bins, density=True, histtype="step", color=c, linewidth=2, label=lab)
axes[1].set_xlabel("Access points heard per scan (of 520)")
axes[1].set_title(f"Each scan hears ~{heard_tr.sum(1).mean():.0f} APs: {100 * (1 - heard_tr.mean()):.1f}% missing")
axes[1].legend()
save(fig, "eda_rssi_distribution")

# figure: AP locality
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
freq = np.sort(heard_tr.mean(0))[::-1] * 100
axes[0].plot(np.arange(1, len(freq) + 1), freq, color=SERIES[0])
axes[0].set_yscale("log")
axes[0].set_xlabel("Access point rank")
axes[0].set_ylabel("% of training scans that hear it")
axes[0].set_title("A few APs are common, most are rare")
k = [summary["aps_by_number_of_buildings_heard_in"][str(i)] for i in range(4)]
axes[1].bar(range(4), k, color=SERIES[0], width=0.6)
for i, v in enumerate(k):
    axes[1].text(i, v, str(v), ha="center", va="bottom", color=INK_2)
axes[1].set_xticks(range(4), ["never heard", "1 building", "2 buildings", "3 buildings"])
axes[1].set_ylabel("Number of APs")
axes[1].set_title("Most APs are heard in only one building")
save(fig, "eda_ap_locality")

# figure: floors per building
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, b in zip(axes, range(3)):
    s = train[train.BUILDINGID == b]
    for f in sorted(s.FLOOR.unique()):
        t = s[s.FLOOR == f]
        ax.scatter(t.LONGITUDE, t.LATITUDE, s=5, color=FLOOR_COLORS[f], alpha=0.7, linewidths=0, label=f"Floor {f}")
    ax.set_title(f"Building {b}")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
axes[2].legend(markerscale=3, loc="lower left", fontsize=8)
fig.suptitle("Floors are stacked on the same (x, y) footprint, so floor must come from signal, not position",
             fontsize=10, color=INK_2)
save(fig, "eda_floors_per_building")

# figure: users and phones
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
u = train.USERID.value_counts().sort_index()
axes[0].bar(u.index.astype(str), u.values, color=SERIES[0])
axes[0].set_xlabel("User ID (train)")
axes[0].set_ylabel("Scans")
axes[0].set_title("Collection effort is uneven across users")
p = pd.DataFrame({"Train": train.PHONEID.value_counts(), "Test": test.PHONEID.value_counts()}).fillna(0).sort_index()
x = np.arange(len(p))
axes[1].bar(x - 0.2, p.Train / p.Train.sum() * 100, 0.38, color=SERIES[0], label="Train")
axes[1].bar(x + 0.2, p.Test / p.Test.sum() * 100, 0.38, color=SERIES[1], label="Test")
axes[1].set_xticks(x, p.index.astype(int).astype(str), fontsize=8)
axes[1].set_xlabel("Phone ID")
axes[1].set_ylabel("Share of scans (%)")
axes[1].set_title("Test uses different phones in different proportions")
axes[1].legend()
save(fig, "eda_users_phones")

# figure: suspicious readings
fig, ax = plt.subplots(figsize=(7, 3.4))
by_phone = pd.Series(strong.any(1), index=train.index).groupby(train.PHONEID).mean() * 100
ax.bar(by_phone.index.astype(str), by_phone.values, color=SERIES[0])
ax.set_xlabel("Phone ID")
ax.set_ylabel("% scans with an RSSI above -30 dBm")
ax.set_title("Implausibly strong readings come almost entirely from phone 19")
save(fig, "eda_suspicious_readings")

# figure: PCA
fe = FingerprintFeatures(add_indicators=False, add_summary=False).fit(W_tr)
pca = PCA(n_components=50, random_state=0).fit(fe.transform(W_tr))
Z_tr, Z_te = pca.transform(fe.transform(W_tr)), pca.transform(fe.transform(W_te))
summary["pca_explained_variance_first_50"] = float(pca.explained_variance_ratio_.sum())
summary["pca_components_for_90pct"] = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.9) + 1) \
    if pca.explained_variance_ratio_.sum() >= 0.9 else ">50"
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for b, c in BUILDING_COLORS.items():
    m = train.BUILDINGID.to_numpy() == b
    axes[0].scatter(Z_tr[m, 0], Z_tr[m, 1], s=4, color=c, alpha=0.5, linewidths=0, label=f"Building {b}")
for b, c in BUILDING_COLORS.items():
    m = test.BUILDINGID.to_numpy() == b
    axes[0].scatter(Z_te[m, 0], Z_te[m, 1], s=12, color=c, marker="x", linewidths=0.8)
axes[0].set_title("PCA of powed RSSI: buildings separate cleanly\n(dots = train, crosses = test)")
axes[0].set_xlabel("PC 1")
axes[0].set_ylabel("PC 2")
axes[0].legend(markerscale=3)
axes[1].plot(np.arange(1, 51), np.cumsum(pca.explained_variance_ratio_) * 100, color=SERIES[0])
axes[1].set_xlabel("Number of components")
axes[1].set_ylabel("Cumulative variance explained (%)")
axes[1].set_title("Variance is spread across many directions")
save(fig, "eda_pca")

# figure: sparsity heatmap
fig, ax = plt.subplots(figsize=(11, 3.2))
order = np.argsort(train.BUILDINGID.to_numpy() * 10 + train.FLOOR.to_numpy(), kind="stable")
idx = order[:: max(1, len(order) // 400)]
used = np.flatnonzero(heard_tr.any(0))
img = np.where(heard_tr[np.ix_(idx, used)], W_tr[np.ix_(idx, used)], np.nan)
im = ax.imshow(img, aspect="auto", cmap=SEQ, interpolation="nearest", vmin=-100, vmax=-30)
ax.set_xlabel("Access point (only those heard in training)")
ax.set_ylabel("Scans sorted by building, floor")
ax.set_title("The fingerprint matrix is ~96% empty and block-structured by building")
ax.grid(False)
fig.colorbar(im, ax=ax, label="RSSI (dBm); blank = not heard")
save(fig, "eda_sparsity_heatmap")

out = ROOT / "results" / "eda_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
