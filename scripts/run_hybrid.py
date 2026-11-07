"""Phase 2: hybrid ML + DL models.

Owners: Aditya Samadhiya (Random Forest side) and Rishi Raj (network side).

Two candidates, both with a reason behind them:
  A. Network trunk + Random Forest. The multitask network learns a compact
     representation of the 465 signals (its trunk output). A per-building
     Random Forest is then trained on that representation instead of the raw
     signals. Reason: the forest was best on floor in Phase 1 but could not
     learn interactions between hundreds of routers; the trunk gives it a
     better input.
  B. Split by target. The network for building and floor (where it is
     strongest), the Phase 1 per-building Random Forest for coordinates.
     Reason: use each model where it won.

Evaluated with user-grouped CV first, then once on the test set. Writes
results/hybrid.csv. A hybrid only counts if it beats both of its parts.

To do:
  - Rishi: settle the network settings (results/dl_best.json) from the
    regularisation and ablation results
  - Aditya: run the hybrids, add the figure, and write docs/hybrid.md with the
    reasoning and whether each one earned its place
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from indoorloc.data import ROOT, load_splits, split_xy
from indoorloc.deep import MultitaskModel
from indoorloc.metrics import summarize
from indoorloc.models import HierarchicalModel
from indoorloc.validation import cv_eval, user_folds

RES = ROOT / "results"
train, test, _ = load_splits()
Xtr, ytr, mtr = split_xy(train)
Xte, yte, mte = split_xy(test)
groups = mtr["USERID"].to_numpy()
folds = user_folds(groups)

best_file = RES / "dl_best.json"
settings = json.load(open(best_file)) if best_file.exists() else {}
COORDS = ["LONGITUDE", "LATITUDE"]


class TrunkForest:
    """Hybrid A: network trunk features, then per-building Random Forests."""

    def fit(self, X, y, groups=None):
        self.net_ = MultitaskModel(**settings).fit(X, y, groups=groups)
        E = self.net_.embed(X)
        rf = dict(n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=42)
        self.building_ = RandomForestClassifier(**rf).fit(E, y["BUILDINGID"])
        self.per_building_ = {}
        for b in np.unique(y["BUILDINGID"]):
            m = (y["BUILDINGID"] == b).to_numpy()
            self.per_building_[b] = (RandomForestClassifier(**rf).fit(E[m], y["FLOOR"][m]),
                                     RandomForestRegressor(**rf).fit(E[m], y[COORDS][m]))
        return self

    def predict(self, X):
        E = self.net_.embed(X)
        b = self.building_.predict(E)
        floor = np.zeros(len(E), dtype=int)
        xy = np.zeros((len(E), 2))
        for bid, (fclf, creg) in self.per_building_.items():
            m = b == bid
            if m.any():
                floor[m] = fclf.predict(E[m])
                xy[m] = creg.predict(E[m])
        return {"BUILDINGID": b, "FLOOR": floor, "LONGITUDE": xy[:, 0], "LATITUDE": xy[:, 1]}


class SplitByTarget:
    """Hybrid B: network for building and floor, Random Forest for coordinates."""

    def fit(self, X, y, groups=None):
        self.net_ = MultitaskModel(**settings).fit(X, y, groups=groups)
        self.rf_ = HierarchicalModel("rf", max_features="sqrt", n_estimators=300).fit(X, y)
        return self

    def predict(self, X):
        pn, pr = self.net_.predict(X), self.rf_.predict(X)
        return {"BUILDINGID": pn["BUILDINGID"], "FLOOR": pn["FLOOR"], "LONGITUDE": pr["LONGITUDE"], "LATITUDE": pr["LATITUDE"]}


class NetOnly:
    def fit(self, X, y, groups=None):
        self.m_ = MultitaskModel(**settings).fit(X, y, groups=groups)
        return self

    def predict(self, X):
        return self.m_.predict(X)


class ForestOnly:
    def fit(self, X, y, groups=None):
        self.m_ = HierarchicalModel("rf", max_features="sqrt", n_estimators=300).fit(X, y)
        return self

    def predict(self, X):
        return self.m_.predict(X)


rows = []
for name, cls in [("Multitask MLP alone", NetOnly), ("Random Forest alone", ForestOnly),
                  ("Hybrid A: trunk + forest", TrunkForest), ("Hybrid B: net for class, forest for xy", SplitByTarget)]:
    cv = cv_eval(lambda cls=cls: cls(), Xtr, ytr, groups, folds)
    m = cls().fit(Xtr, ytr, groups=groups)
    te = summarize(yte, m.predict(Xte))
    rows.append({"model": name, **{f"cv_{k}": v for k, v in cv.items()}, **{f"test_{k}": v for k, v in te.items()}})
    print(f"{name:40s} cv floor {cv['floor_acc']:.3f} err {cv['mean_error_m']:.2f} | test floor {te['floor_acc']:.3f} err {te['mean_error_m']:.2f}", flush=True)

pd.DataFrame(rows).to_csv(RES / "hybrid.csv", index=False)
