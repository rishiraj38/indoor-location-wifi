"""Location models.

Each model exposes `fit(X_raw, y)` and `predict(X_raw) -> dict` with keys
BUILDINGID, FLOOR, LONGITUDE, LATITUDE. Feature engineering lives inside the
model so that cross-validation refits it on every training fold.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from .features import FingerprintFeatures

SEED = 42


class EncodedClassifier:
    """Maps arbitrary labels to 0..k-1 (XGBoost requires it) and handles the
    single-class case that can occur inside one building's subset."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.enc_ = LabelEncoder().fit(y)
        if len(self.enc_.classes_) == 1:
            self.model_ = DummyClassifier(strategy="most_frequent").fit(X, y)
        else:
            self.model_ = clone(self.estimator).fit(X, self.enc_.transform(y))
        return self

    def predict(self, X):
        if isinstance(self.model_, DummyClassifier):
            return self.model_.predict(X)
        return self.enc_.inverse_transform(np.asarray(self.model_.predict(X)).ravel().astype(int))


class TwoTargetRegressor:
    """Fits one regressor per coordinate. Coordinates are centred on the
    training mean first, which keeps tree leaf values small and well scaled."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, Y):
        Y = np.asarray(Y, dtype=float)
        self.mean_ = Y.mean(axis=0)
        self.models_ = [clone(self.estimator).fit(X, Y[:, j] - self.mean_[j]) for j in range(Y.shape[1])]
        return self

    def predict(self, X):
        return np.column_stack([m.predict(X) for m in self.models_]) + self.mean_


def estimators(kind: str, **params):
    """(classifier, regressor) pair for a model family."""
    if kind == "rf":
        common = dict(n_estimators=params.get("n_estimators", 300), n_jobs=-1, random_state=SEED,
                      max_features=params.get("max_features", "sqrt"))
        return RandomForestClassifier(**common), RandomForestRegressor(**common)
    if kind == "xgb":
        common = dict(n_estimators=params.get("n_estimators", 400), max_depth=params.get("max_depth", 8),
                      learning_rate=params.get("learning_rate", 0.1), subsample=0.8,
                      colsample_bytree=params.get("colsample_bytree", 0.5), tree_method="hist",
                      n_jobs=-1, random_state=SEED)
        return XGBClassifier(**common), XGBRegressor(**common)
    if kind == "knn":
        k = params.get("k", 5)
        return (KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1),
                KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1))
    raise ValueError(kind)


COORDS = ["LONGITUDE", "LATITUDE"]


class IndependentModel:
    """AML baseline from the brief: one classifier for building, one for floor
    and one regressor for coordinates, each trained on every scan."""

    def __init__(self, kind="rf", features=None, **params):
        self.kind, self.params = kind, params
        self.features = features or FingerprintFeatures()

    def fit(self, X, y: pd.DataFrame):
        self.fe_ = clone(self.features).fit(X)
        Z = self.fe_.transform(X)
        clf, reg = estimators(self.kind, **self.params)
        self.building_ = EncodedClassifier(clf).fit(Z, y["BUILDINGID"])
        self.floor_ = EncodedClassifier(clf).fit(Z, y["FLOOR"])
        self.coords_ = TwoTargetRegressor(reg).fit(Z, y[COORDS])
        return self

    def predict(self, X):
        Z = self.fe_.transform(X)
        xy = self.coords_.predict(Z)
        return {"BUILDINGID": self.building_.predict(Z), "FLOOR": self.floor_.predict(Z),
                "LONGITUDE": xy[:, 0], "LATITUDE": xy[:, 1]}


class HierarchicalModel:
    """Predict building first, then route each scan to a floor classifier and
    coordinate regressor trained only on that building's scans.

    Buildings are far apart and share few access points, so a per-building
    model solves a much easier, more local problem.
    """

    def __init__(self, kind="xgb", features=None, **params):
        self.kind, self.params = kind, params
        self.features = features or FingerprintFeatures()

    def fit(self, X, y: pd.DataFrame):
        self.fe_ = clone(self.features).fit(X)
        Z = self.fe_.transform(X)
        clf, reg = estimators(self.kind, **self.params)
        self.building_ = EncodedClassifier(clf).fit(Z, y["BUILDINGID"])
        self.per_building_ = {}
        for b in np.unique(y["BUILDINGID"]):
            m = (y["BUILDINGID"] == b).to_numpy()
            self.per_building_[b] = (
                EncodedClassifier(clf).fit(Z[m], y["FLOOR"][m]),
                TwoTargetRegressor(reg).fit(Z[m], y[COORDS][m]),
            )
        return self

    def predict(self, X):
        Z = self.fe_.transform(X)
        b = self.building_.predict(Z)
        floor = np.zeros(len(Z), dtype=int)
        xy = np.zeros((len(Z), 2))
        for bid, (fclf, creg) in self.per_building_.items():
            m = b == bid
            if m.any():
                floor[m] = fclf.predict(Z[m])
                xy[m] = creg.predict(Z[m])
        return {"BUILDINGID": b, "FLOOR": floor, "LONGITUDE": xy[:, 0], "LATITUDE": xy[:, 1]}


class CentroidBaseline:
    """Sanity floor: majority building/floor, mean coordinates."""

    def fit(self, X, y):
        self.b_ = y["BUILDINGID"].mode()[0]
        self.f_ = y["FLOOR"].mode()[0]
        self.xy_ = y[COORDS].mean().to_numpy()
        return self

    def predict(self, X):
        n = len(X)
        return {"BUILDINGID": np.full(n, self.b_), "FLOOR": np.full(n, self.f_),
                "LONGITUDE": np.full(n, self.xy_[0]), "LATITUDE": np.full(n, self.xy_[1])}
