"""Location models.

Each model has fit(X_raw, y) and predict(X_raw), which returns a dict with
BUILDINGID, FLOOR, LONGITUDE, LATITUDE. The features are built inside the
model so they get refitted on each CV training fold.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .features import FingerprintFeatures

SEED = 42


class EncodedClassifier:
    """Encodes labels as 0..k-1. Also handles the case where only one class
    is present (can happen inside one building)."""

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
    """One regressor per coordinate. We subtract the training mean first
    because the raw coordinates are very large numbers."""

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
    """Return (classifier, regressor) for the given model type."""
    if kind == "rf":
        common = dict(n_estimators=params.get("n_estimators", 300), n_jobs=-1, random_state=SEED,
                      max_features=params.get("max_features", "sqrt"))
        return RandomForestClassifier(**common), RandomForestRegressor(**common)
    if kind == "linear":
        # Logistic regression for building/floor, linear regression for
        # coordinates. Scaling is needed for logistic regression to converge.
        return (make_pipeline(StandardScaler(), LogisticRegression(C=params.get("C", 1.0), max_iter=2000)),
                make_pipeline(StandardScaler(), LinearRegression()))
    if kind == "knn":
        k = params.get("k", 5)
        return (KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1),
                KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1))
    raise ValueError(kind)


COORDS = ["LONGITUDE", "LATITUDE"]


class IndependentModel:
    """Baseline from the brief: separate models for building, floor and
    coordinates, all trained on every scan."""

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
    """Predict the building first, then use floor and coordinate models
    trained only on that building's scans.

    Buildings share few APs, so each per-building model has an easier job.
    """

    def __init__(self, kind="rf", features=None, **params):
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
