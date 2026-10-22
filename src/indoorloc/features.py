"""Feature engineering for sparse Wi-Fi RSSI fingerprints.

Every statistic used here (which access points to keep, the RSSI floor, the
scaling range) is learned from the training set only and then applied
unchanged to the test set, so no information leaks from test into training.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .data import NOT_DETECTED


class FingerprintFeatures(BaseEstimator, TransformerMixin):
    """Turn raw RSSI rows into model-ready features.

    Parameters
    ----------
    representation : {"positive", "powed", "raw"}
        How detected RSSI values are rescaled (Torres-Sospedra et al., 2015):
        * positive: rssi - floor, so "not heard" is 0 and stronger is larger.
        * powed:    positive representation normalised to [0, 1] then raised to
                    `beta`, which compresses weak, noisy readings.
        * raw:      dBm with "not heard" replaced by `floor` (a plain imputation).
    add_indicators : bool
        Append one binary "was this AP heard" column per kept AP.
    add_summary : bool
        Append per-scan statistics (count heard, strongest, mean, std, top-k
        strongest AP ids encoded as their signal ranks).
    min_detections : int
        Drop APs heard fewer than this many times in training. APs never seen in
        training carry no information for any model.
    """

    def __init__(self, representation="powed", beta=np.e, add_indicators=True,
                 add_summary=True, min_detections=1, top_k=3):
        self.representation = representation
        self.beta = beta
        self.add_indicators = add_indicators
        self.add_summary = add_summary
        self.min_detections = min_detections
        self.top_k = top_k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        heard = X != NOT_DETECTED
        self.keep_ = np.flatnonzero(heard.sum(axis=0) >= self.min_detections)
        observed = X[heard]
        # One dBm below the weakest reading in training, so "not heard" sits
        # just under every real measurement rather than at an arbitrary -110.
        self.floor_ = float(observed.min()) - 1.0
        self.ceiling_ = float(observed.max())
        return self

    def _scale(self, X, heard):
        if self.representation == "raw":
            return np.where(heard, X, self.floor_)
        pos = np.where(heard, X - self.floor_, 0.0)
        if self.representation == "positive":
            return pos
        if self.representation == "powed":
            span = self.ceiling_ - self.floor_
            return np.clip(pos / span, 0.0, 1.0) ** self.beta
        raise ValueError(f"unknown representation {self.representation!r}")

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)[:, self.keep_]
        heard = X != NOT_DETECTED
        parts = [self._scale(X, heard)]

        if self.add_indicators:
            parts.append(heard.astype(np.float32))

        if self.add_summary:
            masked = np.where(heard, X, np.nan)
            count = heard.sum(axis=1, keepdims=True)
            with np.errstate(all="ignore"):
                strongest = np.nan_to_num(np.nanmax(masked, axis=1, keepdims=True), nan=self.floor_)
                mean = np.nan_to_num(np.nanmean(masked, axis=1, keepdims=True), nan=self.floor_)
                std = np.nan_to_num(np.nanstd(masked, axis=1, keepdims=True), nan=0.0)
            # Index of the k strongest APs. Trees can split on these ids, which
            # captures "which AP am I closest to" without one-hot blow-up.
            order = np.argsort(np.where(heard, -X, np.inf), axis=1)[:, : self.top_k]
            order = np.where(np.take_along_axis(heard, order, axis=1), order, -1)
            parts += [count, strongest, mean, std, order.astype(np.float32)]

        return np.hstack(parts).astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        aps = [f"WAP{i + 1:03d}" for i in self.keep_]
        names = [f"{a}_rssi" for a in aps]
        if self.add_indicators:
            names += [f"{a}_heard" for a in aps]
        if self.add_summary:
            names += ["n_heard", "max_rssi", "mean_rssi", "std_rssi"]
            names += [f"top{k + 1}_ap" for k in range(self.top_k)]
        return np.array(names)
