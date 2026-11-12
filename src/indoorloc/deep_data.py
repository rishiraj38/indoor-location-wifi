"""Data preparation for the neural network.

Owner: Yash Yadav.

Everything here is fitted on training data only, like FingerprintFeatures.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class RSSIScaler:
    """Scale the model-ready RSSI matrix (values from -105 to 0) for the network.

    method="minmax": (x - min) / (max - min), so every value is in [0, 1]
    method="standard": (x - mean) / std per column
    """

    def __init__(self, method="minmax"):
        self.method = method

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.method == "minmax":
            self.lo_, self.hi_ = float(X.min()), float(X.max())
        elif self.method == "standard":
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0) + 1e-6
        else:
            raise ValueError(self.method)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.method == "minmax":
            return (X - self.lo_) / (self.hi_ - self.lo_)
        return (X - self.mean_) / self.std_


class CoordScaler:
    """Centre and scale the coordinates so the regression loss is well behaved.

    Raw coordinates are around 4,864,000, which would make the MSE huge.
    """

    def fit(self, Y):
        Y = np.asarray(Y, dtype=np.float64)
        self.mean_ = Y.mean(axis=0)
        self.std_ = Y.std(axis=0)
        return self

    def transform(self, Y):
        return ((np.asarray(Y, dtype=np.float64) - self.mean_) / self.std_).astype(np.float32)

    def inverse(self, Z):
        return np.asarray(Z, dtype=np.float64) * self.std_ + self.mean_


def make_loader(X, y_building, y_floor, y_xy, batch_size=256, shuffle=True, drop_last=False):
    """Wrap numpy arrays as a PyTorch DataLoader."""
    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(np.asarray(y_building), dtype=torch.long),
        torch.as_tensor(np.asarray(y_floor), dtype=torch.long),
        torch.as_tensor(y_xy, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


class EarlyStopping:
    """Stop training when the validation loss has not improved for `patience` epochs.

    Keeps a copy of the best weights so the model can be restored.
    """

    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad_epochs = 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
