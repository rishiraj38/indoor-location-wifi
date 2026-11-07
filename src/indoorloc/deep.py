"""Multitask neural network for building, floor and position.

Owner: Rishi Raj.

One network: a shared trunk of fully connected layers, then three heads:
  building (3 classes, softmax), floor (5 classes, softmax), coordinates (2 numbers).
Loss = w_b * CE(building) + w_f * CE(floor) + w_xy * MSE(scaled coordinates).

MultitaskModel has the same fit(X, y) / predict(X) interface as the Phase 1
models in models.py, so the Phase 1 metrics and scripts work unchanged.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .deep_data import CoordScaler, EarlyStopping, RSSIScaler, make_loader
from .features import FingerprintFeatures

SEED = 42
N_BUILDINGS = 3
N_FLOORS = 5


class MultitaskMLP(nn.Module):
    def __init__(self, n_inputs, hidden=(512, 256), dropout=0.3):
        super().__init__()
        layers, prev = [], n_inputs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head_building = nn.Linear(prev, N_BUILDINGS)
        self.head_floor = nn.Linear(prev, N_FLOORS)
        self.head_xy = nn.Linear(prev, 2)

    def forward(self, x):
        z = self.trunk(x)
        return self.head_building(z), self.head_floor(z), self.head_xy(z)

    def embed(self, x):
        """The shared trunk output. Used by the hybrid model."""
        return self.trunk(x)


class MultitaskModel:
    """Train the multitask MLP and predict like the Phase 1 models.

    tasks: which heads get a loss. ("building", "floor", "coords") is the full
    multitask model; a single entry gives a single-task network for ablations.
    """

    def __init__(self, features=None, scaler="minmax", hidden=(512, 256), dropout=0.3,
                 lr=1e-3, weight_decay=1e-4, batch_size=256, epochs=200, patience=15,
                 loss_weights=(1.0, 1.0, 1.0), tasks=("building", "floor", "coords"),
                 val_fraction=0.1, seed=SEED, verbose=False):
        self.features = features or FingerprintFeatures(representation="raw", add_indicators=False, add_summary=False)
        self.scaler = scaler
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.loss_weights = loss_weights
        self.tasks = tasks
        self.val_fraction = val_fraction
        self.seed = seed
        self.verbose = verbose

    def _prepare(self, X):
        return self.sc_.transform(self.fe_.transform(X))

    def fit(self, X, y: pd.DataFrame, groups=None):
        """groups: USERID per row. If given, the early-stopping validation split
        holds out whole users, like the Phase 1 cross-validation."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.fe_ = clone(self.features).fit(X)
        Z = self.fe_.transform(X)
        self.sc_ = RSSIScaler(self.scaler).fit(Z)
        Z = self.sc_.transform(Z)
        self.cs_ = CoordScaler().fit(y[["LONGITUDE", "LATITUDE"]])
        xy = self.cs_.transform(y[["LONGITUDE", "LATITUDE"]])
        yb = y["BUILDINGID"].to_numpy()
        yf = y["FLOOR"].to_numpy()

        idx = np.arange(len(Z))
        if groups is not None:
            tr, va = next(GroupShuffleSplit(n_splits=1, test_size=self.val_fraction, random_state=self.seed).split(idx, groups=groups))
        else:
            tr, va = train_test_split(idx, test_size=self.val_fraction, random_state=self.seed)

        train_loader = make_loader(Z[tr], yb[tr], yf[tr], xy[tr], self.batch_size, shuffle=True)
        val_loader = make_loader(Z[va], yb[va], yf[va], xy[va], 1024, shuffle=False)

        self.model_ = MultitaskMLP(Z.shape[1], self.hidden, self.dropout)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        stopper = EarlyStopping(self.patience)
        self.history_ = []

        for epoch in range(self.epochs):
            self.model_.train()
            train_loss = 0.0
            for xb, bb, fb, cb in train_loader:
                opt.zero_grad()
                loss = self._loss(self.model_(xb), bb, fb, cb)
                loss.backward()
                opt.step()
                train_loss += loss.item() * len(xb)
            val_loss = self._evaluate(val_loader)
            self.history_.append({"epoch": epoch + 1, "train_loss": train_loss / len(tr), "val_loss": val_loss})
            if self.verbose:
                print(f"epoch {epoch + 1:3d}  train {train_loss / len(tr):.4f}  val {val_loss:.4f}")
            if stopper.step(val_loss, self.model_):
                break
        stopper.restore(self.model_)
        self.epochs_run_ = len(self.history_)
        return self

    def _loss(self, outputs, yb, yf, cxy):
        ob, of, oxy = outputs
        wb, wf, wxy = self.loss_weights
        loss = 0.0
        if "building" in self.tasks:
            loss = loss + wb * nn.functional.cross_entropy(ob, yb)
        if "floor" in self.tasks:
            loss = loss + wf * nn.functional.cross_entropy(of, yf)
        if "coords" in self.tasks:
            loss = loss + wxy * nn.functional.mse_loss(oxy, cxy)
        return loss

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model_.eval()
        total, n = 0.0, 0
        for xb, bb, fb, cb in loader:
            total += self._loss(self.model_(xb), bb, fb, cb).item() * len(xb)
            n += len(xb)
        return total / n

    @torch.no_grad()
    def predict(self, X):
        self.model_.eval()
        Z = torch.as_tensor(self._prepare(X), dtype=torch.float32)
        ob, of, oxy = self.model_(Z)
        xy = self.cs_.inverse(oxy.numpy())
        return {
            "BUILDINGID": ob.argmax(1).numpy(),
            "FLOOR": of.argmax(1).numpy(),
            "LONGITUDE": xy[:, 0],
            "LATITUDE": xy[:, 1],
        }

    @torch.no_grad()
    def embed(self, X):
        """Trunk features for every scan, for the hybrid model."""
        self.model_.eval()
        Z = torch.as_tensor(self._prepare(X), dtype=torch.float32)
        return self.model_.embed(Z).numpy()
