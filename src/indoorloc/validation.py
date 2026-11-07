"""User-grouped cross-validation, shared by the Phase 2 scripts.

Same protocol as Phase 1: 5 folds, grouped by USERID, so the users in a
validation fold never appear in its training fold.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .metrics import summarize


def user_folds(groups, n_splits=5):
    groups = np.asarray(groups)
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def cv_eval(make_model, X, y, groups, folds, pass_groups=True):
    """Train a fresh model on each fold and average the metrics.

    pass_groups=True hands the fold's USERIDs to fit(), which the neural
    network uses to hold out whole users for early stopping.
    """
    groups = np.asarray(groups)
    rows = []
    for tr, va in folds:
        model = make_model()
        y_tr = y.iloc[tr].reset_index(drop=True)
        if pass_groups:
            model.fit(X[tr], y_tr, groups=groups[tr])
        else:
            model.fit(X[tr], y_tr)
        rows.append(summarize(y.iloc[va].reset_index(drop=True), model.predict(X[va])))
    return pd.DataFrame(rows).mean().to_dict()
