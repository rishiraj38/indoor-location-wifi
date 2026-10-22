"""Loading and cleaning the UJIIndoorLoc fingerprint database."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "UJIndoorLoc"

N_WAPS = 520
NOT_DETECTED = 100  # dataset convention: +100 dBm means "access point not heard"
WAP_COLS = [f"WAP{i:03d}" for i in range(1, N_WAPS + 1)]
META_COLS = ["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID",
             "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"]


def load_raw(split: str) -> pd.DataFrame:
    """Read `train` (trainingData.csv) or `test` (validationData.csv) as published."""
    name = {"train": "trainingData.csv", "test": "validationData.csv"}[split]
    return pd.read_csv(RAW_DIR / name)


def clean_train(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop duplicate rows and scans where no AP was heard.

    We only clean the training set. The test set is kept as published so our
    numbers can be compared with other papers.
    """
    report = {"rows_in": len(df)}

    dup = df.duplicated()
    report["duplicates_dropped"] = int(dup.sum())
    df = df[~dup]

    empty = (df[WAP_COLS] == NOT_DETECTED).all(axis=1)
    report["empty_scans_dropped"] = int(empty.sum())
    df = df[~empty]

    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def split_xy(df: pd.DataFrame):
    """Return (rssi matrix, targets frame, metadata frame)."""
    X = df[WAP_COLS].to_numpy(dtype=np.float32)
    y = df[["BUILDINGID", "FLOOR", "LONGITUDE", "LATITUDE"]].copy()
    meta = df[["USERID", "PHONEID", "TIMESTAMP", "SPACEID", "RELATIVEPOSITION"]].copy()
    return X, y, meta


def load_splits():
    """Load the cleaned train set and the original test set."""
    train, report = clean_train(load_raw("train"))
    test = load_raw("test")
    return train, test, report
