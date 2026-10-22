"""Evaluation metrics."""

import numpy as np
import pandas as pd


def positioning_errors(y_true: pd.DataFrame, pred: dict) -> np.ndarray:
    """2-D distance error in metres (the coordinates are already in metres)."""
    dx = y_true["LONGITUDE"].to_numpy() - pred["LONGITUDE"]
    dy = y_true["LATITUDE"].to_numpy() - pred["LATITUDE"]
    return np.hypot(dx, dy)


def summarize(y_true: pd.DataFrame, pred: dict) -> dict:
    """Main metrics for one model on one split.

    IPIN 2015 score = 2-D error + 50 m per wrong building + 4 m per wrong floor.
    """
    b_ok = y_true["BUILDINGID"].to_numpy() == pred["BUILDINGID"]
    f_ok = y_true["FLOOR"].to_numpy() == pred["FLOOR"]
    err = positioning_errors(y_true, pred)
    return {
        "building_acc": b_ok.mean(),
        "floor_acc": f_ok.mean(),
        "building_floor_acc": (b_ok & f_ok).mean(),
        "mae_longitude_m": np.abs(y_true["LONGITUDE"].to_numpy() - pred["LONGITUDE"]).mean(),
        "mae_latitude_m": np.abs(y_true["LATITUDE"].to_numpy() - pred["LATITUDE"]).mean(),
        "mean_error_m": err.mean(),
        "median_error_m": np.median(err),
        "p75_error_m": np.percentile(err, 75),
        "p95_error_m": np.percentile(err, 95),
        "ipin_score_m": (err + 50 * ~b_ok + 4 * ~f_ok).mean(),
    }


def hierarchical_breakdown(y_true: pd.DataFrame, pred: dict) -> pd.DataFrame:
    """Coordinate error split by whether building and floor were right,
    to see how much error comes from getting the building/floor wrong."""
    b_ok = y_true["BUILDINGID"].to_numpy() == pred["BUILDINGID"]
    f_ok = y_true["FLOOR"].to_numpy() == pred["FLOOR"]
    err = positioning_errors(y_true, pred)
    case = np.select(
        [b_ok & f_ok, b_ok & ~f_ok, ~b_ok],
        ["building ok, floor ok", "building ok, floor wrong", "building wrong"], default="",
    )
    df = pd.DataFrame({"case": case, "error_m": err})
    out = df.groupby("case")["error_m"].agg(["count", "mean", "median"])
    out["share_of_total_error"] = df.groupby("case")["error_m"].sum() / err.sum()
    return out.reset_index()


def per_group_error(y_true: pd.DataFrame, pred: dict, groups: pd.Series) -> pd.DataFrame:
    """Accuracy and error per group (building, floor, phone, ...)."""
    b_ok = y_true["BUILDINGID"].to_numpy() == pred["BUILDINGID"]
    f_ok = y_true["FLOOR"].to_numpy() == pred["FLOOR"]
    df = pd.DataFrame({
        groups.name: groups.to_numpy(),
        "error_m": positioning_errors(y_true, pred),
        "floor_ok": b_ok & f_ok,
    })
    return df.groupby(groups.name).agg(
        n=("error_m", "size"),
        mean_error_m=("error_m", "mean"),
        median_error_m=("error_m", "median"),
        building_floor_acc=("floor_ok", "mean"),
    ).reset_index()
