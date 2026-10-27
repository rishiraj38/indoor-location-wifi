"""Write the cleaned and model-ready CSV files to data/processed/.

train_cleaned.csv       training scans with duplicates and empty scans removed (values unchanged)
train_model_ready.csv   cleaned training scans, +100 replaced with -105, never-heard APs dropped
test_model_ready.csv    the official validation file with the same transformation (fitted on train)

Data cleaning pipeline authored by: Yash Raj (2401010514)
  - Fit-on-train-only strategy to prevent leakage of test statistics.
  - Sentinel replacement (-105 dBm) and AP-column pruning persisted in processing_info.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from indoorloc.data import META_COLS, ROOT, WAP_COLS, load_splits
from indoorloc.features import FingerprintFeatures

OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

train, test, report = load_splits()
train.to_csv(OUT / "train_cleaned.csv", index=False)

fe = FingerprintFeatures(representation="raw", add_indicators=False, add_summary=False)
fe.fit(train[WAP_COLS].to_numpy())
kept = [WAP_COLS[i] for i in fe.keep_]


def model_ready(df):
    X = pd.DataFrame(fe.transform(df[WAP_COLS].to_numpy()).astype(int), columns=kept)
    return pd.concat([X, df[META_COLS].reset_index(drop=True)], axis=1)


model_ready(train).to_csv(OUT / "train_model_ready.csv", index=False)
model_ready(test).to_csv(OUT / "test_model_ready.csv", index=False)

info = {**report, "not_heard_value": fe.floor_, "aps_kept": len(kept),
        "aps_dropped": sorted(set(WAP_COLS) - set(kept))}
(OUT / "processing_info.json").write_text(json.dumps(info, indent=2))
print(json.dumps({k: v for k, v in info.items() if k != "aps_dropped"}, indent=2))
for f in sorted(OUT.glob("*.csv")):
    print(f.name, round(f.stat().st_size / 1e6, 1), "MB")
