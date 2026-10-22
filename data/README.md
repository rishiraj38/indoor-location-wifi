# Data

The raw data is not committed. Download it with:

```bash
python scripts/download_data.py
```

This fetches **UJIIndoorLoc** (UCI ML Repository, dataset 310) into `data/raw/UJIndoorLoc/`:

| File | Rows | Used as |
|---|---|---|
| `trainingData.csv` | 19,937 | training set (cleaned: duplicates and empty scans removed) |
| `validationData.csv` | 1,111 | held-out test set (left exactly as published) |

Columns: `WAP001`–`WAP520` are RSSI values in dBm (`+100` means the access point was not detected),
followed by `LONGITUDE`, `LATITUDE` (metres, UTM), `FLOOR`, `BUILDINGID`, `SPACEID`,
`RELATIVEPOSITION`, `USERID`, `PHONEID`, `TIMESTAMP`.

Source: Torres-Sospedra et al., *UJIIndoorLoc: A new multi-building and multi-floor database for
WLAN fingerprint-based indoor localization problems*, IPIN 2014.
https://archive.ics.uci.edu/dataset/310/ujiindoorloc
