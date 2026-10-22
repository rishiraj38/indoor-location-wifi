# Data Dictionary

Source: UJIIndoorLoc, UCI Machine Learning Repository (dataset 310), collected at Universitat Jaume I in 2013.

## Files

| File | Rows | Columns | Description |
|---|---|---|---|
| `data/raw/UJIndoorLoc/trainingData.csv` | 19,937 | 529 | Original training file, unchanged |
| `data/raw/UJIndoorLoc/validationData.csv` | 1,111 | 529 | Original validation file, used as our test set, unchanged |
| `data/processed/train_cleaned.csv` | 19,227 | 529 | Training file after removing duplicates and empty scans |
| `data/processed/train_model_ready.csv` | 19,227 | 474 | Cleaned training data with +100 replaced by -105 and 55 never-heard APs dropped |
| `data/processed/test_model_ready.csv` | 1,111 | 474 | Test data with the same transformation (learned on training data) |
| `data/processed/processing_info.json` | | | Row counts, the value used for "not heard" and the list of dropped APs |

## Columns

| Column | Type | Range | Meaning |
|---|---|---|---|
| `WAP001` ... `WAP520` | integer | -104 to 0, or +100 | Signal strength (RSSI) in dBm from each Wi-Fi access point. Closer to 0 is stronger. **+100 = not detected** |
| `LONGITUDE` | float | -7696 to -7300 | Position in metres (UTM projection), east-west |
| `LATITUDE` | float | 4864745 to 4865017 | Position in metres (UTM projection), north-south |
| `FLOOR` | integer | 0 to 4 | Floor number |
| `BUILDINGID` | integer | 0, 1, 2 | Building number |
| `SPACEID` | integer | 1 to 254 | Room or office id (0 in the test file) |
| `RELATIVEPOSITION` | integer | 1, 2 | 1 = inside the room, 2 = outside, in the corridor (0 in the test file) |
| `USERID` | integer | 1 to 18 | Person who took the scan (0 in the test file) |
| `PHONEID` | integer | 0 to 24 | Phone model used |
| `TIMESTAMP` | integer | | Unix time of the scan |

In the model-ready files the `WAP` columns run from -105 to 0. -105 means not detected.

## Targets

| Target | Problem type |
|---|---|
| `BUILDINGID` | classification (3 classes) |
| `FLOOR` | classification (up to 5 classes) |
| `LONGITUDE`, `LATITUDE` | regression (metres) |

`SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID` and `TIMESTAMP` are not used as model inputs. `USERID` is used to split
cross-validation folds, and `PHONEID` is used for the error-by-phone analysis.
