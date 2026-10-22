# Data

```
data/
├── raw/UJIndoorLoc/
│   ├── trainingData.csv       original training file (19,937 scans)
│   └── validationData.csv     original validation file (1,111 scans), our test set
└── processed/
    ├── train_cleaned.csv      duplicates and empty scans removed (19,227 scans)
    ├── train_model_ready.csv  +100 replaced by -105, 55 never-heard APs dropped
    ├── test_model_ready.csv   test file with the same transformation
    └── processing_info.json   row counts and the list of dropped APs
```

- Column meanings: [`docs/data_dictionary.md`](../docs/data_dictionary.md)
- What was removed or changed and why: [`docs/cleaning_log.md`](../docs/cleaning_log.md)
- To rebuild the processed files: `python scripts/make_processed_data.py`
- To download the raw files again: `python scripts/download_data.py`

Source: UJIIndoorLoc, UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/310/ujiindoorloc
(Torres-Sospedra et al., IPIN 2014).
