# Cleaning and Preprocessing Log

Produced by `scripts/make_processed_data.py`. The numbers are also saved in `data/processed/processing_info.json`.

## Step 1: training set cleaning (`train_cleaned.csv`)

| Step | Rows removed | Rows left | Reason |
|---|---|---|---|
| Original training file | | 19,937 | |
| Remove exact duplicate rows | 637 | 19,300 | Copies add no information and over-weight those spots |
| Remove scans where no AP was heard | 73 | 19,227 | All 520 values are +100, so there is nothing to learn from |

The test file (`validationData.csv`) is not cleaned, so results can be compared with published work.

Kept on purpose: 492 scans that contain a reading above -30 dBm (430 of them from phone 19). Such readings are
unrealistic indoors, but removing them did not change the result (mean error 9.46 m without them vs 9.37 m with
them), so all of them stay in.

## Step 2: model-ready data (`train_model_ready.csv`, `test_model_ready.csv`)

| Step | Detail | Reason |
|---|---|---|
| Replace +100 with -105 | -105 = weakest real reading in training (-104) minus 1 | +100 means "not heard", i.e. too weak; as a number it would look like the strongest signal |
| Drop 55 APs | APs never heard in any training scan | They are -105 in every training row and carry no information |
| Result | 465 AP columns + 9 information columns | |

Both values (-105 and the list of kept APs) are learned from the training data only and then applied unchanged to
the test data. This avoids leakage from the test set.

## Checks

- The model-ready files contain no +100 values. All AP values are between -105 and 0.
- Row counts: train 19,227, test 1,111.
- All 55 dropped APs are heard in at least one test scan, even though no training scan hears them. They
  were probably added or switched on in the four months between the two collections. They are still dropped,
  because a model cannot learn anything about an AP it never saw in training.
