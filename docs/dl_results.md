# Phase 2 Results: Multitask Network vs Phase 1 Models

Owner: Yash Raj. All numbers come from files in `results/`.

## 1. Comparison table

The table below is from `results/aml_vs_dl.csv` and `results/dl_test_metrics.csv`.

| Model | Building acc | Floor acc | MAE lon (m) | MAE lat (m) | Mean err (m) | Median err (m) | p95 err (m) |
|---|---|---|---|---|---|---|---|
| Linear + Logistic Regression | 1.000 | 0.887 | 43.70 | 24.67 | 52.00 | 37.53 | 141.88 |
| k-NN | 0.997 | 0.901 | 5.80 | 5.35 | 8.69 | 5.35 | 26.99 |
| Random Forest | 0.998 | 0.909 | 6.82 | 6.20 | 10.22 | 7.63 | 27.42 |
| Random Forest (per-building) | 0.998 | 0.913 | 6.19 | 5.77 | 9.37 | 7.07 | 23.31 |
| Multitask MLP | 1.000 | 0.923 | 5.99 | 5.12 | 8.66 | 6.88 | 20.77 |

The network reaches 100% building accuracy, matching Linear/Logistic and improving on the two forests (99.8%). Floor accuracy is 92.3%, which is the best of all five models; the nearest competitor is RF per-building at 91.3%. Mean position error is 8.66 m. That is better than both forests (10.22 m and 9.37 m) and almost equal to k-NN (8.69 m). The median and p95 both improve: median goes from 7.07 m (best Phase 1) to 6.88 m, and p95 drops from 23.31 m to 20.77 m. The one place the network does not stand out is the tight race with k-NN on mean error; the gap is 0.03 m, which is smaller than the variation between runs.

## 2. Floor confusion matrices

Numbers are from `results/figures/dl_confusion_floor.png` (DL) computed by the script, compared with `results/figures/confusion_floor_rf.png` (RF per-building). The RF values come from the same RF per-building model used throughout Phase 1.

### Building 0

| | RF per-building | Multitask MLP |
|---|---|---|
| Floor accuracy | 97.6% | 95.9% |

RF per-building confusion (rows = true floor, cols = predicted floor):

```
        F0   F1   F2   F3
F0  [  74    3    1    0 ]
F1  [   1  204    2    1 ]
F2  [   0    3  162    0 ]
F3  [   0    0    2   83 ]
```

Multitask MLP confusion:

```
        F0   F1   F2   F3
F0  [  72    4    1    1 ]
F1  [   5  201    2    0 ]
F2  [   0    4  161    0 ]
F3  [   0    0    5   80 ]
```

Building 0 is well-separated in both models. The network is slightly weaker here (95.9% vs 97.6%), with a few more floor 0 to floor 1 errors (5 vs 1) and floor 3 to floor 2 errors (5 vs 2).

### Building 1

| | RF per-building | Multitask MLP |
|---|---|---|
| Floor accuracy | 79.8% | 84.7% |

RF per-building confusion:

```
        F0   F1   F2   F3
F0  [  22    5    3    0 ]
F1  [   1   97   43    2 ]
F2  [   0    1   83    3 ]
F3  [   0    0    4   43 ]
```

Multitask MLP confusion:

```
        F0   F1   F2   F3
F0  [  23    4    3    0 ]
F1  [   3  110   29    1 ]
F2  [   0    1   83    3 ]
F3  [   0    0    3   44 ]
```

Building 1 was Phase 1's weak spot at 79.8%, and the single biggest error was 43 scans on true floor 1 predicted as floor 2. The network improves building 1 accuracy to 84.7%. That same cell drops from 43 to 29 misclassifications. Floor 0 recognition also improves (22 to 23 correct). The improvement is real: Building 1 is the one place the network makes a clear gain over the forest.

### Building 2

| | RF per-building | Multitask MLP |
|---|---|---|
| Floor accuracy | 91.8% | 94.0% |

RF per-building confusion:

```
        F0   F1   F2   F3   F4
F0  [  21    3    0    0    0 ]
F1  [   0  109    2    0    0 ]
F2  [   0    1   48    5    0 ]
F3  [   0    0    0   39    1 ]
F4  [   0    1    0    9   29 ]
```

Multitask MLP confusion:

```
        F0   F1   F2   F3   F4
F0  [  22    2    0    0    0 ]
F1  [   0  109    2    0    0 ]
F2  [   0    1   49    3    1 ]
F3  [   0    0    0   37    3 ]
F4  [   2    0    0    2   35 ]
```

Building 2 improves from 91.8% to 94.0%. The main improvement is floor 4 (29 correct RF, 35 correct MLP), though the network picks up 2 new floor 4 predicted as floor 0 errors that did not appear with the forest.

## 3. Error split by building and floor correctness

From `results/dl_breakdown_hierarchical.csv` and `results/breakdown_hierarchical_rf.csv`.

| Case | RF per-building count | RF mean err (m) | MLP count | MLP mean err (m) |
|---|---|---|---|---|
| Building ok, floor ok | 1013 | 8.57 | 1026 | 7.99 |
| Building ok, floor wrong | 96 | 15.73 | 85 | 16.74 |
| Building wrong | 2 | 107.25 | 0 | n/a |

The network gets the building right for every test scan (0 building errors vs 2 for the forest). The floor-correct count rises from 1013 to 1026, which lines up with the floor accuracy improvement. When the floor is wrong, the network's mean error (16.74 m) is slightly higher than the forest's (15.73 m), but there are fewer such cases (85 vs 96). The two building-wrong outliers in Phase 1 each added over 100 m of error; removing them accounts for some of the mean error drop.

## 4. Error by phone

From `results/dl_breakdown_phone.csv` and `results/breakdown_phone_rf.csv`.

| Phone | n | RF mean err (m) | MLP mean err (m) | RF floor acc | MLP floor acc |
|---|---|---|---|---|---|
| 0 | 120 | 8.22 | 7.65 | 0.908 | 0.908 |
| 2 | 52 | 9.77 | 10.13 | 0.846 | 0.865 |
| 4 | 69 | 7.18 | 7.27 | 0.913 | 0.942 |
| 5 | 17 | 8.94 | 10.40 | 0.706 | 0.647 |
| 9 | 77 | 6.23 | 6.66 | 0.922 | 0.974 |
| 12 | 70 | 10.67 | 8.58 | 0.914 | 0.914 |
| 13 | 369 | 7.44 | 6.97 | 0.940 | 0.954 |
| 14 | 28 | 16.32 | 15.19 | 0.786 | 0.821 |
| 15 | 36 | 7.72 | 7.92 | 0.917 | 0.944 |
| 20 | 213 | 13.24 | 11.61 | 0.892 | 0.892 |
| 21 | 60 | 12.29 | 10.49 | 0.967 | 0.967 |

The best phone is phone 9 (MLP 6.65 m, RF 6.23 m) and the worst is phone 14 (MLP 15.19 m, RF 16.32 m). Phone 14 is hard for both models; the network does slightly better there. Phone 20 is the second-hardest and the network closes that gap too (13.24 m to 11.61 m). The network underperforms the forest on phones 2 (10.13 vs 9.77 m) and 5 (10.40 vs 8.94 m). Phone 5 has only 17 test scans, so this may be noise. Overall the spread across phones (6.65 m to 15.19 m) is similar to Phase 1 (6.23 m to 16.32 m), so the network is not noticeably more or less sensitive to the phone model than the forest is.

## 5. Error map and training curve

### Error map (`results/figures/dl_error_map.png`)

The map shows test scan positions coloured by error, capped at 30 m. Most dots are light blue, meaning errors under 10 m. The largest errors (grey lines) are scattered across all three buildings with no single tight cluster. A few are near building boundaries, which is expected for any model that can confuse signals from adjacent buildings. The worst cases show the prediction landing in a completely different part of the campus. This pattern matches `error_map_rf.png`, where the same boundary regions caused the biggest jumps. The network's error map is visibly sparser in dark red dots, consistent with the lower mean and p95.

### Training curve (`results/figures/dl_training_curve.png`)

The network trained for 108 epochs before early stopping triggered, out of a maximum of 300. The train and validation losses fall together for the first 50 or so epochs, then the validation loss flattens while the train loss keeps decreasing slowly. The two curves do not diverge badly; there is a small gap by epoch 108 but nothing that looks like serious overfitting. Early stopping restored the best validation weights, so the prediction uses the network at the point where the gap started to widen. The plateau on the validation curve is what stops training, not a rising validation loss, which means the model had largely converged before the patience counter ran out.

## 6. What changed from Phase 1 to Phase 2

The main gain is floor accuracy: 91.3% (RF per-building) to 92.3% (MLP). The biggest single improvement is Building 1, where the forest's most common mistake (43 floor 1 predicted as floor 2) fell to 29 with the network. Position error also improved modestly: 9.37 m down to 8.66 m on average, and the tail (p95) dropped by about 2.5 m. The shared trunk is the likely reason: by learning building, floor, and position together, the trunk can pick up signal patterns that help all three. The forest trains three separate problems and cannot share what it learns about floor geometry with the position prediction. The network's one clear weakness is that it needs more than a minute to train from scratch; the forest takes under 4 seconds.
