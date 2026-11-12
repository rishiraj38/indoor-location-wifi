# Ablation Studies for the Multitask Network

Every number here comes from `results/dl_ablations.csv`, evaluated with 5-fold cross-validation grouped by user on the training set. The comparison figure is saved in `results/figures/dl_ablations.png`.

One note on the baseline. The default settings (from `results/dl_best.json`) give 82.9% floor accuracy and 11.08 m here, while the same settings gave 83.1% and 11.17 m in `docs/dl_architecture.md`. The two runs were made on different computers. The smallest trunk, (256,), reproduces the earlier number to the last digit, and the three default rows inside this run are identical to each other, so the gap comes from floating-point differences between machines on the larger networks, not from different settings. It is well inside the tie band of 0.1 m and one percentage point.

## 1. Multitask vs Single-Task Networks

We compare one multitask network (a shared trunk that outputs building, floor and coordinates) against three single-task networks. Each single-task network has its own trunk with the same architecture and trains on only one loss.

| Setting | Building acc (%) | Floor acc (%) | Mean error (m) |
|---|---|---|---|
| Multitask (3 heads) | 96.5 | 82.9 | 11.08 |
| Three single-task networks | 98.3 | 83.2 | 11.50 |

Floor accuracy is almost identical (82.9% vs 83.2%, within noise). Mean position error is better with the multitask network (11.08 m vs 11.50 m). Building accuracy goes the other way: the single-task building network reaches 98.3% against 96.5% for the shared trunk, a loss of 1.8 points, which is outside the tie band.

This partly supports the idea behind multitask learning (Caruana 1997): training the three tasks together lets the trunk learn shared signal patterns that help position prediction, but the shared trunk gives up some building accuracy to do it. It also means we only need to train and store one network instead of three.

## 2. Loss Weights

The multitask loss combines two cross-entropy terms with coordinate mean squared error:
`loss = w_b * CE(building) + w_f * CE(floor) + w_xy * MSE(scaled coordinates)`.

We tested different weight combinations (w_b, w_f, w_xy) to see how balancing the targets affects results:

| Loss weights (b, f, xy) | Building acc (%) | Floor acc (%) | Mean error (m) |
|---|---|---|---|
| (1, 1, 1) | 98.5 | 81.7 | 13.35 |
| (1, 2, 1) | 98.0 | 82.2 | 14.15 |
| (1, 1, 3) | 96.9 | 82.0 | 12.77 |
| (1, 3, 3) | 96.6 | 83.5 | 14.50 |
| (0.5, 1, 5) | 96.3 | 84.2 | 11.25 |
| Default baseline (1, 1, 20) | 96.5 | 82.9 | 11.08 |

When the coordinate weight is low (1 or 3), position error is high (12.77 m to 14.50 m) because classification losses dominate the gradients. Giving coordinates higher weight (5 or 20) brings the position error down to around 11 m without hurting floor accuracy. The (0.5, 1, 5) setting has the best floor accuracy of the group (84.2%, 1.4 points above the default) at a cost of 0.16 m, so it would be a fair alternative; weight 20 was chosen earlier by the sweep in `results/dl_tuning.csv`, where weight 30 was a tie (12.48 m vs 12.52 m) and the simpler value was kept.

## 3. Trunk Size

We tested four trunk architectures to see how trunk depth and width affect performance.

| Trunk architecture | Building acc (%) | Floor acc (%) | Mean error (m) |
|---|---|---|---|
| (256,) | 95.6 | 82.6 | 15.89 |
| (512, 256) | 96.4 | 83.7 | 12.55 |
| (512, 256, 128) | 93.9 | 81.8 | 11.62 |
| (1024, 512, 256) | 96.5 | 82.9 | 11.08 |

A single hidden layer of 256 units is clearly too small, giving 15.89 m mean error. Widening the trunk to 1024, 512, 256 gives the best position error (11.08 m) and keeps floor accuracy high (82.9%). A narrower three-layer trunk (512, 256, 128) drops building accuracy to 93.9%.

## 4. Training Length

Our default model uses early stopping on a random 10% validation split of each training fold, with patience 30 and up to 300 epochs. For comparison we trained the same model with a fixed cap of 100 epochs (patience set to 10000 so the stopper never fires). Note that `MultitaskModel.fit` always restores the weights from the epoch with the lowest validation loss, so both rows keep their best epoch; the difference between them is only how long training is allowed to run.

| Setting | Building acc (%) | Floor acc (%) | Mean error (m) | Median error (m) |
|---|---|---|---|---|
| Default (patience 30, up to 300 epochs) | 96.5 | 82.9 | 11.08 | 7.82 |
| 100-epoch cap (best epoch kept) | 96.5 | 82.4 | 11.57 | 8.35 |

With the 100-epoch cap the mean position error is 0.48 m worse and the median 0.53 m worse; floor accuracy is 0.5 percentage points lower, which is a tie. Both settings use the same seed, so their first 100 epochs are identical, and the three default rows in this file are identical to each other, which shows the runs are deterministic. The default can therefore only score differently if some folds found their best validation epoch after epoch 100. (On the full training set the final network stopped at epoch 108, so its best epoch was 78, inside the cap; the folds vary.) The lesson is that the training budget must be long enough for early stopping to find the best epoch, not that a fixed length overfits. A run that truly skips early stopping (no restore of the best weights) would need a switch in `deep.py` and was not part of this study.

## 5. Overall Conclusions

The ablation studies confirm three things about the model:
First, the multitask network matches three separate networks on floor accuracy and beats them on position error, at the cost of 1.8 points of building accuracy, with one model to train instead of three.
Second, the coordinate loss needs a larger weight (5 or more) so the regression head gets enough gradient signal alongside the classification heads; with weight 1 the position error is 2.3 m worse.
Third, a wide trunk (1024, 512, 256) gives the lowest position error, and training must be allowed to run long enough (past 100 epochs on some folds) for early stopping to find the best epoch.
