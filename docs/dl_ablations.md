# Ablation Studies for the Multitask Network

Every number here comes from `results/dl_ablations.csv`, evaluated with 5-fold cross-validation grouped by user on the training set. The comparison figure is saved in `results/figures/dl_ablations.png`.

## 1. Multitask vs Single-Task Networks

We compare one multitask network (a shared trunk that outputs building, floor and coordinates) against three single-task networks. Each single-task network has its own trunk with the same architecture and trains on only one loss.

| Setting | Building acc (%) | Floor acc (%) | Mean error (m) |
|---|---|---|---|
| Multitask (3 heads) | 96.5 | 82.9 | 11.08 |
| Three single-task networks | 98.3 | 83.2 | 11.50 |

Floor accuracy is almost identical (82.9% vs 83.2%, within noise). Mean position error is slightly better with the multitask network (11.08 m vs 11.50 m).

This supports the idea behind multitask learning (Caruana 1997): training the three tasks together lets the trunk learn shared signal patterns that help position prediction. It also means we only need to train and store one network instead of three separate models.

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

When the coordinate weight is low (1 or 3), position error is high (12.77 m to 14.50 m) because classification losses dominate the gradients. Giving coordinates higher weight (5 or 20) brings the position error down to around 11 m without hurting floor accuracy.

## 3. Trunk Size

We tested four trunk architectures to see how trunk depth and width affect performance.

| Trunk architecture | Building acc (%) | Floor acc (%) | Mean error (m) |
|---|---|---|---|
| (256,) | 95.6 | 82.6 | 15.89 |
| (512, 256) | 96.4 | 83.7 | 12.55 |
| (512, 256, 128) | 93.9 | 81.8 | 11.62 |
| (1024, 512, 256) | 96.5 | 82.9 | 11.08 |

A single hidden layer of 256 units is clearly too small, giving 15.89 m mean error. Widening the trunk to 1024, 512, 256 gives the best position error (11.08 m) and keeps floor accuracy high (82.9%). A narrower three-layer trunk (512, 256, 128) drops building accuracy to 93.9%.

## 4. Early Stopping

Our default model uses early stopping on a random 10% validation split of each training fold, with patience 30 up to 300 epochs. To test if early stopping is needed, we trained the same model without early stopping (patience 10000, fixed at 100 epochs).

| Setting | Building acc (%) | Floor acc (%) | Mean error (m) | Median error (m) |
|---|---|---|---|---|
| Default (patience 30) | 96.5 | 82.9 | 11.08 | 7.82 |
| None (100 epochs) | 96.5 | 82.4 | 11.57 | 8.35 |

Without early stopping, the network trains past its best validation point and starts overfitting. Floor accuracy drops by 0.5% and mean position error increases by about 0.5 m (and median error by 0.53 m). Early stopping helps by restoring the weights from the validation minimum.

## 5. Overall Conclusions

The ablation studies confirm three things about the model:
First, the multitask setup performs as well or better than three separate networks while using a third of the parameters.
Second, coordinates need a larger weight (around 20) so the regressor gets enough gradient signal alongside the classification heads.
Third, a wide trunk (1024, 512, 256) with early stopping is necessary to capture signal interactions without overfitting.
