# The Network: Design Choices and What We Tried

Owner: Rishi Raj. Every number here comes from `results/dl_tuning.csv` and `results/dl_architecture.csv`,
both produced with 5-fold cross-validation grouped by user, the same protocol as Phase 1.

## 1. The design

One network for all three targets.

- Input: the 465 signal values after the Phase 1 preprocessing (+100 replaced by -105, never-heard access
  points dropped), scaled to 0 to 1 with the training minimum and maximum.
- Trunk: fully connected layers with ReLU and dropout, shared by all three outputs.
- Heads: building (3 outputs, softmax), floor (5 outputs, softmax), coordinates (2 outputs, linear).
- Loss: `cross-entropy(building) + cross-entropy(floor) + w * loss(coordinates)`. Coordinates are centred
  and scaled with the training mean and standard deviation before the loss, because raw values are around
  4,864,000 and would swamp everything else.
- Training: Adam, learning rate 0.001, weight decay 0.0001, batches of 256, early stopping on a validation
  part of the training data, the best weights restored.

Why a shared trunk: the Phase 1 models learned building, floor and position separately. The brief asks
for a shared trunk so the three tasks can share what they learn (Caruana 1997). The gap we saw in Phase 1
was floor accuracy, and floor depends on the same signal patterns as position.

## 2. Where to stop training: a problem we hit

Our first version held out whole users for the early-stopping check, to match the outer cross-validation.
That broke training. With only one or two users held out, the validation loss was at its minimum after the
first epoch and never improved, so early stopping restored the network to barely trained weights. On one
fold this gave 35% floor accuracy and 131 m position error, and running 60 or 150 epochs changed nothing,
because the restored weights were always from epoch 1.

Switching the early-stopping check to a random 10% of the training scans fixed it: the same fold gave 95%
floor accuracy. The validation set is still inside the training fold, so nothing leaks from the outer
validation users or from the test set. The lesson: an early-stopping signal needs to be representative of
the training data; one or two people's scans are not.

## 3. Loss balance and patience

With equal weights the two cross-entropy terms dominate and the coordinate head is under-trained. We swept
the coordinate weight and the patience:

| Coordinate weight | Patience | Building | Floor | Mean error (m) |
|---|---|---|---|---|
| 1 | 15 | 97.1% | 83.0% | 14.80 |
| 1 | 30 | 97.3% | 83.6% | 13.83 |
| 3 | 15 | 96.3% | 84.2% | 13.12 |
| 3 | 30 | 95.4% | 83.4% | 13.21 |
| 5 | 15 | 95.7% | 83.7% | 12.91 |
| 5 | 30 | 96.7% | 82.9% | 12.60 |
| 10 | 15 | 98.2% | 83.1% | 12.71 |
| 10 | 30 | 96.3% | 82.4% | 12.78 |
| 20 | 15 | 96.0% | 83.7% | 12.72 |
| 20 | 30 | 96.5% | 83.5% | 12.52 |
| 30 | 30 | 96.2% | 83.9% | 12.48 |
| 50 | 30 | 98.2% | 83.5% | 12.79 |

Floor accuracy is flat across the table, and every row is well above the Phase 1 cross-validation numbers
(k-NN 74.1%, per-building Random Forest 71.3%; the Phase 1 tuning rows in `results/tuning_cv.csv` come from
the per-building model, see `scripts/run_experiments.py`). Position error improves as the coordinate weight
rises and levels off from weight 5 upward; weights 30 and 50 were added afterwards to confirm that 20 is not
just the edge of the grid (30 is the same within noise, 50 is worse). We use weight 20 and patience 30:
12.52 m, which sits between k-NN (11.78 m) and the forest (13.55 m) in cross-validation.

## 4. Architecture variants

Starting from that setting, each variant changes one thing.

| Variant | Building | Floor | Mean error (m) |
|---|---|---|---|
| Baseline: 512-256, MSE | 96.5% | 83.5% | 12.52 |
| Batch normalisation | 97.1% | 82.7% | 12.64 |
| Huber coordinate loss (delta 0.1) | 96.3% | 83.5% | 12.33 |
| Wider trunk: 1024-512-256 | 96.1% | 83.1% | 11.17 |
| Wider trunk + Huber | 95.6% | 82.2% | 11.08 |
| Smaller trunk: 256 | 95.6% | 82.6% | 15.89 |
| More dropout: 0.5 | 96.8% | 83.9% | 13.19 |

The wider trunk is the change that matters: 12.52 m to 11.17 m at no cost on floor. A single hidden layer of
256 units is clearly too small. Batch normalisation makes no difference and more dropout hurts position.

The Huber loss needs a word of care. Our first run used PyTorch's default threshold, which in our scaled
coordinate units only switches to the linear part for errors above about 124 m, so it was MSE in disguise
(12.72 m, the same as MSE at that weight). With the threshold set to 0.1 scaled units (about 12 m by 7 m) it is
a real Huber loss and does slightly better than MSE at the baseline width (12.33 m vs 12.52 m). With the wide
trunk the two are a tie: 11.08 m vs 11.17 m on position, 82.2% vs 83.1% on floor. We keep MSE as the simpler
choice. With the wider trunk the network beats both Phase 1 models in cross-validation on position (k-NN
11.78 m, forest 13.55 m) as well as on floor.

## 5. Final settings

Recorded in `results/dl_best.json` and used by every later script.

| Setting | Value |
|---|---|
| Trunk | 1024, 512, 256 units, ReLU |
| Dropout | 0.3 |
| Weight decay | 0.0001 |
| Loss weights (building, floor, coordinates) | 1, 1, 20 |
| Coordinate loss | mean squared error on scaled coordinates |
| Early stopping | random 10% of training scans, patience 30, up to 300 epochs |
| Input scaling | min-max to 0 to 1 |
| Optimiser | Adam, learning rate 0.001, batch 256 |

Yash Yadav's regularisation sweep and Aditya's ablations start from these settings and change one thing at a time.
