# Input scaling and regularisation

## 1. Input scaling for neural networks vs Random Forest

A decision tree only compares a feature value with a threshold at each split. The scale of the feature does not change the ordering of values or where splits are placed, so scaling inputs does not change the splits or predictions of a Random Forest.

A neural network multiplies each input by a weight and sums the products across features. If inputs have different scales or wide ranges, weights on large inputs receive much larger gradient updates during backpropagation. This creates an ill-conditioned loss surface where gradient descent oscillates and learns slowly. Scaling inputs to a similar range keeps the gradients balanced and stabilizes training.

## 2. Scaling methods

Min-max scaling shifts and scales all inputs to [0, 1] using the single minimum and maximum of the training matrix (-105 and 0), so the not-heard value maps exactly to 0.

Standard scaling subtracts the training mean and divides by the training standard deviation for each column to give zero mean and unit variance.

Powed scaling maps values between the training minimum and maximum to [0, 1], clips to the unit interval, and raises the result to the power math.e to push weak signals toward zero.

## 3. Dropout

During training, dropout randomly turns off a chosen fraction of hidden units on each forward pass. This stops units from relying on each other to fix errors, forcing the network to learn multiple independent paths for the same information and reducing overfitting.

In our network dropout is the `nn.Dropout` layer after every trunk layer in `src/indoorloc/deep.py`, and it is only active during training. We tried dropout rates of 0.0, 0.3 and 0.5.

## 4. Weight decay

Weight decay adds a penalty proportional to the squared magnitude of each weight to the loss function. During training, this penalty pulls weights toward zero, which stops any single connection from becoming too large and smooths the decision boundaries.

In our network weight decay is the `weight_decay` argument of the Adam optimiser in `MultitaskModel`. We tried 0 and 1e-4.

## 5. Results

The table below shows 5-fold cross-validation grouped by user, from `results/dl_regularisation.csv`. Every run starts from the settings in `results/dl_best.json` (trunk 1024-512-256, loss weights 1/1/20, patience 30) and changes only the scaler, the dropout rate and the weight decay. Each run took 2.3 to 5.8 minutes on one laptop CPU; the 18 runs add up to 80 minutes.

| Scaler | Dropout | Weight decay | Floor accuracy (%) | Mean error (m) | Median error (m) |
|---|---|---|---|---|---|
| minmax | 0.0 | 0 | 82.5 | 12.40 | 8.98 |
| minmax | 0.0 | 1e-4 | 81.3 | 12.18 | 8.92 |
| minmax | 0.3 | 0 | 82.1 | 11.76 | 7.81 |
| minmax | 0.3 | 1e-4 | 83.1 | 11.17 | 7.87 |
| minmax | 0.5 | 0 | 81.4 | 11.77 | 7.72 |
| minmax | 0.5 | 1e-4 | 83.0 | 11.17 | 7.88 |
| standard | 0.0 | 0 | 73.9 | 21.79 | 12.22 |
| standard | 0.0 | 1e-4 | 75.9 | 20.54 | 11.71 |
| standard | 0.3 | 0 | 75.1 | 22.91 | 14.42 |
| standard | 0.3 | 1e-4 | 74.5 | 21.46 | 11.18 |
| standard | 0.5 | 0 | 73.7 | 24.28 | 15.58 |
| standard | 0.5 | 1e-4 | 72.9 | 21.46 | 13.05 |
| powed | 0.0 | 0 | 78.0 | 19.39 | 10.59 |
| powed | 0.0 | 1e-4 | 78.8 | 15.68 | 10.38 |
| powed | 0.3 | 0 | 82.6 | 15.10 | 8.68 |
| powed | 0.3 | 1e-4 | 81.0 | 14.12 | 9.04 |
| powed | 0.5 | 0 | 83.1 | 13.82 | 8.57 |
| powed | 0.5 | 1e-4 | 80.2 | 14.01 | 9.31 |

The best row is min-max scaling with dropout 0.3 and weight decay 1e-4: 83.1% floor accuracy and 11.17 m mean error. That is the current default, so the default survives. The row with dropout 0.5 and weight decay 1e-4 is a tie (83.0% and 11.17 m); under the tie-break rule from `docs/dl_architecture.md` (differences under about 0.1 m or one percentage point are a tie, keep the simpler or earlier setting) the earlier setting, dropout 0.3, is kept.

What the table says, in order of effect size:

1. The scaler matters most. Standard scaling is far worse than min-max in every setting (20.5 to 24.3 m against 11.2 to 12.4 m, and floor accuracy of 72.9 to 75.9% against 81.3 to 83.1%). Standardising each router column divides by that column's standard deviation, and most columns are nearly constant at the not-heard value, so the few scans where a router is heard become very large inputs. Min-max keeps every input between 0 and 1 and the not-heard value exactly at 0.
2. Powed scaling sits in between (13.8 to 19.4 m). It did not help the Random Forest in Phase 1 either (14.01 m against 13.62 m for raw signals in `results/ablation_features.csv`), and it is worse again here. Raising the scaled signal to the power e pushes weak signals almost to 0, and both models seem to get something out of those weak signals. The best powed row (dropout 0.5, no weight decay) reaches the same floor accuracy as the default, 83.1%, but 2.65 m more position error.
3. Dropout and weight decay help together. With min-max, no regularisation gives 12.40 m; dropout alone gives 11.76 m; weight decay alone 12.18 m; both together 11.17 m. Floor accuracy does not follow the same pattern: each regulariser on its own lowers it slightly (82.5% with neither, 82.1% with dropout alone, 81.3% with weight decay alone) and only the combination raises it, to 83.1%.
4. With min-max, dropout 0.3 and 0.5 are indistinguishable once weight decay is on (11.17 m both), so the network is not sensitive to the exact rate there. With powed the rate still matters (15.10 m against 13.82 m without weight decay).

## 6. Early stopping

Our model stops on a random 10% of the training scans with patience 30. The training-length ablation in docs/dl_ablations.md (Aditya's) already shows what happens with a shorter budget, so patience was not re-tuned here.

## 7. Grid reduction

The grid is 18 combinations (3 scalers, 3 dropout rates, 2 weight decays), about 80 minutes of training. The planned grid had dropout 0.2 and weight decay 1e-5 and 1e-3 as well, which would have doubled the time; they were dropped to fit the schedule. Dropout 0.2 sits between two values that tie with min-max, so little is lost there. Weight decay 1e-3 is a genuinely different setting that remains untested.
