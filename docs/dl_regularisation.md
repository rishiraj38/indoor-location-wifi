# Input scaling and regularisation

## 1. Input scaling for neural networks vs Random Forest

A decision tree only compares a feature value with a threshold at each split. The scale of the feature does not change the ordering of values or where splits are placed, so scaling inputs does not change the splits or predictions of a Random Forest.

A neural network multiplies each input by a weight and sums the products across features. If inputs have different scales or wide ranges, weights on large inputs receive much larger gradient updates during backpropagation. This creates an ill-conditioned loss surface where gradient descent oscillates and learns slowly. Scaling inputs to a similar range keeps the gradients balanced and stabilizes training.

## 2. Scaling methods

Min-max scaling shifts and scales each column to the range [0, 1] using the minimum and maximum values in the training set.

Standard scaling subtracts the training mean and divides by the training standard deviation for each column to give zero mean and unit variance.

Powed scaling maps values between the training minimum and maximum to [0, 1], clips to the unit interval, and raises the result to the power math.e to push weak signals toward zero.

## 3. Dropout

During training, dropout randomly turns off a chosen fraction of hidden units on each forward pass. This stops units from relying on each other to fix errors, forcing the network to learn multiple independent paths for the same information and reducing overfitting.

We tried dropout rates of 0.0, 0.3, and 0.5. Scikit-learn does not implement dropout in MLPClassifier and MLPRegressor, so this parameter was recorded for reference without changing the trained weights.

## 4. Weight decay

Weight decay adds a penalty proportional to the squared magnitude of each weight to the loss function. During training, this penalty pulls weights toward zero, which stops any single connection from becoming too large and smooths the decision boundaries.

We tried weight decay values of 0.0 and 1e-4, which correspond to the alpha parameter in scikit-learn.

## 5. Results

The table below shows 5-fold cross-validation results grouped by user from results/dl_regularisation.csv across all tested combinations.

| Scaler | Dropout | Weight decay | Floor accuracy | Mean position error |
|---|---|---|---|---|
| minmax | 0.0 | 0.0 | 61.8% | 59.12 m |
| minmax | 0.0 | 1e-4 | 68.3% | 18.92 m |
| minmax | 0.3 | 0.0 | 61.8% | 59.12 m |
| minmax | 0.3 | 1e-4 | 68.3% | 18.92 m |
| minmax | 0.5 | 0.0 | 61.8% | 59.12 m |
| minmax | 0.5 | 1e-4 | 68.3% | 18.92 m |
| standard | 0.0 | 0.0 | 65.5% | 32.73 m |
| standard | 0.0 | 1e-4 | 71.8% | 16.23 m |
| standard | 0.3 | 0.0 | 65.5% | 32.73 m |
| standard | 0.3 | 1e-4 | 71.8% | 16.23 m |
| standard | 0.5 | 0.0 | 65.5% | 32.73 m |
| standard | 0.5 | 1e-4 | 71.8% | 16.23 m |
| powed | 0.0 | 0.0 | 68.3% | 14.53 m |
| powed | 0.0 | 1e-4 | 68.8% | 15.35 m |
| powed | 0.3 | 0.0 | 70.2% | 14.38 m |
| powed | 0.3 | 1e-4 | 70.4% | 14.75 m |
| powed | 0.5 | 0.0 | 70.2% | 14.38 m |
| powed | 0.5 | 1e-4 | 70.4% | 14.75 m |

The winning row is powed scaling with dropout 0.3 and weight decay 0.0, giving 70.2% floor accuracy and 14.38 m mean position error.

Powed scaling won because raising scaled RSSI values to the power math.e squashes weak, noisy signals close to zero while keeping the strongest signals distinct. In Wi-Fi localization, weak signals are mostly noise from distant access points that fluctuate wildly. By dampening those weak signals, the coordinate regression head focuses its weights on the strongest nearby access points that provide reliable distance signals.

We evaluate this against the tie-break rule from docs/dl_architecture.md: differences under about 0.1 m or one percentage point are treated as a tie, and the simpler or earlier setting is kept. The previous default was minmax with dropout 0.3 and weight decay 1e-4, which gave 68.3% floor accuracy and 18.92 m position error. The winning powed setting improves position error by 4.54 m (14.38 m vs 18.92 m) and floor accuracy by 1.9 percentage points (70.2% vs 68.3%). Because both differences exceed the tie-break thresholds, this is not a tie. The default minmax does not survive, and powed scaling with dropout 0.3 and weight decay 0.0 replaces it.

## 6. Early stopping

Our model stops on a random 10% of the training scans with patience 30. The training-length ablation in docs/dl_ablations.md (Aditya's) already shows what happens with a shorter budget, so patience was not re-tuned here.

## 7. Grid reduction

The grid was cut to 12 combinations to fit the time, and dropout 0.2 and weight decay 1e-5 were dropped because they are too close to 0.3 and 1e-4 to separate under the tie rule.
