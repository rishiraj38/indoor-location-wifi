# Theory Notes

## 1. The problem

A Wi-Fi scan is a list of 520 numbers, one for each access point (AP). Each number is the received
signal strength (RSSI) in dBm. From one scan we predict three things:

| Target | Type of problem | Metric |
|---|---|---|
| Building (0, 1, 2) | classification | accuracy |
| Floor (0 to 4) | classification | accuracy |
| Position (x, y) in metres | regression | mean distance error in metres |

## 2. The "+100" problem (missing values)

RSSI values are negative. −30 dBm is a very strong signal and −100 dBm is barely heard. The dataset
uses +100 when an AP was not heard at all. If we use +100 as a normal number, it looks stronger
than any real signal, which is the opposite of what it means. "Not heard" really means "too weak to
detect". So we replace it with −105 dBm, which is just below the weakest real reading (−104). We
work out this value from the training data only and then use the same value on the test data.

## 3. Linear and Logistic Regression (baseline)

* Linear regression predicts x (and, in a separate model, y) as a weighted sum of all AP signals:
  `x̂ = w₀ + w₁·RSSI₁ + … + w₄₆₅·RSSI₄₆₅`. The weights are picked to make the squared error as small
  as possible.
* Logistic regression predicts building and floor. For each class it computes a weighted sum. Softmax
  then turns these sums into probabilities. The model is trained with cross-entropy loss.
* We standardise the inputs (mean 0, std 1, using training statistics) so that all APs are on the
  same scale.

Why does linear regression do badly on position? Signal strength falls off with the logarithm of
distance, and walls block it. So position is not a straight-line function of RSSI. A linear model can
only fit one flat surface through 465 dimensions. Logistic regression does well on building and
floor, because you can tell the classes apart by which APs are strong, and a weighted sum picks
that up well.

## 4. k-Nearest Neighbours (k-NN)

This is the classic fingerprinting method. There is no training step. We just store every training
scan. For a new scan:
1. Compute the Euclidean distance to every stored scan.
2. Take the k closest ones.
3. Building and floor come from their vote (weighted by distance). Position is their average
   (also weighted by distance).

* With a small k the predictions are noisy. With a large k we average over points that are far away,
  which blurs the position. We pick k with cross-validation.
* Weak points: every AP counts the same in the distance, and prediction is slow because each new scan
  is compared with all 19k training scans.

## 5. Decision trees

A tree asks yes/no questions such as "is WAP117 stronger than −80 dBm?". At each node it tries many
(AP, threshold) pairs. It keeps the one that makes the two child groups the most pure:
* classification: Gini impurity `G = 1 − Σ pₖ²` (0 means everything is one class)
* regression: the variance of the targets (the split that lowers it the most is chosen)

A leaf predicts the most common class (classification) or the mean position of its training scans
(regression). A single deep tree memorises the training data. This means it overfits (high variance).

## 6. Random Forest

A random forest is many trees (300 in our case). Each tree is made different in two ways:
1. Bootstrap sample: each tree trains on a random sample of scans drawn with replacement.
2. Random features: at each split, the tree only looks at a random subset of APs (`max_features`,
   for example √465, which is about 21).

To predict, the forest takes a majority vote (building, floor) or an average (position). Each tree
overfits in its own way, so when we average many of them their mistakes partly cancel out. The
variance goes down and the bias stays low. In formula form: if we have B trees, each with variance σ²
and correlation ρ between them, their average has variance `ρσ² + (1−ρ)σ²/B`. Adding more trees makes
the second term smaller, and using random features makes ρ smaller.

Some properties that matter for our data:
* Trees only compare values with thresholds. Shifting or rescaling the inputs changes nothing. This
  is why the "raw" and "positive" features give the same results.
* The predicted position is an average of real training positions, so it always falls inside the
  area that was surveyed.
* Feature importance tells us how much each AP lowered impurity, summed over all trees.

## 7. Plain vs per-building Random Forest

* Plain: three separate forests trained on all scans (one for building, one for floor, one for position).
* Per-building (hierarchical): first predict the building (about 99.8% correct). Then use a floor
  forest and a position forest that were trained only on that building's scans. Each sub-problem is
  smaller, and APs from the other buildings no longer get in the way. The risk is that a wrong
  building sends the scan to the wrong sub-model. We measure this with the error breakdown.

In both versions the three tasks do not share anything they learn. Phase 2 will change this with one
neural network that has a shared trunk and three output heads (multitask learning, Caruana 1997).

## 8. How we validate

* Scans are taken in bursts: same person, same spot, a few seconds apart. A random train/validation
  split would put almost identical scans on both sides and make the accuracy look better than it is.
* GroupKFold by USERID (5 folds): each fold leaves out whole users. We make all our choices
  (features, k, max_features) this way, using only the training data.
* Test set: the official validation file. It was collected 4 months later by different users, and a
  lot of it comes from phones that are not in the training data. We use it only once, at the end.

## 9. Why the CV and test numbers differ

Our CV errors are higher than our test errors. We looked at two possible reasons:
* Location coverage is not the reason. In CV, held-out scans are about as close to a training
  reference point (9.9% are more than 5 m away) as on the test set (7.7%).
* New phones are the likely reason. In training, 17 of the 18 users had their own phone, so leaving
  out a user almost always means leaving out a phone too. On the test set, 35.7% of scans come from
  phones that are in the training data. Different phones read the same signal differently.

So CV is a harder test than the test set. We still use it to choose models, because choosing by test
score would be cheating.
