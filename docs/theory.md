# Theory Notes

## 1. The problem

A Wi-Fi scan is a list of 520 numbers, one per access point (AP): the received signal strength
(RSSI) in dBm. From one scan we predict three things:

| Target | Type of problem | Metric |
|---|---|---|
| Building (0, 1, 2) | classification | accuracy |
| Floor (0–4) | classification | accuracy |
| Position (x, y) in metres | regression | mean distance error in metres |

Combined score (from the IPIN indoor-positioning competition):
`score = distance error + 50 m if building wrong + 4 m if floor wrong`.

## 2. The "+100" problem (missing values)

RSSI is negative: −30 dBm is very strong, −100 dBm is barely heard. The dataset writes **+100**
when an AP was **not heard at all**. Used as a number, +100 would look *stronger* than any real
signal, the opposite of what it means. "Not heard" really means "too weak to detect", so we replace it
with **−105 dBm**, just below the weakest real reading (−104). The value is computed from the
training data only and then applied unchanged to the test data.

## 3. Linear and Logistic Regression (baseline)

* **Linear regression** predicts x (and separately y) as a weighted sum of all AP signals:
  `x̂ = w₀ + w₁·RSSI₁ + … + w₄₆₅·RSSI₄₆₅`, with weights chosen to minimise the squared error.
* **Logistic regression** predicts building and floor. For each class it computes a weighted sum,
  then turns the sums into probabilities with softmax, and is trained with cross-entropy loss.
* Inputs are standardised (mean 0, std 1, using training statistics) so all APs are on the same scale.

**Why linear regression does badly on position:** signal strength falls off with the *logarithm* of
distance and is blocked by walls, so position is **not** a straight-line function of RSSI. A linear
model can only draw one flat surface through 465 dimensions. Logistic regression does well on
building and floor because each class can be separated by "which APs are strong", which a weighted
sum captures well.

## 4. k-Nearest Neighbours (k-NN)

The classic fingerprinting method. No training: store every training scan. For a new scan:
1. Compute the Euclidean distance to every stored scan.
2. Take the k closest.
3. Building and floor = their (distance-weighted) vote; position = their (distance-weighted) average.

* Small k gives noisy predictions; large k averages over far-away points and blurs the position. We choose k by
  cross-validation.
* Weakness: every AP counts equally in the distance, and prediction is slow because it compares
  against all 19k scans.

## 5. Decision trees

A tree asks yes/no questions such as "is WAP117 stronger than −80 dBm?". At each node it tries
many (AP, threshold) pairs and keeps the one that makes the two child groups **purest**:
* classification: **Gini impurity** `G = 1 − Σ pₖ²` (0 means all one class)
* regression: **variance** of the targets (the split that most reduces it wins)

A leaf predicts the majority class (classification) or the mean position of its training scans
(regression). One deep tree memorises the training data, so it **overfits** (high variance).

## 6. Random Forest

Many trees (300 here), each made different in two ways:
1. **Bootstrap sample:** each tree trains on a random sample of scans drawn *with replacement*.
2. **Random features:** at each split, only a random subset of APs is considered (`max_features`,
   for example √465 ≈ 21).

Prediction = majority vote (building, floor) or average (position). Averaging many different,
overfit trees cancels out their individual mistakes: the **variance goes down while the bias stays low**.
Formally, B trees with variance σ² and correlation ρ give an average with variance
`ρσ² + (1−ρ)σ²/B`. More trees shrink the second term, and random features lower ρ.

Useful properties here:
* Trees only compare values against thresholds, so shifting or rescaling the inputs changes nothing.
  This is why "raw" and "positive" features give identical results.
* The position prediction is an average of real training positions, so it always lands inside the
  surveyed area.
* **Feature importance**: how much each AP reduced impurity across all trees.

## 7. Plain vs per-building Random Forest

* **Plain:** three separate forests trained on all scans (building, floor, position).
* **Per-building (hierarchical):** predict the building first (≈ 99.8% correct), then use a floor
  forest and a position forest trained only on that building's scans. Each sub-problem is smaller
  and the irrelevant APs of other buildings drop out. The risk is that a wrong building sends the scan to the
  wrong sub-model, which we measure with the error breakdown.

Neither version lets the three tasks share what they learn. Phase 2 does that with one neural
network that has a shared trunk and three output heads (multitask learning, Caruana 1997).

## 8. How we validate

* Scans are taken in bursts (same person, same spot, seconds apart). A random train/validation split
  would put near-identical scans on both sides and overstate accuracy.
* **GroupKFold by USERID (5 folds):** each fold holds out whole users. All choices (features, k,
  max_features) are made this way, on training data only.
* **Test set** = the official validation file: collected 4 months later by different users, much of
  it on phones not seen in training. It is used **once**, at the end.

## 9. Why the CV and test numbers differ

CV errors are higher than test errors. We checked two explanations:
* **Location coverage — not the cause.** Held-out scans are about as close to a training reference
  point in CV (9.9% more than 5 m away) as on the test set (7.7%).
* **New phones — the likely cause.** In training, 17 of 18 users had their own phone, so a held-out
  user almost always means an unseen phone. On the test set, 35.7% of scans come from phones that are in
  training. Different phones read the same signal differently.

So CV is a stricter test. It is still the honest way to choose models, because choosing by test score
would be cheating.
