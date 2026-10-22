# Theory Notes (for the report and the viva)

## 1. Problem formulation

A scan is a vector **x** ∈ ℝ⁵²⁰, one RSSI value per access point, with most entries "not heard".
We learn three functions:

| Target | Type | Loss / metric |
|---|---|---|
| building *b* ∈ {0,1,2} | multiclass classification | cross-entropy (Gini for trees) / accuracy |
| floor *f* ∈ {0..4} | multiclass classification | cross-entropy / accuracy |
| (x, y) in metres | 2-output regression | squared error / mean Euclidean error |

Combined metric (IPIN competition): `score = ‖p − p̂‖₂ + 50·[b ≠ b̂] + 4·[f ≠ f̂]`, i.e. a wrong
building costs as much as 50 m of error and a wrong floor 4 m (roughly one storey of vertical
distance).

## 2. Missing data: why "+100" must not be used as a number

RSSI is negative dBm; −30 is very strong, −100 is the edge of detection. The dataset writes
+100 for "not heard". Fed directly to a model, +100 looks *stronger than any real signal*, the
exact opposite of its meaning. The data is **missing not at random (MNAR)**: an AP is missing
*because* it is far away. So the right imputation is a value just below the weakest real reading
(we use min − 1 = −105 dBm), learned from training data only. We also add a binary "heard"
indicator per AP so a model can use the presence pattern directly.

## 3. Representations (Torres-Sospedra et al., 2015)

Let `m` = weakest observed RSSI − 1 (here −105).

* **raw**: `x` if heard, else `m`.
* **positive**: `x − m` if heard, else 0. Range 0..105.
* **powed**: `(positive / (max − m))^β`, β = e. Raising to a power > 1 compresses weak values
  towards 0 and keeps strong ones distinct. Weak readings are the noisiest, so this down-weights noise.

**Why raw and positive give identical results for both models:**
* Decision trees split on thresholds (`x_j ≤ t`). Any strictly increasing transform of a feature
  preserves the order of values, so the same partitions exist and the tree is identical. Trees are
  *invariant to monotone transforms*. Powed is monotone too, but the *imputed value* moves relative
  to the real values, which can change splits slightly; that is why powed differs a little.
* k-NN uses Euclidean distance, `‖a − b‖`. Positive = raw − m for every coordinate (including the
  imputed ones), and subtracting the same constant from both vectors does not change their
  difference. Powed is non-linear, so it **does** change k-NN distances.

## 4. k-Nearest Neighbours

Predict from the k training scans closest in feature space. Building and floor are decided by a
distance-weighted vote (weight 1/d); coordinates are the distance-weighted mean of the neighbours'
coordinates.
* Non-parametric: no training, but O(n·d) per query (here 19k × 465).
* Small k → low bias, high variance (noisy); large k → averages across distant reference
  points and blurs location. We choose k by user-grouped CV.
* Asymptotically, 1-NN error ≤ 2 × Bayes error (Cover & Hart, 1967).
* Weakness: all APs weigh equally in the distance, and it cannot learn which APs matter.

## 5. Decision trees and Random Forests

A CART tree greedily chooses the split (feature j, threshold t) that most reduces impurity:
* classification: Gini `G = 1 − Σ_k p_k²`
* regression: variance (MSE) of the targets in each child

A deep tree has low bias but high variance. **Random Forest** (Breiman, 2001) averages B trees,
each trained on a bootstrap sample with a random subset of `max_features` features tried at each
split. For B trees with variance σ² and pairwise correlation ρ, the variance of the average is

  `ρσ² + (1 − ρ)σ²/B`

Bagging drives the second term to 0; random feature selection lowers ρ. Bias is unchanged.
Out-of-bag samples (≈ 36.8% = (1 − 1/n)ⁿ → 1/e) give a free validation estimate.

Regression trees predict the **mean of the training targets in a leaf**, so an RF can only output
convex combinations of training coordinates. It can never predict a point outside the surveyed
area, which suits positioning.

## 6. Gradient boosting and XGBoost

Boosting builds an additive model `F_M(x) = Σ_m η·h_m(x)`, where each new tree `h_m` fits the negative
gradient of the loss at the current prediction (Friedman, 2001), with learning rate η.

XGBoost (Chen & Guestrin, 2016) minimises the regularised objective

  `L = Σ_i l(y_i, ŷ_i) + Σ_m [γ·T_m + ½·λ·‖w_m‖²]`

using a **second-order Taylor expansion**. With gradient `g_i` and Hessian `h_i`, the optimal leaf
weight is `w* = −Σg / (Σh + λ)` and a split's gain is

  `½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ) ] − γ`

**Sparsity-aware splits:** at every split XGBoost learns a *default direction* for missing values.
Our imputed −105 plays the same role explicitly.

Bias–variance: boosting mainly reduces **bias** (sequential correction); RF mainly reduces **variance**.

## 7. Independent vs hierarchical modelling

*Independent:* three models on all data. The coordinate regressor must learn one global mapping
across three buildings that share almost no APs.

*Hierarchical:* predict building (≈ 99.8% accurate), then use a floor classifier and coordinate
regressor trained only on that building. Each sub-problem is smaller and more homogeneous, and
irrelevant APs drop out naturally. The cost: a building error propagates (cascade error), which the
hierarchical breakdown measures.

This is a hand-designed way of sharing information between tasks. Phase 2's multitask MLP
(Caruana, 1997) learns a shared representation instead: one trunk, three heads, with loss
`L = λ_b·CE_b + λ_f·CE_f + λ_xy·MSE_xy`.

## 8. Validation design (and why it matters)

* Scans come in bursts (same user, same spot, seconds apart). A random split puts near-duplicate
  scans in both train and validation, so it measures memorisation.
* **GroupKFold by USERID**: no user appears in both the training and validation folds. This simulates new people
  and new walks.
* **Official validation file as test**: collected 4 months later (Sept–Oct vs May–June 2013) by
  different users, partly with phones not in training. Used **once**, at the end. All model and feature choices
  are made by CV. Using test scores to choose models would be test-set leakage.
* Normalisation and imputation statistics are fitted on training data only (`fit` on train,
  `transform` on test) inside each CV fold.

## 9. Why CV and test can disagree (an observed result)

CV errors are higher than test errors, and the model ranking can differ. We checked two explanations:

* **Location coverage — not the cause.** The distance from a held-out scan to the nearest training
  reference point is about the same in CV (9.9% of scans > 5 m away) and on the test set (7.7%):
  several users surveyed the same reference points.
* **Device novelty — the likely cause.** In the training data every user carried their own phone
  (17 of 18 users have a unique phone). So holding out a user almost always means holding out a
  *device*: close to 100% of CV validation scans come from a phone the model never saw. On the
  test set, 35.7% of scans come from phones that are in training. Different phones report
  systematically different RSSI for the same place, which hurts raw-distance methods like k-NN most.

So user-grouped CV is a *stricter* test (new user + new device + 20% less radio map). It is still
the honest basis for choosing models, since the test set cannot be used for that, and we report
both numbers side by side.
