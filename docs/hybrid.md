# Hybrid Models: Combining the Neural Network and Random Forest

Every number here comes from `results/hybrid.csv`. We evaluate all models with 5-fold cross-validation grouped by user, and once on the test set.

The forest in this file uses the Phase 1 tree settings (300 trees, sqrt features, per building) but the default feature set of `FingerprintFeatures` (powed signals with heard flags and summary columns), not the raw dBm features Phase 1 finally chose. That is why the "Random Forest alone" row (90.9% floor, 9.53 m on test) is a little different from the Phase 1 result in `results/test_metrics.csv` (91.3%, 9.37 m). The comparison inside this file is still fair, because every hybrid is measured against the same forest and the same network it is built from. The network's own numbers also differ slightly from `docs/dl_architecture.md` because this run was made on a different computer (see the note in `docs/dl_ablations.md`).

## 1. Why Try a Hybrid?

In Phase 1, the per-building Random Forest achieved our best floor accuracy (91.3% on test), but tree splits are axis-aligned and cannot easily capture complex joint patterns across 465 access points. The multitask neural network learns a shared 256-dimensional continuous representation in its trunk, but its final heads are simple linear layers.

A hybrid aims to combine the strengths of both approaches. We test two designs:
1. Hybrid A: use the network trunk as a feature extractor, then train per-building Random Forests on those features.
2. Hybrid B: split by target, using the network for building and floor, and the Random Forest for coordinates.

## 2. Hybrid A: Trunk Embeddings into Random Forest

### What it is and the reasoning
We train the multitask network on the training scans. We then call `embed(X)` to extract the 256-dimensional activations from the end of the shared trunk. A per-building Random Forest is then fitted on these 256 trunk features instead of the raw 465 signal values.

The reasoning is that the neural network compresses noisy and sparse signal vectors into an informative representation, which should make the Random Forest's split decisions easier.

### Results
Here are the cross-validation and test set numbers for Hybrid A alongside its two constituent models:

Cross-validation (5-fold by user):

| Model | Building acc (%) | Floor acc (%) | Mean error (m) | Median error (m) | p95 error (m) |
|---|---|---|---|---|---|
| Multitask MLP alone | 96.5 | 82.9 | 11.08 | 7.82 | 32.0 |
| Random Forest alone | 99.3 | 72.1 | 14.47 | 9.72 | 41.3 |
| Hybrid A: trunk + forest | 96.6 | 83.0 | 11.05 | 6.44 | 35.2 |

Test set:

| Model | Building acc (%) | Floor acc (%) | Mean error (m) | Median error (m) |
|---|---|---|---|---|
| Multitask MLP alone | 99.9 | 92.6 | 8.57 | 6.77 |
| Random Forest alone | 99.9 | 90.9 | 9.53 | 7.45 |
| Hybrid A: trunk + forest | 100.0 | 92.7 | 7.75 | 5.73 |

In cross-validation, Hybrid A clearly beats the Random Forest alone (83.0% vs 72.1% floor accuracy, 11.05 m vs 14.47 m mean error). Compared to the Multitask MLP alone, floor accuracy is 83.0% vs 82.9% and mean error is 11.05 m vs 11.08 m. Its median error improves from 7.82 m to 6.44 m, but its worst-case errors get larger (p95 35.2 m vs 32.0 m), and on building accuracy both trunk-based models sit below the forest on the signal features (99.3%).

On the test set, Hybrid A reaches 7.75 m mean error (and 5.73 m median), which is better than the network (8.57 m) and the forest (9.53 m).

## 3. Hybrid B: Split by Target

### What it is and the reasoning
Hybrid B uses each model where it was strongest. The multitask network predicts the building and floor, while a per-building Random Forest predicts the longitude and latitude coordinates.

### Results

Cross-validation (5-fold by user):

| Model | Floor acc (%) | Mean error (m) | Median error (m) |
|---|---|---|---|
| Multitask MLP alone | 82.9 | 11.08 | 7.82 |
| Random Forest alone | 72.1 | 14.47 | 9.72 |
| Hybrid B: net for class, forest for xy | 82.9 | 14.47 | 9.72 |

Test set:

| Model | Building acc (%) | Floor acc (%) | Mean error (m) | Median error (m) |
|---|---|---|---|---|
| Multitask MLP alone | 99.9 | 92.6 | 8.57 | 6.77 |
| Random Forest alone | 99.9 | 90.9 | 9.53 | 7.45 |
| Hybrid B: net for class, forest for xy | 99.9 | 92.6 | 9.53 | 7.45 |

Because Hybrid B simply copies classification from the network and regression from the forest, its metrics are identical to the network on building and floor and identical to the forest on coordinates.

## 4. Verdict

The project guidelines state that a hybrid only earns credit if it is justified and clearly better than both of its parts, judged by cross-validation. Differences within 0.1 m or 1 percentage point are counted as a tie under the project tie-break rule.

Applying this rule strictly:
- Hybrid B does not beat both parts. It matches the network on floor and matches the forest on position, giving no overall benefit over the network alone.
- Hybrid A beats the Random Forest by a wide margin (11.05 m vs 14.47 m). However, in cross-validation it is essentially a tie with the Multitask MLP alone: 83.0% vs 82.9% floor accuracy (0.1 percentage point difference) and 11.05 m vs 11.08 m mean position error (0.03 m difference). Even though Hybrid A achieves a noticeably lower median error in CV (6.44 m vs 7.82 m) and achieves the best overall test error (7.75 m), the tie-break rule on mean CV error means we cannot claim it clearly beats the network in model selection.
- Between Hybrid A and the standalone multitask network, the standalone network remains the preferred model for deployment. It requires running only one neural network during inference, avoiding the overhead of training and evaluating an ensemble of 300 trees per building on top of the network trunk.
