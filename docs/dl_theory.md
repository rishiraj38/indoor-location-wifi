# Deep Learning Theory Notes (Phase 2)

Owner: Kaustubh Ranjan Mishra (Rishi Raj checked the parts about our model).

Written for the viva: every team member should be able to explain each section in a minute.

## 1. A fully connected layer

A fully connected (dense) layer takes a vector of inputs x, multiplies it by a weight matrix W,
adds a bias vector b, and passes the result through an activation function:

    z = W x + b
    h = activation(z)

x has one number per input feature. W has one row per output unit and one column per input.
Each output unit computes a weighted sum of all inputs, then applies the activation.

A hidden layer is any layer between the input and the output. Its outputs are not directly observed;
they are intermediate representations the network builds up. Stacking several hidden layers lets the
network compose simple features (which APs are strong) into complex ones (which part of which floor
is this scan from).

Why ReLU? The rectified linear unit replaces every negative value with zero:

    ReLU(z) = max(0, z)

ReLU is cheap to compute and its gradient is 1 for positive inputs, so it does not shrink gradients
as a scan passes through many layers (the vanishing gradient problem that sigmoid and tanh suffer
from at large values). In practice, networks with ReLU train faster and to lower error than those
with sigmoid activations on the same kind of task.

## 2. Our architecture

Input. Each training scan starts as 520 RSSI values in dBm. The 55 APs never heard in training
are dropped, and the +100 not-detected values are replaced with -105. That leaves 465 features.
Before the network sees them, they are scaled to [0, 1] with min-max scaling fitted on the training
set (minimum -105, maximum 0), so the not-heard value maps exactly to 0.

Shared trunk. Three fully connected layers with ReLU activation and dropout:

    465 inputs, then hidden layers of 1024, 512 and 256 units

After each of the three layers, a dropout layer with rate 0.3 zeros out 30% of the units during
training. The final 256-unit layer is the trunk output: a compact representation of the scan that is
shared by all three heads.

Three heads. Three separate linear layers read from the 256-unit trunk output:

| Head        | Output size | Activation | What it predicts                   |
|-------------|-------------|------------|------------------------------------|
| Building    | 3           | Softmax    | Probability for each of 3 buildings|
| Floor       | 5           | Softmax    | Probability for each of 5 floors   |
| Coordinates | 2           | None (linear) | Scaled longitude and latitude   |

Why one trunk instead of three networks? Three separate networks each see only one target.
The trunk is trained by gradients from all three heads at once. What it learns for building
classification (which APs are active) also improves floor and position prediction, because the
same signals carry all three answers. Caruana (1997) calls this inductive transfer. Our ablation
partly supports it: the shared trunk reaches 11.08 m mean position error against 11.50 m for three
single-task networks with the same trunk size, floor accuracy is a tie (82.9% vs 83.2%), and building
accuracy is 1.8 points worse with the shared trunk (96.5% vs 98.3%). Sharing helps the hard target
(position) and costs a little on the easy one (building).

All settings are recorded in `results/dl_best.json`.

## 3. The loss

Softmax turns a vector of raw scores into a probability distribution. For a head with K classes,
if the scores are s_1, ..., s_K, the probability for class k is:

    p_k = exp(s_k) / sum_j exp(s_j)

All p_k are positive and sum to 1. The class with the highest score gets the highest probability.
Softmax is used for the building and floor heads.

Cross-entropy loss measures how far the predicted distribution is from the true one-hot label
(1 for the correct class, 0 for all others). For one scan with true class c:

    CE = -log(p_c)

If the network is confident and correct, p_c is close to 1 and CE is near 0. If the network
assigns a low probability to the right class, -log(p_c) is large. The optimiser minimises this,
which pushes p_c toward 1.

Mean squared error (MSE) for coordinates measures the average squared distance between the
predicted and true scaled values:

    MSE = (1/(2N)) * sum_i [ (x_pred_i - x_true_i)^2 + (y_pred_i - y_true_i)^2 ]

(the mean over both coordinates of all N scans, which is what PyTorch's mse_loss computes)

Why scale the coordinates first? Raw UTM latitude in this dataset is around 4,864,000 and
longitude around -7,500. A freshly initialised network outputs values near 0, so its first errors
would be millions of metres and their squares around 10^13, while the cross-entropy terms are of
order 1. The coordinate term would swallow the loss and the gradients, and the building and floor
heads would barely train. We centre and scale the coordinates with the training mean and standard
deviation (the CoordScaler class) before computing the loss, so all three terms are on a similar
scale from the first step. Predictions are unscaled back to metres for reporting.

Combined loss. The total loss is a weighted sum of the three terms:

    L = CE(building) + CE(floor) + 20 * MSE(scaled coordinates)

The weight 20 on the coordinate term was chosen by a grid search (see `docs/dl_architecture.md`,
section 3). With equal weights (weight 1) the two classification losses dominate and the coordinate
head is under-trained. In the tuning sweep (512-256 trunk, patience 30) weight 1 gives 13.83 m and
weight 20 gives 12.52 m; weights 20 and 30 tie (12.52 m vs 12.48 m), so the simpler value is kept.
The ablation repeats the comparison with the final trunk: 13.35 m at weight 1 against 11.08 m at
weight 20.

## 4. Training

Backpropagation is the algorithm that computes how much each weight contributed to the loss.
Starting from the loss, it applies the chain rule of calculus layer by layer, working backwards
from the output heads through the trunk to the input. Each weight receives a gradient: a number
that says how much the loss would increase if that weight increased by a small amount. The optimiser
then adjusts every weight in the direction that decreases the loss.

Adam (Kingma and Ba, 2015) is an adaptive optimiser. Plain gradient descent updates every
weight by the same step size (learning rate) times its gradient. Adam keeps a running mean (m)
and a running variance (v) of the gradient for each weight separately:

    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient^2
    weight = weight - lr * m / (sqrt(v) + epsilon)

A weight that receives large or erratic gradients gets a large v, so its effective step size
(lr / sqrt(v)) shrinks automatically. A weight with small, consistent gradients takes larger
steps. This is useful for our sparse inputs: most APs are not heard in most scans, so their
weight gradients are zero in those batches and very large in the few batches where the AP appears.
Adam handles this variation much better than plain SGD. We use learning rate 0.001, the default
recommended in the paper, with batch size 256 and up to 300 epochs.

Batches and epochs. On each training step (iteration) the network sees 256 randomly sampled
scans, computes the loss and gradients for that batch, and updates all weights once. One epoch is
one pass through the full training set. In cross-validation each fold trains on four fifths of the 19,227 scans
minus the 10% early-stopping part, about 13,800 scans, so one epoch is about 54 iterations of
batch size 256; the final run on the full training set has about 17,300 scans and 68 iterations
per epoch.

## 5. Regularisation

Dropout (rate 0.3). During each forward pass in training, every hidden unit in the trunk is
independently turned off with probability 0.3. The units that remain active have their outputs
scaled up by 1/(1-0.3) so the expected sum is unchanged. This stops any unit from relying on a
specific partner to fix its errors, which forces the trunk to learn multiple independent paths for
the same information. At test time, dropout is turned off and the full network is used. The
regularisation sweep in `docs/dl_regularisation.md` shows that adding dropout 0.3 reduces mean
position error from 12.40 m to 11.76 m, and combining it with weight decay brings it further to
11.17 m.

Weight decay (1e-4). The optimiser adds a small penalty proportional to the sum of squared
weights to the loss:

    L_total = L + (lambda / 2) * sum_w w^2

where lambda = 0.0001 (PyTorch adds lambda times the weight to each gradient, which is the
derivative of this penalty). During each update, this penalty pulls every weight slightly toward zero,
preventing any single connection from growing very large and dominating the prediction. Weight
decay is equivalent to L2 regularisation. In PyTorch it is set via the `weight_decay` argument
of the Adam optimiser. Without weight decay, mean position error is 12.40 m (no regularisation)
or 11.76 m (dropout only); with weight decay added on top of dropout it falls to 11.17 m.

Early stopping (patience 30). We hold out a random 10% of each fold's training scans as a
validation set, separate from the outer 5-fold cross-validation. After every epoch, the network's
validation loss is recorded. The EarlyStopping class keeps a copy of the weights from the epoch with
the lowest validation loss so far; an epoch only counts as an improvement if the loss falls by at
least 1e-4. If 30 epochs pass without an improvement, training stops and the saved best weights
are put back. This prevents training from continuing past the
point of best generalisation.

The problem we hit. Our first implementation held out whole users for the early-stopping check,
to match the outer cross-validation design. With only one or two users held out, the validation loss was
too noisy to stop on: in every fold its lowest value was at epoch 1 and later epochs never beat it by
the minimum step. Training ran on for the patience window (15 epochs, the default at the time) and
then restored the epoch-1 weights, which were barely trained. On one fold this
produced 35% floor accuracy and 131 m position error. Switching to a random 10% split (which
samples from all users and positions in proportion) fixed the problem: the same fold gave 95%
floor accuracy. The lesson is ours rather than Prechelt's: the early-stopping signal must be large and mixed
enough to be usable. One or two people's scans are not.

On the final training run on the full training set, the network stopped at epoch 108 (best weights
were from epoch 78). The 100-epoch cap ablation in `docs/dl_ablations.md` shows that cutting
training short raises mean error by 0.48 m (11.57 m vs 11.08 m), confirming that some folds
benefit from training past epoch 100.

## 6. Overfitting and the bias-variance picture

A neural network with 465 inputs and 1024 + 512 + 256 hidden units has about 1.1 million parameters
(1,135,882 including the three heads). Trained on
roughly 19,000 scans, it has enough capacity to memorise every training example, which would give
near-zero training loss but poor test performance. This is overfitting: the model fits the
noise in the training data rather than the underlying pattern.

We check for overfitting in two ways:

Training curve. The figure `results/figures/dl_training_curve.png` shows training loss and
validation loss over the 108 training epochs. Both fall steeply in the first few epochs and then
level off. The validation loss stays below the training loss throughout. That is expected with
dropout, because dropout is active when the training loss is measured (some units are off) and
inactive when the validation loss is measured (all units on, so the full network is evaluated).
The two curves never diverge. Because the random 10% shares reference points with the training
scans, a flat validation curve shows that training is stable, not by itself that the network
generalises to new users; that is what the grouped cross-validation measures.

CV vs test gap. Cross-validation floor accuracy is about 83%; test floor accuracy is 92.3%.
This is not overfitting in the usual sense: CV is harder. In training, almost every held-out user
carried their own phone, so leaving out a user means leaving out a device too. On the test set,
35.7% of scans come from phones that appear in training. Different phones read the same signal
differently (the per-phone breakdown in `docs/dl_results.md` shows errors ranging from 6.66 m
to 15.19 m across phones). CV measures generalisation to new devices; the test set is partly an
easier problem. We still use CV to choose hyperparameters, because using test scores to make
choices would be data leakage.

## 7. Why the hybrid should (or should not) help

The shared trunk converts a 465-dimensional sparse RSSI vector into a 256-dimensional dense
representation. Each of the 256 trunk outputs is a ReLU activation: a non-linear function of all
465 inputs, trained to be useful for predicting building, floor and position all at once. Similar
rooms (same floor, same part of the building) should map to similar points in this 256-dimensional
space, because the trunk had to make them predictable.

A Random Forest makes axis-aligned splits: at each node it picks one feature and a threshold.
On raw 465-dimensional RSSI vectors, the forest must split on individual noisy APs. Because each
AP is heard in only a subset of scans and its value varies with phone model and time of day, any
single AP is a noisy signal. On the 256-dimensional trunk space, each feature is a stable
combination of many APs learned by the network, so forest splits are on more informative and
less noisy inputs. This is the reasoning behind Hybrid A in `docs/hybrid.md`.

What the results say. In 5-fold cross-validation, Hybrid A (forest on trunk features) reaches
11.05 m mean error and 83.0% floor accuracy. The standalone network reaches 11.08 m and 82.9%.
Under the project tie-break rule (differences under 0.1 m or one percentage point are a tie),
the two are tied in CV. On the official test set, Hybrid A reaches 7.75 m, compared with 8.57 m
for the network alone in that run (8.66 m in the final run, the difference is floating point) and
9.53 m for the forest. Note that the forest in the hybrid run uses the default feature set (powed
signals with flags), not the raw dBm features of the Phase 1 forest (9.37 m on test). The test result is better, but
because we cannot use the test set for model selection, we cannot claim Hybrid A is definitively
superior. The lesson is: the trunk features are better inputs for a forest than the signal values
(11.05 m against 14.47 m in CV), but the network's own output heads are already good enough that
the forest gains little extra in CV.
