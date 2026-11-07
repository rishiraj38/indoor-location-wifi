# Deep Learning Theory Notes (Phase 2)

Owner: Kaustubh Ranjan Mishra (Rishi Raj checks the parts about our model).

Written for the viva: every team member should be able to explain each section in a minute.
Keep it plain, like `theory.md`.

## 1. A fully connected layer

- `z = W x + b`, then an activation. Why we use ReLU. What a "hidden layer" is.

## 2. Our architecture

- Input: 465 scaled signal values. Trunk: two hidden layers (sizes from the ablation results).
- Three heads: building (3 outputs, softmax), floor (5 outputs, softmax), coordinates (2 outputs, linear).
- Why one shared trunk instead of three networks (link to Caruana).

## 3. The loss

- Cross-entropy for building and floor: formula and what it rewards.
- Mean squared error for the coordinates, on scaled coordinates: why we scale them.
- Combined loss `L = w_b CE_b + w_f CE_f + w_xy MSE_xy`: what the weights do (from the ablation results).

## 4. Training

- Backpropagation in one paragraph: compute the loss, compute gradients layer by layer, update weights.
- Adam optimiser, learning rate, batches, epochs.
- Early stopping on held-out users: what it prevents and how our `EarlyStopping` class does it.

## 5. Regularisation

- Dropout, weight decay, early stopping: what each does, with our chosen values and why.

## 6. Overfitting and the bias-variance picture

- How a network overfits 19k scans, how we checked (training curve, CV vs test).

## 7. Why the hybrid should (or should not) help

- The trunk as a learned feature extractor; why a forest on top can do better than on raw signals.
