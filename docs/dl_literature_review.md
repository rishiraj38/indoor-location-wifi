# Deep Learning Literature Review (Phase 2)

Owner: Kaustubh Ranjan Mishra.

Target: about one page. Plain language. For each paper: what they did, what result they got on
UJIIndoorLoc (if any), and what we take from it. Add the full reference at the bottom in the same style
as `literature_review.md`.

## 1. Multitask learning

Caruana (1997) introduced the idea of training a single model on several related tasks at the same time, with shared hidden layers and separate output heads for each task. In the standard setup, the lower layers of the network are shared across all tasks and the top layer branches into one head per task. The shared layers learn a representation that must be useful for every task at once. This is called hard parameter sharing.

The benefit is a form of inductive transfer. What the network learns for one task carries into the other tasks through the shared weights. Each task also acts as a regulariser for the others: the shared trunk cannot overfit to any single task's noise, because it must keep all tasks happy at the same time. Caruana showed that this consistently reduces generalisation error compared with training the same tasks separately, and that gains are largest when the tasks are related and each one has limited training data.

Our indoor positioning problem maps directly onto this idea. Building and floor classification depend on which access points are heard and how strongly, and so does position regression. In Phase 1, we trained three separate models. A shared trunk can learn from all three signals at once. The three heads are: building (3 outputs, softmax), floor (5 outputs, softmax) and coordinates (2 outputs, linear). The ablation in `docs/dl_ablations.md` confirms the direction: the multitask network beats three single-task networks on position (11.08 m vs 11.50 m) while matching them on floor accuracy.

## 2. Neural networks on UJIIndoorLoc

**Nowicki and Wietrzykowski (2017)** used a stacked autoencoder followed by a feed-forward classifier on UJIIndoorLoc. The autoencoder compressed the 520-dimensional RSSI vector into a lower-dimensional representation in an unsupervised step, then a small classifier predicted building and floor. They treated +100 (not detected) as a valid input number after shifting, without the explicit floor-below-threshold replacement that we use. Their best configuration reached roughly 92% combined building-and-floor accuracy. We take from this paper the confirmation that deep features from raw RSSI can outperform k-NN on building and floor, but note they did not predict coordinates and trained two networks in sequence rather than one joint network.

**Kim, Lee and Huang (2018)** proposed a single scalable deep network with an autoencoder front end and a multi-label classifier for building, floor and position on UJIIndoorLoc. Their network achieved near state-of-the-art building and floor hit rates with lower computational cost than ensemble methods. What they showed us is that combining all three targets in one network is feasible and practical; our simpler MLP trunk replaces their autoencoder pre-training stage and is trained end-to-end with a combined loss.

**Song et al. (2019), CNNLoc**, combined a stacked autoencoder with a one-dimensional convolutional neural network. On UJIIndoorLoc they reported 100% building accuracy, approximately 95.92% floor accuracy, and a mean position error of about 11.78 m. They evaluated on the official validation file. We take from CNNLoc the floor accuracy level to aim for and the observation that an autoencoder pre-training step helps when inputs are sparse. Our plain MLP trunk is simpler but trained end-to-end; on the same test file we reach 92.3% floor accuracy, which is below CNNLoc's floor result but substantially better than our Phase 1 baseline (91.3%).

## 3. Regularisation and training

**Dropout (Srivastava et al., 2014)** is a regularisation method that randomly sets a fraction of hidden units to zero on each forward pass during training. Because units cannot rely on specific neighbours to correct their errors, each unit must learn a useful feature on its own, which forces the network to learn redundant and independent representations. Srivastava et al. showed that a dropout rate of 0.5 was effective on many benchmark tasks. With 465 sparse inputs and about 19,000 training scans our network is at risk of memorising the training data, so we use dropout (rate 0.3) after every trunk layer. The regularisation sweep in `docs/dl_regularisation.md` confirms this: without dropout the mean position error is 12.40 m; with dropout 0.3 and weight decay it falls to 11.17 m.

**Kingma and Ba (2015), Adam**, is an adaptive gradient descent optimiser. Plain gradient descent uses the same learning rate for every parameter. Adam tracks a running mean and a running variance of the gradient for each parameter and divides the update by the square root of the variance. Parameters that receive large or erratic gradients get smaller steps; parameters with small or consistent gradients get relatively larger steps. This makes training more stable and faster to converge than plain SGD, particularly on sparse and high-dimensional inputs like our RSSI vectors. We use Adam with learning rate 0.001, the default suggested by Kingma and Ba.

**Prechelt (1998), early stopping**, monitors a validation loss during training and stops when the loss stops improving. Without early stopping a network can keep fitting the training data after the point of best generalisation, which raises test error without lowering training error. Prechelt showed that the right stopping criterion must be based on a held-out set that is representative of the training distribution. We use a random 10% of the training scans as the early-stopping validation set, with patience 30, meaning we stop if the validation loss has not improved for 30 consecutive epochs. Using whole users instead (to match our outer cross-validation) broke training in our first attempt: with one or two held-out users, the validation loss hit its minimum after epoch 1 and never recovered, restoring barely trained weights. The random split fixed this, as described in `docs/dl_architecture.md` (section 2).

## 4. Hybrid ML + DL

HybLoc (Akram et al., 2018), already reviewed in `docs/literature_review.md`, shows that a divide-and-conquer strategy (cluster the radio map, then train a forest per cluster) can outperform a single global forest. The idea is relevant here because our hybrid design (Aditya's `scripts/run_hybrid.py`) does something analogous: use the network trunk as a feature extractor and train a forest on those 256-dimensional features instead of the raw 465 RSSI signals. The network compresses noisy, sparse signals into a denser representation; the forest then makes axis-aligned splits in that space. This combination targets the main weakness of a plain forest on raw RSSI: it must split on individual noisy APs rather than on stable combinations learned from all three tasks.

## 5. What this review tells us about our design

- **Why a shared trunk:** The Phase 1 baselines treated building, floor and coordinates as three independent problems. Caruana (1997) and the results in Kim et al. (2018) show that a single shared network can learn representations that help all three targets at the same time. Our ablation confirms this direction (11.08 m vs 11.50 m for multitask vs single-task on position).
- **Why dropout, weight decay and early stopping:** With roughly 19,000 training scans and over 1.5 million network parameters, the network can memorise the training set. Srivastava et al. (2014) and Prechelt (1998) provide the theoretical and empirical basis for each regulariser we use. Our sweep shows they all help and work best together.
- **Why a hybrid is worth testing:** The network learns a compact shared representation; a forest can exploit that representation differently from the network's own linear heads. The HybLoc idea (Akram et al., 2018) and the general combination of learned features with tree models is well-established in the indoor localisation literature. Our Hybrid A result (7.75 m on test) is better than either model alone, though it does not clearly beat the network in cross-validation under the project tie-break rule.
- **Floor accuracy gap:** CNNLoc (Song et al., 2019) reaches about 95.92% floor accuracy on UJIIndoorLoc. Phase 1 reached 91.3%. Our network reaches 92.3%. The gap to CNNLoc likely reflects the autoencoder pre-training they use; end-to-end training of a plain MLP trunk on a sparse input is harder. This is a direction for future work.

## References

1. R. Caruana, "Multitask learning," *Machine Learning*, vol. 28, no. 1, pp. 41–75, 1997.
2. J. Torres-Sospedra et al., "UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems," *IPIN*, 2014, pp. 261–270.
3. M. Nowicki and J. Wietrzykowski, "Low-effort place recognition with WiFi fingerprints using deep learning," in *Automation 2017*, Springer AISC vol. 550, 2017.
4. K. S. Kim, S. Lee, and K. Huang, "A scalable deep neural network architecture for multi-building and multi-floor indoor localization based on Wi-Fi fingerprinting," *Big Data Analytics*, vol. 3, no. 4, 2018.
5. X. Song et al., "A novel convolutional neural network based indoor localization framework with WiFi fingerprinting," *IEEE Access*, vol. 7, pp. 122095–122106, 2019.
6. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *Journal of Machine Learning Research*, vol. 15, no. 1, pp. 1929–1958, 2014.
7. D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. Int. Conf. on Learning Representations (ICLR)*, 2015.
8. L. Prechelt, "Early stopping, but when?," in *Neural Networks: Tricks of the Trade*, Springer LNCS vol. 1524, pp. 55–69, 1998.
9. B. A. Akram, A. H. Akbar, and O. Shafiq, "HybLoc: Hybrid indoor Wi-Fi localization using soft clustering-based random decision forest ensembles," *IEEE Access*, vol. 6, pp. 38251–38272, 2018.
