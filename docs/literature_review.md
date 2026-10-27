# Literature Review: Wi-Fi Fingerprinting for Indoor Location

## 1. Why indoor positioning is a learning problem

GPS does not work well indoors, because walls and ceilings block and reflect the satellite signal.
But most buildings already have Wi-Fi access points (APs). A phone can measure the received signal
strength indicator (RSSI) of every AP it hears, without connecting to any of them. RSSI drops with
distance. Indoors, though, it also depends on walls, floors, furniture, people and signals bouncing
around (multipath). Because of this, formula-based propagation models (log-distance path loss,
trilateration) are not very accurate indoors [Zafari 2019].

Fingerprinting does not try to model how the signal travels. It works in two phases:

1. Offline phase. Walk around the building and, at known positions, record the RSSI value of every
   AP. This collection is called the radio map (our training set).
2. Online phase. Compare a new scan with the radio map and guess the location from the most similar
   fingerprints.

So the online phase is a supervised learning problem. The input is the RSSI vector. The outputs are
the building (classification), the floor (classification) and the (x, y) coordinates (regression).

## 2. Main prior work

| Work | What it did | Why it matters for us |
|---|---|---|
| Bahl & Padmanabhan, RADAR (INFOCOM 2000) | First Wi-Fi fingerprinting system. Uses nearest neighbour in signal space. Median error of about 2 to 3 m on one floor. | Where the k-NN fingerprinting baseline we use comes from. |
| Youssef & Agrawala, Horus (MobiSys 2005) | Probabilistic fingerprinting. Models the RSSI distribution at each location instead of one mean value. | Shows that RSSI is noisy, which is a reason to use stable features and ensembles. |
| Torres-Sospedra et al., UJIIndoorLoc (IPIN 2014) | Publishes our dataset: 3 buildings, 4 to 5 floors, 520 APs, 25 phones, 18 users. Their k-NN baseline gets an 89.9% floor hit rate and 7.9 m mean error. | Our dataset. It defines the train/validation split and the +100 "not detected" value. |
| Torres-Sospedra et al., Expert Syst. Appl. 2015 | Compares 51 distance metrics and 4 data representations (positive, normalized, exponential, powed) for k-NN. The powed representation with Sørensen or Euclidean distance works best. | Where our `positive` and `powed` features come from. We test if the gain also shows up with trees. |
| Torres-Sospedra et al., IPIN 2016 competition (Sensors 2017) | Compared indoor positioning methods on building and floor accuracy and position error. | The same measures we report. |
| Bozkurt et al. (INISTA 2015) | Compares k-NN, decision trees, naive Bayes, Bagging, AdaBoost and others on UJIIndoorLoc. k-NN and ensembles do best. | Backs up our choice of k-NN and tree ensembles as baselines. |
| Akram et al., HybLoc (IEEE Access 2018) | Splits the radio map into clusters, then trains a random forest for each cluster. | Same idea as our hierarchical model: break the problem into smaller local ones. |
| Breiman, Random Forests (2001) | Bagged trees that are made less correlated. Lowers variance and gives feature importance for free. | Our main tree model. |
| Cover & Hart (1967) | Nearest-neighbour classification. The 1-NN error is at most twice the best possible (Bayes) error. | The theory behind k-NN. |
| Caruana, Multitask Learning (1997) | Training related tasks with a shared representation helps the model generalise, because what it learns for one task carries over to the others (inductive transfer). | The reason for Phase 2: one shared network for building, floor and coordinates. |
| Nowicki & Wietrzykowski (2017); Kim et al. (2018) | Stacked autoencoder + MLP on UJIIndoorLoc for building and floor (Kim et al. use one multi-label output). | Deep learning baselines for Phase 2. |
| Song et al., CNNLoc (IEEE Access 2019) | Autoencoder + 1-D CNN. About 96% floor accuracy on UJIIndoorLoc. | A reference for how well deep learning can do. |

## 3. Gaps that shaped our design

1. Missing values are not missing at random. "Not heard" means the signal is below what the phone
   can pick up, so the real value is very weak, not unknown. If a pipeline treats +100 as a normal
   number, the model thinks undetected APs are the strongest ones. We fill in a value just below the
   weakest observed one. We also tested adding "heard" indicator columns, but they did not help.
2. Evaluation leakage. Random K-fold splits put scans from the same walk, user and phone into both
   train and validation, so the accuracy looks better than it really is. We tune with GroupKFold by
   user. We test on the official validation file, which was collected 4 months later by different
   users, partly with different phones.
3. Independent targets. Most ML baselines train separate models for building, floor and
   coordinates. They cannot use the fact that the building limits which floors and locations are
   possible. We compare an independent design with a hierarchical one (building first, then a model
   for each building). In Phase 2 we replace this hand-built hierarchy with a shared representation
   that the model learns (multitask MLP).
4. Different devices. Different phones report different RSSI values at the same place. We split
   the error by phone to measure this.

## References

1. P. Bahl and V. N. Padmanabhan, "RADAR: An in-building RF-based user location and tracking system," *IEEE INFOCOM*, 2000.
2. M. Youssef and A. Agrawala, "The Horus WLAN location determination system," *MobiSys*, 2005.
3. J. Torres-Sospedra et al., "UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems," *IPIN*, 2014, pp. 261–270.
4. J. Torres-Sospedra, R. Montoliu, S. Trilles, Ó. Belmonte, and J. Huerta, "Comprehensive analysis of distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems," *Expert Systems with Applications*, vol. 42, no. 23, pp. 9263–9278, 2015.
5. J. Torres-Sospedra et al., "The smartphone-based offline indoor location competition at IPIN 2016: Analysis and future work," *Sensors*, vol. 17, no. 3, 557, 2017.
6. S. Bozkurt, G. Elibol, S. Gunal, and U. Yayan, "A comparative study on machine learning algorithms for indoor positioning," *INISTA*, 2015.
7. B. A. Akram, A. H. Akbar, and O. Shafiq, "HybLoc: Hybrid indoor Wi-Fi localization using soft clustering-based random decision forest ensembles," *IEEE Access*, vol. 6, pp. 38251–38272, 2018.
8. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
9. T. Cover and P. Hart, "Nearest neighbor pattern classification," *IEEE Trans. Information Theory*, vol. 13, no. 1, pp. 21–27, 1967.
10. R. Caruana, "Multitask learning," *Machine Learning*, vol. 28, no. 1, pp. 41–75, 1997.
11. M. Nowicki and J. Wietrzykowski, "Low-effort place recognition with WiFi fingerprints using deep learning," *Automation 2017*, Springer AISC vol. 550, 2017.
12. K. S. Kim, S. Lee, and K. Huang, "A scalable deep neural network architecture for multi-building and multi-floor indoor localization based on Wi-Fi fingerprinting," *Big Data Analytics*, vol. 3, 4, 2018.
13. X. Song et al., "A novel convolutional neural network based indoor localization framework with WiFi fingerprinting," *IEEE Access*, vol. 7, 2019.
14. F. Zafari, A. Gkelias, and K. K. Leung, "A survey of indoor localization systems and technologies," *IEEE Communications Surveys & Tutorials*, vol. 21, no. 3, pp. 2568–2599, 2019.
15. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.
