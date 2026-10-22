# Literature Review — Wi-Fi Fingerprinting for Indoor Location

## 1. Why indoor positioning is a learning problem

GPS does not work reliably indoors: walls and ceilings block and reflect the satellite signal. Buildings
already contain Wi-Fi access points (APs), and a phone can measure the **received signal strength
indicator (RSSI)** of every AP it hears without connecting to any of them. RSSI drops with distance,
but indoors the drop also depends on walls, floors, furniture, people and multipath reflections, so
closed-form propagation models (log-distance path loss, trilateration) are inaccurate [Zafari 2019].

**Fingerprinting** avoids modelling propagation at all:

1. *Offline phase.* Walk the building, and at known positions record the vector of RSSI values from
   every AP. This is the radio map (the training set).
2. *Online phase.* A new scan is compared with the radio map and the location is inferred from the
   most similar fingerprints.

Seen this way, the online phase is supervised learning: input = RSSI vector, outputs = building
(classification), floor (classification) and (x, y) coordinates (regression).

## 2. Key prior work

| Work | Contribution | Relevance to us |
|---|---|---|
| **Bahl & Padmanabhan, RADAR (INFOCOM 2000)** | First Wi-Fi fingerprinting system; nearest neighbour in signal space; ~2–3 m median error in one floor. | Origin of the k-NN fingerprinting baseline we use. |
| **Youssef & Agrawala, Horus (MobiSys 2005)** | Probabilistic fingerprinting: models the RSSI distribution per location instead of one mean value. | Shows RSSI is noisy; motivates robust features and ensembles. |
| **Torres-Sospedra et al., UJIIndoorLoc (IPIN 2014)** | Publishes our dataset: 3 buildings, 4–5 floors, 520 APs, 25 phones, 18 users; k-NN baseline gives 89.9% floor hit rate and 7.9 m mean error. | Our dataset; defines the train/validation split and the +100 "not detected" convention. |
| **Torres-Sospedra et al., Expert Syst. Appl. 2015** | Compares 51 distance metrics and 4 data representations (positive, normalized, exponential, powed) for k-NN. The *powed* representation with Sørensen/Euclidean distance works best. | Source of our `positive` and `powed` features; we test whether the gain carries over to trees. |
| **Torres-Sospedra et al., IPIN 2016 competition (Sensors 2017)** | Common evaluation: 2-D error + 50 m per wrong building + 4 m per wrong floor. | We report this "IPIN score" so all three targets are summarised in one number. |
| **Bozkurt et al. (INISTA 2015)** | Compares k-NN, decision trees, naive Bayes, Bagging, AdaBoost etc. on UJIIndoorLoc; k-NN and ensembles are strongest. | Supports k-NN and tree ensembles as the right ML baselines. |
| **Akram et al., HybLoc (IEEE Access 2018)** | Clusters the radio map, then trains a random forest per cluster. | Same idea as our hierarchical model: split the problem, solve smaller local ones. |
| **Breiman, Random Forests (2001)** | Bagged decorrelated trees; variance reduction; built-in feature importance. | Our main tree model. |
| **Friedman (2001); Chen & Guestrin, XGBoost (KDD 2016)** | Gradient boosting: fit trees to the gradient of the loss; XGBoost adds second-order updates, regularisation and sparsity-aware splits. | Our boosting model; its sparsity-aware split finding suits 96%-missing RSSI. |
| **Caruana, Multitask Learning (1997)** | Training related tasks with a shared representation improves generalisation through inductive transfer. | Motivates Phase 2: a shared network for building, floor and coordinates. |
| **Nowicki & Wietrzykowski (2017); Kim et al. (2018)** | Stacked autoencoder + MLP on UJIIndoorLoc for building/floor (Kim et al. use one multi-label output). | Deep learning baselines for Phase 2. |
| **Song et al., CNNLoc (IEEE Access 2019)** | Autoencoder + 1-D CNN; ~96% floor accuracy on UJIIndoorLoc. | Upper reference point for what DL achieves. |

## 3. Gaps that shape our design

1. **Missing values are not missing at random.** "Not heard" means the signal is below the phone's
   sensitivity, so the RSSI is *very weak*, not unknown. Treating +100 as a number (as naive
   pipelines do) tells a model that undetected APs are the strongest. We impute just below the weakest
   observed value and add explicit "heard" indicators.
2. **Evaluation leakage.** Random K-fold splits put scans from the same walk, user and phone in both
   train and validation and overstate accuracy. We tune with **GroupKFold by user** and test on the
   official validation file, which was collected 4 months later by different users with partly
   different phones.
3. **Independent targets.** Most ML baselines train separate models for building, floor and
   coordinates, so they cannot exploit the fact that the building constrains the floor and the
   location. We compare an independent and a **hierarchical** (building → per-building model) design.
   Phase 2 replaces this hand-built hierarchy with a learned shared representation (multitask MLP).
4. **Device heterogeneity.** Different phones report different RSSI for the same place. We break
   down error by phone to measure this.

## References

1. P. Bahl and V. N. Padmanabhan, "RADAR: An in-building RF-based user location and tracking system," *IEEE INFOCOM*, 2000.
2. M. Youssef and A. Agrawala, "The Horus WLAN location determination system," *MobiSys*, 2005.
3. J. Torres-Sospedra et al., "UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems," *IPIN*, 2014, pp. 261–270.
4. J. Torres-Sospedra, R. Montoliu, S. Trilles, Ó. Belmonte, and J. Huerta, "Comprehensive analysis of distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems," *Expert Systems with Applications*, vol. 42, no. 23, pp. 9263–9278, 2015.
5. J. Torres-Sospedra et al., "The smartphone-based offline indoor location competition at IPIN 2016: Analysis and future work," *Sensors*, vol. 17, no. 3, 557, 2017.
6. S. Bozkurt, G. Elibol, S. Gunal, and U. Yayan, "A comparative study on machine learning algorithms for indoor positioning," *INISTA*, 2015.
7. B. A. Akram, A. H. Akbar, and O. Shafiq, "HybLoc: Hybrid indoor Wi-Fi localization using soft clustering-based random decision forest ensembles," *IEEE Access*, vol. 6, pp. 38251–38272, 2018.
8. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
9. J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.
10. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *KDD*, 2016.
11. T. Cover and P. Hart, "Nearest neighbor pattern classification," *IEEE Trans. Information Theory*, vol. 13, no. 1, pp. 21–27, 1967.
12. R. Caruana, "Multitask learning," *Machine Learning*, vol. 28, no. 1, pp. 41–75, 1997.
13. M. Nowicki and J. Wietrzykowski, "Low-effort place recognition with WiFi fingerprints using deep learning," *Automation 2017*, Springer AISC vol. 550, 2017.
14. K. S. Kim, S. Lee, and K. Huang, "A scalable deep neural network architecture for multi-building and multi-floor indoor localization based on Wi-Fi fingerprinting," *Big Data Analytics*, vol. 3, 4, 2018.
15. X. Song et al., "A novel convolutional neural network based indoor localization framework with WiFi fingerprinting," *IEEE Access*, vol. 7, 2019.
16. F. Zafari, A. Gkelias, and K. K. Leung, "A survey of indoor localization systems and technologies," *IEEE Communications Surveys & Tutorials*, vol. 21, no. 3, pp. 2568–2599, 2019.
17. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.
