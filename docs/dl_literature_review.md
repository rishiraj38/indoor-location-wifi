# Deep Learning Literature Review (Phase 2)

Owner: Kaustubh Ranjan Mishra.

Target: about one page. Plain language. For each paper: what they did, what result they got on
UJIIndoorLoc (if any), and what we take from it. Add the full reference at the bottom in the same style
as `literature_review.md`.

## 1. Multitask learning

- Caruana (1997), Multitask Learning. To write: the idea (shared hidden layers, several outputs trained
  together), why it helps (the extra tasks act as extra training signal and a regulariser), and how
  our building / floor / position heads map onto it.

## 2. Neural networks on UJIIndoorLoc

- Nowicki and Wietrzykowski (2017): stacked autoencoder + classifier for building and floor.
- Kim, Lee and Huang (2018): one network for building, floor and position, autoencoder front end.
- Song et al. (2019), CNNLoc: autoencoder + 1-D CNN; reported about 95% floor accuracy.
- To write for each: input handling (how they treat "not detected"), architecture, reported numbers,
  and how our simpler MLP compares.

## 3. Regularisation and training

- Dropout (Srivastava et al., 2014), weight decay, early stopping. One short paragraph each:
  what it does and why it matters with 465 inputs and 19k training scans.

## 4. Hybrid ML + DL

- One or two papers that combine learned features with tree models or nearest neighbours, to
  justify the hybrid in `scripts/run_hybrid.py`. HybLoc (Akram et al., 2018) is already in the Phase 1 review.

## 5. What this review tells us

- Three or four bullet points: the gaps the Phase 1 models left, what the literature suggests, and
  what our Phase 2 design does about it.

## References

(to add)
