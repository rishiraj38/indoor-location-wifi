# How we work on this repository

This is a five-member course project (Advanced Machine Learning, Project 9, Section D). These are the
rules we followed so that every result can be reproduced and every change can be traced to a person.

## Setting up

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/make_processed_data.py
```

## Running the project

| Step | Command | Phase |
|---|---|---|
| EDA figures | `python scripts/run_eda.py` | 1 |
| All classical experiments | `python scripts/run_experiments.py` | 1 |
| LaTeX tables for the Phase 1 report | `python scripts/make_tables.py` | 1 |
| Loss weight and patience sweep | `python scripts/tune_network.py` | 2 |
| Architecture comparison | `python scripts/compare_architectures.py` | 2 |
| Scaling and regularisation sweep | `python scripts/run_dl_regularisation.py` | 2 |
| Ablation studies | `python scripts/run_dl_ablations.py` | 2 |
| Hybrid models | `python scripts/run_hybrid.py` | 2 |
| Final test run and Phase 2 figures | `python scripts/run_dl_analysis.py` | 2 |
| Notebooks | `jupyter notebook notebooks/` | both |

## Rules

- Every model choice is made by 5-fold cross-validation grouped by user on the training file. The
  official validation file is the test set and is used once, for reporting.
- All randomness uses seed 42 (`random_state=42`, `torch.manual_seed(42)`).
- Every number in a document comes from a file in `results/`. If a script changes, its result files
  are regenerated and the documents updated.
- Phase 2 scripts read their settings from `results/dl_best.json`, so every result describes the
  same network.
- Data files, virtual environments and build output are not committed (see `.gitignore`).
- Plain writing in code comments and documents, so that every member can explain every line.

## Branches and pull requests

Each member works on their own branch (`phase2-<name>`) and opens a pull request to `main`. The
pull request is reviewed against the rules above, fixed if needed, approved and merged with a merge
commit, so each member's commits stay under their own name. Nobody pushes to `main` directly, and
history is never rewritten.
