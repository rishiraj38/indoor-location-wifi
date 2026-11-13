# Contributing

Thank you for your interest in contributing to this project!

## Setting up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the project

| Step | Command |
|---|---|
| Download raw data | `python scripts/download_data.py` |
| Build processed data | `python scripts/make_processed_data.py` |
| Run EDA figures | `python scripts/run_eda.py` |
| Run all experiments | `python scripts/run_experiments.py` |
| Generate LaTeX tables | `python scripts/make_tables.py` |
| Open the Phase 1 notebook | `jupyter notebook notebooks/phase1_ml_baseline.ipynb` |
| Phase 2 experiments | see the table under Phase 2 workflow |
| Open the Phase 2 notebook | `jupyter notebook notebooks/phase2_deep_learning.ipynb` |

## Branch naming

- `feature/<short-description>`: new feature or experiment
- `fix/<short-description>`: bug or data fix
- `docs/<short-description>`: documentation-only change
- `phase2-<name>`: a member's Phase 2 branch

## Coding style

- Follow PEP 8.
- All randomness must use `random_state=42` so results are reproducible.
- Keep functions short and well-documented.
- Do not commit raw or processed data files (see `.gitignore`).

## Submitting changes

1. Fork the repository and create your branch from `main`.
2. Make your changes and ensure the notebook runs end-to-end without errors.
3. Open a pull request with a clear description of what was changed and why.

## Phase 2 workflow

1. Update your copy: `git checkout main && git pull`
2. Make your branch: `git checkout -b phase2-<yourname>`
3. Edit only the files you own. Commit small and often.
4. Push the branch: `git push -u origin phase2-<yourname>`
5. Open a pull request to `main` on GitHub. Rishi reviews it on the pull request and merges once it is approved.

Never push to `main` directly, never `git push --force`, never rebase or rewrite shared history.
If `git push` is rejected, run `git pull` and push again; do not force it.

Phase 2 scripts:

| Step | Command | Owner |
|---|---|---|
| Loss weight and patience sweep | `python scripts/tune_network.py` | Rishi Raj |
| Architecture comparison | `python scripts/compare_architectures.py` | Rishi Raj |
| Scaling and regularisation sweep | `python scripts/run_dl_regularisation.py` | Yash Yadav |
| Ablation studies | `python scripts/run_dl_ablations.py` | Aditya Samadhiya |
| Hybrid models | `python scripts/run_hybrid.py` | Aditya Samadhiya, Rishi Raj |
| Final results and error analysis | `python scripts/run_dl_analysis.py` | Yash Raj |
