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
| Open the notebook | `jupyter notebook notebooks/phase1_ml_baseline.ipynb` |

## Branch naming

- `feature/<short-description>` — new feature or experiment
- `fix/<short-description>` — bug or data fix
- `docs/<short-description>` — documentation-only change

## Coding style

- Follow PEP 8.
- All randomness must use `random_state=42` so results are reproducible.
- Keep functions short and well-documented.
- Do not commit raw or processed data files (see `.gitignore`).

## Submitting changes

1. Fork the repository and create your branch from `main`.
2. Make your changes and ensure the notebook runs end-to-end without errors.
3. Open a pull request with a clear description of what was changed and why.
