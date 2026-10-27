# Environment & Setup Notes

## Python version

The project was developed and tested with **Python 3.11**.  
Earlier versions (3.9+) should also work, but have not been verified.

## Virtual environment

Always activate the virtual environment before running any script or notebook:

```bash
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
```

## Package versions

Pinned versions in `requirements.txt` were chosen for reproducibility.
If you update any package, re-run the full notebook and check that results match
the values in `results/`.

## Data directory

The `data/` directory is **not** stored in the repository.  
Run `python scripts/download_data.py` to fetch the UJIIndoorLoc dataset from UCI.  
After that, `python scripts/make_processed_data.py` builds the cleaned files under
`data/processed/`.

Expected directory layout after setup:

```
data/
  raw/
    UJIIndoorLoc/
      trainingData.csv
      validationData.csv
  processed/
    train.csv
    test.csv
```

## Reproducibility

All scripts and the notebook set `random_state=42` wherever randomness is used.
Do **not** change this value if you want to reproduce the published results exactly.

## Common issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: indoorloc` | Run `pip install -e .` or ensure `src/` is on `PYTHONPATH` |
| Missing figures in notebook | Run `python scripts/run_eda.py` first |
| `tectonic` not found for PDF build | Install from <https://tectonic-typesetting.github.io> or upload `report/` to Overleaf |
