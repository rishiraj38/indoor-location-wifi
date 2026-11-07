# Phase 2 Plan: Deep Learning

## What Phase 2 asks for

From the project brief: train a shared MLP trunk with classification heads for building and floor and a
regression head for coordinates. Compare it fairly with the Phase 1 models: same split (official test file,
cross-validation by user), same metrics (building and floor accuracy, coordinate MAE, hierarchical error
breakdown). Deliver a notebook with the ML vs DL comparison, floor confusion matrices and coordinate error maps.

Graded on: architecture logic, hybrid innovation (bonus), ablation studies, DL literature review, DL dataset
and regularisation, technical validation, theoretical rigor, GitHub repo and code quality, presentation,
viva (individual), IEEE report.

## Who does what

| Member | Owns | Files | Done when |
|---|---|---|---|
| Rishi Raj | Network architecture, training loop, final settings, notebook, integration | `src/indoorloc/deep.py`, `notebooks/phase2_deep_learning.ipynb`, `results/dl_best.json` | The network trains, beats Phase 1 on floor, and the notebook runs end to end |
| Yash Yadav | Input scaling and regularisation | `src/indoorloc/deep_data.py`, `scripts/run_dl_regularisation.py`, `docs/dl_regularisation.md` | `results/dl_regularisation.csv` exists, a best setting is chosen and explained |
| Aditya Samadhiya | Ablation studies and the hybrid | `scripts/run_dl_ablations.py`, `scripts/run_hybrid.py`, `docs/dl_ablations.md`, `docs/hybrid.md` | Ablation table and figure exist; each ablation explained; hybrid results in `results/hybrid.csv` with a verdict |
| Yash Raj | Test results, error analysis, ML vs DL comparison figures | `scripts/run_dl_analysis.py`, `docs/dl_results.md` | Confusion matrices, error map, training curve and comparison figure exist and match Phase 1 style |
| Kaustubh Ranjan Mishra | DL literature review, theory notes, report writing | `docs/dl_literature_review.md`, `docs/dl_theory.md`, `report/phase2.tex` (sections I to IV) | Both docs complete; report draft compiles |

Everyone: your own slides and your part of the viva.

## Order of work

1. Week 1: Yash Yadav runs the regularisation grid and picks a setting. Rishi fixes the loss weights and
   patience so position error is competitive. Kaustubh drafts the literature review and theory notes.
2. Week 1 to 2: Aditya runs ablations and the hybrid. Yash Raj runs the final analysis once
   `results/dl_best.json` exists.
3. Week 2: report, slides, notebook, viva practice. Everyone reads every doc.

Scripts are written so they run as they are. Your job is to run them, read the output, improve the parts
listed under "To do" at the top of your file, and write the short doc that explains the result.

## How the code fits together

- `MultitaskModel(...).fit(X, y, groups=USERID)` and `.predict(X)` behave like the Phase 1 models, so
  `summarize`, `hierarchical_breakdown` and `per_group_error` from `metrics.py` work unchanged.
- `validation.user_folds` and `validation.cv_eval` are the shared cross-validation. Use them; do not
  write a new split.
- `MultitaskModel.embed(X)` returns the trunk features, used by the hybrid.
- Settings flow through small JSON files in `results/`: `dl_best_regularisation.json` (Yash Yadav) feeds
  the ablations; `dl_best.json` (Rishi, after the ablations) feeds the analysis and the hybrid.

## Rules

- Work on your own branch (`phase2-<yourname>`), push it, open a pull request to `main`. Rishi merges.
  Never push to `main` directly, never force-push, never rebase shared history.
- Only edit the files you own. If you need a change in someone else's file, ask them in the group.
- Keep `random_state=42` / `seed=42`. Do not change Phase 1 code or results.
- Plain writing, like the Phase 1 docs: short sentences, no decorative symbols, no "authored by" lines in
  code (git history records who wrote what).
- Commit small and often, with a message that says what changed, for example
  "Add weight decay sweep to regularisation script".
- Every number you write in a doc must come from a results file in `results/`.
