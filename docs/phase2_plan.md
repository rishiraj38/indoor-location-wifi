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
| Rishi Raj | Network architecture, training loop, final settings, notebook, integration, merging, form submission | `src/indoorloc/deep.py`, `scripts/tune_network.py`, `results/dl_best.json`, `notebooks/phase2_deep_learning.ipynb`, `report/phase2.tex` (sections V to VII) | The network beats Phase 1 on floor and is competitive on position; the notebook runs end to end |
| Yash Yadav | Input scaling and regularisation: min-max vs standard vs powed scaling; dropout, weight decay, early-stopping patience | `src/indoorloc/deep_data.py`, `scripts/run_dl_regularisation.py`, `docs/dl_regularisation.md` | `results/dl_regularisation.csv` and `dl_best_regularisation.json` exist; one-page doc explains the choice |
| Aditya Samadhiya | Ablations (multitask vs single-task, loss weights, trunk size) and the hybrid (network features into the Random Forest; network for floor plus forest for position) | `scripts/run_dl_ablations.py`, `scripts/run_hybrid.py`, `docs/dl_ablations.md`, `docs/hybrid.md` | Ablation table and figure exist; each ablation explained; hybrid results in `results/hybrid.csv` with an honest verdict |
| Yash Raj | Final test run and error analysis of the network: confusion matrices, error map, training curve, error per phone, ML vs DL comparison figures | `scripts/run_dl_analysis.py`, `docs/dl_results.md` | All figures and result files exist, matching Phase 1 style; doc compares with Phase 1 |
| Kaustubh Ranjan Mishra | DL literature review, theory notes, first four sections of the report | `docs/dl_literature_review.md`, `docs/dl_theory.md`, `report/phase2.tex` (sections I to IV) | Both docs complete; report draft compiles |

Everyone: your own slides and your part of the viva.

## Deadline and order of work

Evaluation: Thursday 19 November 2026, 09:30 to 09:55, room A411 (team T108). The Google Form with the
presentation (PPTX or PDF, under 10 MB), the report (PDF from LaTeX, under 10 MB) and the GitHub link must be
submitted by 08:30 that morning, once, by one member using a university email. Rishi submits.

Dependencies decide the order:

1. First: Yash Yadav's regularisation results (Aditya's ablations read `results/dl_best_regularisation.json`) and
   Kaustubh's docs (nobody waits on him). Rishi tunes the loss balance in parallel.
2. Then: Rishi writes the final network settings to `results/dl_best.json`.
3. Then: Aditya's ablations and hybrid, and Yash Raj's final analysis (both read `dl_best.json`).
4. Last: report, slides, notebook, viva practice. Everyone reads every doc.

Needed by: steps 1 and 2 by 11 November, step 3 by 15 November, everything merged by 17 November, leaving a day to check.

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
