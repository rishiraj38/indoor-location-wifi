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

Rishi runs every experiment and owns the model; the others each own one small code change, one written
document built from the results files, and their own slides and viva part.

| Member | Owns | Files | Done when |
|---|---|---|---|
| Rishi Raj | Network architecture and training; loss balance and patience tuning; the regularisation, ablation and hybrid runs; final settings; final test run and all figures; notebook; report results and discussion; deck; submission | `src/indoorloc/deep.py`, `scripts/tune_network.py`, `scripts/run_dl_*.py`, `scripts/run_hybrid.py`, `results/dl_best.json`, `notebooks/phase2_deep_learning.ipynb`, `report/phase2.tex` (V to VII) | Network beats Phase 1 on floor and matches it on position; all result files and figures exist; notebook and report build |
| Yash Yadav | Scaling and regularisation write-up, plus the powed scaling option | `src/indoorloc/deep_data.py` (add `method="powed"` to `RSSIScaler`), `docs/dl_regularisation.md` | Powed option merged and included in the sweep; one-page doc explains what each setting does and which won, with numbers from `results/dl_regularisation.csv` |
| Aditya Samadhiya | Ablation and hybrid write-ups, plus one extra ablation | `scripts/run_dl_ablations.py` (add the "no early stopping" ablation), `docs/dl_ablations.md`, `docs/hybrid.md` | Extra ablation merged; both docs explain each result from `results/dl_ablations.csv` and `results/hybrid.csv`, with an honest hybrid verdict |
| Yash Raj | Results write-up and figure check | `docs/dl_results.md`, any figure fixes in `scripts/run_dl_analysis.py` | Doc compares Phase 1 and Phase 2 side by side from `results/aml_vs_dl.csv`; figures match Phase 1 style |
| Kaustubh Ranjan Mishra | DL literature review, theory notes, report sections I to IV | `docs/dl_literature_review.md`, `docs/dl_theory.md`, `report/phase2.tex` (I to IV) | Both docs complete; report draft compiles |

Everyone: your own slides and your part of the viva.

## Deadline and order of work

Evaluation: Thursday 19 November 2026, 09:30 to 09:55, room A411 (team T108). The Google Form with the
presentation (PPTX or PDF, under 10 MB), the report (PDF from LaTeX, under 10 MB) and the GitHub link must be
submitted by 08:30 that morning, once, by one member using a university email.

Dependencies decide the order, not weeks:

1. Rishi: tuning and regularisation runs, then `results/dl_best.json`, then ablations, hybrid and the final analysis. Results files appear in `results/` as each run finishes.
2. Yash Yadav, Aditya, Yash Raj: your small code change can go in any time; your doc is written once the matching results file exists. Kaustubh: start now, nothing blocks you.
3. Last: report, slides, notebook, viva practice. Everyone reads every doc.

Needed by: all result files by 13 November; everyone's docs and code by 15 November; everything merged by 17 November, leaving a day to check.

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
