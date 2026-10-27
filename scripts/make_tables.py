"""Write LaTeX tables for the report from the CSVs in results/."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "report" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)
    print(OUT / f"{name}.tex")


def fmt_pct(v):
    return f"{100 * v:.2f}"


# Main results
t = pd.read_csv(RES / "test_metrics.csv")
best_err = t.mean_error_m.min()
best_bf = t.building_floor_acc.max()
rows = []
for _, r in t.iterrows():
    err = f"{r.mean_error_m:.2f}"
    bf = fmt_pct(r.building_floor_acc)
    if r.mean_error_m == best_err:
        err = rf"\textbf{{{err}}}"
    if r.building_floor_acc == best_bf:
        bf = rf"\textbf{{{bf}}}"
    rows.append(f"{r.model} & {fmt_pct(r.building_acc)} & {fmt_pct(r.floor_acc)} & {bf} & "
                f"{r.mae_longitude_m:.2f} & {r.mae_latitude_m:.2f} & "
                f"{err} & {r.median_error_m:.2f} & {r.p75_error_m:.2f} \\\\")
write("main_results", r"""\begin{table*}[t]
\centering
\caption{Test-set results (official UJIIndoorLoc validation file, 1{,}111 scans). Accuracies in \%, errors in metres. MAE lon./lat. $=$ mean absolute error of each coordinate.}
\label{tab:main}
\begin{tabular}{lcccccccc}
\toprule
Model & Building & Floor & Bldg+Floor & MAE lon. & MAE lat. & Mean err. & Median & P75 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# Feature comparison
a = pd.read_csv(RES / "ablation_features.csv")
blocks = []
for model, sub in a.groupby("model", sort=False):
    blocks.append(rf"\multicolumn{{4}}{{l}}{{\emph{{{model}}}}} \\")
    blocks += [f"{r.features} & {r.cv_mean_error_m:.2f} & {r.test_mean_error_m:.2f} & "
               f"{fmt_pct(r.test_building_floor_acc)} \\\\" for _, r in sub.iterrows()]
    blocks.append(r"\midrule")
write("ablation", r"""\begin{table}[t]
\centering
\caption{Feature comparison. CV $=$ 5-fold GroupKFold by user on the training set (used to choose); test shown for reference only.}
\label{tab:ablation}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccc}
\toprule
Features & CV err. (m) & Test err. (m) & Test B+F \% \\
\midrule
""" + "\n".join(blocks[:-1]) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# Tuning
tu = pd.read_csv(RES / "tuning_cv.csv")
rows = []
for _, r in tu.iterrows():
    ps = ", ".join(f"{k.replace('_', r'\_')}={v}" for k, v in json.loads(r.params).items())
    rows.append(f"{r.model} & {ps} & {fmt_pct(r.building_floor_acc)} & {r.mean_error_m:.2f} \\\\")
write("tuning", r"""\begin{table}[t]
\centering
\caption{Hyper-parameter search with user-grouped 5-fold CV on the training set.}
\label{tab:tuning}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcc}
\toprule
Model & Parameters & B+F \% & Mean err. (m) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# Hierarchical breakdown (best tree model)
s = json.loads((RES / "summary.json").read_text())
h = pd.read_csv(RES / "breakdown_hierarchical_rf.csv")
rows = [f"{r.case} & {r['count']} & {r['mean']:.2f} & {r['median']:.2f} & {100 * r.share_of_total_error:.1f} \\\\"
        for _, r in h.iterrows()]
write("breakdown", r"""\begin{table}[t]
\centering
\caption{Error split by whether building and floor were right, """ + s["main_model"] + r""" (test set).}
\label{tab:breakdown}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
Case & $n$ & Mean (m) & Median (m) & Share of total error \% \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
