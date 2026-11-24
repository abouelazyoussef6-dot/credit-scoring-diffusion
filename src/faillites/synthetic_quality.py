"""Étude 2 — qualité des faillites synthétiques et utilité en remplacement (TSTR).

Même protocole que src.synthetic_quality, sur les 5 plis de la répétition 0, avec une
ligne de contrôle (vraies faillites mises de côté). Les générateurs ne sont pas
réentraînés : ce sont ceux de l'expérience principale (cache).

Usage : python -m src.faillites.synthetic_quality   (après src.faillites.experiment)
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.faillites.config import FEATURES, FIG, LABELS, N_FOLDS, TAB
from src.faillites.data import load_dev, split_xy
from src.faillites.features import Prep, slog
from src.faillites.generators import GENERATORS, get_synthetic
from src.faillites.quality import quality_report
from src.faillites.resampling import smote_blocks
from src.faillites.tuning import load_params
from src.metrics import MAIN_METRIC, evaluate
from src.models import make_classifier
from src.plots import METHOD_COLORS, set_style
from src.resampling import n_to_add

warnings.filterwarnings("ignore")
METHODS = ["SMOTE"] + GENERATORS
CLFS = ("LogReg", "XGBoost", "LightGBM")
SHOWN = ["X2", "X29", "X46", "X27"]


def smote_raw(Xtr, y, prep, n, seed):
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    N, C, _ = smote_blocks(Xtr, y, prep, (n_pos + n) / n_neg, seed)
    return prep.inverse_blocks(N[len(Xtr):], C[len(Xtr):])


def replacement_utility(A_neg, X_def, B, yva, seed, params):
    X = np.vstack([A_neg, X_def]); y = np.r_[np.zeros(len(A_neg), dtype=int), np.ones(len(X_def), dtype=int)]
    return {f"PR-AUC {k}": evaluate(yva, make_classifier(k, seed, params=params[k]).fit(X, y)
                                    .predict_proba(B)[:, 1])[MAIN_METRIC] for k in CLFS}


def fig_marginals(real, fakes):
    fig, axes = plt.subplots(len(SHOWN), len(METHODS), figsize=(14, 2.3 * len(SHOWN)))
    for i, c in enumerate(SHOWN):
        r = slog(real[c].dropna())
        lo, hi = np.percentile(r, [1, 99])
        bins = np.linspace(lo, hi, 31)
        for j, m in enumerate(METHODS):
            ax = axes[i, j]
            ax.hist(r.clip(lo, hi), bins=bins, density=True, color="#cccccc", label="Vraies faillites")
            ax.hist(slog(fakes[m][c].dropna()).clip(lo, hi), bins=bins, density=True, histtype="step", lw=2,
                    color=METHOD_COLORS[m], label=m)
            ax.set_yticks([]); ax.grid(False)
            if i == 0:
                ax.set_title(m)
            if j == 0:
                ax.set_ylabel(f"{c}\n" + LABELS[c][:28], fontsize=8)
    fig.suptitle("Distributions de quatre ratios (échelle logarithmique signée) : vraies faillites (gris) "
                 "et faillites synthétiques (trait)", y=1.0)
    fig.tight_layout(); fig.savefig(FIG / "qualite_distributions.png"); plt.close(fig)


def fig_correlations(real, fakes):
    fig, axes = plt.subplots(1, 1 + len(METHODS), figsize=(16, 3.6))
    ref = real[FEATURES].corr("spearman").values
    for ax, (name, d) in zip(axes, [("Vraies faillites", real)] + [(m, fakes[m]) for m in METHODS]):
        corr = d[FEATURES].corr("spearman").values
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        gap = np.nanmean(np.abs(corr - ref)[np.triu_indices(len(FEATURES), k=1)])
        ax.set_title(name if name == "Vraies faillites" else f"{name}\nécart moyen : {gap:.2f}", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Corrélation de Spearman")
    fig.savefig(FIG / "qualite_correlations.png"); plt.close(fig)


def main() -> None:
    set_style()
    dev, params = load_dev(), load_params()
    rows, tstr, first = [], [], None
    for k in range(N_FOLDS):
        Xtr, ytr = split_xy(dev[dev["fold_r0"] != k]); Xva, yva = split_xy(dev[dev["fold_r0"] == k])
        y = ytr.values
        real, holdout = Xtr[y == 1], Xva[yva.values == 1]
        prep = Prep().fit(Xtr)
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        n_syn = n_to_add("éq.", n_pos, n_neg)
        fakes = {"SMOTE": smote_raw(Xtr, y, prep, n_syn, k)}
        for g in GENERATORS:
            fakes[g] = get_synthetic(g, f"r0f{k}", real, n_syn, 1000 * 0 + k)
        ctrl = real.sample(len(holdout), random_state=k)
        rows.append({"pli": k, "méthode": "Vraies faillites inédites (contrôle)",
                     **quality_report(real.drop(ctrl.index), holdout, ctrl, prep)})
        A, B = prep.transform(Xtr), prep.transform(Xva)
        tstr.append({"pli": k, "faillites d'entraînement": "Vraies faillites",
                     **replacement_utility(A[y == 0], A[y == 1], B, yva.values, k, params)})
        for m in METHODS:
            rows.append({"pli": k, "méthode": m, **quality_report(real, holdout, fakes[m], prep)})
            tstr.append({"pli": k, "faillites d'entraînement": m, **replacement_utility(
                A[y == 0], prep.transform(fakes[m].iloc[:n_pos]), B, yva.values, k, params)})
        if first is None:
            first = (real, fakes)
        print(f"pli {k} terminé", flush=True)

    runs = pd.DataFrame(rows)
    runs.round(4).to_csv(TAB / "synthetic_quality_runs.csv", index=False)
    table = (runs.groupby("méthode").mean(numeric_only=True).drop(columns="pli")
             .reindex(["Vraies faillites inédites (contrôle)"] + METHODS))
    table.round(3).to_csv(TAB / "synthetic_quality.csv")
    print(table.round(3).to_string())
    t = pd.DataFrame(tstr)
    t.round(4).to_csv(TAB / "tstr_runs.csv", index=False)
    tt = (t.groupby("faillites d'entraînement").mean(numeric_only=True).drop(columns="pli")
          .reindex(["Vraies faillites"] + METHODS))
    for kind in CLFS:
        tt[f"part conservée {kind} (%)"] = 100 * tt[f"PR-AUC {kind}"] / tt.loc["Vraies faillites", f"PR-AUC {kind}"]
    tt.round(3).to_csv(TAB / "tstr.csv")
    print("\n", tt.round(3).to_string())
    fig_marginals(*first); fig_correlations(*first)


if __name__ == "__main__":
    main()
