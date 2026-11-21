"""Qualité des défauts synthétiques : ressemblent-ils aux vrais, sans les recopier ?

L'expérience principale mesure l'UTILITÉ en complément (ajouter des défauts synthétiques
aide-t-il le classifieur ?). Ce script mesure, sur les 5 plis de la première répétition :

1. la fidélité et la confidentialité (définitions dans src.quality). Pour chaque pli, les
   générateurs ont été entraînés sur les défauts d'entraînement, et les défauts du pli de
   validation servent de référence « vrais clients jamais vus ». Une ligne de contrôle
   (« Vrais défauts inédits ») donne les valeurs obtenues par de vrais défauts mis de
   côté : c'est le meilleur résultat qu'un générateur puisse viser ;

2. l'utilité « en remplacement » (protocole TSTR, train on synthetic, test on real) : on
   retire TOUS les vrais défauts de l'entraînement et on les remplace par le même nombre
   de défauts synthétiques. Le classifieur n'a donc jamais vu un vrai défaut ; on l'évalue
   sur de vraies demandes. C'est le cas d'usage du partage de données : transmettre des
   dossiers synthétiques à la place des vrais. Le rapport entre cette PR-AUC et celle
   obtenue avec les vrais défauts dit quelle part de l'information utile le générateur a
   su conserver.

SMOTE est évalué de la même façon : ses points, créés dans l'espace des classifieurs,
sont ramenés au format brut.

Usage : python -m src.synthetic_quality   (après src.experiment)
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.decomposition import PCA

from src.config import FIG_DIR, LABELS, MISSING_LABEL, N_FOLDS, NUM_COLS, TAB_DIR
from src.features import IND_COLS, N_CAT_BLOCK, N_NUM, Preprocessor, load_dev, split_xy
from src.generators import GENERATORS, get_synthetic
from src.metrics import MAIN_METRIC, evaluate
from src.models import make_classifier
from src.plots import METHOD_COLORS, set_style
from src.quality import quality_report
from src.resampling import n_to_add

warnings.filterwarnings("ignore")
METHODS = ["SMOTE"] + GENERATORS
CLFS = ("LogReg", "XGBoost", "LightGBM")


def smote_raw(Xtr: pd.DataFrame, y: np.ndarray, prep: Preprocessor, n: int, seed: int) -> pd.DataFrame:
    """Défauts créés par SMOTE-NC, ramenés au format brut des demandes."""
    N, C = prep.blocks(Xtr)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    Z, _ = SMOTENC(categorical_features=list(range(N_NUM, N_NUM + N_CAT_BLOCK)),
                   sampling_strategy=(n_pos + n) / n_neg, random_state=seed).fit_resample(np.hstack([N, C]), y)
    Z = Z[len(N):]
    return prep.inverse_blocks(Z[:, :N_NUM], Z[:, N_NUM:].astype(int))


def replacement_utility(A_neg, X_defaults, B, yva, seed) -> dict:
    """PR-AUC sur la validation quand l'entraînement contient `X_defaults` pour seuls défauts."""
    X = np.vstack([A_neg, X_defaults])
    y = np.r_[np.zeros(len(A_neg), dtype=int), np.ones(len(X_defaults), dtype=int)]
    return {f"PR-AUC {kind}": evaluate(yva, make_classifier(kind, seed).fit(X, y).predict_proba(B)[:, 1])[MAIN_METRIC]
            for kind in CLFS}


def fig_marginals(real: pd.DataFrame, fakes: dict) -> None:
    cols = ["DEBTINC", "CLAGE", "DELINQ", "VALUE"]
    fig, axes = plt.subplots(len(METHODS), len(cols), figsize=(12.5, 8.2), sharex="col")
    for i, m in enumerate(METHODS):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            r, f = real[c].dropna(), fakes[m][c].dropna()
            if c == "VALUE":
                r, f = np.log10(r), np.log10(f.clip(lower=1))
            lo, hi = r.quantile(0.005), r.quantile(0.995)
            bins = np.arange(-0.5, 8.5) if c == "DELINQ" else np.linspace(lo, hi, 31)
            ax.hist(r.clip(bins[0], bins[-1]), bins=bins, density=True, color="#bdbdbd", label="Vrais défauts")
            ax.hist(f.clip(bins[0], bins[-1]), bins=bins, density=True, histtype="step", lw=1.8,
                    color=METHOD_COLORS[m], label=m)
            ax.set_yticks([]); ax.grid(False)
            if i == 0:
                ax.set_title({"VALUE": "Valeur du bien (log10)"}.get(c, LABELS[c]), fontsize=9)
            if j == 0:
                ax.set_ylabel(m, fontsize=11)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Distributions des vrais défauts (gris) et des défauts synthétiques (trait)", y=0.995)
    fig.tight_layout(); fig.savefig(FIG_DIR / "qualite_distributions.png"); plt.close(fig)


def fig_categorical(real: pd.DataFrame, fakes: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [7, 9]})
    sets = [("Vrais défauts", real, "#8a8a8a")] + [(m, fakes[m], METHOD_COLORS[m]) for m in METHODS]
    w = 0.16
    jobs = ["Other", "ProfExe", "Mgr", "Office", "Self", "Sales", MISSING_LABEL]
    for i, (name, d, color) in enumerate(sets):
        p = d["JOB"].fillna(MISSING_LABEL).value_counts(normalize=True).reindex(jobs).fillna(0)
        axes[0].bar(np.arange(len(jobs)) + (i - 2) * w, 100 * p, w, color=color, label=name)
        miss = 100 * d[IND_COLS].isna().mean()
        axes[1].bar(np.arange(len(IND_COLS)) + (i - 2) * w, miss, w, color=color, label=name)
    axes[0].set_xticks(range(len(jobs))); axes[0].set_xticklabels(jobs)
    axes[0].set_title("Catégorie professionnelle (% des défauts)"); axes[0].legend(fontsize=8)
    axes[1].set_xticks(range(len(IND_COLS))); axes[1].set_xticklabels(IND_COLS)
    axes[1].set_title("Valeurs manquantes par variable (% des défauts)")
    for ax in axes: ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "qualite_categorielles.png"); plt.close(fig)


def fig_correlations(real: pd.DataFrame, fakes: dict) -> None:
    fig, axes = plt.subplots(1, 1 + len(METHODS), figsize=(16, 3.8))
    ref = real[NUM_COLS].corr("spearman").values
    for ax, (name, d) in zip(axes, [("Vrais défauts", real)] + [(m, fakes[m]) for m in METHODS]):
        corr = d[NUM_COLS].corr("spearman").values
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(NUM_COLS))); ax.set_xticklabels(NUM_COLS, rotation=90, fontsize=6)
        ax.set_yticks(range(len(NUM_COLS))); ax.set_yticklabels(NUM_COLS if name == "Vrais défauts" else [], fontsize=6)
        ax.grid(False)
        gap = np.nanmean(np.abs(corr - ref)[np.triu_indices(len(NUM_COLS), k=1)])
        ax.set_title(name if name == "Vrais défauts" else f"{name}\nécart moyen : {gap:.2f}", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Corrélation de Spearman")
    fig.savefig(FIG_DIR / "qualite_correlations.png"); plt.close(fig)


def fig_projection(real: pd.DataFrame, fakes: dict, prep: Preprocessor) -> None:
    R = prep.transform(real)
    pca = PCA(n_components=2).fit(R)
    P = pca.transform(R)
    fig, axes = plt.subplots(1, len(METHODS), figsize=(15, 3.9), sharex=True, sharey=True)
    for ax, m in zip(axes, METHODS):
        F = pca.transform(prep.transform(fakes[m].sample(min(len(real), len(fakes[m])), random_state=0)))
        ax.scatter(F[:, 0], F[:, 1], s=10, color=METHOD_COLORS[m], alpha=0.55, linewidths=0, label=m)
        ax.scatter(P[:, 0], P[:, 1], s=10, facecolors="none", edgecolors="#333", linewidths=0.5, label="Vrais défauts")
        ax.set_title(m); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlim(np.percentile(P[:, 0], 0.5) - 1, np.percentile(P[:, 0], 99.5) + 1)
        ax.set_ylim(np.percentile(P[:, 1], 0.5) - 1, np.percentile(P[:, 1], 99.5) + 1)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Vrais et faux défauts projetés sur les deux premiers axes principaux "
                 f"({100 * pca.explained_variance_ratio_.sum():.0f} % de la variance des vrais défauts)", y=1.0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "qualite_projection.png"); plt.close(fig)


def main() -> None:
    set_style()
    dev = load_dev()
    rows, tstr, first = [], [], None
    for k in range(N_FOLDS):
        Xtr, ytr = split_xy(dev[dev["fold_r0"] != k])
        Xva, yva = split_xy(dev[dev["fold_r0"] == k])
        y = ytr.values
        real, holdout = Xtr[y == 1], Xva[yva.values == 1]
        prep = Preprocessor().fit(Xtr)
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        n_syn = n_to_add("éq.", n_pos, n_neg)
        fakes = {"SMOTE": smote_raw(Xtr, y, prep, n_syn, k)}
        for g in GENERATORS:
            fakes[g] = get_synthetic(g, f"r0f{k}", real, n_syn, k)

        # Contrôle : de vrais défauts mis de côté, traités comme s'ils étaient synthétiques
        ctrl = real.sample(len(holdout), random_state=k)
        rows.append({"pli": k, "méthode": "Vrais défauts inédits (contrôle)",
                     **quality_report(real.drop(ctrl.index), holdout, ctrl, prep)})
        A, B = prep.transform(Xtr), prep.transform(Xva)
        A_neg = A[y == 0]
        tstr.append({"pli": k, "défauts d'entraînement": "Vrais défauts",
                     **replacement_utility(A_neg, A[y == 1], B, yva.values, k)})
        for m in METHODS:
            rows.append({"pli": k, "méthode": m, **quality_report(real, holdout, fakes[m], prep)})
            tstr.append({"pli": k, "défauts d'entraînement": m, **replacement_utility(
                A_neg, prep.transform(fakes[m].iloc[:n_pos]), B, yva.values, k)})
        if first is None:
            first = (real, fakes, prep)
        print(f"pli {k} terminé", flush=True)

    runs = pd.DataFrame(rows)
    runs.round(4).to_csv(TAB_DIR / "synthetic_quality_runs.csv", index=False)
    table = (runs.groupby("méthode").mean(numeric_only=True).drop(columns="pli")
             .reindex(["Vrais défauts inédits (contrôle)"] + METHODS))
    table.round(3).to_csv(TAB_DIR / "synthetic_quality.csv")
    print("\nMoyenne sur les 5 plis. La ligne de contrôle donne les valeurs obtenues par de vrais défauts.\n")
    print(table.round(3).to_string())

    t = pd.DataFrame(tstr)
    t.round(4).to_csv(TAB_DIR / "tstr_runs.csv", index=False)
    tt = (t.groupby("défauts d'entraînement").mean(numeric_only=True).drop(columns="pli")
          .reindex(["Vrais défauts"] + METHODS))
    for kind in CLFS:
        tt[f"part conservée {kind} (%)"] = 100 * tt[f"PR-AUC {kind}"] / tt.loc["Vrais défauts", f"PR-AUC {kind}"]
    tt.round(3).to_csv(TAB_DIR / "tstr.csv")
    print("\nUtilité en remplacement : l'entraînement ne contient AUCUN vrai défaut, seulement des défauts "
          "synthétiques\n(autant que de vrais défauts retirés). PR-AUC sur de vraies demandes :\n")
    print(tt.round(3).to_string())

    fig_marginals(first[0], first[1])
    fig_categorical(first[0], first[1])
    fig_correlations(first[0], first[1])
    fig_projection(*first)


if __name__ == "__main__":
    main()
