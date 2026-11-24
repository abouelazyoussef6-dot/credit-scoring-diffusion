"""Étude 2 — analyse statistique de la validation croisée (mêmes calculs que src.analysis :
moyennes sur 15 découpages, gain apparié, test t corrigé de Nadeau-Bengio, Holm).

Usage : python -m src.faillites.analysis
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import src.analysis as A
from src.faillites.config import FIG, TAB
from src.metrics import MAIN_METRIC
from src.plots import GREY, METHOD_COLORS, METHOD_MARKERS, set_style

A.FIG_DIR, A.TAB_DIR = FIG, TAB      # mêmes fonctions, dossiers de l'étude 2


def fig_dose(summary) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    for ax, clf in zip(axes, A.CLFS):
        g = summary[summary["classifieur"] == clf]
        base = g[g["méthode"] == "Aucune"].iloc[0]
        ax.axhline(base[MAIN_METRIC], color=GREY, ls="--", lw=1.5, label="Aucune")
        ax.axhspan(base[MAIN_METRIC] - base[f"{MAIN_METRIC} (écart-type)"],
                   base[MAIN_METRIC] + base[f"{MAIN_METRIC} (écart-type)"], color=GREY, alpha=0.10, lw=0)
        for m in ["SMOTE", "TVAE", "CTGAN", "TabDDPM"]:
            d = g[g["méthode"] == m].sort_values("part de défauts (%)")
            ax.plot(d["part de défauts (%)"], d[MAIN_METRIC], color=METHOD_COLORS[m], lw=2, ms=7,
                    marker=METHOD_MARKERS[m], markeredgecolor="white", markeredgewidth=1.5, label=m)
        ax.set_xscale("log"); ax.set_xticks([10, 20, 50]); ax.set_xticklabels(["10", "20", "50"])
        ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlim(8, 56)
        ax.set_xlabel("Part de faillites dans l'entraînement après ajout (%)")
        ax.set_title(clf)
    axes[0].set_ylabel("PR-AUC (validation croisée)")
    axes[0].legend(loc="lower left")
    fig.suptitle("Effet de la quantité de faillites synthétiques ajoutées (4,7 % : aucun ajout ; "
                 "50 % : équilibre parfait ; bande grise : ± 1 écart-type sans ajout)", y=1.0)
    fig.tight_layout(); fig.savefig(FIG / "cv_dose.png"); plt.close(fig)


def main() -> None:
    set_style()
    runs = A.load_runs()
    summary = A.summarize(runs)
    best = A.best_per_method(summary)
    summary.round(4).to_csv(TAB / "cv_summary.csv", index=False)
    best.round(4).to_csv(TAB / "cv_best.csv", index=False)
    A.fig_models(summary); A.fig_threshold(summary); A.fig_gains(best); fig_dose(summary)
    n = runs.groupby("classifieur")[A.KEYS].apply(lambda d: len(d.drop_duplicates())).max()
    print(f"{n} découpages analysés.\n")
    cols = ["config", MAIN_METRIC, f"{MAIN_METRIC} (écart-type)", "ROC-AUC", "KS", "gain", "IC bas", "IC haut",
            "p-valeur", "p-valeur (Holm)", "découpages gagnés"]
    for clf in A.CLFS:
        print(f"=== {clf} : meilleure dose de chaque méthode ===")
        print(best[best["classifieur"] == clf][cols].round(4).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
