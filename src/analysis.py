"""Analyse des résultats de la validation croisée : tableaux, tests statistiques, figures.

Pour chaque classifieur et chaque méthode (avec sa dose) :
- moyenne et écart-type des métriques sur les 15 découpages ;
- gain de PR-AUC par rapport à « Aucune », calculé découpage par découpage (différences
  appariées), avec intervalle de confiance à 95 % et p-valeur du test t corrigé pour
  validation croisée répétée (Nadeau & Bengio) ;
- p-valeurs ajustées par la méthode de Holm quand on compare plusieurs méthodes à la fois
  (plus on fait de comparaisons, plus on a de chances d'en trouver une « significative »
  par hasard).

Usage : python -m src.analysis
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import FIG_DIR, N_FOLDS, TAB_DIR
from src.metrics import MAIN_METRIC, corrected_ttest
from src.plots import FAMILIES, FAMILY_COLOR, GREY, METHOD_COLORS, METHOD_MARKERS, set_style

KEYS = ["répétition", "pli"]
METRICS = ["PR-AUC", "ROC-AUC", "KS", "F1 max", "Rappel (seuil 0,5)",
           "Précision (seuil 0,5)"]
ORDER = ["Aucune", "Pondération", "Sous-échantillonnage", "Duplication", "SMOTE",
         "TVAE", "CTGAN", "TabDDPM"]
CLFS = ["LogReg", "XGBoost", "LightGBM"]


def load_runs() -> pd.DataFrame:
    files = sorted(TAB_DIR.glob("cv_runs_r*.csv"))
    runs = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)
    runs["config"] = np.where(runs["dose"] == "", runs["méthode"],
                              runs["méthode"] + " (" + runs["dose"].astype(str) + ")")
    return runs


def holm(pvalues: pd.Series) -> pd.Series:
    """Ajustement de Holm : contrôle le risque d'au moins une fausse découverte."""
    order = pvalues.sort_values().index
    m, adj, running = len(order), {}, 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[idx]))
        adj[idx] = running
    return pd.Series(adj).reindex(pvalues.index)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for clf, g in runs.groupby("classifieur"):
        base = g[g["méthode"] == "Aucune"].set_index(KEYS)[MAIN_METRIC]
        for (method, dose, config), c in g.groupby(["méthode", "dose", "config"], sort=False):
            row = {"classifieur": clf, "méthode": method, "dose": dose, "config": config,
                   "découpages": len(c), "part de défauts (%)": c["part de défauts (%)"].mean()}
            for m in METRICS:
                row[m], row[f"{m} (écart-type)"] = c[m].mean(), c[m].std()
            if method != "Aucune":
                d = (c.set_index(KEYS)[MAIN_METRIC] - base).dropna().values
                t, p = corrected_ttest(d, N_FOLDS)
                half = stats.t.ppf(0.975, len(d) - 1) * np.sqrt(
                    (1 / len(d) + 1 / (N_FOLDS - 1)) * d.var(ddof=1))
                row.update({"gain": d.mean(), "IC bas": d.mean() - half, "IC haut": d.mean() + half,
                            "p-valeur": p, "découpages gagnés": int((d > 0).sum())})
            rows.append(row)
    out = pd.DataFrame(rows)
    out["_o"] = out["méthode"].map({m: i for i, m in enumerate(ORDER)})
    return out.sort_values(["classifieur", "_o"], kind="stable").drop(columns="_o").reset_index(drop=True)


def best_per_method(summary: pd.DataFrame) -> pd.DataFrame:
    """Meilleure dose de chaque méthode (PR-AUC moyenne), avec p-valeurs ajustées (Holm)."""
    best = summary.loc[summary.groupby(["classifieur", "méthode"], sort=False)[MAIN_METRIC].idxmax()].copy()
    best["p-valeur (Holm)"] = np.nan
    for clf, g in best.groupby("classifieur"):
        p = g["p-valeur"].dropna()
        best.loc[p.index, "p-valeur (Holm)"] = holm(p)
    return best.reset_index(drop=True)


def fig_models(summary: pd.DataFrame) -> None:
    sub = summary[summary["méthode"].isin(["Aucune", "Pondération"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (method, color) in enumerate([("Aucune", GREY), ("Pondération", METHOD_COLORS["Pondération"])]):
        d = sub[sub["méthode"] == method].set_index("classifieur").reindex(CLFS)
        x = np.arange(len(CLFS)) + (i - 0.5) * 0.36
        ax.bar(x, d[MAIN_METRIC], width=0.34, color=color, label=method,
               yerr=d[f"{MAIN_METRIC} (écart-type)"], error_kw=dict(lw=1, capsize=3, ecolor="#444"))
        for xi, v in zip(x, d[MAIN_METRIC]):
            ax.text(xi, 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(range(len(CLFS))); ax.set_xticklabels(["Régression logistique", "XGBoost", "LightGBM"])
    ax.set_ylabel("PR-AUC (validation croisée)"); ax.set_ylim(0, 1.08)
    ax.set_title("Modèle de référence et modèles avancés, sans création de données")
    ax.legend(loc="upper left", ncol=2); ax.grid(axis="x", visible=False)
    fig.savefig(FIG_DIR / "cv_modeles.png"); plt.close(fig)


def fig_gains(best: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
    methods = [m for m in ORDER if m != "Aucune"][::-1]
    for ax, clf in zip(axes, CLFS):
        d = best[(best["classifieur"] == clf)].set_index("méthode").reindex(methods)
        for y, (m, r) in enumerate(d.iterrows()):
            color = FAMILY_COLOR[m]
            ax.plot([r["IC bas"], r["IC haut"]], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.plot(r["gain"], y, "o", color=color, ms=8, markeredgecolor="white", markeredgewidth=1.5)
        ax.axvline(0, color="#444", lw=1)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        base = best[(best["classifieur"] == clf) & (best["méthode"] == "Aucune")][MAIN_METRIC].iloc[0]
        ax.set_title(f"{clf} (sans rééquilibrage : {base:.3f})")
        ax.set_xlabel("Gain de PR-AUC (IC à 95 %)")
        ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=c, marker="o", lw=2, ms=7, markeredgecolor="white", label=name)
               for name, (c, _) in FAMILIES.items()]
    for ax in axes:
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Gain par rapport à l'absence de rééquilibrage, à la meilleure dose de chaque méthode "
                 "(à droite de 0 : la méthode aide)", y=1.0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "cv_gains.png"); plt.close(fig)


def fig_dose(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    for ax, clf in zip(axes, CLFS):
        g = summary[summary["classifieur"] == clf]
        base = g[g["méthode"] == "Aucune"].iloc[0]
        ax.axhline(base[MAIN_METRIC], color=GREY, ls="--", lw=1.5, label="Aucune")
        ax.axhspan(base[MAIN_METRIC] - base[f"{MAIN_METRIC} (écart-type)"],
                   base[MAIN_METRIC] + base[f"{MAIN_METRIC} (écart-type)"], color=GREY, alpha=0.10, lw=0)
        for m in ["SMOTE", "TVAE", "CTGAN", "TabDDPM"]:
            d = g[g["méthode"] == m].sort_values("part de défauts (%)")
            ax.plot(d["part de défauts (%)"], d[MAIN_METRIC], color=METHOD_COLORS[m], lw=2, ms=7,
                    marker=METHOD_MARKERS[m], markeredgecolor="white", markeredgewidth=1.5, label=m)
        ax.set_xlabel("Part de défauts dans l'entraînement après ajout (%)")
        ax.set_title(clf)
    axes[0].set_ylabel("PR-AUC (validation croisée)")
    axes[0].legend(loc="lower left")
    fig.suptitle("Effet de la quantité de défauts synthétiques ajoutés "
                 "(20 % : aucun ajout ; 50 % : équilibre parfait ; bande grise : ± 1 écart-type sans ajout)", y=1.0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "cv_dose.png"); plt.close(fig)


def fig_threshold(summary: pd.DataFrame) -> None:
    """Au seuil naïf de 0,5, le rééquilibrage semble faire des miracles sur le rappel ;
    la PR-AUC montre que le modèle n'est pas meilleur pour autant."""
    configs = ["Aucune", "Pondération", "Sous-échantillonnage (1)", "SMOTE (éq.)", "TabDDPM (éq.)"]
    g = summary[(summary["classifieur"] == "LogReg")].set_index("config").reindex(configs)
    fig, ax = plt.subplots(figsize=(9.4, 3.8))
    x, w = np.arange(len(configs)), 0.26
    for i, (col, color, lab) in enumerate([("Rappel (seuil 0,5)", "#2a78d6", "Rappel au seuil 0,5"),
                                           ("Précision (seuil 0,5)", "#eb6834", "Précision au seuil 0,5"),
                                           (MAIN_METRIC, "#1baf7a", "PR-AUC (indépendante du seuil)")]):
        ax.bar(x + (i - 1) * w, g[col], width=w - 0.02, color=color, label=lab)
        for xi, v in zip(x + (i - 1) * w, g[col]):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(configs); ax.set_ylim(0, 1.18)
    ax.set_title("Régression logistique : rééquilibrer déplace le seuil,\nsans améliorer le classement des demandes", fontsize=10)
    ax.legend(loc="upper center", ncol=3, fontsize=8); ax.grid(axis="x", visible=False)
    fig.savefig(FIG_DIR / "cv_seuil.png"); plt.close(fig)


def main() -> None:
    set_style()
    runs = load_runs()
    summary = summarize(runs)
    best = best_per_method(summary)
    summary.round(4).to_csv(TAB_DIR / "cv_summary.csv", index=False)
    best.round(4).to_csv(TAB_DIR / "cv_best.csv", index=False)
    for f in (fig_models, fig_dose, fig_threshold):
        f(summary)
    fig_gains(best)

    n = runs.groupby("classifieur")[KEYS].apply(lambda d: len(d.drop_duplicates())).max()
    print(f"{n} découpages analysés.\n")
    cols = ["config", MAIN_METRIC, f"{MAIN_METRIC} (écart-type)", "ROC-AUC", "KS", "gain",
            "IC bas", "IC haut", "p-valeur", "p-valeur (Holm)", "découpages gagnés"]
    for clf in CLFS:
        print(f"=== {clf} : meilleure dose de chaque méthode ===")
        print(best[best["classifieur"] == clf][cols].round(4).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
