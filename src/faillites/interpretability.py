"""Étude 2 — interprétabilité du modèle retenu (SHAP sur le jeu de test).

- importance de chaque ratio (valeur + indicateur « manquant » regroupés) ;
- sens des effets pour les ratios les plus importants (nuage SHAP) ;
- comparaison avec le même classifieur entraîné avec des faillites TabDDPM : le modèle
  change-t-il de logique ?

Usage : python -m src.faillites.interpretability   (après src.faillites.final_evaluation)
"""
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.faillites.config import FEATURES, FIG, LABELS, MODEL_FILE, TAB
from src.faillites.data import load_dev, load_test, split_xy
from src.faillites.final_evaluation import fit_predict, parse_dose
from src.plots import AQUA, BLUE, set_style

warnings.filterwarnings("ignore")
TOP = 15



def short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"

def shap_values(model, A):
    if hasattr(model, "coef_"):
        return model.coef_[0] * (A - A.mean(axis=0))
    v = shap.TreeExplainer(model)(A).values
    return v[:, :, 1] if v.ndim == 3 else v


def grouped(values: np.ndarray, names: list[str]) -> pd.Series:
    imp = pd.Series(np.abs(values).mean(axis=0), index=names)
    return imp.groupby(lambda c: c.replace("_manquant", "")).sum().reindex(FEATURES).fillna(0)


def main() -> None:
    set_style()
    bundle = joblib.load(MODEL_FILE)
    prep, model = bundle["prep"], bundle["model"]
    Xte, yte = split_xy(load_test())
    A = prep.transform(Xte)
    V = shap_values(model, A)
    imp = grouped(V, prep.names).sort_values(ascending=False)

    # Même classifieur avec la meilleure dose TabDDPM (validation croisée)
    best = pd.read_csv(TAB / "cv_best.csv", keep_default_na=False)
    row = best[(best["classifieur"] == bundle["classifier"]) & (best["méthode"] == "TabDDPM")].iloc[0]
    Xdev, ydev = split_xy(load_dev())
    m2, _ = fit_predict(bundle["classifier"], "TabDDPM", parse_dose(row["dose"]), Xdev, ydev.values, prep, A, 0, "final")
    imp2 = grouped(shap_values(m2, A), prep.names).reindex(imp.index)
    rho = float(np.corrcoef(imp, imp2)[0, 1])
    top_overlap = len(set(imp.index[:10]) & set(imp2.sort_values(ascending=False).index[:10]))

    table = pd.DataFrame({"ratio": imp.index, "définition": [LABELS[c] for c in imp.index],
                          "importance SHAP": imp.values, "importance SHAP (modèle + TabDDPM)": imp2.values})
    table.round(4).to_csv(TAB / "importance_ratios.csv", index=False)
    print(table.head(TOP).round(3).to_string(index=False))
    print(f"\nCorrélation des importances avec le modèle entraîné sur faillites TabDDPM ({row['config']}) : "
          f"{rho:.2f} ; ratios communs dans les 10 premiers : {top_overlap}/10")
    pd.Series({"correlation_importances_tabddpm": rho, "top10_communs": top_overlap,
               "config_tabddpm": row["config"]}).to_json(TAB / "interpretabilite.json", force_ascii=False)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    d = imp.head(TOP)[::-1]
    ax.barh([f"{c} · {short(LABELS[c], 46)}" for c in d.index], d.values, color=BLUE)
    ax.set_xlabel("Importance SHAP moyenne (log-cote)")
    ax.set_title(f"Ratios les plus utilisés par le modèle retenu ({bundle['classifier']}), jeu de test")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "importance_ratios.png"); plt.close(fig)

    cols = [prep.names.index(c) for c in imp.index[:12]]
    plt.figure()
    shap.summary_plot(V[:, cols], A[:, cols], feature_names=[f"{c} · {short(LABELS[c], 34)}" for c in imp.index[:12]],
                      show=False, plot_size=(9, 5.6), color_bar_label="Valeur du ratio")
    plt.title("Sens des effets : chaque point est une entreprise du test (rouge = valeur élevée)", fontsize=10)
    plt.savefig(FIG / "shap_beeswarm.png", bbox_inches="tight", dpi=150); plt.close()


if __name__ == "__main__":
    main()
