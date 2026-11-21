"""Interprétabilité du modèle final et analyse détaillée de ses erreurs.

1. Importance globale des variables, par deux méthodes indépendantes :
   - SHAP : contribution de chaque variable à chaque prédiction (valeurs de Shapley) ;
     l'importance globale est la moyenne des contributions absolues. Pour une lecture
     métier, les contributions sont regroupées par variable d'origine : la valeur d'une
     variable et son indicateur « manquant » comptent ensemble, de même que les modalités
     d'une variable catégorielle ;
   - importance par permutation : baisse de la PR-AUC sur le test quand on mélange au
     hasard les valeurs d'une variable d'origine (ce qui casse son lien avec la cible).
   L'accord des deux classements est mesuré par une corrélation de rang.

2. Effet des données synthétiques sur le raisonnement du modèle : on compare les
   importances SHAP du modèle final à celles du même classifieur entraîné avec les
   défauts synthétiques du meilleur générateur.

3. Analyse des erreurs au seuil de décision retenu : défauts acceptés à tort (faux
   négatifs) et bons payeurs refusés à tort (faux positifs), avec leur profil et
   l'explication SHAP de cas individuels.

Usage : python -m src.interpretability   (après src.final_evaluation)
"""
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from src.config import CAT_COLS, FIG_DIR, LABELS, MODELS_DIR, NUM_COLS, RAW_COLS, SEED, TAB_DIR, TARGET
from src.features import FEATURES, load_dev, numeric_frame, split_xy
from src.final_evaluation import fit_predict, parse_dose
from src.generators import GENERATORS
from src.plots import AQUA, BLUE, C_BAD, C_GOOD, GREY, ORANGE, RED, set_style

warnings.filterwarnings("ignore")
GROUPS = RAW_COLS + ["LTV"]


def group_of(feature: str) -> str:
    """Variable d'origine d'une colonne de la matrice des classifieurs."""
    return feature.split("=")[0].replace("_manquant", "")


def explain(model, X: pd.DataFrame, display: pd.DataFrame) -> shap.Explanation:
    if hasattr(model, "coef_"):
        exp = shap.LinearExplainer(model, X.values)(X.values)
    else:
        exp = shap.TreeExplainer(model)(X.values)
    values = exp.values[:, :, 1] if exp.values.ndim == 3 else exp.values
    base = np.ravel(exp.base_values)
    base = np.full(len(X), base[-1] if exp.values.ndim == 3 else base[0]) if base.size != len(X) else base
    return shap.Explanation(values=values, base_values=base, data=display.values, feature_names=FEATURES)


def display_frame(raw: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Valeurs lisibles pour les graphiques : unités d'origine pour les variables numériques."""
    D = X.copy()
    for c in NUM_COLS:
        D[c] = raw[c].values
    D["LTV"] = ((raw["LOAN"] + raw["MORTDUE"]) / raw["VALUE"]).values
    return D.round(2)


def grouped(importance: pd.Series) -> pd.Series:
    return importance.groupby([group_of(f) for f in importance.index]).sum().reindex(GROUPS)


def permutation_by_variable(model, prep, raw: pd.DataFrame, y, n_repeats: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    base = average_precision_score(y, model.predict_proba(prep.transform(raw))[:, 1])
    rows = {}
    for c in RAW_COLS:
        drops = []
        for _ in range(n_repeats):
            shuffled = raw.copy()
            shuffled[c] = rng.permutation(shuffled[c].values)
            drops.append(base - average_precision_score(y, model.predict_proba(prep.transform(shuffled))[:, 1]))
        rows[c] = {"Permutation (baisse de PR-AUC)": np.mean(drops), "écart-type permutation": np.std(drops)}
    return pd.DataFrame(rows).T


def main() -> None:
    set_style()
    bundle = joblib.load(MODELS_DIR / "final_model.joblib")
    model, prep, thr = bundle["model"], bundle["prep"], bundle["threshold"]
    test = pd.read_csv(TAB_DIR / "test_predictions.csv")
    raw, y, proba = test[RAW_COLS], test[TARGET].values, test["proba"].values
    X = pd.DataFrame(prep.transform(raw), columns=FEATURES)
    D = display_frame(raw, X)
    print(f"Modèle final : {bundle['classifier']} — {bundle['config']} | seuil = {thr:.3f}\n")

    exp = explain(model, X, D)

    # --- 1. Importance globale ------------------------------------------------------------
    imp_cols = pd.Series(np.abs(exp.values).mean(axis=0), index=FEATURES)
    imp_shap = grouped(imp_cols)
    perm = permutation_by_variable(model, prep, raw, y)
    imp = pd.DataFrame({"signification": pd.Series(LABELS), "SHAP (moyenne |contribution|)": imp_shap}).join(perm)
    imp = imp.reindex(GROUPS).sort_values("SHAP (moyenne |contribution|)", ascending=False)
    imp["rang SHAP"] = imp["SHAP (moyenne |contribution|)"].rank(ascending=False).astype(int)
    imp["rang permutation"] = imp["Permutation (baisse de PR-AUC)"].rank(ascending=False)
    imp.round(4).to_csv(TAB_DIR / "importance_variables.csv")
    imp_cols.sort_values(ascending=False).round(4).to_csv(TAB_DIR / "importance_colonnes.csv", header=["SHAP"])
    both = imp.dropna(subset=["Permutation (baisse de PR-AUC)"])
    rho = spearmanr(both["SHAP (moyenne |contribution|)"], both["Permutation (baisse de PR-AUC)"])[0]
    print("Importance par variable d'origine :\n")
    print(imp.drop(columns="signification").round(4).to_string())
    print(f"\nAccord des deux classements (corrélation de Spearman sur les {len(both)} variables d'origine) : {rho:.2f}")
    share = imp_cols[[f for f in FEATURES if f.endswith("_manquant")]].sum() / imp_cols.sum()
    print(f"Part de l'importance SHAP portée par les indicateurs de valeur manquante : {share:.0%}")

    plt.figure()
    shap.summary_plot(exp.values, X, max_display=12, show=False, plot_size=(8.4, 5.2))
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel("Valeur de la variable"); cbar.set_yticklabels(["Faible", "Élevée"])
    plt.gcf().axes[0].set_xlabel("Contribution SHAP à la prédiction (vers la droite : pousse vers « défaut »)", fontsize=10)
    plt.title("Effet des 12 colonnes les plus importantes (un point = une demande du test)", fontsize=10)
    plt.savefig(FIG_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()

    order = imp.index[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    axes[0].barh(order, imp.loc[order, "SHAP (moyenne |contribution|)"], color=BLUE)
    axes[0].set_title("SHAP : moyenne des contributions absolues")
    axes[1].barh(order, imp.loc[order, "Permutation (baisse de PR-AUC)"].fillna(0), color=AQUA,
                 xerr=imp.loc[order, "écart-type permutation"].fillna(0), error_kw=dict(lw=1, capsize=2, ecolor="#444"))
    axes[1].set_title("Permutation : baisse de PR-AUC sur le test")
    for ax in axes: ax.grid(axis="y", visible=False)
    fig.suptitle(f"Importance des variables d'origine selon deux méthodes (accord des classements : ρ = {rho:.2f})", y=1.0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "importance_variables.png"); plt.close(fig)

    # --- 2. Les données synthétiques changent-elles le raisonnement du modèle ? -----------
    best = pd.read_csv(TAB_DIR / "cv_best.csv", keep_default_na=False)
    g = best[(best["classifieur"] == bundle["classifier"]) & (best["méthode"].isin(GENERATORS))].copy()
    g["PR-AUC"] = pd.to_numeric(g["PR-AUC"])
    gen = g.loc[g["PR-AUC"].idxmax()]
    dev = load_dev(); Xd, yd = split_xy(dev)
    model_aug, _ = fit_predict(bundle["classifier"], gen["méthode"], parse_dose(gen["dose"]), Xd, yd.values,
                               prep, X.values[:5], 0, "final")
    imp_aug = grouped(pd.Series(np.abs(explain(model_aug, X, D).values).mean(axis=0), index=FEATURES))
    rho_aug = spearmanr(imp_shap, imp_aug)[0]
    print(f"\nImportances SHAP du modèle final et du même classifieur entraîné avec {gen['config']} : ρ = {rho_aug:.2f}")
    print("Cinq premières variables — final :", ", ".join(imp_shap.nlargest(5).index),
          "| avec défauts synthétiques :", ", ".join(imp_aug.nlargest(5).index))
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(imp_shap, imp_aug, s=34, color=ORANGE, edgecolor="white", linewidths=0.8, zorder=3)
    lim = [0, max(imp_shap.max(), imp_aug.max()) * 1.12]
    ax.plot(lim, lim, color=GREY, ls="--", lw=1)
    for f in GROUPS:
        ax.annotate(f, (imp_shap[f], imp_aug[f]), xytext=(5, -3), textcoords="offset points", fontsize=7)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"Importance SHAP — modèle final ({bundle['config'].lower()})")
    ax.set_ylabel(f"Importance SHAP — avec {gen['config']}")
    ax.set_title(f"Le modèle s'appuie-t-il sur les mêmes variables ? (ρ = {rho_aug:.2f})", fontsize=10)
    fig.savefig(FIG_DIR / "shap_reel_vs_synthetique.png"); plt.close(fig)
    pd.DataFrame({"final": imp_shap, f"avec {gen['config']}": imp_aug}).round(4).to_csv(
        TAB_DIR / "shap_reel_vs_synthetique.csv")

    # --- 3. Analyse des erreurs au seuil retenu -------------------------------------------
    pred = proba >= thr
    groups = {"Défauts refusés (VP)": pred & (y == 1), "Défauts acceptés à tort (FN)": ~pred & (y == 1),
              "Bons payeurs refusés à tort (FP)": pred & (y == 0), "Bons payeurs acceptés (VN)": ~pred & (y == 0)}
    summary = pd.DataFrame({name: {
        "effectif": int(m.sum()),
        "probabilité médiane": np.median(proba[m]),
        "DEBTINC manquant (%)": 100 * raw.loc[m, "DEBTINC"].isna().mean(),
        "DEBTINC médian": raw.loc[m, "DEBTINC"].median(),
        "au moins 1 retard (DELINQ) (%)": 100 * (raw.loc[m, "DELINQ"] >= 1).mean(),
        "au moins 1 incident (DEROG) (%)": 100 * (raw.loc[m, "DEROG"] >= 1).mean(),
        "CLAGE médian (mois)": raw.loc[m, "CLAGE"].median(),
        "NINQ moyen": raw.loc[m, "NINQ"].mean(),
        "LOAN médian": raw.loc[m, "LOAN"].median(),
        "aucune valeur manquante (%)": 100 * raw.loc[m].notna().all(axis=1).mean(),
    } for name, m in groups.items()}).T
    summary.round(2).to_csv(TAB_DIR / "analyse_erreurs.csv")
    print("\nAnalyse des erreurs sur le test, au seuil fixé sur le développement :\n")
    print(summary.round(1).T.to_string())

    fn = np.flatnonzero(groups["Défauts acceptés à tort (FN)"])
    detail = []
    for i in fn:
        contrib = pd.Series(exp.values[i], index=FEATURES).sort_values()
        detail.append({"probabilité": proba[i], **raw.iloc[i].to_dict(),
                       "colonnes qui tirent vers « remboursera »": ", ".join(contrib.index[:3])})
    pd.DataFrame(detail).round(3).to_csv(TAB_DIR / "defauts_acceptes_a_tort.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bins = np.linspace(0, 1, 41)
    ax.hist(proba[y == 0], bins=bins, color=C_GOOD, alpha=0.6, label=f"Bons payeurs ({int((y == 0).sum())})")
    ax.hist(proba[y == 1], bins=bins, color=C_BAD, alpha=0.6, label=f"Défauts ({int((y == 1).sum())})")
    ax.axvline(thr, color="#333", ls="--", lw=1.2)
    ax.text(thr + 0.01, ax.get_ylim()[1] * 0.92, f"seuil {thr:.2f}\nrefus à droite", fontsize=8)
    ax.set_yscale("log"); ax.set_xlabel("Probabilité de défaut estimée"); ax.set_ylabel("Demandes (échelle log)")
    ax.set_title(f"Scores du modèle final sur le test : {int(groups['Défauts acceptés à tort (FN)'].sum())} défauts "
                 f"acceptés à tort, {int(groups['Bons payeurs refusés à tort (FP)'].sum())} bons payeurs refusés à tort", fontsize=10)
    ax.legend(loc="upper center")
    fig.savefig(FIG_DIR / "analyse_erreurs.png"); plt.close(fig)

    cases = {"vp": ("Défaut correctement refusé", np.flatnonzero(groups["Défauts refusés (VP)"])),
             "fn": ("Défaut accepté à tort", fn),
             "fp": ("Bon payeur refusé à tort", np.flatnonzero(groups["Bons payeurs refusés à tort (FP)"]))}
    for key, (title, cand) in cases.items():
        if len(cand) == 0:
            continue
        i = int(cand[np.argsort(proba[cand])[len(cand) // 2]])     # un cas médian, pas le plus spectaculaire
        plt.figure()
        shap.plots.waterfall(exp[i], max_display=9, show=False)
        for _ax in plt.gcf().axes:
            _labels = [t.get_text() for t in _ax.get_yticklabels()]
            if any("other features" in t for t in _labels):
                _ax.set_yticks(_ax.get_yticks())
                _ax.set_yticklabels([t.replace("other features", "autres colonnes") for t in _labels], fontsize=13)
        plt.title(f"{title} — probabilité estimée {proba[i]:.2f} (seuil {thr:.2f})", fontsize=10)
        plt.savefig(FIG_DIR / f"shap_cas_{key}.png", dpi=150, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    main()
