"""Étude de rareté : que se passe-t-il quand le déséquilibre devient fort ?

Le taux de défaut naturel de ces données est de 20 %. Pour étudier des déséquilibres plus
sévères, on réduit artificiellement le nombre de vrais défauts disponibles pour
l'entraînement (40, 80, 160, 320, puis tous : ~760), en gardant TOUS les bons payeurs.
Le taux de défaut à l'entraînement passe ainsi d'environ 1 % à 20 %.

À chaque niveau, on compare l'entraînement sans ajout, la pondération des classes, et
l'ajout de défauts synthétiques (SMOTE, TVAE, CTGAN, TabDDPM) à deux doses : 1 et 5
défauts synthétiques par vrai défaut. Les générateurs ne voient que les défauts conservés.

L'évaluation se fait toujours sur le même pli de validation complet (vraies demandes,
taux de défaut de 20 %). Seul le nombre de défauts disponibles change : la comparaison est
directe. Les 5 plis de la première répétition servent de répétitions de l'expérience.

Usage :
    python -m src.rarity                    # tout
    python -m src.rarity --folds 0 1 2      # certains plis seulement (calcul en parallèle)
    python -m src.rarity --analyse-only     # tableaux et figure à partir des résultats enregistrés
"""
import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIG_DIR, N_FOLDS, TAB_DIR, TARGET
from src.features import Preprocessor, load_dev, split_xy
from src.generators import GENERATORS, get_synthetic
from src.metrics import MAIN_METRIC, evaluate
from src.models import make_classifier
from src.plots import METHOD_COLORS, METHOD_MARKERS, set_style
from src.resampling import build_training_set

warnings.filterwarnings("ignore")
LEVELS = [40, 80, 160, 320, None]     # None = tous les défauts
DOSES = [1, 5]
ADDING = ["SMOTE"] + GENERATORS
CLFS = ["LogReg", "XGBoost", "LightGBM"]
ORDER = ["40", "80", "160", "320", "tous"]


def run_fold(dev: pd.DataFrame, k: int) -> None:
    out = TAB_DIR / f"rarity_runs_f{k}.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {str(r["niveau"]) for r in rows}
    train, val = dev[dev["fold_r0"] != k], dev[dev["fold_r0"] == k]
    Xva, yva = split_xy(val)
    for level in LEVELS:
        name = "tous" if level is None else str(level)
        if name in done:
            continue
        defaults = train[train[TARGET] == 1]
        if level is not None:
            defaults = defaults.sample(n=level, random_state=k)
        sub = pd.concat([train[train[TARGET] == 0], defaults])
        Xtr, ytr = split_xy(sub)
        y = ytr.values
        prep = Preprocessor().fit(Xtr)
        B = prep.transform(Xva)
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        synth = {g: get_synthetic(g, f"rare{name}_f{k}", Xtr[y == 1], min(max(DOSES) * n_pos, n_neg - n_pos), k)
                 for g in GENERATORS}
        configs = [("Aucune", None), ("Pondération", None)] + [(m, d) for m in ADDING for d in DOSES]
        for method, dose in configs:
            X_fit, y_fit = build_training_set(method, dose, Xtr, y, prep, k, synth.get(method))
            for kind in CLFS:
                clf = make_classifier(kind, k, weighted=(method == "Pondération"), neg_pos_ratio=n_neg / n_pos)
                proba = clf.fit(X_fit, y_fit).predict_proba(B)[:, 1]
                rows.append({"pli": k, "niveau": name, "vrais défauts": n_pos,
                             "taux de défaut à l'entraînement (%)": 100 * n_pos / (n_pos + n_neg),
                             "classifieur": kind, "méthode": method, "dose": dose if dose else 0,
                             **evaluate(yva.values, proba)})
        pd.DataFrame(rows).round(5).to_csv(out, index=False)
        print(f"pli {k} | {n_pos:>3} vrais défauts : terminé", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(N_FOLDS)))
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    set_style()
    if not args.analyse_only:
        dev = load_dev()
        for k in args.folds:
            run_fold(dev, k)

    runs = pd.concat([pd.read_csv(f) for f in sorted(TAB_DIR.glob("rarity_runs_f*.csv"))], ignore_index=True)
    runs["niveau"] = runs["niveau"].astype(str)
    print(f"\n{runs['pli'].nunique()} plis analysés.")
    levels = runs.groupby("niveau")[["vrais défauts", "taux de défaut à l'entraînement (%)"]].mean().reindex(ORDER)
    print("\nNiveaux :\n", levels.round(1).to_string())

    runs["config"] = runs.apply(lambda r: r["méthode"] if r["dose"] == 0 else f"{r['méthode']} (dose {int(r['dose'])})", axis=1)
    idx = ["pli", "niveau", "classifieur"]
    base = runs[runs["méthode"] == "Aucune"].set_index(idx)[MAIN_METRIC]
    runs["gain"] = runs[MAIN_METRIC] - runs.set_index(idx).index.map(base)
    pr = runs.pivot_table(index=["classifieur", "config"], columns="niveau", values=MAIN_METRIC, sort=False)[ORDER]
    gain = runs[runs["méthode"] != "Aucune"].pivot_table(index=["classifieur", "config"], columns="niveau",
                                                        values="gain", sort=False)[ORDER]
    wins = (runs[runs["méthode"] != "Aucune"].assign(win=lambda d: d["gain"] > 0)
            .pivot_table(index=["classifieur", "config"], columns="niveau", values="win", aggfunc="sum", sort=False)[ORDER])
    pr.round(4).to_csv(TAB_DIR / "rarity_pr_auc.csv")
    gain.round(4).to_csv(TAB_DIR / "rarity_gains.csv")
    wins.to_csv(TAB_DIR / "rarity_wins.csv")
    print("\nPR-AUC moyenne selon le nombre de vrais défauts disponibles :\n")
    print(pr.round(3).to_string())
    print("\nGain de PR-AUC par rapport à « Aucune » (> 0 : la méthode aide) :\n")
    print(gain.round(3).to_string())

    for dose in DOSES:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
        for ax, clf in zip(axes, CLFS):
            g = runs[(runs["classifieur"] == clf) & (runs["dose"].isin([0, dose]))]
            for m in ["Aucune", "Pondération"] + ADDING:
                d = g[g["méthode"] == m].groupby("niveau")[MAIN_METRIC].agg(["mean", "std"]).reindex(ORDER)
                ax.plot(levels["vrais défauts"].values, d["mean"], color=METHOD_COLORS[m], lw=2, ms=7,
                        marker=METHOD_MARKERS[m], markeredgecolor="white", markeredgewidth=1.5, label=m,
                        ls="--" if m == "Aucune" else "-")
                if m == "Aucune":
                    ax.fill_between(levels["vrais défauts"].values, d["mean"] - d["std"], d["mean"] + d["std"],
                                    color=METHOD_COLORS[m], alpha=0.12, lw=0)
            ax.set_xscale("log"); ax.set_xticks(levels["vrais défauts"].values)
            ax.set_xticklabels([f"{n:.0f}\n({p:.0f} %)" for n, p in levels.values])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlabel("Vrais défauts à l'entraînement (taux de défaut)")
            ax.set_title(clf)
        axes[0].set_ylabel("PR-AUC sur la validation")
        axes[2].legend(loc="lower right")
        fig.suptitle(f"Étude de rareté — {dose} défaut{'s' if dose > 1 else ''} synthétique{'s' if dose > 1 else ''} "
                     "ajouté" + ("s" if dose > 1 else "") + " par vrai défaut (bande grise : ± 1 écart-type sans ajout)", y=1.0)
        fig.tight_layout(); fig.savefig(FIG_DIR / f"rarete_dose{dose}.png"); plt.close(fig)


if __name__ == "__main__":
    main()
