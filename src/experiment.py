"""Expérience principale : comparaison de toutes les méthodes en validation croisée.

Pour chaque pli de chaque répétition (3 x 5 = 15 découpages du jeu de développement) :
1. le prétraitement (imputation, standardisation) est ajusté sur l'entraînement du pli ;
2. les trois générateurs sont entraînés sur les vrais défauts d'ENTRAÎNEMENT du pli ;
3. chaque méthode construit son jeu d'entraînement (src.resampling) ;
4. les classifieurs sont entraînés puis évalués sur le pli de validation, composé
   uniquement de vraies demandes jamais vues.

Toutes les méthodes sont donc comparées sur exactement les mêmes découpages : les
différences sont appariées, ce qui permet un test statistique (src.analysis).

Les résultats sont enregistrés après chaque pli (reprise possible après interruption) et
les défauts synthétiques sont mis en cache.

Usage :
    python -m src.experiment                 # tout (3 répétitions)
    python -m src.experiment --repeats 0     # une seule répétition
"""
import argparse
import time
import warnings

import pandas as pd

from src.config import N_FOLDS, N_REPEATS, TAB_DIR
from src.features import Preprocessor, load_dev, split_xy
from src.generators import GENERATORS, get_synthetic
from src.metrics import at_threshold, evaluate
from src.models import make_classifier
from src.resampling import build_training_set, n_to_add

warnings.filterwarnings("ignore")

DOSES = [0.5, 1, 2, "éq."]
CONFIGS = (
    [("Aucune", None), ("Pondération", None)]
    + [("Sous-échantillonnage", d) for d in (2, 1)]
    + [("Duplication", d) for d in (1, "éq.")]
    + [(m, d) for m in ["SMOTE"] + GENERATORS for d in DOSES]
)
CLASSIFIERS = ["LogReg", "XGBoost", "LightGBM"]


def run_fold(dev: pd.DataFrame, r: int, k: int) -> list[dict]:
    col = f"fold_r{r}"
    Xtr, ytr = split_xy(dev[dev[col] != k])
    Xva, yva = split_xy(dev[dev[col] == k])
    prep = Preprocessor().fit(Xtr)
    B, y = prep.transform(Xva), ytr.values
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    seed = 1000 * r + k

    defaults = Xtr[y == 1]
    synth = {g: get_synthetic(g, f"r{r}f{k}", defaults, n_to_add("éq.", n_pos, n_neg), seed)
             for g in GENERATORS}

    rows = []
    for method, dose in CONFIGS:
        X_fit, y_fit = build_training_set(method, dose, Xtr, y, prep, seed, synth.get(method))
        for kind in CLASSIFIERS:
            t0 = time.time()
            clf = make_classifier(kind, seed, weighted=(method == "Pondération"),
                                  neg_pos_ratio=n_neg / n_pos)
            proba = clf.fit(X_fit, y_fit).predict_proba(B)[:, 1]
            half = at_threshold(yva.values, proba, 0.5)
            rows.append({
                "répétition": r, "pli": k, "classifieur": kind, "méthode": method,
                "dose": "" if dose is None else str(dose),
                "part de défauts (%)": 100 * y_fit.mean(), "lignes": len(y_fit),
                **evaluate(yva.values, proba),
                # au seuil naïf de 0,5 : montre l'effet « déplacement du seuil » du rééquilibrage
                "Rappel (seuil 0,5)": half["Rappel"], "Précision (seuil 0,5)": half["Précision"],
                "durée (s)": round(time.time() - t0, 2),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", nargs="+", type=int, default=list(range(N_REPEATS)))
    args = parser.parse_args()

    dev = load_dev()
    for r in args.repeats:
        out = TAB_DIR / f"cv_runs_r{r}.csv"
        rows = pd.read_csv(out, keep_default_na=False).to_dict("records") if out.exists() else []
        done = {row["pli"] for row in rows}
        for k in range(N_FOLDS):
            if k in done:
                continue
            t0 = time.time()
            rows += run_fold(dev, r, k)
            pd.DataFrame(rows).round(5).to_csv(out, index=False)
            last = pd.DataFrame(rows)
            base = last[(last["pli"] == k) & (last["méthode"] == "Aucune")]
            print(f"répétition {r} pli {k} terminé en {time.time() - t0:.0f} s | PR-AUC sans "
                  f"rééquilibrage : " + ", ".join(f"{c} {v:.3f}" for c, v in
                                                  zip(base["classifieur"], base["PR-AUC"])),
                  flush=True)


if __name__ == "__main__":
    main()
