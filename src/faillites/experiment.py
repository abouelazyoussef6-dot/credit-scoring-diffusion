"""Étude 2 — comparaison de toutes les méthodes sur les 15 mêmes découpages.

Même protocole que src.experiment. Doses adaptées au fort déséquilibre (4,7 % de
faillites : l'équilibre parfait demande d'ajouter environ 20 fois plus de faillites qu'il
n'y en a) : 1, 2, 5 faillites synthétiques par vraie faillite, et l'équilibre.

Usage : python -m src.faillites.experiment --repeats 0 2   (une liste de répétitions par processus)
"""
import argparse
import time
import warnings

import pandas as pd

from src.faillites.config import N_FOLDS, N_REPEATS, TAB
from src.faillites.data import load_dev, split_xy
from src.faillites.features import Prep
from src.faillites.generators import GENERATORS, get_synthetic
from src.faillites.resampling import build_training_set
from src.faillites.tuning import load_params
from src.metrics import at_threshold, evaluate
from src.models import make_classifier
from src.resampling import n_to_add

warnings.filterwarnings("ignore")

DOSES = [1, 2, 5, "éq."]
CONFIGS = ([("Aucune", None), ("Pondération", None)]
           + [("Sous-échantillonnage", d) for d in (5, 1)]
           + [("Duplication", d) for d in (1, "éq.")]
           + [(m, d) for m in ["SMOTE"] + GENERATORS for d in DOSES])
CLASSIFIERS = ["LogReg", "XGBoost", "LightGBM"]


def run_fold(dev: pd.DataFrame, r: int, k: int, params: dict) -> list[dict]:
    col = f"fold_r{r}"
    Xtr, ytr = split_xy(dev[dev[col] != k]); Xva, yva = split_xy(dev[dev[col] == k])
    prep = Prep().fit(Xtr)
    B, y = prep.transform(Xva), ytr.values
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    seed = 1000 * r + k
    synth = {g: get_synthetic(g, f"r{r}f{k}", Xtr[y == 1], n_to_add("éq.", n_pos, n_neg), seed) for g in GENERATORS}
    rows = []
    for method, dose in CONFIGS:
        X_fit, y_fit = build_training_set(method, dose, Xtr, y, prep, seed, synth.get(method))
        for kind in CLASSIFIERS:
            t0 = time.time()
            clf = make_classifier(kind, seed, weighted=(method == "Pondération"), neg_pos_ratio=n_neg / n_pos,
                                  params=params[kind])
            proba = clf.fit(X_fit, y_fit).predict_proba(B)[:, 1]
            half = at_threshold(yva.values, proba, 0.5)
            rows.append({"répétition": r, "pli": k, "classifieur": kind, "méthode": method,
                         "dose": "" if dose is None else str(dose),
                         "part de défauts (%)": 100 * y_fit.mean(), "lignes": len(y_fit),
                         **evaluate(yva.values, proba),
                         "Rappel (seuil 0,5)": half["Rappel"], "Précision (seuil 0,5)": half["Précision"],
                         "durée (s)": round(time.time() - t0, 2)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", nargs="+", type=int, default=list(range(N_REPEATS)))
    args = parser.parse_args()
    dev, params = load_dev(), load_params()
    for r in args.repeats:
        out = TAB / f"cv_runs_r{r}.csv"
        rows = pd.read_csv(out, keep_default_na=False).to_dict("records") if out.exists() else []
        done = {row["pli"] for row in rows}
        for k in range(N_FOLDS):
            if k in done:
                continue
            t0 = time.time()
            rows += run_fold(dev, r, k, params)
            pd.DataFrame(rows).round(5).to_csv(out, index=False)
            last = pd.DataFrame(rows)
            base = last[(last["pli"] == k) & (last["méthode"] == "Aucune")]
            print(f"répétition {r} pli {k} terminé en {time.time() - t0:.0f} s | PR-AUC sans rééquilibrage : "
                  + ", ".join(f"{c} {v:.3f}" for c, v in zip(base["classifieur"], base["PR-AUC"])), flush=True)


if __name__ == "__main__":
    main()
