"""Réglage des hyperparamètres des classifieurs, par validation croisée.

Recherche aléatoire (graine fixée) sur les 5 plis de la première répétition du jeu de
développement, SANS rééquilibrage, en maximisant la PR-AUC moyenne. Le jeu de test
n'intervient pas.

Les hyperparamètres retenus sont enregistrés dans reports/tables/best_params.json et
réutilisés tels quels pour toutes les méthodes de rééquilibrage : la comparaison porte
sur les données d'entraînement, pas sur le réglage du classifieur. (Limite assumée : un
réglage propre à chaque méthode pourrait la favoriser légèrement ; ce choix évite de
multiplier les calculs et les degrés de liberté.)

Usage : python -m src.tuning
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import N_FOLDS, SEED, TAB_DIR
from src.features import Preprocessor, load_dev, split_xy
from src.models import DEFAULTS, PARAMS_FILE, make_classifier

GRIDS = {
    "LogReg": {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "XGBoost": {"n_estimators": [200, 400, 800], "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [3, 4, 6, 8], "min_child_weight": [1, 3, 5],
                "subsample": [0.7, 1.0], "colsample_bytree": [0.5, 0.8, 1.0]},
    "LightGBM": {"n_estimators": [200, 400, 800], "learning_rate": [0.01, 0.03, 0.05, 0.1],
                 "num_leaves": [7, 15, 31, 63], "min_child_samples": [5, 10, 20, 50],
                 "subsample": [0.7, 1.0], "subsample_freq": [1],
                 "colsample_bytree": [0.5, 0.8, 1.0], "reg_lambda": [0.0, 1.0, 10.0]},
}
N_CANDIDATES = 40


def candidates(kind: str) -> list[dict]:
    grid = GRIDS[kind]
    if kind == "LogReg":
        return [{"C": c} for c in grid["C"]]
    rng = np.random.default_rng(SEED)
    cands = [dict(DEFAULTS[kind])]   # les valeurs par défaut concourent aussi
    while len(cands) < N_CANDIDATES:
        c = {k: v[rng.integers(len(v))] for k, v in grid.items()}
        c = {k: (x.item() if hasattr(x, "item") else x) for k, x in c.items()}
        if c not in cands:
            cands.append(c)
    return cands


def main() -> None:
    dev = load_dev()
    folds = []
    for k in range(N_FOLDS):
        Xtr, ytr = split_xy(dev[dev["fold_r0"] != k])
        Xva, yva = split_xy(dev[dev["fold_r0"] == k])
        prep = Preprocessor().fit(Xtr)
        folds.append((prep.transform(Xtr), ytr.values, prep.transform(Xva), yva.values))

    best, rows = {}, []
    for kind in GRIDS:
        for params in candidates(kind):
            scores = [average_precision_score(yva, make_classifier(kind, SEED, params=params)
                                              .fit(A, ytr).predict_proba(B)[:, 1])
                      for A, ytr, B, yva in folds]
            rows.append({"modèle": kind, "PR-AUC": np.mean(scores), "écart-type": np.std(scores),
                         "paramètres": json.dumps(params)})
            print(f"{kind:<9} PR-AUC = {np.mean(scores):.4f} ± {np.std(scores):.4f}  {params}",
                  flush=True)
        top = max((r for r in rows if r["modèle"] == kind), key=lambda r: r["PR-AUC"])
        best[kind] = json.loads(top["paramètres"])

    pd.DataFrame(rows).round(4).to_csv(TAB_DIR / "tuning.csv", index=False)
    PARAMS_FILE.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("\nHyperparamètres retenus :", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
