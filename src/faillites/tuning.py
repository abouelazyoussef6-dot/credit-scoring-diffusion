"""Étude 2 — réglage des classifieurs (recherche aléatoire, PR-AUC, plis de la répétition 0,
sans rééquilibrage). Mêmes grilles que l'étude 1 (src.tuning).

Usage : python -m src.faillites.tuning
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.faillites.config import N_FOLDS, SEED, TAB
from src.faillites.data import load_dev, split_xy
from src.faillites.features import Prep
from src.models import make_classifier
from src.tuning import GRIDS, candidates

PARAMS_FILE = TAB / "best_params.json"


def load_params() -> dict:
    return json.loads(PARAMS_FILE.read_text(encoding="utf-8"))


def main() -> None:
    dev = load_dev()
    folds = []
    for k in range(N_FOLDS):
        Xtr, ytr = split_xy(dev[dev["fold_r0"] != k]); Xva, yva = split_xy(dev[dev["fold_r0"] == k])
        prep = Prep().fit(Xtr)
        folds.append((prep.transform(Xtr), ytr.values, prep.transform(Xva), yva.values))
    best, rows = {}, []
    for kind in GRIDS:
        for params in candidates(kind):
            scores = [average_precision_score(yva, make_classifier(kind, SEED, params=params)
                                              .fit(A, ytr).predict_proba(B)[:, 1]) for A, ytr, B, yva in folds]
            rows.append({"modèle": kind, "PR-AUC": np.mean(scores), "écart-type": np.std(scores),
                         "paramètres": json.dumps(params)})
            print(f"{kind:<9} PR-AUC = {np.mean(scores):.4f} ± {np.std(scores):.4f}  {params}", flush=True)
        top = max((r for r in rows if r["modèle"] == kind), key=lambda r: r["PR-AUC"])
        best[kind] = json.loads(top["paramètres"])
    pd.DataFrame(rows).round(4).to_csv(TAB / "tuning.csv", index=False)
    PARAMS_FILE.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("\nHyperparamètres retenus :", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
