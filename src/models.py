"""Classifieurs utilisés dans toutes les expériences.

- LogReg   : régression logistique, le modèle de référence simple (baseline).
- XGBoost  : gradient boosting, la référence pratique sur données tabulaires.
- LightGBM : autre implémentation du gradient boosting (comparaison des modèles avancés).

Les hyperparamètres sont ceux trouvés par src.tuning (validation croisée sur le jeu de
développement, sans rééquilibrage) et enregistrés dans reports/tables/best_params.json.
Les mêmes hyperparamètres servent ensuite pour toutes les méthodes de rééquilibrage :
ce qui change d'une expérience à l'autre, ce sont uniquement les données d'entraînement.
"""
import json
import os

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import TAB_DIR

N_JOBS = int(os.environ.get("N_JOBS", "-1"))
PARAMS_FILE = TAB_DIR / "best_params.json"

DEFAULTS = {
    "LogReg": {"C": 1.0},
    "XGBoost": {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 4,
                "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1},
    "LightGBM": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 15,
                 "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
                 "min_child_samples": 20},
}


def get_params(kind: str) -> dict:
    params = dict(DEFAULTS[kind])
    if PARAMS_FILE.exists():
        params.update(json.loads(PARAMS_FILE.read_text(encoding="utf-8")).get(kind, {}))
    return params


def make_classifier(kind: str, seed: int = 0, weighted: bool = False,
                    neg_pos_ratio: float = 1.0, params: dict | None = None):
    """Construit un classifieur.

    weighted=True : les erreurs sur les fraudes coûtent plus cher (pondération par
    l'inverse des fréquences de classe). C'est le rééquilibrage « par le coût », qui ne
    crée aucune donnée.
    """
    p = get_params(kind) if params is None else params
    if kind == "LogReg":
        return LogisticRegression(max_iter=2000, class_weight="balanced" if weighted else None,
                                  random_state=seed, **p)
    if kind == "XGBoost":
        return XGBClassifier(tree_method="hist", eval_metric="aucpr", random_state=seed,
                             n_jobs=N_JOBS, scale_pos_weight=neg_pos_ratio if weighted else 1.0,
                             **p)
    if kind == "LightGBM":
        return LGBMClassifier(random_state=seed, n_jobs=N_JOBS, verbose=-1,
                              scale_pos_weight=neg_pos_ratio if weighted else 1.0, **p)
    raise ValueError(kind)
