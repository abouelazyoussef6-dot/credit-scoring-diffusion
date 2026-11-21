"""Métriques d'évaluation et tests statistiques.

Avec 20 % de défauts, et jusqu'à 1 % dans l'étude de rareté, l'exactitude (accuracy) est
trompeuse : répondre toujours « remboursera » donne déjà 80 %. On utilise :

- PR-AUC (average precision) : aire sous la courbe précision-rappel. C'est la métrique
  principale : elle ne dépend pas du seuil de décision et se concentre sur les défauts.
  Son niveau de hasard est le taux de défaut (0,20 ici), pas 0,5.
- ROC-AUC : probabilité qu'un client défaillant reçoive un score plus élevé qu'un bon
  payeur. Le coefficient de Gini des praticiens du crédit vaut 2 x ROC-AUC - 1.
- Statistique KS (Kolmogorov-Smirnov) : écart maximal entre les fonctions de répartition
  des scores des défaillants et des bons payeurs. C'est la mesure de séparation
  traditionnelle du scoring de crédit.
- F1 maximal : meilleur compromis précision/rappel, chaque méthode ayant droit à son seuil
  optimal. Une méthode ne peut donc pas « gagner » simplement en déplaçant le seuil.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

MAIN_METRIC = "PR-AUC"
COST_FN, COST_FP = 5.0, 1.0     # accepter un futur défaillant coûte 5 fois plus que refuser un bon payeur


def ks_statistic(y_true, proba) -> float:
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(np.max(tpr - fpr))


def cost_threshold(y_true, proba) -> float:
    """Seuil qui minimise le coût total COST_FN x (défauts acceptés) + COST_FP x (bons payeurs refusés)."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    thresholds = np.unique(np.round(proba, 3))
    costs = [COST_FN * ((proba < t) & (y_true == 1)).sum() + COST_FP * ((proba >= t) & (y_true == 0)).sum()
             for t in thresholds]
    return float(thresholds[int(np.argmin(costs))])


def best_f1(y_true, proba) -> tuple[float, float]:
    """(F1 maximal, seuil correspondant)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    i = int(np.argmax(f1[:-1]))
    return float(f1[i]), float(thresholds[i])


def evaluate(y_true, proba) -> dict[str, float]:
    f1, thr = best_f1(y_true, proba)
    return {
        "PR-AUC": float(average_precision_score(y_true, proba)),
        "ROC-AUC": float(roc_auc_score(y_true, proba)),
        "KS": ks_statistic(y_true, proba),
        "F1 max": f1,
        "Seuil F1": thr,
    }


def at_threshold(y_true, proba, threshold: float) -> dict[str, float]:
    """Résultats concrets à un seuil fixé à l'avance (choisi sans regarder ces données)."""
    y_true = np.asarray(y_true)
    pred = np.asarray(proba) >= threshold
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    fn = int((~pred & (y_true == 1)).sum())
    tn = int((~pred & (y_true == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"VP": tp, "FP": fp, "FN": fn, "VN": tn, "Précision": precision, "Rappel": recall,
            "F1": 2 * precision * recall / max(precision + recall, 1e-12),
            "Coût": COST_FN * fn + COST_FP * fp}


def corrected_ttest(diffs, n_folds: int) -> tuple[float, float]:
    """Test t corrigé pour validation croisée répétée (Nadeau & Bengio, 2003).

    Les plis d'une validation croisée partagent une grande partie de leurs données
    d'entraînement : leurs résultats ne sont pas indépendants, et un test t ordinaire
    conclurait trop facilement à une différence. La correction gonfle la variance d'un
    facteur (1/J + n_test/n_train), avec n_test/n_train = 1/(k-1) pour k plis.

    `diffs` : différences appariées (méthode - référence), une par pli et par répétition.
    Retourne (statistique t, p-valeur bilatérale).
    """
    d = np.asarray(diffs, dtype=float)
    j = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / j + 1 / (n_folds - 1)) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=j - 1))


def bootstrap_ci(y_true, probas, metric=average_precision_score, n_boot: int = 1000,
                 seed: int = 0, probas_ref=None) -> tuple[float, float]:
    """Intervalle de confiance à 95 % par bootstrap stratifié (fraudes et non-fraudes
    rééchantillonnées séparément, pour garder le même nombre de fraudes).

    `probas` : liste de vecteurs de probabilités, un par graine d'entraînement ; la
    métrique est moyennée sur les graines à chaque tirage.
    Si `probas_ref` est fourni, l'intervalle porte sur la DIFFÉRENCE avec la référence,
    calculée sur les mêmes tirages et les mêmes graines (bootstrap apparié).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    vals = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        v = np.mean([metric(y_true[idx], np.asarray(p)[idx]) for p in probas])
        if probas_ref is not None:
            v -= np.mean([metric(y_true[idx], np.asarray(p)[idx]) for p in probas_ref])
        vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def fit_calibrator(y_true, proba):
    """Calibration de Platt : régression logistique sur le logit du score brut.

    Les arbres boostés classent bien les demandes mais donnent des probabilités trop
    extrêmes. La transformation est strictement croissante : elle ne change ni le
    classement des demandes ni les métriques de rang (PR-AUC, ROC-AUC, KS).
    """
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1e6, max_iter=1000).fit(_logit(proba), np.asarray(y_true))


def calibrate(calibrator, proba) -> np.ndarray:
    return calibrator.predict_proba(_logit(proba))[:, 1]


def _logit(proba) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)).reshape(-1, 1)
