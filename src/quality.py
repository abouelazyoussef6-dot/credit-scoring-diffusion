"""Mesures de qualité des données synthétiques (utilisées par plusieurs scripts).

Toutes comparent des défauts synthétiques aux VRAIS défauts d'entraînement et, pour la
confidentialité, à de vrais défauts jamais vus par le générateur (le pli de validation).

1. Fidélité des variables numériques — statistique de Kolmogorov-Smirnov (KS), moyennée
   sur les 10 variables : plus grand écart entre les fonctions de répartition des vraies
   et des fausses valeurs (0 = identiques, 1 = disjointes). Calculée sur les valeurs
   renseignées.

2. Fidélité des variables catégorielles et des valeurs manquantes — distance de variation
   totale (TVD), moyennée sur le motif, la profession et les 9 taux de valeurs
   manquantes : part des clients qu'il faudrait changer de modalité pour que les
   proportions coïncident.

3. Fidélité des relations — écart absolu moyen entre les matrices de corrélation de
   Spearman des vrais et des faux défauts. Un générateur peut reproduire chaque variable
   isolément tout en cassant les liens entre elles.

4. Valeurs aux bornes — part des valeurs numériques collées au minimum ou au maximum
   observé, en excès par rapport aux vraies données. Signe d'un générateur mal convergé.

5. Détectabilité — un classifieur essaie de distinguer vrais et faux défauts (validation
   croisée). ROC-AUC de 0,5 : impossible de les distinguer ; 1 : faux évidents.

6. Confidentialité — distance au plus proche voisin (DCR, distance to closest record).
   Pour chaque faux client, on mesure sa distance au vrai client d'entraînement le plus
   proche, et on compare à la même distance mesurée pour de VRAIS défauts inédits (le pli
   de validation), qui sert de référence « sans mémorisation ».
   - Rapport DCR : distance médiane des faux, divisée par celle des vrais inédits.
     ~1 : les faux clients ne sont pas plus proches des données d'entraînement que ne
     l'est un vrai client jamais vu. Proche de 0 : copies.
   - Quasi-copies (%) : part des faux clients plus proches d'un client d'entraînement que
     ne le sont 95 % des vrais clients inédits. Valeur attendue sans mémorisation : 5 %.

Les distances et la détectabilité sont calculées dans l'espace des classifieurs
(variables standardisées, modalités one-hot), avec le prétraitement ajusté sur
l'entraînement.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

from src.config import CAT_COLS, MISSING_LABEL, NUM_COLS, SEED
from src.features import IND_COLS, Preprocessor


def ks_mean(real: pd.DataFrame, fake: pd.DataFrame) -> float:
    scores = [ks_2samp(real[c].dropna(), fake[c].dropna()).statistic
              for c in NUM_COLS if real[c].notna().any() and fake[c].notna().any()]
    return float(np.mean(scores))


def tvd_mean(real: pd.DataFrame, fake: pd.DataFrame) -> float:
    scores = []
    for c in CAT_COLS:
        p = real[c].fillna(MISSING_LABEL).value_counts(normalize=True)
        q = fake[c].fillna(MISSING_LABEL).value_counts(normalize=True)
        scores.append(0.5 * p.subtract(q, fill_value=0).abs().sum())
    scores += [abs(real[c].isna().mean() - fake[c].isna().mean()) for c in IND_COLS]
    return float(np.mean(scores))


def correlation_gap(real: pd.DataFrame, fake: pd.DataFrame) -> float:
    diff = (real[NUM_COLS].corr("spearman") - fake[NUM_COLS].corr("spearman")).abs().values
    return float(np.nanmean(diff[np.triu_indices(len(NUM_COLS), k=1)]))


def bounds_excess(real: pd.DataFrame, fake: pd.DataFrame) -> float:
    cols = ["MORTDUE", "VALUE", "CLAGE", "DEBTINC", "LOAN"]      # variables continues
    lo, hi = real[cols].min(), real[cols].max()
    at = lambda d: ((d[cols] <= lo) | (d[cols] >= hi)).sum().sum() / d[cols].notna().sum().sum()
    return float(at(fake) - at(real))


def detectability(R: np.ndarray, F: np.ndarray) -> float:
    n = min(len(R), len(F))
    rng = np.random.default_rng(SEED)
    X = np.vstack([R[rng.choice(len(R), n, replace=False)], F[rng.choice(len(F), n, replace=False)]])
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, tree_method="hist",
                        random_state=SEED, n_jobs=1)
    return float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())


def privacy(T: np.ndarray, H: np.ndarray, F: np.ndarray) -> dict[str, float]:
    nn = NearestNeighbors(n_neighbors=1).fit(T)
    d_fake = nn.kneighbors(F)[0].ravel()
    d_hold = nn.kneighbors(H)[0].ravel()
    return {
        "Rapport DCR": float(np.median(d_fake) / np.median(d_hold)),
        "Quasi-copies (%)": float(100 * (d_fake < np.percentile(d_hold, 5)).mean()),
        "Copies exactes (%)": float(100 * (d_fake < 1e-9).mean()),
    }


def quality_report(train: pd.DataFrame, holdout: pd.DataFrame, fake: pd.DataFrame,
                   prep: Preprocessor) -> dict[str, float]:
    """`train`, `holdout`, `fake` : défauts au format brut ; `prep` : prétraitement ajusté
    sur toutes les demandes d'entraînement."""
    T, H, F = prep.transform(train), prep.transform(holdout), prep.transform(fake)
    return {
        "KS moyen": ks_mean(train, fake),
        "TVD moyenne": tvd_mean(train, fake),
        "Écart corrélations": correlation_gap(train, fake),
        "Excès aux bornes (%)": 100 * bounds_excess(train, fake),
        "Détectabilité (ROC-AUC)": detectability(T, F),
        **privacy(T, H, F),
    }
