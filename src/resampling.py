"""Construction du jeu d'entraînement pour chaque méthode de traitement du déséquilibre.

Toutes les méthodes reçoivent les mêmes vraies demandes d'entraînement et ne modifient QUE
l'entraînement : la validation et le test restent composés de vraies demandes.

Familles de méthodes :
- Aucune / Pondération : données inchangées (la pondération agit dans le classifieur) ;
- Sous-échantillonnage : on retire au hasard des bons payeurs ;
- Duplication : on recopie au hasard des défauts existants ;
- SMOTE : on crée un défaut entre un défaut et l'un de ses voisins. On utilise la variante
  SMOTE-NC, prévue pour les données mixtes : les variables numériques sont interpolées,
  les modalités (motif, profession, indicateurs de valeur manquante) sont celles de la
  majorité des voisins. Un SMOTE ordinaire créerait des clients « 0,4 fois cadre » ;
- Générateurs profonds : TVAE, CTGAN, TabDDPM (défauts synthétiques fournis par
  src.generators, puis prétraités exactement comme les vrais).

La « dose » règle la quantité ajoutée :
- un nombre m  → m défauts synthétiques ajoutés pour 1 vrai défaut ;
- "éq."        → équilibre parfait (autant de défauts que de bons payeurs), le réglage par
                 défaut de SMOTE dans la pratique ;
- pour le sous-échantillonnage, m = nombre de bons payeurs gardés par défaut.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from src.features import N_CAT_BLOCK, N_NUM, Preprocessor
from src.generators import GENERATORS

NO_RESAMPLING = ("Aucune", "Pondération")


def n_to_add(dose, n_pos: int, n_neg: int) -> int:
    """Nombre de défauts à ajouter, plafonné à l'équilibre parfait."""
    gap = max(n_neg - n_pos, 0)
    return gap if dose == "éq." else min(int(round(float(dose) * n_pos)), gap)


def build_training_set(method: str, dose, X: pd.DataFrame, y: np.ndarray, prep: Preprocessor,
                       seed: int, synth: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (matrice, cible) d'entraînement après application de la méthode.

    X : vraies demandes d'entraînement au format brut ; prep : prétraitement déjà ajusté sur X."""
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    A = prep.transform(X)

    if method in NO_RESAMPLING:
        return A, y

    if method == "Sous-échantillonnage":
        ratio = min(1.0, max(n_pos / n_neg, 1.0 / float(dose)))     # défauts / bons payeurs
        return RandomUnderSampler(sampling_strategy=ratio, random_state=seed).fit_resample(A, y)

    n_add = n_to_add(dose, n_pos, n_neg)
    if n_add == 0:
        return A, y
    ratio = (n_pos + n_add) / n_neg

    if method == "Duplication":
        return RandomOverSampler(sampling_strategy=ratio, random_state=seed).fit_resample(A, y)

    if method == "SMOTE":
        N, C = prep.blocks(X)
        sampler = SMOTENC(categorical_features=list(range(N_NUM, N_NUM + N_CAT_BLOCK)),
                          sampling_strategy=ratio, k_neighbors=min(5, n_pos - 1), random_state=seed)
        Z, y_res = sampler.fit_resample(np.hstack([N, C]), y)
        return prep.design(Z[:, :N_NUM], Z[:, N_NUM:].astype(int)), y_res

    if method in GENERATORS:
        if synth is None or len(synth) < n_add:
            raise ValueError(f"{method} : pas assez de défauts synthétiques ({n_add} demandés)")
        return (np.vstack([A, prep.transform(synth.iloc[:n_add])]),
                np.r_[y, np.ones(n_add, dtype=y.dtype)])

    raise ValueError(method)
