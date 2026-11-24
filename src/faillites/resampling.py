"""Étude 2 — jeu d'entraînement de chaque méthode (même logique que src.resampling).

SMOTE-NC : les 64 ratios (transformés) sont interpolés ; les indicateurs « manquant »
prennent la valeur majoritaire des voisins.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from src.faillites.features import Prep
from src.faillites.generators import GENERATORS
from src.resampling import n_to_add

NO_RESAMPLING = ("Aucune", "Pondération")


def smote_blocks(X: pd.DataFrame, y: np.ndarray, prep: Prep, ratio: float, seed: int):
    N, C = prep.blocks(X)
    n_pos = int((y == 1).sum())
    if C.shape[1]:
        sampler = SMOTENC(categorical_features=list(range(prep.n_num, prep.n_num + C.shape[1])),
                          sampling_strategy=ratio, k_neighbors=min(5, n_pos - 1), random_state=seed)
    else:
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(sampling_strategy=ratio, k_neighbors=min(5, n_pos - 1), random_state=seed)
    Z, y_res = sampler.fit_resample(np.hstack([N, C]), y)
    return Z[:, :prep.n_num], Z[:, prep.n_num:].round().astype(int), y_res


def build_training_set(method: str, dose, X: pd.DataFrame, y: np.ndarray, prep: Prep, seed: int,
                       synth: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    A = prep.transform(X)
    if method in NO_RESAMPLING:
        return A, y
    if method == "Sous-échantillonnage":
        ratio = min(1.0, max(n_pos / n_neg, 1.0 / float(dose)))
        return RandomUnderSampler(sampling_strategy=ratio, random_state=seed).fit_resample(A, y)
    n_add = n_to_add(dose, n_pos, n_neg)
    if n_add == 0:
        return A, y
    ratio = (n_pos + n_add) / n_neg
    if method == "Duplication":
        return RandomOverSampler(sampling_strategy=ratio, random_state=seed).fit_resample(A, y)
    if method == "SMOTE":
        N, C, y_res = smote_blocks(X, y, prep, ratio, seed)
        return prep.design(N, C), y_res
    if method in GENERATORS:
        if synth is None or len(synth) < n_add:
            raise ValueError(f"{method} : pas assez de faillites synthétiques ({n_add} demandées)")
        return np.vstack([A, prep.transform(synth.iloc[:n_add])]), np.r_[y, np.ones(n_add, dtype=y.dtype)]
    raise ValueError(method)
