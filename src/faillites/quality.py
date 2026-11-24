"""Étude 2 — qualité des faillites synthétiques (mêmes mesures que src.quality).

KS et corrélations de Spearman ne dépendent que des rangs : elles se calculent sur les
ratios bruts. Détectabilité et distances (DCR) se calculent dans l'espace des
classifieurs (prétraitement ajusté sur l'entraînement).
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.faillites.config import FEATURES
from src.faillites.features import Prep
from src.quality import detectability, privacy


def ks_mean(real: pd.DataFrame, fake: pd.DataFrame) -> float:
    return float(np.mean([ks_2samp(real[c].dropna(), fake[c].dropna()).statistic
                          for c in FEATURES if real[c].notna().sum() > 1 and fake[c].notna().sum() > 1]))


def missing_gap(real: pd.DataFrame, fake: pd.DataFrame) -> float:
    cols = [c for c in FEATURES if real[c].isna().mean() >= 0.01]
    return float(np.mean([abs(real[c].isna().mean() - fake[c].isna().mean()) for c in cols])) if cols else 0.0


def correlation_gap(real: pd.DataFrame, fake: pd.DataFrame) -> float:
    diff = (real[FEATURES].corr("spearman") - fake[FEATURES].corr("spearman")).abs().values
    return float(np.nanmean(diff[np.triu_indices(len(FEATURES), k=1)]))


def quality_report(train: pd.DataFrame, holdout: pd.DataFrame, fake: pd.DataFrame, prep: Prep) -> dict:
    T, H, F = prep.transform(train), prep.transform(holdout), prep.transform(fake)
    ks, mg = ks_mean(train, fake), missing_gap(train, fake)
    return {"KS moyen": ks, "Écart manquants": mg, "Écart de distribution": (ks + mg) / 2,
            "Écart corrélations": correlation_gap(train, fake),
            "Détectabilité (ROC-AUC)": detectability(T, F), **privacy(T, H, F)}
