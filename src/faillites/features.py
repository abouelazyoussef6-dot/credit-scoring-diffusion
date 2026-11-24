"""Étude 2 — prétraitement des 64 ratios financiers, appris sur l'entraînement seul.

Les ratios ont des queues extrêmes (asymétrie médiane ≈ 50 ; valeurs jusqu'à 10^7) parce
qu'un dénominateur proche de zéro fait exploser un ratio. Étapes :

1. transformation logarithmique signée s(x) = signe(x) · log(1 + |x|) : strictement
   croissante (le classement des entreprises est inchangé, donc sans effet sur les
   arbres), elle ramène les valeurs extrêmes à une échelle raisonnable pour la
   régression logistique et pour les générateurs ;
2. indicateur « manquant » pour chaque ratio manquant dans au moins 1 % des entreprises
   d'entraînement ;
3. imputation par la médiane, puis standardisation (moyenne et écart-type de
   l'entraînement).
"""
import numpy as np
import pandas as pd

from src.faillites.config import FEATURES, MISSING_FLAG_MIN


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def slog_inv(z):
    return np.sign(z) * np.expm1(np.abs(z))


class Prep:
    def fit(self, X: pd.DataFrame) -> "Prep":
        Z = slog(X[FEATURES].astype(float))
        self.ind = [c for c in FEATURES if X[c].isna().mean() >= MISSING_FLAG_MIN]
        self.med = Z.median()
        filled = Z.fillna(self.med)
        self.mu, self.sd = filled.mean(), filled.std(ddof=0).replace(0, 1.0)
        return self

    @property
    def n_num(self) -> int:
        return len(FEATURES)

    @property
    def names(self) -> list[str]:
        return FEATURES + [f"{c}_manquant" for c in self.ind]

    def blocks(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Z = slog(X[FEATURES].astype(float)).fillna(self.med)
        N = ((Z - self.mu) / self.sd).to_numpy()
        C = X[self.ind].isna().astype(int).to_numpy()
        return N, C

    @staticmethod
    def design(N: np.ndarray, C: np.ndarray) -> np.ndarray:
        return np.hstack([N, C]).astype(float)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.design(*self.blocks(X))

    def inverse_blocks(self, N: np.ndarray, C: np.ndarray) -> pd.DataFrame:
        """Retour au format brut (utilisé pour mesurer la qualité des points SMOTE)."""
        Z = N * self.sd.to_numpy() + self.mu.to_numpy()
        out = pd.DataFrame(slog_inv(Z), columns=FEATURES)
        for j, c in enumerate(self.ind):
            out.loc[C[:, j].round() == 1, c] = np.nan
        return out
