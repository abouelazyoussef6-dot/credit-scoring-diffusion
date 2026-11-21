"""Prétraitement des demandes de prêt, sans fuite de données.

Deux étapes bien séparées :

1. Transformations par règle (aucun apprentissage, donc aucun risque de fuite) :
   - indicateurs de valeur manquante. Dans ces données, l'absence d'une information est
     elle-même un signal : quand le ratio dette/revenu n'est pas renseigné, le taux de
     défaut est sept fois plus élevé. On garde donc l'information « était manquant »
     dans une colonne à part, avant d'imputer ;
   - LTV = (prêt + hypothèque restante) / valeur du bien : le ratio d'endettement sur le
     bien, variable classique du crédit immobilier ;
   - logarithme des montants (prêt, hypothèque, valeur), très asymétriques ;
   - modalité « Inconnu » pour les variables catégorielles non renseignées.

2. Transformations apprises, ajustées sur les VRAIES demandes d'entraînement uniquement,
   puis appliquées telles quelles à la validation, au test et aux données synthétiques :
   - imputation des valeurs numériques manquantes par la médiane ;
   - standardisation (moyenne 0, écart-type 1).

Le résultat est une matrice de 30 colonnes : 11 variables numériques, 10 modalités
(encodage one-hot du motif et de la profession) et 9 indicateurs de valeur manquante.
"""
import numpy as np
import pandas as pd

from src.config import (CAT_COLS, DEV_FILE, MISSING_LABEL, MONEY_COLS, NUM_COLS, RAW_COLS,
                        TARGET, TEST_FILE)

NUM_FEATURES = NUM_COLS + ["LTV"]
IND_COLS = [c for c in NUM_COLS if c != "LOAN"]          # LOAN est toujours renseigné
CATEGORIES = {
    "REASON": ["DebtCon", "HomeImp", MISSING_LABEL],
    "JOB": ["Mgr", "Office", "Other", "ProfExe", "Sales", "Self", MISSING_LABEL],
}
FEATURES = (NUM_FEATURES
            + [f"{c}={v}" for c in CAT_COLS for v in CATEGORIES[c]]
            + [f"{c}_manquant" for c in IND_COLS])
N_NUM, N_CAT_BLOCK = len(NUM_FEATURES), len(CAT_COLS) + len(IND_COLS)


def numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Variables numériques après transformations par règle (valeurs manquantes conservées)."""
    num = X[NUM_COLS].astype(float).copy()
    num["LTV"] = (num["LOAN"] + num["MORTDUE"]) / num["VALUE"].where(num["VALUE"] > 0)
    for c in MONEY_COLS:
        num[c] = np.log1p(num[c])
    return num[NUM_FEATURES]


class Preprocessor:
    """Imputation et standardisation ajustées sur l'entraînement ; encodage par règle."""

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        num = numeric_frame(X)
        self.medians_ = num.median()
        filled = num.fillna(self.medians_)
        self.means_, self.stds_ = filled.mean(), filled.std(ddof=0).replace(0, 1.0)
        return self

    def blocks(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(bloc numérique standardisé, bloc catégoriel codé par des entiers).

        Le bloc catégoriel contient le motif, la profession et les indicateurs de valeur
        manquante. Cette forme intermédiaire sert à SMOTE-NC, qui interpole les variables
        numériques mais recopie les modalités catégorielles."""
        num = numeric_frame(X)
        N = ((num.fillna(self.medians_) - self.means_) / self.stds_).to_numpy()
        cats = [X[c].where(X[c].isin(CATEGORIES[c][:-1]), MISSING_LABEL)
                .map({v: i for i, v in enumerate(CATEGORIES[c])}).to_numpy() for c in CAT_COLS]
        inds = [X[c].isna().astype(int).to_numpy() for c in IND_COLS]
        return N, np.column_stack(cats + inds).astype(int)

    @staticmethod
    def design(N: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Matrice finale des classifieurs : numériques, modalités one-hot, indicateurs."""
        parts = [N]
        for j, c in enumerate(CAT_COLS):
            parts.append(np.eye(len(CATEGORIES[c]))[C[:, j]])
        parts.append(C[:, len(CAT_COLS):].astype(float))
        return np.hstack(parts)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.design(*self.blocks(X))

    def inverse_blocks(self, N: np.ndarray, C: np.ndarray) -> pd.DataFrame:
        """Ramène des blocs (par exemple créés par SMOTE-NC) au format brut des demandes."""
        num = pd.DataFrame(N, columns=NUM_FEATURES) * self.stds_ + self.means_
        for c in MONEY_COLS:
            num[c] = np.expm1(num[c])
        X = num[NUM_COLS].copy()
        for j, c in enumerate(CAT_COLS):
            labels = np.array(CATEGORIES[c], dtype=object)[C[:, j]]
            X[c] = pd.Series(labels).where(labels != MISSING_LABEL)
        for j, c in enumerate(IND_COLS):
            X.loc[C[:, len(CAT_COLS) + j] == 1, c] = np.nan
        return X[RAW_COLS]


def load_dev() -> pd.DataFrame:
    """Jeu de développement : variables brutes, cible et colonnes de plis."""
    return pd.read_csv(DEV_FILE)


def load_test() -> pd.DataFrame:
    """Jeu de test. À n'appeler que depuis l'évaluation finale et l'interprétabilité."""
    return pd.read_csv(TEST_FILE)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[RAW_COLS], df[TARGET]
