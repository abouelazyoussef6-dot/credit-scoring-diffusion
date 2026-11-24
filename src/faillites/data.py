"""Étude 2 — chargement, contrôle et découpage des données (premier traitement appliqué).

1. lecture du fichier ARFF d'origine (valeurs manquantes notées « ? ») ;
2. contrôle : 10 503 entreprises et 495 faillites attendues ;
3. suppression des 87 lignes en double exact (mêmes 64 ratios, même issue) : une même
   entreprise ne doit pas se retrouver à la fois à l'entraînement et au test ;
4. découpage stratifié : 20 % de test, mis de côté ; 80 % de développement avec les plis
   de validation croisée (5 plis × 3 répétitions), enregistrés dans le fichier.

Usage : python -m src.faillites.data
"""
import io

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.faillites.config import (DEV_FILE, EXPECTED_BANKRUPT, EXPECTED_ROWS, FEATURES, N_FOLDS,
                                  N_REPEATS, RAW_FILE, SEED, TARGET, TEST_FILE, TEST_FRACTION)


def read_raw() -> pd.DataFrame:
    lines = RAW_FILE.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().lower() == "@data")
    df = pd.read_csv(io.StringIO("\n".join(lines[start + 1:])), header=None, na_values="?")
    df.columns = FEATURES + [TARGET]
    df[TARGET] = df[TARGET].astype(int)
    return df


def load_dev() -> pd.DataFrame:
    return pd.read_csv(DEV_FILE)


def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_FILE)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[TARGET]


def main() -> None:
    raw = read_raw()
    assert len(raw) == EXPECTED_ROWS and raw[TARGET].sum() == EXPECTED_BANKRUPT, "fichier inattendu"
    df = raw.drop_duplicates().reset_index(drop=True)
    print(f"Données d'origine : {len(raw)} entreprises, {raw[TARGET].sum()} faillites "
          f"({raw[TARGET].mean():.2%}) ; doublons exacts retirés : {len(raw) - len(df)}")

    dev, test = train_test_split(df, test_size=TEST_FRACTION, stratify=df[TARGET], random_state=SEED)
    dev, test = dev.reset_index(drop=True), test.reset_index(drop=True)
    for r in range(N_REPEATS):
        folds = np.zeros(len(dev), dtype=int)
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + r)
        for k, (_, va) in enumerate(cv.split(dev, dev[TARGET])):
            folds[va] = k
        dev[f"fold_r{r}"] = folds
    dev.to_csv(DEV_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)
    for name, d in [("développement", dev), ("test", test)]:
        print(f"→ {name:<14} {len(d):>6} entreprises | {int(d[TARGET].sum()):>4} faillites "
              f"({d[TARGET].mean():.2%})")


if __name__ == "__main__":
    main()
