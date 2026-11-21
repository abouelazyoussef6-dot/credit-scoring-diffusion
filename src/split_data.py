"""Découpage des données, AVANT toute analyse et tout prétraitement.

1. Jeu de TEST = 20 % des demandes, tirées au hasard en respectant le taux de défaut
   (découpage stratifié). Il n'est utilisé qu'une seule fois, à la toute fin du projet
   (src.final_evaluation). Le jeu de données ne contient pas de date : un découpage
   chronologique n'est pas possible.

2. Jeu de DÉVELOPPEMENT = les 80 % restants. Avec environ 950 défauts, une unique
   validation serait trop bruitée : on utilise une validation croisée stratifiée à 5 plis,
   répétée 3 fois. Les plis sont tirés une fois pour toutes et enregistrés dans le fichier
   (colonnes fold_r0, fold_r1, fold_r2), afin que toutes les expériences comparent les
   méthodes sur exactement les mêmes découpages.

Les doublons exacts sont vérifiés : une même demande présente à la fois dans
l'entraînement et l'évaluation serait une fuite de données.

Usage : python -m src.split_data
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import (DEV_FILE, N_FOLDS, N_REPEATS, RAW_COLS, RAW_FILE, SEED, TARGET,
                        TEST_FILE, TEST_FRACTION)


def main() -> None:
    df = pd.read_csv(RAW_FILE)[[TARGET] + RAW_COLS]
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Doublons exacts supprimés : {n0 - len(df)}")

    dev, test = train_test_split(df, test_size=TEST_FRACTION, stratify=df[TARGET], random_state=SEED)
    dev, test = dev.reset_index(drop=True), test.reset_index(drop=True)

    for r in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + r)
        dev[f"fold_r{r}"] = -1
        for k, (_, idx) in enumerate(skf.split(dev, dev[TARGET])):
            dev.loc[idx, f"fold_r{r}"] = k

    dev.to_csv(DEV_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)
    for name, d in [("développement", dev), ("test", test)]:
        print(f"→ {name:<14} {len(d):>5} demandes | {int(d[TARGET].sum()):>4} défauts ({d[TARGET].mean():.1%})")
    per_fold = dev.groupby("fold_r0")[TARGET].agg(["size", "sum"])
    print("Plis de la répétition 0 (demandes, défauts) :",
          ", ".join(f"{int(s)}/{int(f)}" for s, f in per_fold.values))


if __name__ == "__main__":
    main()
