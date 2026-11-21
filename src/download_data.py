"""Récupération du jeu de données HMEQ (Home Equity Loans).

5 960 demandes de prêt garanti par un bien immobilier (home equity), avec l'issue du prêt :
BAD = 1 si le client a fait défaut ou a été en retard grave, 0 s'il a remboursé.

Sources :
- Baesens, Rösch, Scheule, « Credit Risk Analytics » (Wiley, 2016), jeu de données mis à
  disposition pour l'enseignement : http://www.creditriskanalytics.net/datasets-private2.html
- copie sur Kaggle : https://www.kaggle.com/datasets/ajay1735/hmeq-data

Deux façons d'obtenir data/raw/hmeq.csv :
1. automatiquement, avec l'API Kaggle (fichier kaggle.json configuré, voir README) ;
2. à la main : télécharger hmeq.csv depuis l'une des deux pages et le placer dans data/raw/.

Le script vérifie ensuite que le fichier est bien celui attendu.

Usage : python -m src.download_data
"""
import subprocess
import sys

import pandas as pd

from src.config import RAW_COLS, RAW_DIR, RAW_FILE, TARGET

EXPECTED_ROWS, EXPECTED_BAD = 5960, 1189


def download() -> None:
    print("Téléchargement depuis Kaggle (ajay1735/hmeq-data)...")
    subprocess.run(["kaggle", "datasets", "download", "-d", "ajay1735/hmeq-data",
                    "--unzip", "-p", str(RAW_DIR)], check=True)


def verify() -> None:
    df = pd.read_csv(RAW_FILE)
    missing = [c for c in [TARGET] + RAW_COLS if c not in df.columns]
    n_rows, n_bad = len(df), int(df[TARGET].sum()) if TARGET in df else -1
    print(f"{n_rows} demandes de prêt | {df.shape[1]} colonnes | {n_bad} défauts ({n_bad / n_rows:.1%})")
    if missing or (n_rows, n_bad) != (EXPECTED_ROWS, EXPECTED_BAD):
        sys.exit(f"Le fichier ne correspond pas au jeu de données attendu (colonnes absentes : {missing}).")
    print("Fichier conforme.")


def main() -> None:
    if not RAW_FILE.exists():
        try:
            download()
        except Exception as e:
            sys.exit(f"Téléchargement automatique impossible ({e}).\n"
                     f"Télécharger hmeq.csv à la main et le placer dans {RAW_DIR}")
    verify()


if __name__ == "__main__":
    main()
