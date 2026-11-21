"""Prépare les petits fichiers dont l'application Streamlit a besoin (dossier app/data).

Ces fichiers sont versionnés dans le dépôt pour que l'application fonctionne sans
retélécharger le jeu de données ni réentraîner les modèles.

Usage : python -m src.build_app_data   (après src.final_evaluation)
"""
import pandas as pd

from src.config import RAW_COLS, ROOT, SEED, TAB_DIR, TARGET
from src.features import Preprocessor, load_dev, split_xy
from src.generators import GENERATORS, get_synthetic
from src.resampling import n_to_add
from src.synthetic_quality import smote_raw

OUT = ROOT / "app" / "data"
N_DEMO, N_SYNTH = 300, 900


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(TAB_DIR / "test_predictions.csv")

    # 1. Dossiers de démonstration : un échantillon du jeu de test (jamais vu à l'entraînement)
    pred.sample(N_DEMO, random_state=SEED)[RAW_COLS + [TARGET]].reset_index(drop=True).to_csv(
        OUT / "demo_clients.csv", index=False)
    # 2. Scores du modèle final sur tout le test (pour le réglage du seuil)
    pred[[TARGET, "proba"]].round(5).to_csv(OUT / "test_scores.csv", index=False)
    # 3. Exemple de fichier à scorer (format d'origine, sans la cible)
    pred.sample(100, random_state=SEED + 1)[RAW_COLS].to_csv(OUT / "exemple_a_scorer.csv", index=False)

    # 4. Échantillons de défauts : vrais (développement) et synthétiques (ceux de l'évaluation finale)
    dev = load_dev(); Xd, yd = split_xy(dev); y = yd.values
    defaults = Xd[y == 1]
    n_eq = n_to_add("éq.", int(y.sum()), int((y == 0).sum()))
    parts = [defaults.assign(méthode="Vrais défauts"),
             smote_raw(Xd, y, Preprocessor().fit(Xd), N_SYNTH, SEED).assign(méthode="SMOTE")]
    for g in GENERATORS:
        parts.append(get_synthetic(g, "final_s0", defaults, n_eq, 0).iloc[:N_SYNTH].assign(méthode=g))
    pd.concat(parts).round(3).to_csv(OUT / "synthetic_samples.csv", index=False)

    for f in sorted(OUT.glob("*.csv")):
        print(f"{f.name:<26} {f.stat().st_size / 1024:>6.0f} Ko")


if __name__ == "__main__":
    main()
