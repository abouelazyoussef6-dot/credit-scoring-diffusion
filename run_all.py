"""Reproduit tout le projet, étape par étape.

    python run_all.py              # tout, dans l'ordre
    python run_all.py --from 5     # reprendre à partir de l'étape 5
    python run_all.py --only 9     # une seule étape

Chaque étape enregistre ses résultats au fur et à mesure : après une interruption,
relancer la même commande reprend là où le calcul s'était arrêté.
Durée totale indicative sur un ordinateur portable récent : 2 h à 3 h 30 (étude 2 : environ 1 h).
    python run_all.py --from 12    # l'étude 2 seule
"""
import argparse
import subprocess
import sys
import time

STEPS = [
    ("src.download_data", "Récupération et vérification du jeu de données HMEQ"),
    ("src.split_data", "Découpage : jeu de test + plis de validation croisée"),
    ("src.tuning", "Réglage des hyperparamètres des classifieurs"),
    ("src.ddpm_tuning", "Réglage de TabDDPM : fidélité contre mémorisation"),
    ("src.experiment", "Comparaison de toutes les méthodes en validation croisée"),
    ("src.analysis", "Tableaux, tests statistiques et figures"),
    ("src.synthetic_quality", "Qualité des défauts synthétiques"),
    ("src.rarity", "Étude de rareté (fort déséquilibre)"),
    ("src.final_evaluation", "Évaluation finale sur le jeu de test"),
    ("src.interpretability", "Interprétabilité (SHAP) et analyse des erreurs"),
    ("src.build_app_data", "Fichiers de l'application"),
    # Étude 2 : faillites d'entreprises (déséquilibre naturel fort, 4,7 %)
    ("src.faillites.data", "Étude 2 — contrôle et découpage des données de faillites"),
    ("src.faillites.build_eda_notebook", "Étude 2 — analyse exploratoire (notebook exécuté)"),
    ("src.faillites.tuning", "Étude 2 — réglage des classifieurs"),
    ("src.faillites.ddpm_tuning", "Étude 2 — réglage de TabDDPM"),
    ("src.faillites.experiment", "Étude 2 — comparaison de toutes les méthodes"),
    ("src.faillites.analysis", "Étude 2 — tableaux, tests statistiques et figures"),
    ("src.faillites.synthetic_quality", "Étude 2 — qualité des faillites synthétiques"),
    ("src.faillites.final_evaluation", "Étude 2 — évaluation finale sur le jeu de test"),
    ("src.faillites.interpretability", "Étude 2 — interprétabilité (SHAP)"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", type=int, default=1)
    parser.add_argument("--only", type=int)
    args = parser.parse_args()

    for i, (module, title) in enumerate(STEPS, start=1):
        if (args.only and i != args.only) or (not args.only and i < args.start):
            continue
        print(f"\n{'=' * 78}\nÉtape {i}/{len(STEPS)} — {title}  (python -m {module})\n{'=' * 78}", flush=True)
        t0 = time.time()
        if subprocess.run([sys.executable, "-m", module]).returncode != 0:
            sys.exit(f"Arrêt : l'étape {i} a échoué. Corriger puis relancer avec --from {i}")
        print(f"Étape {i} terminée en {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
