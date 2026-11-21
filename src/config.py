"""Paramètres globaux du projet : chemins, graine aléatoire, variables, protocole d'évaluation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTH_DIR = PROCESSED_DIR / "synthetic"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TAB_DIR = REPORTS_DIR / "tables"
MODELS_DIR = ROOT / "models"

RAW_FILE = RAW_DIR / "hmeq.csv"
DEV_FILE = PROCESSED_DIR / "dev.csv"    # 80 % : entraînement + validation croisée
TEST_FILE = PROCESSED_DIR / "test.csv"  # 20 % : évaluation finale, une seule fois

SEED = 42
TARGET = "BAD"            # 1 = défaut de paiement, 0 = prêt remboursé

# Protocole d'évaluation
TEST_FRACTION = 0.20
N_FOLDS = 5               # validation croisée stratifiée sur le jeu de développement
N_REPEATS = 3             # répétitions de la validation croisée (15 mesures par méthode)

# Variables d'une demande de prêt
NUM_COLS = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]
CAT_COLS = ["REASON", "JOB"]
RAW_COLS = NUM_COLS + CAT_COLS
INT_COLS = ["LOAN", "DEROG", "DELINQ", "NINQ", "CLNO"]        # valeurs entières
MONEY_COLS = ["LOAN", "MORTDUE", "VALUE"]                      # montants, très asymétriques
MISSING_LABEL = "Inconnu"

LABELS = {
    "LOAN": "Montant du prêt demandé",
    "MORTDUE": "Capital restant dû sur l'hypothèque",
    "VALUE": "Valeur actuelle du bien",
    "REASON": "Motif du prêt",
    "JOB": "Catégorie professionnelle",
    "YOJ": "Ancienneté dans l'emploi (années)",
    "DEROG": "Incidents de paiement majeurs",
    "DELINQ": "Lignes de crédit en retard de paiement",
    "CLAGE": "Âge de la plus ancienne ligne de crédit (mois)",
    "NINQ": "Demandes de crédit récentes",
    "CLNO": "Nombre de lignes de crédit",
    "DEBTINC": "Ratio dette / revenu (%)",
    "LTV": "Ratio (prêt + hypothèque) / valeur du bien",
}

for d in (RAW_DIR, PROCESSED_DIR, SYNTH_DIR, FIG_DIR, TAB_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
