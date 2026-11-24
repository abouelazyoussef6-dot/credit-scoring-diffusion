"""Étude 2 — portefeuille de crédit aux entreprises à fort déséquilibre naturel.

Données : « Polish Companies Bankruptcy » (UCI, Tomczak 2016, CC BY 4.0), fichier de la
3e année : ratios financiers d'une année donnée et faillite (ou non) dans les 3 ans qui
suivent. 10 503 entreprises, 495 faillites (4,7 %), 64 ratios financiers.

Lecture « crédit » : pour une banque qui prête à une entreprise, la faillite de
l'emprunteur est l'événement de défaut. Prédire la faillite à partir des comptes est
exactement le scoring de crédit des entreprises (comme les scores d'Altman).

Tout ce qui est propre à cette étude est rangé à part : données, tableaux, figures et
cache des données synthétiques, pour ne rien modifier de l'étude 1 (HMEQ).
"""
from src.config import FIG_DIR, MODELS_DIR, PROCESSED_DIR, RAW_DIR, SYNTH_DIR, TAB_DIR

RAW_FILE = RAW_DIR / "polish_3year.arff"
DEV_FILE = PROCESSED_DIR / "faillites_dev.csv"
TEST_FILE = PROCESSED_DIR / "faillites_test.csv"
SYNTH = SYNTH_DIR / "faillites"
TAB = TAB_DIR / "faillites"
FIG = FIG_DIR / "faillites"
MODEL_FILE = MODELS_DIR / "faillites_model.joblib"

EXPECTED_ROWS, EXPECTED_BANKRUPT = 10503, 495
TARGET = "FAILLITE"             # 1 = faillite dans les 3 ans (défaut de l'emprunteur)
FEATURES = [f"X{i}" for i in range(1, 65)]

SEED = 42
TEST_FRACTION = 0.20
N_FOLDS, N_REPEATS = 5, 3
MISSING_FLAG_MIN = 0.01         # indicateur « manquant » pour les ratios manquants ≥ 1 % du temps

# Définitions officielles (UCI), traduites
LABELS = {
    "X1": "Résultat net / total actif",
    "X2": "Total dettes / total actif",
    "X3": "Fonds de roulement / total actif",
    "X4": "Actif circulant / dettes à court terme",
    "X5": "[(Trésorerie + titres CT + créances − dettes CT) / (charges d'exploitation − amortissements)] × 365",
    "X6": "Réserves (bénéfices non distribués) / total actif",
    "X7": "EBIT / total actif",
    "X8": "Capitaux propres comptables / total dettes",
    "X9": "Chiffre d'affaires / total actif",
    "X10": "Capitaux propres / total actif",
    "X11": "(Résultat brut + éléments exceptionnels + charges financières) / total actif",
    "X12": "Résultat brut / dettes à court terme",
    "X13": "(Résultat brut + amortissements) / chiffre d'affaires",
    "X14": "(Résultat brut + intérêts) / total actif",
    "X15": "(Total dettes × 365) / (résultat brut + amortissements)",
    "X16": "(Résultat brut + amortissements) / total dettes",
    "X17": "Total actif / total dettes",
    "X18": "Résultat brut / total actif",
    "X19": "Résultat brut / chiffre d'affaires",
    "X20": "(Stocks × 365) / chiffre d'affaires",
    "X21": "Chiffre d'affaires (n) / chiffre d'affaires (n−1)",
    "X22": "Résultat d'exploitation / total actif",
    "X23": "Résultat net / chiffre d'affaires",
    "X24": "Résultat brut (sur 3 ans) / total actif",
    "X25": "(Capitaux propres − capital social) / total actif",
    "X26": "(Résultat net + amortissements) / total dettes",
    "X27": "Résultat d'exploitation / charges financières",
    "X28": "Fonds de roulement / actif immobilisé",
    "X29": "Logarithme du total actif (taille)",
    "X30": "(Total dettes − trésorerie) / chiffre d'affaires",
    "X31": "(Résultat brut + intérêts) / chiffre d'affaires",
    "X32": "(Dettes courantes × 365) / coût des produits vendus",
    "X33": "Charges d'exploitation / dettes à court terme",
    "X34": "Charges d'exploitation / total dettes",
    "X35": "Résultat sur ventes / total actif",
    "X36": "Ventes totales / total actif",
    "X37": "(Actif circulant − stocks) / dettes à long terme",
    "X38": "Capitaux permanents / total actif",
    "X39": "Résultat sur ventes / chiffre d'affaires",
    "X40": "(Actif circulant − stocks − créances) / dettes à court terme",
    "X41": "Total dettes / ((résultat d'exploitation + amortissements) × 12/365)",
    "X42": "Résultat d'exploitation / chiffre d'affaires",
    "X43": "Rotation des créances + des stocks (jours)",
    "X44": "(Créances × 365) / chiffre d'affaires",
    "X45": "Résultat net / stocks",
    "X46": "(Actif circulant − stocks) / dettes à court terme",
    "X47": "(Stocks × 365) / coût des produits vendus",
    "X48": "EBITDA / total actif",
    "X49": "EBITDA / chiffre d'affaires",
    "X50": "Actif circulant / total dettes",
    "X51": "Dettes à court terme / total actif",
    "X52": "(Dettes à court terme × 365) / coût des produits vendus",
    "X53": "Capitaux propres / actif immobilisé",
    "X54": "Capitaux permanents / actif immobilisé",
    "X55": "Fonds de roulement (montant)",
    "X56": "(Chiffre d'affaires − coût des produits vendus) / chiffre d'affaires",
    "X57": "(Actif circulant − stocks − dettes CT) / (CA − résultat brut − amortissements)",
    "X58": "Coûts totaux / ventes totales",
    "X59": "Dettes à long terme / capitaux propres",
    "X60": "Chiffre d'affaires / stocks",
    "X61": "Chiffre d'affaires / créances",
    "X62": "(Dettes à court terme × 365) / chiffre d'affaires",
    "X63": "Chiffre d'affaires / dettes à court terme",
    "X64": "Chiffre d'affaires / actif immobilisé",
}

for d in (SYNTH, TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)
