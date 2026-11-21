# Défauts synthétiques par modèles de diffusion pour le scoring de crédit

Projet de Machine Learning — *Génération de données tabulaires synthétiques par modèles de diffusion pour le
traitement du déséquilibre des classes : application au scoring de crédit.*

**Question :** des défauts de paiement synthétiques produits par un modèle de diffusion (TabDDPM) aident-ils un
modèle de scoring à mieux repérer les vrais défauts, par rapport aux méthodes classiques (pondération,
sous-échantillonnage, duplication, SMOTE), à d'autres générateurs profonds (TVAE, CTGAN) et à l'absence de
rééquilibrage ?

**Application en ligne : https://scoring-credit-diffusion.streamlit.app** · Code : https://github.com/abouelazyoussef6-dot/credit-scoring-diffusion

## Résultats en bref

**Modèles (validation croisée, 15 découpages, sans rééquilibrage)** — la comparaison baseline / modèles avancés :

| Classifieur | PR-AUC | ROC-AUC | KS |
|---|---|---|---|
| Régression logistique (baseline) | 0,791 | 0,907 | 0,69 |
| XGBoost | 0,908 | 0,966 | 0,83 |
| LightGBM | 0,917 | 0,969 | 0,85 |

**Qualité des défauts synthétiques** (moyenne sur 5 plis ; la ligne de contrôle donne ce qu'obtiennent de vrais
défauts jamais vus) :

| Méthode | Écart de distribution (KS) | Écart de corrélations | Détectabilité (idéal 0,5) | Rapport DCR (idéal 1) | Quasi-copies | Performance conservée sans aucun vrai défaut (LogReg / XGBoost / LightGBM) |
|---|---|---|---|---|---|---|
| Vrais défauts (contrôle) | 0,070 | 0,078 | 0,50 | 1,03 | 6 % | 100 % |
| SMOTE-NC | 0,119 | 0,034 | 0,92 | 0,58 | 33 % | 97 / 90 / 91 % |
| TVAE | 0,252 | 0,094 | 0,99 | 0,93 | 2 % | 50 / 59 / 62 % |
| CTGAN | 0,179 | 0,161 | 0,97 | 1,70 | 0 % | 88 / 77 / 80 % |
| **TabDDPM** | **0,045** | 0,052 | **0,85** | **1,07** | 2 % | **98 / 93 / 91 %** |

**Utilité pour le scoring** (gain de PR-AUC par rapport à l'absence de rééquilibrage, meilleure dose de chaque méthode) :

| Méthode | LogReg | XGBoost | LightGBM | LightGBM sur le test |
|---|---|---|---|---|
| Pondération des classes | −0,003 | +0,005 | +0,003 | +0,005 |
| Sous-échantillonnage | −0,003 | −0,029 * | −0,029 * | — |
| Duplication | −0,003 | +0,004 | +0,004 | +0,007 |
| SMOTE-NC | −0,003 | +0,004 | +0,004 | — |
| TVAE | −0,014 * | −0,012 * | −0,008 | +0,002 |
| CTGAN | −0,018 | +0,001 | +0,003 | +0,005 |
| **TabDDPM** | −0,005 | +0,004 | **+0,004** | **+0,013** |

\* écart significatif après correction de Holm (p < 0,05). Aucun gain positif n'est significatif en validation croisée.

Ce qu'il faut retenir :

1. **Le modèle de diffusion produit les meilleurs défauts synthétiques** : distributions et corrélations les plus
   proches des vrais défauts, sans recopie des clients réels, alors que SMOTE reste collé aux points d'origine
   (un tiers de quasi-copies) et que CTGAN perd les liens entre variables.
2. **Le gain en prédiction est réel mais petit.** LightGBM + TabDDPM est la meilleure combinaison en validation
   croisée (0,921 contre 0,917) et sur le test (0,916 contre 0,904 ; écart +0,013, IC à 95 % [+0,002 ; +0,022]).
   L'écart n'est pas significatif en validation croisée : c'est un signal encourageant, pas une preuve.
3. **Un générateur mal ajusté fait perdre de la performance** (TVAE), d'où l'intérêt de mesurer la qualité des
   données synthétiques avant de les utiliser.
4. **Sous fort déséquilibre** (1 à 9 % de défauts à l'entraînement), TabDDPM et CTGAN apportent aux modèles à base
   d'arbres un gain modeste mais régulier (+0,005 à +0,02), ce que SMOTE ne fait pas.
5. **Rééquilibrer déplace surtout le seuil de décision** : au seuil 0,5 le rappel de la régression logistique passe
   de 0,63 à 0,80, mais la PR-AUC ne bouge pas. On obtient le même effet en choisissant le seuil.

**Modèle final** (règle fixée à l'avance : le plus simple quand aucun autre n'est significativement meilleur) :
LightGBM sans rééquilibrage, probabilités calibrées. Sur le test : PR-AUC 0,904 [0,876 ; 0,928], ROC-AUC 0,961,
KS 0,82. Au seuil choisi sur le développement (coût d'un défaut accepté = 5 × coût d'un bon payeur refusé) :
218 défauts refusés sur 238 (rappel 92 %), 96 bons payeurs refusés à tort sur 954.

**Interprétabilité** : les variables décisives sont le ratio dette / revenu (et surtout le fait qu'il ne soit pas
renseigné), les retards de paiement, l'ancienneté de l'historique de crédit et les incidents majeurs. Le modèle
entraîné avec des défauts TabDDPM s'appuie sur les mêmes variables (corrélation des importances : 0,99).

Figures : `reports/figures/` — tableaux : `reports/tables/`.

## Données

**HMEQ (Home Equity)** : 5 960 demandes de prêt garanti par un bien immobilier, dont 1 189 se sont terminées
par un défaut ou un retard grave (19,9 %). Jeu de données de référence en scoring de crédit.

- Source : Baesens, Rösch, Scheule, *Credit Risk Analytics* (Wiley, 2016) — http://www.creditriskanalytics.net
- Copie utilisée : https://www.kaggle.com/datasets/ajay1735/hmeq-data (fichier `hmeq.csv`, 0,4 Mo)

| Variable | Signification | Type | Manquants |
|---|---|---|---|
| `BAD` | 1 = défaut ou retard grave, 0 = prêt remboursé (**cible**) | binaire | 0 % |
| `LOAN` | Montant du prêt demandé | numérique | 0 % |
| `MORTDUE` | Capital restant dû sur l'hypothèque existante | numérique | 9 % |
| `VALUE` | Valeur actuelle du bien | numérique | 2 % |
| `REASON` | Motif : regroupement de dettes (`DebtCon`) ou travaux (`HomeImp`) | catégorielle | 4 % |
| `JOB` | Catégorie professionnelle (6 modalités) | catégorielle | 5 % |
| `YOJ` | Ancienneté dans l'emploi (années) | numérique | 9 % |
| `DEROG` | Nombre d'incidents de paiement majeurs | comptage | 12 % |
| `DELINQ` | Nombre de lignes de crédit en retard de paiement | comptage | 10 % |
| `CLAGE` | Âge de la plus ancienne ligne de crédit (mois) | numérique | 5 % |
| `NINQ` | Nombre de demandes de crédit récentes | comptage | 9 % |
| `CLNO` | Nombre de lignes de crédit | comptage | 4 % |
| `DEBTINC` | Ratio dette / revenu (%) | numérique | 21 % |

Préparation des données :

| Étape | Script | Ce qui est fait |
|---|---|---|
| Vérification | `src/download_data.py` | 5 960 lignes, 13 colonnes, 1 189 défauts attendus |
| Découpage | `src/split_data.py` | test : 20 % stratifié (1 192 demandes, 238 défauts), mis de côté ; développement : 4 768 demandes, 951 défauts ; 5 plis stratifiés × 3 répétitions |
| Variables | `src/features.py` | log(1 + x) des montants, ratio d'endettement sur le bien (LTV), imputation par la médiane, indicateurs de valeur manquante, one-hot, standardisation — le tout appris sur l'entraînement seul |

## Structure du dépôt

```
credit-scoring-diffusion/
├── run_all.py                  # reproduit tout le projet, étape par étape
├── requirements.txt            # environnement complet (versions figées)
├── data/
│   ├── LISEZMOI.md             # source des données et traitements effectués
│   ├── raw/hmeq.csv            # jeu de données (0,4 Mo)
│   └── processed/              # découpages et défauts synthétiques en cache (régénérés par script)
├── notebooks/
│   └── 01_eda.ipynb            # analyse exploratoire, exécutée, sur le jeu de développement
├── src/
│   ├── config.py               # chemins, graine, protocole d'évaluation, dictionnaire des variables
│   ├── download_data.py        # 1. récupération et vérification des données
│   ├── split_data.py           # 2. jeu de test + plis de validation croisée
│   ├── features.py             #    prétraitement sans fuite
│   ├── metrics.py              #    PR-AUC, ROC-AUC, KS, coût, tests statistiques
│   ├── models.py               #    régression logistique, XGBoost, LightGBM
│   ├── tuning.py               # 3. hyperparamètres des classifieurs
│   ├── generators.py           #    TVAE, CTGAN, TabDDPM (données mixtes, valeurs manquantes, cache)
│   ├── quality.py              #    fidélité et confidentialité des données synthétiques
│   ├── ddpm_tuning.py          # 4. réglage de TabDDPM : fidélité contre mémorisation
│   ├── resampling.py           #    construction du jeu d'entraînement de chaque méthode
│   ├── experiment.py           # 5. comparaison de toutes les méthodes (15 découpages)
│   ├── analysis.py             # 6. tableaux, tests statistiques, figures
│   ├── synthetic_quality.py    # 7. qualité des défauts synthétiques
│   ├── rarity.py               # 8. étude de rareté (fort déséquilibre)
│   ├── final_evaluation.py     # 9. évaluation finale sur le test (une seule fois)
│   ├── interpretability.py     # 10. SHAP, importance par permutation, analyse des erreurs
│   └── build_app_data.py       # 11. fichiers de l'application
├── app/
│   ├── streamlit_app.py        # application de démonstration
│   ├── requirements.txt        # dépendances de l'application seule (déploiement)
│   └── data/                   # petits fichiers nécessaires à l'application
├── models/
│   └── final_model.joblib      # modèle final, prétraitement et seuils
└── reports/
    ├── figures/                # toutes les figures
    └── tables/                 # tous les tableaux de résultats (CSV)
```

## Installation

Python **3.11** est requis (la bibliothèque synthcity, qui fournit TabDDPM, ne fonctionne pas avec Python 3.13).

```bash
git clone https://github.com/abouelazyoussef6-dot/credit-scoring-diffusion.git
cd credit-scoring-diffusion
python -m venv .venv            # sous Windows : py -3.11 -m venv .venv
source .venv/bin/activate       # sous Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Le fichier `data/raw/hmeq.csv` est fourni dans le dépôt. S'il manque, `python -m src.download_data` le
télécharge depuis Kaggle (jeton `kaggle.json` nécessaire) ; on peut aussi le déposer à la main dans `data/raw/`.

## Reproduire les résultats

```bash
python run_all.py               # tout, dans l'ordre (1 h 30 à 2 h 30 sur un ordinateur portable)
python run_all.py --from 5      # reprendre à partir d'une étape
python run_all.py --only 9      # une seule étape
```

Chaque étape peut aussi être lancée seule (`python -m src.experiment`, etc.). Les résultats sont enregistrés
au fur et à mesure : après une interruption, relancer la même commande reprend où le calcul s'était arrêté.
Les graines aléatoires sont fixées ; de légers écarts restent possibles d'une machine à l'autre pour les
générateurs (calcul flottant de PyTorch). Tous les résultats du dépôt (tableaux, figures, modèle) sont déjà
calculés : il n'est pas nécessaire de relancer les calculs pour utiliser l'application.

Pour explorer les données : `jupyter notebook notebooks/01_eda.ipynb`.

## Application

Version en ligne, sans installation : **https://scoring-credit-diffusion.streamlit.app**

En local :

```bash
streamlit run app/streamlit_app.py
```

L'application permet de saisir une demande de prêt (ou de partir d'un dossier réel du jeu de test) et d'obtenir
la probabilité de défaut, la décision proposée et son explication (SHAP) ; de régler le seuil de décision en
voyant ses conséquences ; de consulter la comparaison des méthodes et la qualité des défauts synthétiques ; et de
scorer un fichier CSV de demandes.

Elle n'utilise que des fichiers du dépôt (`models/`, `app/data/`, `reports/`) : ni téléchargement ni réentraînement.

**Déploiement sur Streamlit Community Cloud** (gratuit) : pousser le dépôt sur GitHub, puis sur
https://share.streamlit.io choisir le dépôt, la branche et le fichier `app/streamlit_app.py`. Dans
« Advanced settings », sélectionner Python 3.11. Les dépendances de `app/requirements.txt` sont installées
automatiquement.

## Protocole : comment les fuites de données sont évitées

1. **Découpage avant tout prétraitement** : le jeu de test (20 %) est mis de côté dès le départ et n'est utilisé
   qu'une seule fois, à la fin. L'analyse exploratoire ne porte que sur le jeu de développement.
2. **Prétraitement appris sur l'entraînement uniquement** : médianes d'imputation, moyennes et écarts-types sont
   recalculés sur la partie entraînement de chaque pli.
3. **Générateurs entraînés sur les défauts d'entraînement uniquement**, pli par pli.
4. **Évaluation sur de vraies demandes uniquement** : aucune donnée synthétique en validation ni en test.
5. **Aucun choix fait en regardant le test** : hyperparamètres, doses, durée d'entraînement de TabDDPM, modèle
   final et seuil de décision sont fixés par validation croisée sur le jeu de développement.
6. **Comparaisons appariées** : toutes les méthodes sont évaluées sur les mêmes 15 découpages ; les écarts sont
   testés par le test t corrigé de Nadeau et Bengio, avec correction de Holm pour les comparaisons multiples.

## Limites

- **Un seul jeu de données**, de taille modeste (5 960 demandes) : les conclusions ne se généralisent pas
  automatiquement à d'autres portefeuilles de crédit.
- **Déséquilibre naturel modéré (20 %)** : le fort déséquilibre est obtenu en retirant de vrais défauts de
  l'entraînement (étude de rareté, 5 plis seulement, sans test statistique).
- **Gain non significatif** : sur le test, l'intervalle de l'écart TabDDPM − sans rééquilibrage exclut zéro, mais
  15 comparaisons y sont faites sans correction ; la validation croisée, elle, ne conclut pas.
- **TVAE et CTGAN sont utilisés avec leurs réglages par défaut** (300 époques) ; seul TabDDPM a fait l'objet d'un
  réglage de sa durée d'entraînement. Un réglage fin pourrait améliorer TVAE et CTGAN.
- **TabDDPM place trop de valeurs aux bornes** observées (3 % des valeurs), visible sur les histogrammes.
- **Valeurs manquantes très informatives** : elles reflètent le processus de collecte autant que le risque ;
  un tel modèle ne serait pas transposable tel quel.
- **Biais de sélection** : seules les demandes acceptées à l'époque figurent dans les données (pas d'inférence
  des refusés). Aucune dimension temporelle : le test est un tirage aléatoire, pas une période future.
- **Confidentialité** : les mesures de distance (DCR) sont des indicateurs, pas une garantie formelle
  (pas de confidentialité différentielle).

## Références

- Baesens, Rösch, Scheule (2016). *Credit Risk Analytics*. Wiley (jeu de données HMEQ).
- Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR.
- Xu et al. (2019). *Modeling Tabular Data using Conditional GAN* (CTGAN, TVAE). NeurIPS.
- Ho, Jain, Abbeel (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.
- Kotelnikov et al. (2023). *TabDDPM: Modelling Tabular Data with Diffusion Models*. ICML.
- Qian, Cebere, van der Schaar (2023). *Synthcity*. NeurIPS Datasets and Benchmarks.
- Chen, Guestrin (2016). *XGBoost*. KDD. — Ke et al. (2017). *LightGBM*. NeurIPS.
- Lundberg, Lee (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS.
- Nadeau, Bengio (2003). *Inference for the Generalization Error*. Machine Learning.
