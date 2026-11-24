# Données du projet

## Source

**HMEQ (Home Equity)** — demandes de prêt garanti par un bien immobilier.

- Origine : Baesens, B., Rösch, D., Scheule, H. (2016). *Credit Risk Analytics*. Wiley.
  http://www.creditriskanalytics.net
- Copie utilisée : https://www.kaggle.com/datasets/ajay1735/hmeq-data (fichier `hmeq.csv`)
- Contrôles à l'ouverture (`src/download_data.py`) : 5 960 lignes, 13 colonnes, 1 189 défauts (19,9 %),
  aucun doublon exact.

Le dictionnaire des variables figure dans le `README.md` à la racine du dépôt.

## Fichiers

| Fichier | Contenu | Produit par |
|---|---|---|
| `raw/hmeq.csv` | Données brutes, telles que publiées, non modifiées | téléchargement |
| `processed/test.csv` | Jeu de test : 1 192 demandes (238 défauts), mis de côté avant tout traitement | `src/split_data.py` |
| `processed/dev.csv` | Jeu de développement : 4 768 demandes (951 défauts) + colonnes `fold_r0`, `fold_r1`, `fold_r2` donnant le pli (0 à 4) de chaque demande pour les 3 répétitions de validation croisée | `src/split_data.py` |
| `processed/synthetic/` | Cache des défauts synthétiques générés (non versionné, recréé à la demande) | `src/generators.py` |

## Traitements effectués

1. **Découpage** (premier traitement, sur les données brutes) : 80 % développement / 20 % test, stratifié
   sur la cible, graine 42. Le jeu de test n'est utilisé qu'une fois, pour l'évaluation finale.
2. **Aucune ligne supprimée ni modifiée** dans les fichiers : les valeurs manquantes sont conservées telles
   quelles, car elles sont informatives (62 % de défauts quand le ratio dette / revenu manque, contre 9 %).
3. **Prétraitement à la volée** (`src/features.py`), appris sur la partie entraînement de chaque pli :
   - variable construite LTV = (LOAN + MORTDUE) / VALUE ;
   - log(1 + x) des montants (LOAN, MORTDUE, VALUE) ;
   - un indicateur « valeur manquante » pour chacune des 9 variables numériques incomplètes ;
   - imputation par la médiane, puis standardisation (moyenne et écart-type de l'entraînement) ;
   - encodage one-hot du motif et de la profession, « non renseigné » formant une modalité.
4. **Données synthétiques** (`src/generators.py`) : les générateurs sont entraînés uniquement sur les
   défauts de la partie entraînement de chaque pli. Les valeurs manquantes y sont représentées par une
   colonne « oui / non », puis restaurées après génération ; les comptages sont arrondis à l'entier, les
   montants de prêt à la centaine, et toutes les valeurs bornées au domaine observé.

## Étude 2 : faillites d'entreprises

**Polish Companies Bankruptcy** — Tomczak, S. (2016). UCI Machine Learning Repository,
https://doi.org/10.24432/C5F600, licence CC BY 4.0. Article associé : Zięba, Tomczak et Tomczak (2016),
*Expert Systems with Applications*, 58, 93-101.

- Fichier utilisé : `raw/polish_3year.arff` (3ᵉ année : ratios d'un exercice, statut de faillite trois ans plus
  tard), tel que publié, non modifié.
- Contrôles à l'ouverture (`src/faillites/data.py`) : 10 503 entreprises, 64 ratios (X1 à X64), 495 faillites
  (4,7 %). Valeurs manquantes notées « ? » dans le fichier.

| Fichier | Contenu | Produit par |
|---|---|---|
| `processed/faillites_test.csv` | Jeu de test : 2 084 entreprises (99 faillites), mis de côté avant tout traitement | `src/faillites/data.py` |
| `processed/faillites_dev.csv` | Développement : 8 332 entreprises (394 faillites) + colonnes `fold_r0..2` | `src/faillites/data.py` |
| `processed/synthetic/faillites/` | Cache des faillites synthétiques (non versionné) | `src/faillites/generators.py` |

Traitements :

1. **Suppression de 87 doublons exacts** (mêmes 64 ratios, même issue), pour qu'une même entreprise ne se
   retrouve pas à la fois à l'entraînement et au test ; puis découpage stratifié 80 / 20, graine 42.
2. **Prétraitement à la volée** (`src/faillites/features.py`), appris sur l'entraînement de chaque pli :
   logarithme signé sign(x)·log(1 + |x|) des 64 ratios, indicateur « valeur manquante » pour chaque ratio manquant
   dans au moins 1 % des entreprises, imputation par la médiane, standardisation.
3. **Faillites synthétiques** : générateurs entraînés sur les seules faillites d'entraînement du pli, sur
   l'échelle du logarithme signé ; retour à l'échelle d'origine, valeurs bornées au domaine observé, valeurs
   manquantes restaurées d'après l'indicateur généré.
