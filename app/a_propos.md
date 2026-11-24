### Le projet

**Équipe :** Abouelaz Youssef, Sissi Youssef, Ettahiri Mohamed Yassine, Achbani Ismail
(École Centrale Casablanca).

**Sujet :** génération de données tabulaires synthétiques par modèles de diffusion pour le traitement du
déséquilibre des classes — application au scoring de crédit.

**Données :** HMEQ (*Home Equity*), 5 960 demandes de prêt garanti par un bien immobilier, dont 19,9 % se sont
terminées par un défaut. Source : Baesens, Rösch et Scheule, *Credit Risk Analytics* (creditriskanalytics.net) ;
copie sur Kaggle (`ajay1735/hmeq-data`).

**Méthodes comparées :** aucun rééquilibrage, pondération des classes, sous-échantillonnage, duplication,
SMOTE-NC, et trois générateurs profonds : TVAE (auto-encodeur variationnel), CTGAN (réseau antagoniste) et
TabDDPM (modèle de diffusion). Chaque méthode est testée avec trois classifieurs : régression logistique
(modèle simple de référence), XGBoost et LightGBM.

### Ce que montrent les résultats

1. **TabDDPM produit les défauts synthétiques les plus fidèles.** Écart moyen de distribution de 0,045 (0,12 pour
   SMOTE, 0,18 pour CTGAN, 0,25 pour TVAE), corrélations entre variables préservées, et pas de recopie des
   clients réels (rapport de distances 1,07, contre 0,58 pour SMOTE dont un tiers des points sont des
   quasi-copies). Un modèle entraîné **sans aucun vrai défaut**, uniquement avec ceux de TabDDPM, conserve 91 à
   98 % de la performance.
2. **Le gain en prédiction est faible.** En validation croisée, la meilleure combinaison est LightGBM + TabDDPM
   (PR-AUC 0,921 contre 0,917 sans rééquilibrage), mais l'écart n'est pas statistiquement significatif. Sur le
   jeu de test, la même combinaison obtient 0,916 contre 0,904 (écart + 0,013, intervalle à 95 % : + 0,002 à
   + 0,022) : un signal encourageant, pas une preuve.
3. **Les générateurs mal ajustés dégradent le modèle.** TVAE fait baisser la PR-AUC avec les trois classifieurs.
   La fidélité des données synthétiques se vérifie avant de s'en servir.
4. **Quand les défauts sont rares (1 à 9 %)**, TabDDPM et CTGAN apportent un gain modeste mais régulier aux
   modèles à base d'arbres (+ 0,005 à + 0,02 de PR-AUC), là où SMOTE n'apporte rien.
5. **Rééquilibrer déplace surtout le seuil.** Avec un seuil fixe de 0,5, toutes les méthodes augmentent le
   rappel ; mais on obtient le même effet, sans donnée synthétique, en choisissant le seuil.

6. **Étude 2 — faillites d'entreprises (4,7 % de défauts naturels, 10 503 entreprises, 64 ratios).** Le même
   protocole confirme ces constats : TabDDPM produit de loin les faillites synthétiques les plus fidèles (écart
   0,025 contre 0,16 à 0,22), sans recopie, et c'est le seul à aider LightGBM (+0,010 de PR-AUC, non
   significatif). Voir l'onglet « Étude 2 ».

### Le modèle de cette application

LightGBM entraîné sur les données réelles, sans rééquilibrage : la règle fixée à l'avance retient le modèle le
plus simple quand aucun autre n'est significativement meilleur en validation croisée. Ses probabilités sont
calibrées (méthode de Platt). Sur le jeu de test : PR-AUC 0,904, ROC-AUC 0,961, KS 0,82. Au seuil retenu
(coût d'un défaut accepté = 5 fois celui d'un bon payeur refusé), il refuse 92 % des défauts.

### Garanties de méthode

Jeu de test mis de côté avant tout traitement et utilisé une seule fois ; prétraitement et générateurs appris
sur les données d'entraînement uniquement ; évaluation sur de vraies demandes uniquement ; tous les choix
(hyperparamètres, doses, modèle final, seuil) faits par validation croisée.

### Limites

Deux jeux de données de taille moyenne (le principal est ancien) ; valeurs manquantes très informatives (un ratio dette /
revenu non renseigné suffit presque à prédire le défaut), ce qui reflète le processus de collecte autant que le
risque ; pas de dimension temporelle ; les demandes refusées à l'époque sont absentes des données. Cet outil
est une démonstration pédagogique, pas un système de décision de crédit.
